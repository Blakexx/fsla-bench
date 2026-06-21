"""The canonical-MQAR-grid sweep as a `fleet` Job.

This is the manager-side description of WHAT to run; `fleet` owns HOW (provision/control/sync). The
matching box-side run command lives in the mqar image (docker/mqar.Dockerfile sets FLEET_RUN_CMD to
`run_rla_sweep.py --config canonical_mqar_grid`, and FLEET_VERIFY_CMD to the fla-rola kernel check).
"""
import glob, importlib, json, os
from fleet import Job

GRID_TIERS = os.environ.get("GRID_TIERS", "12")
# Manager-side homes for results + the durable bad-machine blocklist.
RESULTS_DIR = os.environ.get("ROLA_BENCH_RESULTS", "/home/blake/rola-bench/results/mqar")  # scientific output (jsonl, git-tracked)
RUNTIME_DIR = os.environ.get("ROLA_BENCH_RUNTIME", "/home/blake/rola-bench/runtime/mqar")  # fleet ops state (bad_hosts, logs) — gitignored
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v1")


def work_list():
    """Every run_id in the grid at the configured GRID_TIERS (reloaded so the tier env applies).
    Optional manager-side filters (run_id = `grid-<family>-<shape>-...`): FAMILIES / SHAPES env
    (comma lists) restrict which cells the dispatcher hands out — e.g. FAMILIES=hedgehog,gla,gdn
    SHAPES=square,heads for a 4090-safe baseline subset (wide shapes OOM at batch 128 on 4090)."""
    os.environ["GRID_TIERS"] = GRID_TIERS
    from rola_bench.mqar.run_rla_sweep import load_configs
    ids = [c.run_id for c in load_configs("canonical")[0]]
    fams = os.environ.get("FAMILIES")
    shps = os.environ.get("SHAPES")
    if fams:
        fset = set(fams.split(","))
        ids = [r for r in ids if r.split("-")[1] in fset]
    if shps:
        sset = set(shps.split(","))
        ids = [r for r in ids if len(r.split("-")) > 2 and r.split("-")[2] in sset]
    excl = os.environ.get("EXCLUDE_SHAPES")  # e.g. EXCLUDE_SHAPES=wide -> 4090-safe full grid
    if excl:
        eset = set(excl.split(","))
        ids = [r for r in ids if not (len(r.split("-")) > 2 and r.split("-")[2] in eset)]
    exfam = os.environ.get("EXCLUDE_FAMILIES")  # e.g. EXCLUDE_FAMILIES=sse,gdn -> drop those families
    if exfam:
        xset = set(exfam.split(","))
        ids = [r for r in ids if r.split("-")[1] not in xset]
    return ids


def done_ids():
    """run_ids with an ok row in any fleet_results ndjson — reused, never recomputed."""
    have = set()
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.jsonl")):
        try:
            for line in open(f):
                j = json.loads(line)
                if j.get("ok") and j.get("run_id"):
                    have.add(j["run_id"])
        except Exception:
            pass
    return have


mqar_job = Job(
    name="canonical-mqar-grid",
    image=IMAGE,
    # 4090 default (cheap, fits square/heads); override for the wide / Based cells that OOM at
    # batch 128 on a 4090: OFFER_QUERY="gpu_name=A100_PCIE num_gpus=1 rentable=true reliability>0.97".
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "canonical.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    env={"GRID_TIERS": GRID_TIERS},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=1200,
    batch=1,                  # one cell per batch: the slowest high-nc square/heads cells take ~3.6h for
                              # 40 epochs; batched 5-at-a-time (5 LRs of one config) they'd need ~18h and
                              # never fit any timeout. batch=1 lets each slow cell finish inside per_batch_timeout.
    per_batch_timeout=18000,  # 5h: comfortably exceeds the slowest single cell (~3.6h at ~4 it/s on nc256 square)
)
