"""The canonical-MQAR-grid sweep as a `fleet` Job.

This is the manager-side description of WHAT to run; `fleet` owns HOW (provision/control/sync). The
matching box-side run command lives in the mqar image (docker/mqar.Dockerfile sets FLEET_RUN_CMD to
`run_rla_sweep.py --config canonical_mqar_grid`, and FLEET_VERIFY_CMD to the fla-rola kernel check).
"""
import glob, importlib, json, os
from fleet import Job

GRID_TIERS = os.environ.get("GRID_TIERS", "12")
# Manager-side homes for results + the durable bad-machine blocklist.
RESULTS_DIR = os.environ.get("ROLA_BENCH_RESULTS", "/home/blake/rola-bench/fleet_results")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v1")


def work_list():
    """Every run_id in the grid at the configured GRID_TIERS (reloaded so the tier env applies)."""
    os.environ["GRID_TIERS"] = GRID_TIERS
    from rola_bench.mqar.experiments import canonical_mqar_grid as g
    importlib.reload(g)
    return [c.run_id for c in g.configs]


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
    offer_query="gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500",
    results_path=os.path.join(RESULTS_DIR, "results_dispatch.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    env={"GRID_TIERS": GRID_TIERS},
    bad_hosts=os.path.join(RESULTS_DIR, "bad_hosts.txt"),
    idle_timeout=1200,
    batch=5,
)
