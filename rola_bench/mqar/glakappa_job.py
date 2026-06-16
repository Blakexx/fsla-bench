"""Kappa-GLA re-run as a `fleet` Job — re-runs the routed-GLA ladder with the KAPPA variant
(rola-gla-kappa-asym), the canonical match to routed-RLA's kappa-asym (the original grid used the
unnormalized scalar-sym). Reuses the MQAR image + run_rla_sweep with --config gla_kappa_rerun, an
isolated label/results dir so it never collides with the canonical-mqar-grid fleet, and a distinct
run_id tag (glakappa) so the preserved scalar-sym rows are untouched.

Launch:  python launch.py glakappa --boxes 6
"""
import glob, importlib, json, os
from fleet import Job

# Writes into the UNIFIED results/mqar dir (run_ids grid-routed-glakappa-* are distinct from the
# grid-routed-gla-* scalar-sym rows — no collision; the run_id is what dedups, not the directory).
# A distinct dispatch filename keeps it separable but co-located so similarity/analysis read one dir.
RESULTS_DIR = os.environ.get("ROLA_GLAKAPPA_RESULTS", "/home/blake/rola-bench/results/mqar")
RUNTIME_DIR = os.environ.get("ROLA_GLAKAPPA_RUNTIME", "/home/blake/rola-bench/runtime/glakappa")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v10")  # v10 = + gla_kappa_rerun config


def work_list():
    os.environ["GRID_TIERS"] = "all"
    from rola_bench.mqar.experiments import gla_kappa_rerun as g
    importlib.reload(g)
    return [c.run_id for c in g.configs]


def done_ids():
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


glakappa_job = Job(
    name="rola-glakappa",                                      # isolated label
    image=IMAGE,
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "glakappa_dispatch.jsonl"),   # results/mqar/glakappa_dispatch.jsonl
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    env={"FLEET_RUN_CMD": "python run_rla_sweep.py --config gla_kappa_rerun --results $FLEET_RESULTS",
         "GRID_TIERS": "all"},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=1200,
    batch=1,                  # one cell per batch (high-nc cells take ~hours at 40 epochs)
    per_batch_timeout=18000,  # 5h, matches the canonical mqar grid
)
