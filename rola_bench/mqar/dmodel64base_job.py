"""d_model=64 matched-state baselines (RLA-wide + Hedgehog-square at nc256) as a `fleet` Job.

Pairs with the rola-d64 routed sweep: does routing beat the monolith forms at tiny d_model=64?
Isolated label + distinct run_id tag (grid-{method}-{shape}-d64-*). Requires v16+ image (ships
the dmodel64_baselines config).

Launch:  python launch.py d64base --boxes 2
"""
import glob, importlib, json, os
from fleet import Job

RESULTS_DIR = os.environ.get("ROLA_D64BASE_RESULTS", "/home/blake/rola-bench/results/mqar")
RUNTIME_DIR = os.environ.get("ROLA_D64BASE_RUNTIME", "/home/blake/rola-bench/runtime/dmodel64base")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v16")


def work_list():
    os.environ["GRID_TIERS"] = "all"
    from rola_bench.mqar.experiments import dmodel64_baselines as g
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


d64base_job = Job(
    name="rola-d64base",
    image=IMAGE,
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "dmodel64base_dispatch.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    env={"FLEET_RUN_CMD": "python run_rla_sweep.py --config dmodel64_baselines --results $FLEET_RESULTS",
         "GRID_TIERS": "all"},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=1200,
    batch=1,
    per_batch_timeout=18000,
)
