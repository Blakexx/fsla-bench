"""d_model=64, nc=256 RoLA-RLA-kappa ablation as a `fleet` Job — 5-LR sweep, 1 seed (task #111).

Isolated label + distinct run_id tag (grid-routed-rla-d64-*) so it never collides with the canonical
d_model=128 rows. Requires the v14+ image (ships the dmodel64_ablation config).

Launch:  python launch.py dmodel64 --boxes 2
"""
import glob, importlib, json, os
from fleet import Job

RESULTS_DIR = os.environ.get("ROLA_D64_RESULTS", "/home/blake/rola-bench/results/mqar")
RUNTIME_DIR = os.environ.get("ROLA_D64_RUNTIME", "/home/blake/rola-bench/runtime/dmodel64")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v14")


def work_list():
    from rola_bench.mqar.run_rla_sweep import load_configs
    return [c.run_id for c in load_configs("dmodel64")[0]]


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


dmodel64_job = Job(
    name="rola-d64",
    image=IMAGE,
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "dmodel64_dispatch.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    env={"FLEET_RUN_CMD": "python run_rla_sweep.py --config dmodel64 --results $FLEET_RESULTS",
         "GRID_TIERS": "all"},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=1200,
    batch=1,
    per_batch_timeout=18000,
)
