"""Router-bias ablation as a `fleet` Job — does a learned router bias help RoLA-RLA-kappa at nc128?

4 cells: routed RLA-kappa, nc128, best LR (3e-3), router_bias={False,True} x seeds={1337,2024},
paired so the only difference is the bias (see router_bias_ablation.py). Isolated label +
distinct run_id tag (rlabias0/rlabias1) so it never collides with the canonical nc128 rows.

REQUIRES image v13+ (router_bias was added to rola.py AFTER v12; on v12 the kwarg is silently
swallowed by RoLA's **kwargs, making bias=True a no-op).

Launch:  python launch.py rlabias --boxes 2
"""
import glob, importlib, json, os
from fleet import Job

RESULTS_DIR = os.environ.get("ROLA_RLABIAS_RESULTS", "/home/blake/rola-bench/results/mqar")
RUNTIME_DIR = os.environ.get("ROLA_RLABIAS_RUNTIME", "/home/blake/rola-bench/runtime/rlabias")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v13")  # v13 = + router_bias flag


def work_list():
    from rola_bench.mqar.run_rla_sweep import load_configs
    return [c.run_id for c in load_configs("router_bias")[0]]


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


rlabias_job = Job(
    name="rola-rlabias",
    image=IMAGE,
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "router_bias.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    env={"FLEET_RUN_CMD": "python run_rla_sweep.py --config router_bias --results $FLEET_RESULTS",
         "GRID_TIERS": "all"},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=1200,
    batch=1,
    per_batch_timeout=18000,
)
