"""Limited norm x symmetry MQAR ablation as a `fleet` Job (config: normsym_resweep, 72 cells).

The equivalence-section sweep: 3 normalized RoLA-RLA modes x {sym,asym} x nc{16,64,256} x 4 LR.
Inter-RoLA only. Requires the rola-bench image that bakes normsym_resweep.py (v17+).

Launch:  python launch.py normsym --boxes 6
"""
import glob, importlib, json, os
from fleet import Job

RESULTS_DIR = os.environ.get("ROLA_NORMSYM_RESULTS", "/home/blake/rola-bench/results/mqar")
RUNTIME_DIR = os.environ.get("ROLA_NORMSYM_RUNTIME", "/home/blake/rola-bench/runtime/normsym")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v17")


def work_list():
    from rola_bench.mqar.experiments import normsym_resweep as g
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


normsym_job = Job(
    name="rola-normsym",
    image=IMAGE,
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "normsym_dispatch.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    env={"FLEET_RUN_CMD": "python run_rla_sweep.py --config normsym_resweep --results $FLEET_RESULTS"},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=1200,
    batch=6,                  # MQAR cells are minutes — batch several per box (72 cells / 6 ~ 12 batches)
    per_batch_timeout=10800,
)
