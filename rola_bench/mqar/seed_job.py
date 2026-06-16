"""Seed-confirmation sweep as a `fleet` Job — re-runs the best cell per (kernel, rung) at extra seeds
for error bars on the headline. Reuses the MQAR image + run_rla_sweep, but with --config seed_confirm
(overriding the image's default canonical_mqar_grid via FLEET_RUN_CMD) and an isolated label/results
dir so it never collides with the canonical-mqar-grid fleet.

Launch:  python launch.py seed --boxes 6
"""
import glob, importlib, json, os
from fleet import Job

RESULTS_DIR = os.environ.get("ROLA_SEED_RESULTS", "/home/blake/rola-bench/results/seed")
RUNTIME_DIR = os.environ.get("ROLA_SEED_RUNTIME", "/home/blake/rola-bench/runtime/seed")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v9")  # v9 = seed_confirm configs_envs + data.seed fix
SEEDS = os.environ.get("SEED_CONFIRM_SEEDS", "2024,777")   # ADDITIONAL seeds (existing grid is s1337)


def work_list():
    os.environ["SEED_CONFIRM_SEEDS"] = SEEDS
    os.environ["GRID_TIERS"] = "all"
    from rola_bench.mqar.experiments import seed_confirm as s
    importlib.reload(s)
    return [c.run_id for c in s.configs]


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


seed_job = Job(
    name="rola-seed",                                          # isolated label
    image=IMAGE,
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "seed_dispatch.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    env={"FLEET_RUN_CMD": "python run_rla_sweep.py --config seed_confirm --results $FLEET_RESULTS",
         "GRID_TIERS": "all", "SEED_CONFIRM_SEEDS": SEEDS},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=1200,
    batch=1,                  # one cell per batch (high-nc square/heads cells take ~hours at 40 epochs)
    per_batch_timeout=18000,  # 5h, matches the canonical mqar grid
)
