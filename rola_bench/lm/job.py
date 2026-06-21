"""LM benchmark job builder — iso-backbone RoLA-vs-attention comparison, on the global layer.

`lm_job(config)` builds the fleet.Job via rola_bench.jobs.make_job (uniform results/lm/<config>/
<run_id>.jsonl store + resume). Work-list = the arm registry (arms.ARMS, env LM_ARMS) x seeds,
run_id "<arm>_s<seed>". The box runs run_lm.py, which trains each arm and appends its result row.
LM keeps preemption-safe checkpoint resume (payload_for + pull_in_progress + no_kill).

Needs the rola-lm image (flash-attn for the `attn` ceiling arm). Launch: launch.py lm default
"""
import os

from rola_bench.jobs import make_job
from rola_bench.lm import arms

CKPT_DIR = os.environ.get("ROLA_LM_CKPTS", "/mnt/d/rola_data/lm_ckpts")

# Forward LM budget/LR + geometry knobs from the launching shell to the box (run_lm/arms read them
# from box env). Default to the headline; the d_model=128 small-backbone diagnostic overrides them.
_FWD = {k: os.environ[k] for k in
        ("LM_LR", "LM_MAX_STEPS", "LM_BLOCK_SIZE", "LM_BATCH_SIZE", "LM_GRAD_ACCUM",
         "LM_DATASET", "LM_DATASET_CONFIG", "LM_GRAD_CKPT", "LM_SAVE_STEPS",
         "LM_HIDDEN", "LM_ROLA_NC", "LM_ROLA_DQK", "LM_ROLA_DV", "LM_ROLA_NH",
         "LM_GDN_NH", "LM_ATTN_HEADS", "LM_MAX_POS", "LM_LAYERS",
         "LM_MAX_TRAIN_SAMPLES") if k in os.environ}


def _payload_for(run_id):
    """Resume payload = the last in-progress checkpoint tar pulled for this run_id (untarred + resumed
    on the box). None on first run."""
    p = os.path.join(CKPT_DIR, run_id + ".pt")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return fh.read()


def lm_job(config="default"):
    arm_list = [a for a in os.environ.get("LM_ARMS", ",".join(arms.ARMS)).split(",") if a]
    seeds = [s for s in os.environ.get("LM_SEEDS", "1337,2024").split(",") if s]   # >=2 for error bars
    fleet = {
        "image": os.environ.get("ROLA_LM_IMAGE", "blakeresearch/rola-lm:v2"),
        "idle_timeout": 1800,
        "batch": 1,                  # one arm×seed per box (a full LM training)
        "per_batch_timeout": 43200,  # 12h log cadence; under no_kill a longer run keeps going
        "no_kill": bool(int(os.environ.get("LM_NO_KILL", "1"))),   # pause+reattach, never destroy live work
        "pull_in_progress": True,    # periodically pull each box's in-progress ckpt (preemption-safe)
        "live_pull_every": int(os.environ.get("LM_LIVE_PULL", "900")),
        "env": dict(_FWD),
    }
    return make_job("lm", config, lambda: [f"{a}_s{s}" for a in arm_list for s in seeds],
                    "python run_lm.py", fleet, ckpt_dir=CKPT_DIR, payload_for=_payload_for)
