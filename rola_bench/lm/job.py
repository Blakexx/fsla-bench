"""The LM benchmark as a `fleet` Job — the iso-backbone RoLA-vs-attention comparison.

Work-list = the 5-arm registry (arms.ARMS) x seeds, run_id "<arm>_s<seed>". The box runs
run_lm.py (FLEET_RUN_CMD), which trains each arm via train_lm.py and appends its result row
{run_id, ok, params_m, state_floats, ppl, train_tok_per_s, peak_mib, ...} to $FLEET_RESULTS,
saving the checkpoint dir under $FLEET_CKPT_DIR/<run_id> for the post-hoc recall-downstream pass.

Needs the rola-lm image with flash-attn (the softmax-attention ceiling arm `attn` requires it;
the RoLA/GDN arms don't). Image tag rola-lm:v2.
"""
import glob, json, os
from fleet import Job
from rola_bench.lm import arms

SEEDS = [s for s in os.environ.get("LM_SEEDS", "1337,2024").split(",") if s]   # >=2 seeds for error bars
ARMS = [a for a in os.environ.get("LM_ARMS", ",".join(arms.ARMS)).split(",") if a]
RESULTS_DIR = os.environ.get("ROLA_LM_RESULTS", "/home/blake/rola-bench/results/lm")
RUNTIME_DIR = os.environ.get("ROLA_LM_RUNTIME", "/home/blake/rola-bench/runtime/lm")
CKPT_DIR = os.environ.get("ROLA_LM_CKPTS", "/mnt/d/rola_data/lm_ckpts")
IMAGE = os.environ.get("ROLA_LM_IMAGE", "blakeresearch/rola-lm:v2")  # v2 = + flash-attn (attn ceiling arm)


# Forward LM budget/LR + geometry knobs from the launching shell to the box (run_lm/arms read them
# from box env). Geometry knobs (LM_HIDDEN/LM_ROLA_*/LM_GDN_NH/LM_ATTN_HEADS) default to the Option-Y
# headline; the d_model=128 small-backbone diagnostic overrides them.
_FWD = {k: os.environ[k] for k in
        ("LM_LR", "LM_MAX_STEPS", "LM_BLOCK_SIZE", "LM_BATCH_SIZE", "LM_GRAD_ACCUM",
         "LM_DATASET", "LM_DATASET_CONFIG", "LM_GRAD_CKPT", "LM_SAVE_STEPS",
         "LM_HIDDEN", "LM_ROLA_NC", "LM_ROLA_DQK", "LM_ROLA_DV", "LM_ROLA_NH",
         "LM_GDN_NH", "LM_ATTN_HEADS", "LM_MAX_POS", "LM_LAYERS",
         "LM_MAX_TRAIN_SAMPLES") if k in os.environ}


def work_list():
    return [f"{a}_s{s}" for a in ARMS for s in SEEDS]


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


def payload_for(run_id):
    """Resume payload = the last in-progress checkpoint tar the dispatcher pulled for this run_id
    (CKPT_DIR/<run_id>.pt). Shipped on (re-)dispatch → the box untars + resumes. None on first run."""
    p = os.path.join(CKPT_DIR, run_id + ".pt")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return fh.read()


lm_job = Job(
    name=os.environ.get("ROLA_LM_LABEL", "rola-lm"),   # override to run a 2nd LM fleet alongside the suite
    image=IMAGE,
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "lm_dispatch.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    payload_for=payload_for,        # ship the last resume tar back on re-dispatch (off-box backstop)
    pull_in_progress=True,          # periodically pull each box's in-progress checkpoint (preemption-safe)
    live_pull_every=int(os.environ.get("LM_LIVE_PULL", "900")),  # min secs between in-progress ckpt pulls;
    # env-overridable (set small for the resume kill-test; 900s/15min for real runs — optimizer ckpt is large)
    # Resilience: on-demand boxes keep their disk, so NEVER destroy a host with incomplete work — PAUSE
    # (vast.stop) it and reattach (resume-from-local-disk) instead. Guards against the dispatcher's own
    # transient-unreachability false-positives + survives a dispatcher restart. state_path persists the
    # host<->run map for reattach. (LM_NO_KILL=0 reverts to destroy-on-drain.)
    no_kill=bool(int(os.environ.get("LM_NO_KILL", "1"))),
    state_path=os.path.join(RUNTIME_DIR, "lm_hosts.json"),
    env={"FLEET_RUN_CMD": "python run_lm.py", **_FWD},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=1800,
    batch=1,                  # one arm×seed per box (each is a full LM training)
    per_batch_timeout=43200,  # 12h log/monitor cadence — under no_kill a longer alive run keeps going
)
