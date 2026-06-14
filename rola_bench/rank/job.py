"""Post-hoc rank evaluation as a `fleet` Job.

Unlike perf, rank IS naturally per-cell — one trained checkpoint -> one rank measurement — so it
fans out across boxes like the MQAR sweep. The wrinkle: the box needs the trained `.pt` to score
it. We use the generic payload channel (fleet.Job.payload_for): the dispatcher base64s each cell's
checkpoint into the /run JSON and the box writes it to $FLEET_PAYLOAD_DIR/<run_id> before running.
That keeps fleet's "send JSON, execute" contract intact (no upload side-endpoint), at the cost of a
few MB of base64 per cell — negligible vs a GPU-minute.

work_list = every checkpoint we have whose family has a faithful extractor (rola / fla-linear /
gla / based / gdn / mha). SSE has no extractor yet, so its checkpoints are excluded rather than
silently failing forever (a fail row would never enter done_ids and would be retried endlessly).
"""
import glob, json, os
from fleet import Job

CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
RESULTS_DIR = os.environ.get("ROLA_BENCH_RANK_RESULTS", "/home/blake/rola-bench/results/rank")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v2")  # v2 = suite + payload base

# Families the evaluator can faithfully extract (run_id prefix -> has-extractor). SSE excluded.
_NO_EXTRACTOR_PREFIXES = ("grid-sse-",)


def _has_extractor(run_id):
    return not any(run_id.startswith(p) for p in _NO_EXTRACTOR_PREFIXES)


def work_list():
    """Every checkpoint on disk (<run_id>.pt) with a supported family."""
    ids = []
    for p in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt"))):
        rid = os.path.basename(p)[:-3]            # strip .pt
        if _has_extractor(rid):
            ids.append(rid)
    return ids


def done_ids():
    """run_ids with an ok rank row already recorded."""
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
    """Raw bytes of this cell's checkpoint, shipped (base64) in the /run payload. None if missing
    (the box will record ok=False rather than crash)."""
    p = os.path.join(CKPT_DIR, run_id + ".pt")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return fh.read()


rank_job = Job(
    name="rola-rank",
    image=IMAGE,
    # Rank runs each model's real kernel forward on a small fixed eval batch — a 4090 is plenty
    # and cheap. Override OFFER_QUERY for a specific card.
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "rank_dispatch.jsonl"),
    ckpt_dir=os.path.join(RESULTS_DIR, "_unused_ckpts"),    # rank produces no checkpoints
    work_list=work_list,
    done_ids=done_ids,
    payload_for=payload_for,                                # <-- ships the .pt to the box
    env={"FLEET_RUN_CMD": "python -u -m rola_bench.rank.run_rank",
         "GRID_TIERS": "all",                               # so every cell's config is reconstructable
         "RANK_BATCH": os.environ.get("RANK_BATCH", "4"),
         "RANK_SEQ_LEN": os.environ.get("RANK_SEQ_LEN", "2048")},
    bad_hosts=os.path.join(RESULTS_DIR, "bad_hosts.txt"),
    idle_timeout=900,
    batch=4,                  # a few ckpts per box; each is a forward + SVD, seconds-to-minutes
    per_batch_timeout=3600,
)
