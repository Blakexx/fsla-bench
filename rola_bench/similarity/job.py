"""Post-hoc SIMILARITY-matrix evaluation as a `fleet` Job (formerly "rank").

Measures properties of each trained model's final attention/similarity matrix — rank (how many
unique selection blends) AND sharpness/diffuseness (how peaked each blend is), pre- and
post-causal-mask. See similarity/evaluator.py.

Selection strategy: we do NOT score every checkpoint. For each CELL (= run_id minus its LR token,
i.e. one (kernel, shape, state) point) we take the **best-accuracy** run's checkpoint and score
that one. This dedups the 5-LR axis to the model that actually learned the task — the similarity
structure of the *best* model per cell is the scientifically meaningful object — and cuts the eval
~5x. Each cell's .pt is shipped to the box via the generic base64 payload channel
(fleet.Job.payload_for -> $FLEET_PAYLOAD_DIR/<run_id>). SSE has no faithful extractor -> excluded.
"""
import glob, json, os, re
from fleet import Job

CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
MQAR_RESULTS_DIR = os.environ.get("ROLA_BENCH_RESULTS", "/home/blake/rola-bench/fleet_results")
RESULTS_DIR = os.environ.get("ROLA_BENCH_SIM_RESULTS", "/home/blake/rola-bench/results/similarity")
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v4")  # v4 = similarity rename + sharpness

_NO_EXTRACTOR_PREFIXES = ("grid-sse-",)            # SSE has no faithful extractor


def _has_extractor(run_id):
    return not any(run_id.startswith(p) for p in _NO_EXTRACTOR_PREFIXES)


def _cell_key(run_id):
    """Grouping key = (KERNEL, nc-RUNG) -- shape, LR, exact-st and seed all collapsed.

    We pick the single best-accuracy checkpoint per (kernel, state-rung) regardless of how it spent
    the state (wide/square/heads) -- we just want the best model at that kernel+state. Key on the nc
    RUNG (the RoLA-matched n_s level), not the realized `st` float (which differs by dim rounding
    across shapes). The RoLA method (routed-rla/gla) stays distinct from the rla/gla MONOLITH
    baselines -- those are different kernels we compare."""
    nc = re.search(r'-nc(\d+)', run_id)
    nc = nc.group(1) if nc else "na"
    p = run_id.split("-")
    if run_id.startswith("grid-routed-"):
        kernel = "routed-" + p[2]            # routed-rla / routed-gla (the proposed method)
    elif run_id.startswith("grid-mha"):
        kernel = "mha"
    else:
        kernel = p[1]                        # rla / gla / based / hedgehog (monolith); shape p[2] dropped
    return f"{kernel}-nc{nc}"


def _mqar_accuracy():
    """run_id -> best accuracy, from the MQAR result ndjson."""
    acc = {}
    for f in glob.glob(os.path.join(MQAR_RESULTS_DIR, "*.jsonl")):
        try:
            for line in open(f):
                j = json.loads(line)
                if j.get("ok") and j.get("run_id"):
                    a = j.get("max_acc") or j.get("best_overall") or j.get("final_overall") or 0.0
                    acc[j["run_id"]] = max(acc.get(j["run_id"], -1.0), a)
        except Exception:
            pass
    return acc


def work_list():
    """Best-accuracy checkpoint per cell (LR collapsed): one similarity-eval per (kernel,shape,state)."""
    have_pt = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(CKPT_DIR, "*.pt"))}
    best = {}                                       # cell_key -> (acc, run_id)
    for rid, a in _mqar_accuracy().items():
        if rid not in have_pt or not _has_extractor(rid):
            continue
        k = _cell_key(rid)
        if a > best.get(k, (-1.0, None))[0]:
            best[k] = (a, rid)
    return [rid for _, rid in best.values()]


def done_ids():
    """run_ids with an ok similarity row already recorded."""
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
    """Raw bytes of this cell's best checkpoint, shipped (base64) in the /run payload."""
    p = os.path.join(CKPT_DIR, run_id + ".pt")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return fh.read()


sim_job = Job(
    name="rola-similarity",
    image=IMAGE,
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "similarity_dispatch.jsonl"),
    ckpt_dir=os.path.join(RESULTS_DIR, "_unused_ckpts"),    # similarity produces no checkpoints
    work_list=work_list,
    done_ids=done_ids,
    payload_for=payload_for,                                # ships the best .pt per cell to the box
    env={"FLEET_RUN_CMD": "python -u -m rola_bench.similarity.run_sim",
         "GRID_TIERS": "all",                               # so every cell's config is reconstructable
         "RANK_BATCH": os.environ.get("RANK_BATCH", "4"),
         "RANK_SEQ_LEN": os.environ.get("RANK_SEQ_LEN", "2048")},
    bad_hosts=os.path.join(RESULTS_DIR, "bad_hosts.txt"),
    idle_timeout=900,
    batch=4,
    per_batch_timeout=3600,
)
