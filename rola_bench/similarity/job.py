"""Similarity-matrix eval job builder, on the global layer (formerly "rank").

Measures each trained model's attention/similarity matrix — rank + sharpness/diffuseness, pre/post
causal mask (see similarity/evaluator.py). Selection: per CELL (= kernel, nc-rung) take the
best-accuracy run's checkpoint from the MQAR store and score that one (dedups the LR axis, ~5x
cheaper). Each cell's .pt ships to the box via the payload channel. SSE has no faithful extractor.

Results: results/similarity/<config>/<run_id>.jsonl. Launch: launch.py similarity main
"""
import glob
import json
import os
import re

from rola_bench.jobs import make_job, RESULTS_ROOT

CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
MQAR_DIR = os.path.join(RESULTS_ROOT, "mqar")            # the unified MQAR store (dir/run_id)
_NO_EXTRACTOR_PREFIXES = ("grid-sse-",)                  # SSE has no faithful extractor


def _has_extractor(run_id):
    return not any(run_id.startswith(p) for p in _NO_EXTRACTOR_PREFIXES)


def _cell_key(run_id):
    """Grouping key = (KERNEL, nc-RUNG); shape, LR, exact-st, seed collapsed. Best model per kernel+
    state, keyed on the nc rung (not the realized st float). routed-rla/gla stay distinct from the
    rla/gla MONOLITH baselines."""
    nc = re.search(r'-nc(\d+)', run_id)
    nc = nc.group(1) if nc else "na"
    p = run_id.split("-")
    if run_id.startswith("grid-routed-"):
        kernel = "routed-" + p[2]
    elif run_id.startswith("grid-mha"):
        kernel = "mha"
    else:
        kernel = p[1]
    return f"{kernel}-nc{nc}"


def _mqar_accuracy():
    """run_id -> best accuracy, from the unified MQAR store (results/mqar/<config>/<run_id>.jsonl)."""
    acc = {}
    for f in glob.glob(os.path.join(MQAR_DIR, "**", "*.jsonl"), recursive=True):
        try:
            for line in open(f):
                j = json.loads(line)
                if j.get("ok") and j.get("run_id"):
                    a = j.get("max_acc") or j.get("best_overall") or j.get("final_overall") or 0.0
                    acc[j["run_id"]] = max(acc.get(j["run_id"], -1.0), a)
        except Exception:
            pass
    return acc


def _work_list():
    """Best-accuracy checkpoint per cell (LR+shape collapsed), restricted to SIM_RUNGS (default
    16,64,256) + mha (rung-independent)."""
    rungs = set(os.environ.get("SIM_RUNGS", "16,64,256").split(","))
    have_pt = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(CKPT_DIR, "*.pt"))}
    best = {}
    for rid, a in _mqar_accuracy().items():
        if rid not in have_pt or not _has_extractor(rid):
            continue
        k = _cell_key(rid)
        nc = k.rsplit("-nc", 1)[-1]
        if nc not in rungs and not k.startswith("mha"):
            continue
        if a > best.get(k, (-1.0, None))[0]:
            best[k] = (a, rid)
    return [rid for _, rid in best.values()]


def _payload_for(run_id):
    p = os.path.join(CKPT_DIR, run_id + ".pt")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return fh.read()


def sim_job(config="main"):
    fleet = {
        "idle_timeout": 900,
        "batch": 1,                  # one ckpt per POST body (~42MB base64) — less uplink contention
        "per_batch_timeout": 3600,
        "post_timeout": int(os.environ.get("SIM_POST_TIMEOUT", "1800")),
        "env": {"GRID_TIERS": "all",      # every cell's config must be reconstructable
                "RANK_FRAC": os.environ.get("RANK_FRAC", "0.25"),
                "SIM_RUNGS": os.environ.get("SIM_RUNGS", "16,64,256")},
    }
    return make_job("similarity", config, _work_list, "python -u -m rola_bench.similarity.run_sim",
                    fleet, payload_for=_payload_for)
