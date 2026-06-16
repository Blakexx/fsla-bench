"""Box-side runner for the rank fleet job (FLEET_RUN_CMD -> this module).

For each run_id in RUN_IDS: rebuild the EXACT model from its canonical-grid config, load the
checkpoint the dispatcher shipped (base64 -> $FLEET_PAYLOAD_DIR/<run_id>), auto-detect the model
family, and measure the rank of its FINAL attention matrix BOTH pre-causal-mask (causal=False, the
rank-law object) and post-causal-mask (causal=True, the literal MQAR operator). One ndjson row per
(run_id, causal) plus an ok summary row, appended to $FLEET_RESULTS.

Generic by construction: the only model-specific knowledge lives in the evaluator's extractors;
here we just dispatch on which family's predicate matches the rebuilt model. SSE has no extractor
(excluded upstream in the job); if one slips through we record ok=False rather than guess.
"""
import json
import math
import os
import sys
import traceback

import torch

from rola_bench.similarity import evaluator as E

RESULTS = os.environ.get("FLEET_RESULTS", "/workspace/res.jsonl")
PAYLOAD_DIR = os.environ.get("FLEET_PAYLOAD_DIR", "/workspace/payloads")
RANK_FRAC = float(os.environ.get("RANK_FRAC", "0.25"))     # fraction of each test slice's examples to eval
MEM_BUDGET = float(os.environ.get("RANK_MEM_BUDGET", "3e9"))  # bytes for the [chunk,H,L,L] W tensor (4090-safe)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# per-(kernel) metric -> rank_stats key whose per-slice `_dist` we aggregate across chunks.
_METRICS = ("eff_rank", "stable_rank", "rank_1e-02", "sharp_diffuseness", "sharp_eff_key_frac", "sharp_top1")


def _agg(vals):
    """Summarize a concatenated per-slice distribution -> mean + percentiles (compact)."""
    if not vals:
        return None
    s = sorted(vals); n = len(s)
    q = lambda p: s[min(n - 1, int(p * (n - 1) + 0.5))]
    return {"mean": round(sum(s) / n, 3), "p10": round(q(0.1), 3), "p50": round(q(0.5), 3),
            "p90": round(q(0.9), 3), "min": round(s[0], 3), "max": round(s[-1], 3), "n": n}


def _chunk_size(L, n_heads):
    """Largest example-chunk whose [chunk, H, L, L] fp32 W fits MEM_BUDGET. n_heads is the model's
    ACTUAL head count (probed per cell): heads-shape baselines reach H~140, so a hardcoded H=8 would
    under-budget by ~17x and OOM the heavy slices."""
    per = max(1, n_heads * L * L * 4)
    return max(1, min(64, int(MEM_BUDGET // per)))


def _append(row):
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def _configs_by_runid():
    """run_id -> TrainConfig for the full grid (GRID_TIERS=all set by the job env)."""
    import importlib
    from rola_bench.mqar.experiments import canonical_mqar_grid as g
    importlib.reload(g)
    return {c.run_id: c for c in g.configs}


def _build_model(cfg):
    from zoology.model import LanguageModel
    return LanguageModel(cfg.model)


def _pick_extractor_builder(model):
    """Return (family, builder(layer_index, causal)->extractor) by matching the rebuilt model's
    layer-0 mixer against each family predicate. None if no faithful extractor applies."""
    # RoLA first (its mixer also exposes q/k projections that could look linear-attn-ish).
    try:
        E._find_rola_mixer(model, 0); return "rola", E.build_rola_attention_extractor
    except Exception:
        pass
    mods = list(model.modules())
    checks = [
        ("mha", E._is_mha, E.build_softmax_attention_extractor),
        ("gdn", E._is_gdn, E.build_gdn_extractor),
        ("gla", E._is_gla, E.build_gla_extractor),
        ("based", E._is_based, E.build_based_extractor),
        ("fla_linear", E._is_fla_linear_attn, E.build_fla_linear_attn_extractor),
    ]
    for fam, pred, builder in checks:
        if any(pred(m) for m in mods):
            return fam, builder
    return None, None


def _rank_one(run_id, cfg):
    """Rank/sharpness DISTRIBUTION over the real MQAR test set (per kv-density slice), at RANK_FRAC
    sampling, memory-chunked. One row per (run_id, causal) carrying a per-slice summary dict."""
    from rola_bench.mqar.experiments.rola_router_width_v2 import TEST_CONFIGS
    ckpt = os.path.join(PAYLOAD_DIR, run_id)
    if not os.path.exists(ckpt):
        _append({"run_id": run_id, "ok": False, "error": "checkpoint payload missing"})
        return
    model = _build_model(cfg).to(DEV)
    sd = E._load_state_dict(ckpt, map_location=DEV, state_dict_key="model")
    model.load_state_dict(sd, strict=True)
    model.eval()
    family, builder = _pick_extractor_builder(model)
    if builder is None:
        _append({"run_id": run_id, "ok": False, "error": "no faithful extractor for this family"})
        return
    vocab = cfg.model.vocab_size
    wrote_ok = True
    for causal in (False, True):
        try:
            extractor = builder(layer_index=0, causal=causal)
            # Probe this family's head count H once (W is [b,H,L,L]) on the SMALLEST-L slice, 1 example,
            # so the per-slice memory-chunk size reflects the real H (heads-shape baselines reach ~140).
            _sc0 = min(TEST_CONFIGS, key=lambda c: c.input_seq_len)
            with torch.no_grad():
                _probe = E.make_mqar_eval(vocab, _sc0.input_seq_len, _sc0.num_kv_pairs, 1, seed=1234, device=DEV)
                n_heads = int(extractor(model, _probe).shape[1])
            per_slice = {}
            for sc in TEST_CONFIGS:                                   # one entry per kv-density slice
                L, kv = sc.input_seq_len, sc.num_kv_pairs
                n = max(1, math.ceil(RANK_FRAC * sc.num_examples))
                batch = E.make_mqar_eval(vocab, L, kv, n, seed=1234, device=DEV)  # [n, L] real MQAR
                cs = _chunk_size(L, n_heads)
                acc = {m: [] for m in _METRICS}
                with torch.no_grad():
                    for i in range(0, n, cs):
                        attn = extractor(model, batch[i:i + cs])      # [b,H,L,L]
                        st = E.rank_stats(attn, max_b=cs, max_l=L, emit_spectrum=False)
                        for m in _METRICS:
                            acc[m].extend(st.get(m + "_dist", []))
                        del attn
                        if DEV == "cuda":
                            torch.cuda.empty_cache()
                per_slice[f"kv{kv}"] = {"L": L, "n_examples": n, **{m: _agg(acc[m]) for m in _METRICS}}
            _append({"run_id": run_id, "ok": True, "family": family, "causal": causal,
                     "frac": RANK_FRAC, "per_slice": per_slice})
        except Exception as e:  # noqa: BLE001
            wrote_ok = False
            _append({"run_id": run_id, "ok": False, "family": family, "causal": causal,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
    return wrote_ok


def main():
    run_ids = [r for r in os.environ.get("RUN_IDS", "").split(",") if r]
    by_id = _configs_by_runid()
    for rid in run_ids:
        cfg = by_id.get(rid)
        if cfg is None:
            _append({"run_id": rid, "ok": False, "error": "run_id not in canonical grid (GRID_TIERS=all?)"})
            print(f"[rank] {rid}: no config", flush=True)
            continue
        try:
            print(f"[rank] {rid} on {DEV} (frac={RANK_FRAC}, rungs per SIM_RUNGS)", flush=True)
            _rank_one(rid, cfg)
            print(f"[rank] {rid} done", flush=True)
        except Exception as e:  # noqa: BLE001 -- never abort the batch on one cell
            _append({"run_id": rid, "ok": False,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
            print(f"[rank] {rid} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
