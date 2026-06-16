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
import os
import sys
import traceback

import torch

from rola_bench.similarity import evaluator as E

RESULTS = os.environ.get("FLEET_RESULTS", "/workspace/res.jsonl")
PAYLOAD_DIR = os.environ.get("FLEET_PAYLOAD_DIR", "/workspace/payloads")
BATCH = int(os.environ.get("RANK_BATCH", "4"))
SEQ_LEN = int(os.environ.get("RANK_SEQ_LEN", "2048"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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
    eval_batch = E.make_eval_batch(BATCH, SEQ_LEN, vocab, seed=1234, device=DEV)
    wrote_ok = True
    for causal in (False, True):
        try:
            extractor = builder(layer_index=0, causal=causal)
            attn = extractor(model, eval_batch)
            stats = E.rank_stats(attn, max_b=BATCH, max_l=SEQ_LEN, emit_spectrum=False)
            _append({"run_id": run_id, "ok": True, "family": family, "causal": causal,
                     "seq_len": SEQ_LEN, "batch": BATCH, **stats})
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
            print(f"[rank] {rid} on {DEV} (B={BATCH} L={SEQ_LEN})", flush=True)
            _rank_one(rid, cfg)
            print(f"[rank] {rid} done", flush=True)
        except Exception as e:  # noqa: BLE001 -- never abort the batch on one cell
            _append({"run_id": rid, "ok": False,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
            print(f"[rank] {rid} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
