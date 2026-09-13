"""Box-side runner for the similarity fleet job (FLEET_RUN_CMD -> this module).

DECOUPLED PROBE ARCHITECTURE. Each run_id is "<cell>::<probe>": the cell identifies the trained
model (its canonical-grid config + shipped checkpoint), the probe selects WHICH measurement to
compute on it. The fleet dedups by full run_id, so each (cell, probe) is tracked independently —
the expensive `rank` probe runs exactly once, and adding a cheap probe later (e.g. `routing`)
dispatches only the new "<cell>::<probe>" ids without re-running rank.

Legacy un-suffixed run_ids ("<cell>") are treated as "<cell>::rank" so the existing rank rows
remain valid (no re-run). Checkpoints are looked up by the CELL (suffix stripped), so all probes
of one cell share its single shipped .pt.

Probes (registry below):
  rank     - rank + sharpness DISTRIBUTION of the FINAL attention matrix over the real MQAR test
             set (per kv slice, both pre/post causal mask). EXPENSIVE (per-slice SVD). Any model
             family with a faithful extractor.
  routing  - empirical state-usage distribution of RoLA's read/write routers over real tokens
             (does it collapse to one state?). CHEAP (forward only). RoLA-only.
"""
import json
import math
import os
import sys
import traceback

import torch

from rola_bench.similarity import evaluator as E

import base64
import zlib

RESULTS = os.environ.get("FLEET_RESULTS", "/workspace/res.jsonl")
PAYLOAD_DIR = os.environ.get("FLEET_PAYLOAD_DIR", "/workspace/payloads")


def _b64_f16(arr):
    """ndarray -> base64(zlib(float16 bytes)). Compact binary that rides INSIDE the jsonl row, so heavy
    2D artifacts (the [nc x nc] joint) return through the generic fleet json-pull like any other result."""
    import numpy as np
    return base64.b64encode(zlib.compress(arr.astype(np.float16).tobytes())).decode("ascii")


def _unb64_f16(s, shape):
    import numpy as np
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype=np.float16).astype(np.float32).reshape(shape)
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


def _parse_runid(rid):
    """'<cell>::<probe>' -> (cell, probe). Legacy un-suffixed -> ('<rid>', 'rank')."""
    if "::" in rid:
        cell, probe = rid.rsplit("::", 1)
        return cell, probe
    return rid, "rank"


def _ckpt_path(rid):
    """Find the cell's checkpoint. Cloud ships it under PAYLOAD_DIR/<run_id>; locally the files are
    named PAYLOAD_DIR/<cell>.pt. Try all three so the SAME runner works in both."""
    cell = _parse_runid(rid)[0]
    for cand in (rid, cell, cell + ".pt"):
        p = os.path.join(PAYLOAD_DIR, cand)
        if os.path.exists(p):
            return p
    return None


def _configs_by_runid():
    """run_id -> TrainConfig. The canonical grid (GRID_TIERS=all) PLUS the kappa-GLA re-run
    (grid-routed-glakappa-*), which lives in a separate config module so the box can rebuild
    those models for the similarity probe."""
    import importlib
    from rola_bench.mqar.experiments import canonical_mqar_grid as g
    importlib.reload(g)
    by_id = {c.run_id: c for c in g.configs}
    from rola_bench.mqar.experiments import gla_kappa_rerun as gk   # kappa-GLA cells (grid-routed-glakappa-*)
    importlib.reload(gk)
    for c in gk.configs:
        by_id[c.run_id] = c
    try:
        from rola_bench.mqar.experiments import router_bias_ablation as rb  # router-bias cells (grid-routed-rlabias*)
        importlib.reload(rb)
        for c in rb.configs:
            by_id[c.run_id] = c
    except Exception:
        pass
    return by_id


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


# ------------------------------------------------------------------------------------------------
# PROBES. Each: probe(run_id, cfg, model, vocab) -> bool ok; appends its own row(s) carrying a
# `probe` field. Setup (build model + load ckpt) is done ONCE by main() and shared across the probe.
# ------------------------------------------------------------------------------------------------

def _probe_rank(run_id, cfg, model, vocab):
    """Rank/sharpness DISTRIBUTION over the real MQAR test set (per kv-density slice), at RANK_FRAC
    sampling, memory-chunked. One row per (run_id, causal). EXPENSIVE (per-slice SVD)."""
    from rola_bench.mqar.experiments.rola_router_width_v2 import TEST_CONFIGS
    family, builder = _pick_extractor_builder(model)
    if builder is None:
        _append({"run_id": run_id, "ok": False, "probe": "rank", "error": "no faithful extractor for this family"})
        return False
    wrote_ok = True
    for causal in (False, True):
        try:
            extractor = builder(layer_index=0, causal=causal)
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
            _append({"run_id": run_id, "ok": True, "probe": "rank", "family": family, "causal": causal,
                     "frac": RANK_FRAC, "per_slice": per_slice})
        except Exception as e:  # noqa: BLE001
            wrote_ok = False
            _append({"run_id": run_id, "ok": False, "probe": "rank", "family": family, "causal": causal,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
    return wrote_ok


def _mean_mass(g):
    """g: [B,L,H,nc] softmax routing weights -> [H, nc] mean routing distribution per head."""
    return g.mean(dim=(0, 1))


def _joint_stats(wH, rH):
    """Joint of write vs read per-state mass (both [H, nc]) — answers 'does read avoid the dominant
    write (trash) state?'. Per head: read mass AT that head's top-1/top-3 write states, and the
    Pearson corr between the write and read state-distributions. Averaged over heads."""
    H, nc = wH.shape
    r_top1, r_top3, corrs = [], [], []
    for h in range(H):
        w, r = wH[h], rH[h]
        order = torch.argsort(w, descending=True)
        r_top1.append(r[order[0]].item())
        r_top3.append(r[order[:3]].sum().item())
        wc, rc = w - w.mean(), r - r.mean()
        denom = (wc.norm() * rc.norm()).clamp_min(1e-12)
        corrs.append((wc @ rc / denom).item())
    f = lambda xs: round(sum(xs) / len(xs), 4)
    return {"read_at_write_top1": f(r_top1),   # read mass on the single most-written state (~0 => read avoids trash)
            "read_at_write_top3": f(r_top3),   # read mass on the top-3 most-written states
            "corr_write_read": f(corrs),       # Pearson(write_dist, read_dist) across states (<0 => anti-aligned)
            "uniform_ref": round(3.0 / nc, 4)} # read mass top-3 would be if read were uniform (baseline)


def _routing_summary(mean_mass, argmax_counts, nc):
    """mean_mass: [H, nc] accumulated mean soft mass per head; argmax_counts: [H, nc] hard-route token
    counts per head. Per-head-averaged scalars: 'does routing collapse to one state?'."""
    H = mean_mass.shape[0]
    p = mean_mass.clamp_min(1e-12)
    ent = (-(p * p.log()).sum(-1) / math.log(nc))       # [H] normalized entropy (0=one state, 1=uniform)
    eff = 1.0 / p.pow(2).sum(-1)                         # [H] participation ratio (effective #states)
    top1_mass = mean_mass.max(-1).values                # [H] biggest state's mean mass
    am_top1, am_active = [], []
    for h in range(H):
        cnt = argmax_counts[h]
        tot = cnt.sum().clamp_min(1)
        am_top1.append((cnt.max() / tot).item())        # frac of tokens hard-routed to the single top state
        am_active.append(int((cnt > 0.001 * tot).sum().item()))  # #states winning >=0.1% of tokens
    f = lambda t: round(float(t), 4)
    return {
        "mean_mass_top1": f(top1_mass.mean()),
        "mean_mass_entropy": f(ent.mean()),
        "eff_states": f(eff.mean()),
        "argmax_top1_frac": round(sum(am_top1) / H, 4),
        "argmax_active_states": round(sum(am_active) / H, 2),
    }


def _probe_routing(run_id, cfg, model, vocab):
    """Empirical RoLA routing distribution over real MQAR tokens (per kv slice), write + read routers.
    CHEAP (forward only, no SVD), RoLA-only. The forward is CHUNKED over examples and all statistics
    (per-head mean mass, argmax counts, the [nc,nc] joint) are ACCUMULATED — so high RANK_FRAC / long
    L never OOMs (works on a 4090 and at frac=1.0)."""
    from rola_bench.mqar.experiments.rola_router_width_v2 import TEST_CONFIGS
    try:
        mixer = E._find_rola_mixer(model, 0)
    except Exception:
        _append({"run_id": run_id, "ok": False, "probe": "routing",
                 "error": "routing probe is RoLA-only; no RoLA mixer in this model"})
        return False
    import numpy as _np
    H, nc = int(mixer.n_heads), int(mixer.num_chunks)
    cap = {}

    def mk_hook(name):
        def hook(_m, _i, out):
            cap[name] = out.detach()
        return hook

    handles = [mixer.write_router.register_forward_hook(mk_hook("write"))]
    has_read = getattr(mixer, "read_router", None) is not None
    if has_read:
        handles.append(mixer.read_router.register_forward_hook(mk_hook("read")))
    whichs = ["write"] + (["read"] if has_read else [])
    CH = int(os.environ.get("ROUTING_FWD_CHUNK", "8"))      # examples per forward (mem-safe)
    maxL = max(s.input_seq_len for s in TEST_CONFIGS)
    try:
        per_slice = {}
        for sc in TEST_CONFIGS:
            L, kv = sc.input_seq_len, sc.num_kv_pairs
            n = max(1, math.ceil(RANK_FRAC * sc.num_examples))
            batch = E.make_mqar_eval(vocab, L, kv, n, seed=1234, device=DEV)
            dense = (L == maxL)
            mass = {w: torch.zeros(H, nc) for w in whichs}      # accumulated sum of soft mass over tokens
            cnts = {w: torch.zeros(H, nc) for w in whichs}      # accumulated hard-argmax counts
            jointM = _np.zeros((H, nc, nc), dtype=_np.float64) if (dense and has_read) else None
            tok = 0
            for i in range(0, n, CH):
                cap.clear()
                with torch.no_grad():
                    model(batch[i:i + CH])                      # populates cap via the router hooks
                bsz = min(CH, n - i); tok += bsz * L
                cg = {}
                for w in whichs:
                    lo = cap.get(w)
                    if lo is None:
                        continue
                    g = torch.softmax(lo.view(bsz, L, H, nc).float(), dim=-1)   # [b,L,H,nc]
                    cg[w] = g
                    mass[w] += g.sum(dim=(0, 1)).cpu()
                    am = g.argmax(-1)                            # [b,L,H]
                    for h in range(H):
                        cnts[w][h] += torch.bincount(am[:, :, h].reshape(-1), minlength=nc).float().cpu()
                if jointM is not None and "write" in cg and "read" in cg:
                    T = bsz * L
                    for h in range(H):
                        Wt = cg["write"][:, :, h, :].reshape(T, nc)
                        Rt = cg["read"][:, :, h, :].reshape(T, nc)
                        jointM[h] += (Wt.transpose(0, 1) @ Rt).double().cpu().numpy()  # accumulate, /tok later
                del cg
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            mm = {w: (mass[w] / max(tok, 1)) for w in whichs}   # [H,nc] mean soft mass
            entry = {"L": L, "n_examples": n}
            entry["write"] = _routing_summary(mm["write"], cnts["write"], nc)
            entry["read"] = _routing_summary(mm["read"], cnts["read"], nc) if has_read else "tied(=write)"
            if has_read:
                entry["joint"] = _joint_stats(mm["write"], mm["read"])   # read avoid the dominant write state?
            if dense:
                # PER-HEAD vectors (NOT averaged — heads specialize to different states; a fixed-index
                # average blurs the per-head sparsity). plot_routing aligns heads by per-head write-rank.
                vecs = {"kv": kv, "L": L, "n_heads": H}
                for w in whichs:
                    cf = cnts[w] / cnts[w].sum(-1, keepdim=True).clamp_min(1)
                    vecs[f"{w}_mass"] = [[round(x, 6) for x in row] for row in mm[w].tolist()]
                    vecs[f"{w}_argmax_frac"] = [[round(x, 6) for x in row] for row in cf.tolist()]
                # JOINT write x read co-occurrence M[h,i,j]=<write_i read_j>=Wh^T Rh / tok (per head),
                # encoded base64(zlib(float16)) IN the row so it rides the generic fleet json-pull.
                if jointM is not None:
                    M = (jointM / max(tok, 1)).astype(_np.float32)
                    vecs["joint_b64"] = _b64_f16(M)
                    vecs["joint_shape"] = list(M.shape)
                entry["vectors"] = vecs
            per_slice[f"kv{kv}"] = entry
        _append({"run_id": run_id, "ok": True, "probe": "routing", "family": "rola",
                 "n_heads": H, "num_chunks": nc, "frac": RANK_FRAC, "per_slice": per_slice})
        return True
    except Exception as e:  # noqa: BLE001
        _append({"run_id": run_id, "ok": False, "probe": "routing",
                 "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
        return False
    finally:
        for h in handles:
            h.remove()


PROBES = {
    "rank": _probe_rank,        # expensive: per-slice SVD of the attention matrix (any family)
    "routing": _probe_routing,  # cheap: RoLA router state-usage distribution (forward only)
}


def main():
    run_ids = [r for r in os.environ.get("RUN_IDS", "").split(",") if r]
    by_id = _configs_by_runid()
    for rid in run_ids:
        cell, probe = _parse_runid(rid)
        cfg = by_id.get(cell)
        if cfg is None:
            _append({"run_id": rid, "ok": False, "error": "cell not in canonical grid (GRID_TIERS=all?)"})
            print(f"[sim] {rid}: no config", flush=True); continue
        if probe not in PROBES:
            _append({"run_id": rid, "ok": False, "error": f"unknown probe '{probe}'; have {list(PROBES)}"})
            print(f"[sim] {rid}: unknown probe", flush=True); continue
        ck = _ckpt_path(rid)
        if ck is None:
            _append({"run_id": rid, "ok": False, "probe": probe, "error": "checkpoint payload missing"})
            print(f"[sim] {rid}: no ckpt", flush=True); continue
        try:
            model = _build_model(cfg).to(DEV)
            sd = E._load_state_dict(ck, map_location=DEV, state_dict_key="model")
            model.load_state_dict(sd, strict=True)
            model.eval()
            print(f"[sim] {rid} probe={probe} on {DEV} (frac={RANK_FRAC})", flush=True)
            PROBES[probe](rid, cfg, model, cfg.model.vocab_size)
            print(f"[sim] {rid} done", flush=True)
        except Exception as e:  # noqa: BLE001 -- never abort the batch on one cell
            _append({"run_id": rid, "ok": False, "probe": probe,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
            print(f"[sim] {rid} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
