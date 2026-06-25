"""RoLA SCALING benchmark — per-architecture characterization, NOT cross-architecture comparison.

Arms: **RoLA** (rla + gla inner kernels, swept over routing structure {flat,square,tree}), its
matched-state **wide-monolith** equivalent (nc=1, all the state in one wide head — the "what if you
didn't route" ablation, RoLA's own control, NOT a zoo baseline), and **softmax attention**. Rationale
for excluding the rest: GDN / GLA / Based / ... are the SAME complexity class as RoLA (O(L) compute,
bounded state), so on an L-axis they are parallel lines with different constants — "the constants" is
a matched-state COMPARISON question (see perf/comparison), not a scaling-shape one. Attention is the
ONLY baseline that scales DIFFERENTLY in L (O(L^2) compute, O(L) growing KV cache), the entire reason
RoLA exists. The wide monolith earns its place as RoLA's structural ablation (matched state,
routed-vs-not), not as a competitor.

IMPORTANT: the numbers here are NOT a fairness verdict. RoLA sitting at X<100% of attention's KV
cache is the bounded-state VALUE PROPOSITION (the whole point), not an unfair tilt — bounded-vs-
unbounded is the legitimate contrast. The output is the SHAPE of each curve and where RoLA crosses
attention; head-to-head verdicts at matched state live in the (separate) comparison benchmark.

EVERYTHING IS LAYER-vs-LAYER: every arm instantiates its CANONICAL nn.Module at d_model=DM and times
`module(x)` — no hand-assembled op-on-synthetic-tensor scaffold (the user was burnt by op-level
benches). The decode arm too: RoLA's layer `forward(use_cache=True)` recurrent path vs the attention
KV-cache decode, both timed as one fused layer region.

Axes:
  * L        sequence length (the primary scaling axis)
  * X%       RoLA content state as a fraction of attention's KV cache -> nc = round(2*L*X/d), d_qk=d_v=d
  * d_qk     feature dim {16,64,128} -- the occupancy / operating-point knob (16 = the starved corner)
  * routing  the per-head factor topology {flat,square,tree} the RoLA layer routes through in-kernel
             (flat = dense router; square/tree = factored). RoLA-only axis (monolith/attn don't route).
  * bench    training (chunked fwd+bwd) | prefill (chunked fwd-only) | decode (recurrent, per-token)

attention depends only on L (no X / d_qk / routing axis) -> timed once per (bench, L) and reused as the
reference curve. Each cell is wrapped so an OOM/error writes a sentinel and the sweep continues;
the CSV is written in a `finally`. Use small PERF_B (e.g. 2) so long L fits.

This module STANDS ALONE: it owns its copies of the shared helpers (_CausalMHA / _instantiate /
_e2e_rep) — no import from the archived perf/scratch/{chunked,recurrent}.py.

Run:  python -m rola_bench.perf.scaling [--benches train,prefill,decode]
            [--lens 1024,4096,16384,65536] [--xs 10,25,50,100] [--dqks 16,64,128]
            [--routings flat,square,tree] [--reps N] [--out DIR]
"""
import argparse
import importlib
import os
import warnings
from pathlib import Path

import torch

from zoology.mixers.rola import RoLAMixer
from rola_bench.mqar.instances import rola_instance

from . import _common as C
import rola_bench.mqar.experiments.canonical_baselines as cb

DM = 512  # d_model backbone for the scaling layers
ROUTINGS = ("flat", "square", "tree")  # the per-head factor topologies the RoLA layer supports


def nc_for(L, x_pct, d):
    """RoLA states to hit X% of attention's KV cache at d_qk=d_v=d, from X = nc*d/(2L)."""
    return max(1, round(2 * L * (x_pct / 100.0) / d))


def _mib(*dims):
    n = 1
    for x in dims:
        n *= x
    return round(n * 4 / 2 ** 20, 1)


# ------------------------------------------------------------------ shared helpers (own copies)
def _instantiate(kernel_cfg, DM):
    """Build a canonical baseline module from a baseline_cell kernel dict. `name` is a dotted import
    path; kwargs already carry hidden_size for the fla classes, so d_model is injected only for the
    zoology mixers (which take d_model= and have neither key in their kwargs). Relocated here so the
    scaling path stands alone (was imported from the now-archived chunked.py)."""
    mod_path, cls_name = kernel_cfg["name"].rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_path), cls_name)
    kwargs = dict(kernel_cfg["kwargs"])
    if "hidden_size" not in kwargs and "d_model" not in kwargs:
        kwargs["d_model"] = DM
    return cls(**kwargs).to(C.DEV, C.DT)


class _CausalMHA(torch.nn.Module):
    """Canonical causal multi-head attention via SDPA (flash backend on A100/bf16) — the O(L^2)
    reference for the scaling sweep (RoLA's O(L) crossover + the within-window ceiling). Relocated
    here so scaling.py stands alone. Supports a forward `kv` cache for the layer-level decode arm:
    pass `cache=(k_cache,v_cache)` and `use_cache=True` to append the new token and attend to the
    full grown KV (the O(L)-growing-cache decode RoLA's bounded state is measured against)."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.nh, self.hd = n_heads, d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, self.nh, self.hd).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            kc, vc = cache
            k = torch.cat([kc, k], dim=2)
            v = torch.cat([vc, v], dim=2)
        # prefill (L>1) is causal; single-token decode attends to the whole cached past (no mask).
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(L > 1))
        o = self.proj(o.transpose(1, 2).reshape(B, L, D))
        if use_cache:
            return o, (k, v)
        return o


def _e2e_rep(forward, leaves, module):
    """One fused fwd+bwd rep -> (fwd_ms, fwd_bwd_ms, peak_mib, act_mib). act_mib = bytes RETAINED for
    backward (saved tensors, deduped by storage) — the memory that accumulates across depth; peak_mib
    is the transient max. Clears x AND param grads per rep. Relocated here from chunked.py."""
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    probe = C.ActivationProbe()
    with probe:
        e0.record(); o = forward(leaves); e1.record()
    torch.cuda.synchronize()
    act = probe.mib                                       # retained-for-backward (across-depth) memory
    o.float().sum().backward(); e2.record(); torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    leaves["x"].grad = None
    module.zero_grad(set_to_none=True)
    return e0.elapsed_time(e1), e0.elapsed_time(e2), peak, act


# ------------------------------------------------------------------ builders
def build_rola(gla, nc, d, routing="flat"):
    """Routed RoLA LAYER at d_qk=d_v=d, nc states, `routing` factor topology, perf backbone. Returns
    the nn.Module (layer_idx=0 so it can carry a KV cache for the layer-level decode arm)."""
    kw = rola_instance("rola-gla-kappa-asym" if gla else "rola-rla-kappa-asym",
                       d_qk=d, d_v=d, num_chunks=nc, n_heads=C.H, routing=routing)
    return RoLAMixer(d_model=DM, layer_idx=0, **kw).to(C.DEV, C.DT)


def build_monolith(gla, nc, d):
    """The matched-state WIDE monolith (nc=1, one big state of size H·nc·d·d, d_k widened to nc·d) —
    RoLA's routed-vs-not ablation. Reuses canonical_baselines.baseline_cell at the perf shape so the
    state exactly matches build_rola(gla,nc,d). Returns the module, or None if infeasible."""
    cb.DMODEL = DM; cb.BASE_H = C.H; cb.DV = d
    cb.REF = lambda _nc: C.H * _nc * d * d           # match RoLA's content state at this (nc,d) cell
    cell = cb.baseline_cell("gla" if gla else "rla", "wide", nc)
    if cell is None:
        return None
    kernel_cfg, _state = cell
    return _instantiate(kernel_cfg, DM)


def _fwd(module, x):
    o = module(x)
    return o[0] if isinstance(o, tuple) else o


# ------------------------------------------------------------------ timing reps
def _train_cell(module, L):
    """fwd+bwd median over reps -> (fwd_ms, fb_ms, peak_mib, act_mib)."""
    x = torch.randn(C.B, L, DM, device=C.DEV, dtype=C.DT)
    leaves = {"x": x.detach().requires_grad_()}
    fwd = lambda lv: _fwd(module, lv["x"])
    recs = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*format mismatch.*")
        for r in range(C.WARMUP + C.REPS):
            rec = _e2e_rep(fwd, leaves, module)
            if r >= C.WARMUP:
                recs.append(rec)
    f, fb, pk, ac = zip(*recs)
    return round(C.median(f), 3), round(C.median(fb), 3), round(max(pk)), round(max(ac))


def _prefill_cell(module, L):
    """forward-only (inference_mode, no backward) median -> (fwd_ms, peak_mib)."""
    x = torch.randn(C.B, L, DM, device=C.DEV, dtype=C.DT)
    times, peak = [], 0
    with torch.inference_mode():
        for _ in range(C.WARMUP):
            _fwd(module, x)
        torch.cuda.synchronize()
        for _ in range(C.REPS):
            torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); _fwd(module, x); e1.record(); torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1))
            peak = max(peak, (torch.cuda.max_memory_allocated() - base) / 2 ** 20)
    times.sort()
    return round(times[len(times) // 2], 3), round(peak)


def _new_cache():
    """A fresh fla_rola Cache for the layer-level decode arm (carries the per-layer recurrent state)."""
    from fla_rola.models.utils import Cache
    return Cache()


_SEED_PREFILL = 256   # tokens used to SEED the recurrent state (fixed-size; seeding doesn't need full L)


def _decode_layer_step(module, L):
    """LAYER-LEVEL decode: SEED the layer's recurrent state from a prefill, then time the single-token
    decode step `module.forward(x1, past_key_values=cache, use_cache=True)` — the shipped recurrent
    decode path (RoLA auto-selects fused_recurrent for L<=64). The recurrent state is FIXED-SIZE (it
    does NOT grow with steps nor with the prefill length — that's the whole point vs attention's
    growing KV), so seeding from a bounded prefill is faithful and each timed step runs into the SAME
    persisted cache: a drift-free steady-state seeded decode (no per-step clone, no op-level call).

    ASSERTS the cache is actually seeded after the prefill (len>0) — a fail-loud guard against a
    silently-dropped cache kwarg (the zoology wrapper used to swallow it, which secretly measured a
    COLD un-seeded step; #40 gate-B). Returns ms/token."""
    if getattr(module, "layer_idx", None) is None:
        module.layer_idx = 0
    seed_len = min(max(1, L), _SEED_PREFILL)
    x_pre = torch.randn(C.B, seed_len, DM, device=C.DEV, dtype=C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    with torch.inference_mode():
        cache = _new_cache()
        module(x_pre, past_key_values=cache, use_cache=True)        # prefill -> seeds recurrent_state
        if len(cache) == 0:
            raise RuntimeError(
                "decode arm: cache NOT seeded after prefill (len==0) — the layer dropped the "
                "past_key_values/use_cache kwargs, so this would measure a COLD un-seeded step. "
                "Check the module's forward threads the cache (zoology RoLAMixer wrapper).")
        step = lambda: module(x1, past_key_values=cache, use_cache=True)
        return round(C.time_forward(step, steps=100, warmup=20), 4)


def _decode_rola_cell(module, L):
    """RoLA layer decode -> (ms/token, state_mib). state_mib is the FIXED recurrent footprint."""
    ms = _decode_layer_step(module, L)
    floats = module.get_stats()["state_floats"]
    return ms, round(C.B * floats * 4 / 2 ** 20, 1)


def _decode_monolith_cell(module, L):
    """Matched-state monolith layer decode -> (ms/token, None) via its own use_cache path."""
    return _decode_layer_step(module, L), None


def _decode_attn_cell(L):
    """LAYER-LEVEL attention decode: one query token through _CausalMHA attending to an L-long cached
    KV (the O(L)-GROWING cache RoLA's bounded state is measured against). Reports (ms/token, cache_mib)."""
    hd = DM // C.H
    m = _CausalMHA(DM, C.H).to(C.DEV, C.DT)
    kc = torch.randn(C.B, C.H, L, hd, device=C.DEV, dtype=C.DT)
    vc = torch.randn(C.B, C.H, L, hd, device=C.DEV, dtype=C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    with torch.inference_mode():
        step = lambda: m(x1, cache=(kc, vc), use_cache=False)
        ms = C.time_forward(step, steps=100, warmup=20)
    return round(ms, 4), _mib(C.B, C.H, L, hd, 2)  # K+V cache (the O(L)-growing footprint)


# ------------------------------------------------------------------ sweep
def _row(**kw):
    base = dict(bench=None, L=None, X_pct=None, d_qk=None, nc=None, routing=None, arm=None,
                fwd_ms=None, fb_ms=None, peak_mib=None, act_mib=None, decode_ms=None, state_mib=None)
    base.update(kw)
    return base


def sweep(benches, lens, xs, dqks, rows, routings=ROUTINGS):
    for bench in benches:
        for L in lens:
            C.set_shape(L=L)
            # ---- attention reference: depends only on L (no X / d_qk / routing axis) ----
            try:
                if bench == "decode":
                    ms, mib = _decode_attn_cell(L)
                    rows.append(_row(bench=bench, L=L, arm="attn", decode_ms=ms, state_mib=mib))
                else:
                    m = _CausalMHA(DM, C.H).to(C.DEV, C.DT)
                    if bench == "train":
                        fwd, fb, pk, ac = _train_cell(m, L)
                        rows.append(_row(bench=bench, L=L, arm="attn", fwd_ms=fwd, fb_ms=fb, peak_mib=pk, act_mib=ac))
                    else:  # prefill
                        fwd, pk = _prefill_cell(m, L)
                        rows.append(_row(bench=bench, L=L, arm="attn", fwd_ms=fwd, peak_mib=pk))
                    del m
                print(rows[-1], flush=True)
            except torch.cuda.OutOfMemoryError:
                rows.append(_row(bench=bench, L=L, arm="attn", fwd_ms="OOM", decode_ms="OOM")); torch.cuda.empty_cache()
                print(f"OOM attn {bench} L{L}", flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append(_row(bench=bench, L=L, arm="attn", fwd_ms="ERR")); torch.cuda.empty_cache()
                print(f"ERR attn {bench} L{L}: {type(e).__name__}: {e}", flush=True)

            # ---- RoLA (routing sweep) + its matched-state wide-monolith ablation: (X%, d_qk) grid ----
            for d in dqks:
                for x_pct in xs:
                    nc = nc_for(L, x_pct, d)
                    for kind in ("rola", "monolith"):
                        # routing is a RoLA-ONLY axis; the monolith doesn't route -> a single None pass.
                        kind_routings = routings if kind == "rola" else (None,)
                        for routing in kind_routings:
                            for gla, var in ((False, "rla"), (True, "gla")):
                                arm = f"{kind}-{var}"
                                try:
                                    C.set_shape(L=L, DQK=d, DV=d)
                                    if kind == "rola":
                                        m = build_rola(gla, nc, d, routing=routing)
                                    else:
                                        m = build_monolith(gla, nc, d)
                                    if m is None:
                                        continue                   # baseline_cell declared it infeasible
                                    if bench == "decode":
                                        fn = _decode_rola_cell if kind == "rola" else _decode_monolith_cell
                                        ms, mib = fn(m, L)
                                        rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc,
                                                         routing=routing, arm=arm, decode_ms=ms, state_mib=mib))
                                    elif bench == "train":
                                        fwd, fb, pk, ac = _train_cell(m, L)
                                        rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc,
                                                         routing=routing, arm=arm,
                                                         fwd_ms=fwd, fb_ms=fb, peak_mib=pk, act_mib=ac))
                                    else:
                                        fwd, pk = _prefill_cell(m, L)
                                        rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc,
                                                         routing=routing, arm=arm, fwd_ms=fwd, peak_mib=pk))
                                    del m
                                    print(rows[-1], flush=True)
                                except torch.cuda.OutOfMemoryError:
                                    rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc,
                                                     routing=routing, arm=arm, fwd_ms="OOM"))
                                    torch.cuda.empty_cache()
                                    print(f"OOM {arm} {bench} L{L} X{x_pct} d{d} r{routing} nc{nc}", flush=True)
                                except Exception as e:  # noqa: BLE001
                                    rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc,
                                                     routing=routing, arm=arm, fwd_ms="ERR"))
                                    torch.cuda.empty_cache()
                                    print(f"ERR {arm} {bench} L{L} X{x_pct} d{d} r{routing} nc{nc}: "
                                          f"{type(e).__name__}: {e}", flush=True)
    return rows


def warm(lens, xs, dqks, progress=True):
    """Pre-warm the Triton cache (chunk_rola_routed forward autotune) for the RoLA shapes the sweep
    will hit, BEFORE the timed cells, so they run WARM (no cold-compile time folded into a timing).
    Integrates fla_rola.precompile.warm_gate (repointed to chunk_rola_routed by #44). Best-effort: a
    warm failure prints and the sweep proceeds (cells just pay cold-compile on first touch)."""
    try:
        from fla_rola.precompile import warm_gate, ShapeSpec
    except Exception as e:  # noqa: BLE001
        print(f"# warm_gate unavailable ({type(e).__name__}: {e}); cells will cold-compile", flush=True)
        return None
    specs, seen = [], set()
    for L in lens:
        for d in dqks:
            for x_pct in xs:
                nc = nc_for(L, x_pct, d)
                for path in ("rla", "gla"):
                    key = (path, d, d, nc)
                    if key not in seen:
                        seen.add(key)
                        specs.append(ShapeSpec(path=path, dqk=d, dv=d, nc=nc, norm="kappa"))
    print(f"# warm_gate: {len(specs)} RoLA shapes", flush=True)
    summary = warm_gate(specs, progress=progress)
    print(f"# warm_gate done: {summary.get('n_warmed', '?')} warmed in "
          f"{summary.get('total_wall_s', 0):.1f}s", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="RoLA scaling benchmark (RoLA vs attention; per-arch curves).")
    ap.add_argument("--benches", default="train,prefill,decode", help="comma: train,prefill,decode")
    ap.add_argument("--lens", default=os.environ.get("SCALE_LENS", "1024,4096,16384,65536"))
    ap.add_argument("--xs", default=os.environ.get("SCALE_XS", "10,25,50,100"), help="state %% of KV cache")
    ap.add_argument("--dqks", default=os.environ.get("SCALE_DQKS", "16,64,128"))
    ap.add_argument("--routings", default=os.environ.get("SCALE_ROUTINGS", "flat,square,tree"),
                    help="comma: RoLA per-head factor topologies to sweep")
    ap.add_argument("--reps", type=int, default=C.REPS)
    ap.add_argument("--no-warm", action="store_true", help="skip the pre-sweep warm_gate")
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    C.require_cuda()
    C.REPS = a.reps
    benches = [b for b in a.benches.split(",") if b.strip()]
    lens = [int(v) for v in a.lens.split(",") if v.strip()]
    xs = [float(v) for v in a.xs.split(",") if v.strip()]
    dqks = [int(v) for v in a.dqks.split(",") if v.strip()]
    routings = [r for r in a.routings.split(",") if r.strip()]
    print(f"# scaling | {C.gpu_name()} | B={C.B} H={C.H} d_model={DM} bf16 | benches={benches} "
          f"| L={lens} X%={xs} d_qk={dqks} routing={routings} | RoLA(rla,gla) + attn", flush=True)
    if not a.no_warm:
        warm(lens, xs, dqks)
    rows = []
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    try:
        sweep(benches, lens, xs, dqks, rows, routings=routings)
    finally:
        if rows:
            C.write_csv(out / "scaling.csv", rows)
            print(f"\nWROTE {out}/scaling.csv  [{C.gpu_name()}]  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
