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
  * nc       RoLA states per head; X% is recorded as the realized fraction of attention's KV cache
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
            [--lens 1024,4096,16384,65536] [--ncs 16,64,256,1024] [--dqks 16,64,128]
            [--routings flat,square,tree] [--reps N] [--out DIR]

WARMUP reps absorb each cell's first-touch Triton cold-compile, so cells time warm — no separate
pre-sweep cache-warm pass (the old warm_gate was net-negative: its capture phase serialized the
autotune in-process so the parallel phase only re-hit cache, and it warmed the raw-forward kernels
while this bench fires the kappa-norm kernels — 0/4 overlap).
"""
import argparse
import importlib
import math
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


def _nearest_even_pow2(n):
    """Nearest nc that is BOTH a perfect square AND a power of two — i.e. an EVEN power of two
    (1,4,16,64,256,1024,...) = 4^m. Such an nc is the one value valid for every routing topology at
    once: flat (any), square (sqrt = 2^m is an integer), tree (2^(2m) is a power of two). So all three
    routings in a cell run at the SAME matched state. Rounded in log space (geometric nearest)."""
    if n <= 1:
        return 1
    m = round(math.log2(n) / 2.0)        # 4^m == 2^(2m)
    return 1 << (2 * m)


def nc_for(L, x_pct, d):
    """RoLA states to hit ~X% of attention's KV cache at d_qk=d_v=d (X = nc*d/(2L)), SNAPPED to the
    nearest even-power-of-two so the single nc is valid for flat|square|tree alike (perfect-square AND
    power-of-two — see _nearest_even_pow2). The snap rule is consistent across all cells; realized X =
    nc*d/(2L) follows from the snapped nc (recorded via the row's nc field — no silent corruption)."""
    raw = max(1, round(2 * L * (x_pct / 100.0) / d))
    return _nearest_even_pow2(raw)


def _infeasible_excs():
    """(CUDA OOM, Triton OutOfResources) — the "this config can't run on THIS GPU" signals: out of
    memory, or a kernel that won't fit the GPU's shared-mem/registers at this nc. A cell that raises
    one is an infeasible-frontier SKIP (sentinel row), not a bug — matching the same rule warm uses.
    Any OTHER exception is a real bug (still recorded per-cell, but as ERR, and printed loudly)."""
    oor = ()
    for m in ("triton.runtime.errors", "triton.compiler.errors"):
        try:
            oor = (__import__(m, fromlist=["OutOfResources"]).OutOfResources,)
            break
        except Exception:  # noqa: BLE001
            pass
    return (torch.cuda.OutOfMemoryError,) + oor


_INFEASIBLE = _infeasible_excs()


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
            t, p = _prefill_rep(module, x)
            times.append(t)
            peak = max(peak, p)
    times.sort()
    return round(times[len(times) // 2], 3), round(peak)


def _prefill_rep(module, x):
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); _fwd(module, x); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1), (torch.cuda.max_memory_allocated() - base) / 2 ** 20


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
    COLD un-seeded step; #40 gate-B). Returns (ms/token, peak_mib) — peak_mib is the MEASURED peak
    allocated above the pre-cache baseline during the decode steps (resident state + per-step
    transients); the chunked seed's spike is reset out so it reflects decode, not the prefill. Carried
    alongside the analytic state size as an overhead/sanity cross-check (literature reports ms/token)."""
    if getattr(module, "layer_idx", None) is None:
        module.layer_idx = 0
    seed_len = min(max(1, L), _SEED_PREFILL)
    x_pre = torch.randn(C.B, seed_len, DM, device=C.DEV, dtype=C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated()   # pre-cache baseline (params only)
    with torch.inference_mode():
        cache = _new_cache()
        module(x_pre, past_key_values=cache, use_cache=True)        # prefill -> seeds recurrent_state
        if len(cache) == 0:
            raise RuntimeError(
                "decode arm: cache NOT seeded after prefill (len==0) — the layer dropped the "
                "past_key_values/use_cache kwargs, so this would measure a COLD un-seeded step. "
                "Check the module's forward threads the cache (zoology RoLAMixer wrapper).")
        step = lambda: module(x1, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()  # forget the chunked-seed spike
        ms = round(C.time_forward(step, steps=100, warmup=20), 4)
        peak = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    return ms, peak


def _decode_rola_cell(module, L):
    """RoLA layer decode -> (ms/token, state_mib_analytic, peak_mib_measured). state_mib is the FIXED
    recurrent footprint (the comparison axis); peak_mib is the measured decode-step peak."""
    ms, peak = _decode_layer_step(module, L)
    floats = module.get_stats()["state_floats"]
    return ms, round(C.B * floats * 4 / 2 ** 20, 1), peak


def _decode_monolith_cell(module, L):
    """Matched-state monolith layer decode -> (ms/token, None, peak_mib_measured) via its own use_cache
    path. No clean analytic state axis (the wide nc=1 state), so the analytic slot is None."""
    ms, peak = _decode_layer_step(module, L)
    return ms, None, peak


def _decode_attn_cell(L):
    """LAYER-LEVEL attention decode: one query token through _CausalMHA attending to an L-long cached
    KV (the O(L)-GROWING cache RoLA's bounded state is measured against). Reports
    (ms/token, cache_mib_analytic, peak_mib_measured) — cache_mib is the analytic K+V footprint (the
    comparison axis), peak_mib the measured decode-step peak (cache + SDPA transient)."""
    hd = DM // C.H
    m = _CausalMHA(DM, C.H).to(C.DEV, C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated()   # pre-cache baseline (params only)
    kc = torch.randn(C.B, C.H, L, hd, device=C.DEV, dtype=C.DT)
    vc = torch.randn(C.B, C.H, L, hd, device=C.DEV, dtype=C.DT)
    with torch.inference_mode():
        step = lambda: m(x1, cache=(kc, vc), use_cache=False)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ms = C.time_forward(step, steps=100, warmup=20)
        peak = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    return round(ms, 4), _mib(C.B, C.H, L, hd, 2), peak  # analytic K+V cache + measured peak


# ------------------------------------------------------------------ sweep
def _row(**kw):
    base = dict(bench=None, L=None, state_axis=None, X_pct=None, d_qk=None, nc=None, routing=None, arm=None,
                fwd_ms=None, fb_ms=None, peak_mib=None, act_mib=None, decode_ms=None, state_mib=None)
    base.update(kw)
    return base


def _arm_specs(routings):
    idx = 0
    for kind in ("rola", "monolith"):
        kind_routings = routings if kind == "rola" else (None,)
        for routing in kind_routings:
            for gla, var in ((False, "rla"), (True, "gla")):
                yield dict(idx=idx, kind=kind, routing=routing, gla=gla, arm=f"{kind}-{var}")
                idx += 1


def _append_error(rows, bench, L, x_pct, d, nc, spec, label):
    row = _row(bench=bench, L=L, state_axis="nc", X_pct=x_pct, d_qk=d, nc=nc,
               routing=spec["routing"], arm=spec["arm"])
    if bench == "decode":
        row["decode_ms"] = label
    else:
        row["fwd_ms"] = label
        if bench == "train":
            row["fb_ms"] = label
    rows.append(row)
    return row


def _build_scaling_arm(spec, L, d, nc):
    C.set_shape(L=L, DQK=d, DV=d)
    if spec["kind"] == "rola":
        return build_rola(spec["gla"], nc, d, routing=spec["routing"])
    return build_monolith(spec["gla"], nc, d)


def _summarize_train(rows, L, x_pct, d, nc, spec, recs):
    f, fb, pk, ac = zip(*recs)
    row = _row(bench="train", L=L, state_axis="nc", X_pct=x_pct, d_qk=d, nc=nc,
               routing=spec["routing"], arm=spec["arm"],
               fwd_ms=round(C.median(f), 3), fb_ms=round(C.median(fb), 3),
               peak_mib=round(max(pk)), act_mib=round(max(ac)))
    rows.append(row); print(row, flush=True)


def _summarize_prefill(rows, L, x_pct, d, nc, spec, recs):
    f, pk = zip(*recs)
    row = _row(bench="prefill", L=L, state_axis="nc", X_pct=x_pct, d_qk=d, nc=nc,
               routing=spec["routing"], arm=spec["arm"],
               fwd_ms=round(C.median(f), 3), peak_mib=round(max(pk)))
    rows.append(row); print(row, flush=True)


def _timed_sample(bench, L, d, nc, spec):
    """Run one isolated timing sample for an arm.

    Modules are intentionally built per sample so each arm's peak/OOM reflects only that arm, while the
    outer loop still cycles arms round-robin to share clock/thermal drift.
    """
    seed = 10_000 + L + d * 31 + nc * 17 + spec["idx"]
    torch.manual_seed(seed)
    module = _build_scaling_arm(spec, L, d, nc)
    if module is None:
        return None
    torch.manual_seed(seed + 1)
    x = torch.randn(C.B, L, DM, device=C.DEV, dtype=C.DT)
    if bench == "train":
        leaves = {"x": x.detach().requires_grad_()}
        forward = lambda lv: _fwd(module, lv["x"])
        return _e2e_rep(forward, leaves, module)
    return _prefill_rep(module, x)


def _timed_group(bench, L, x_pct, d, nc, routings, rows):
    """Time comparable RoLA/monolith arms round-robin for one scaling cell.

    Keeping the same `(bench,L,d,nc)` arms in round-robin order makes thermal/clock drift affect their
    ratios similarly. Each sample builds and frees only that arm so peak memory and OOM sentinels remain
    per-arm, matching the old one-arm-at-a-time memory semantics.
    """
    active = [dict(key=f"{spec['arm']}:{spec['routing']}", spec=spec, data=[])
              for spec in _arm_specs(routings)]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*format mismatch.*")
        for r in range(C.WARMUP + C.REPS):
            still_active = []
            for rec in active:
                spec = rec["spec"]
                try:
                    sample = _timed_sample(bench, L, d, nc, spec)
                    if sample is None:
                        continue
                    if r >= C.WARMUP:
                        rec["data"].append(sample)
                    still_active.append(rec)
                except _INFEASIBLE:
                    row = _append_error(rows, bench, L, x_pct, d, nc, spec, "OOM")
                    torch.cuda.empty_cache()
                    print(f"OOM {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}", flush=True)
                    print(row, flush=True)
                except Exception as e:  # noqa: BLE001
                    row = _append_error(rows, bench, L, x_pct, d, nc, spec, "ERR")
                    torch.cuda.empty_cache()
                    print(f"ERR {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    print(row, flush=True)
                finally:
                    torch.cuda.empty_cache()
            active = still_active
            if not active:
                break

    for rec in active:
        if not rec["data"]:
            continue
        if bench == "train":
            _summarize_train(rows, L, x_pct, d, nc, rec["spec"], rec["data"])
        else:
            _summarize_prefill(rows, L, x_pct, d, nc, rec["spec"], rec["data"])


def sweep(benches, lens, ncs, dqks, rows, routings=ROUTINGS):
    for bench in benches:
        for L in lens:
            C.set_shape(L=L)
            # ---- attention reference: depends only on L (no X / d_qk / routing axis) ----
            try:
                if bench == "decode":
                    ms, mib, pk = _decode_attn_cell(L)
                    rows.append(_row(bench=bench, L=L, arm="attn", decode_ms=ms, state_mib=mib, peak_mib=pk))
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
            except _INFEASIBLE:
                rows.append(_row(bench=bench, L=L, arm="attn", fwd_ms="OOM", decode_ms="OOM")); torch.cuda.empty_cache()
                print(f"OOM attn {bench} L{L}", flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append(_row(bench=bench, L=L, arm="attn", fwd_ms="ERR")); torch.cuda.empty_cache()
                print(f"ERR attn {bench} L{L}: {type(e).__name__}: {e}", flush=True)

            # ---- RoLA (routing sweep) + its matched-state wide-monolith ablation: (X%, d_qk) grid ----
            for d in dqks:
                for nc in ncs:
                    # BOUNDED-STATE REGIME ONLY. RoLA content state (nc·d²) vs attention's KV cache
                    # (2·L·d): include the cell ONLY while state < KV, i.e. nc·d < 2L. At/above that
                    # (nc·d >= 2L) RoLA holds MORE than the full KV cache — the regime it isn't for —
                    # so skip it. nc is a fixed sweep value (16/64/256/1024, each even-pow2 -> valid
                    # for flat|square|tree); x_pct is the realized state-as-%-of-KV, recorded per row.
                    if nc * d >= 2 * L:
                        continue
                    x_pct = round(100.0 * nc * d / (2 * L), 1)
                    if bench in ("train", "prefill"):
                        _timed_group(bench, L, x_pct, d, nc, routings, rows)
                        continue
                    for spec in _arm_specs(routings):
                        try:
                            m = _build_scaling_arm(spec, L, d, nc)
                            if m is None:
                                continue                   # baseline_cell declared this cell infeasible
                            fn = _decode_rola_cell if spec["kind"] == "rola" else _decode_monolith_cell
                            ms, mib, pk = fn(m, L)
                            rows.append(_row(bench=bench, L=L, state_axis="nc", X_pct=x_pct, d_qk=d, nc=nc,
                                             routing=spec["routing"], arm=spec["arm"],
                                             decode_ms=ms, state_mib=mib, peak_mib=pk))
                            del m
                            print(rows[-1], flush=True)
                        except _INFEASIBLE:
                            row = _append_error(rows, bench, L, x_pct, d, nc, spec, "OOM")
                            torch.cuda.empty_cache()
                            print(f"OOM {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}", flush=True)
                            print(row, flush=True)
                        except Exception as e:  # noqa: BLE001
                            row = _append_error(rows, bench, L, x_pct, d, nc, spec, "ERR")
                            torch.cuda.empty_cache()
                            print(f"ERR {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}: "
                                  f"{type(e).__name__}: {e}", flush=True)
                            print(row, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="RoLA scaling benchmark (RoLA vs attention; per-arch curves).")
    ap.add_argument("--benches", default="train,prefill,decode", help="comma: train,prefill,decode")
    ap.add_argument("--lens", default=os.environ.get("SCALE_LENS", "1024,4096,16384,65536"))
    ap.add_argument("--ncs", default=os.environ.get("SCALE_NCS", "16,64,256,1024"),
                    help="fixed states_per_head to sweep (cells kept only while state<KV: nc*d<2L)")
    ap.add_argument("--dqks", default=os.environ.get("SCALE_DQKS", "16,64,128"))
    ap.add_argument("--routings", default=os.environ.get("SCALE_ROUTINGS", "flat,square,tree"),
                    help="comma: RoLA per-head factor topologies to sweep")
    ap.add_argument("--reps", type=int, default=C.REPS)
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    C.require_cuda()
    C.REPS = a.reps
    benches = [b for b in a.benches.split(",") if b.strip()]
    lens = [int(v) for v in a.lens.split(",") if v.strip()]
    ncs = [int(v) for v in a.ncs.split(",") if v.strip()]
    dqks = [int(v) for v in a.dqks.split(",") if v.strip()]
    routings = [r for r in a.routings.split(",") if r.strip()]
    print(f"# scaling | {C.gpu_name()} | B={C.B} H={C.H} d_model={DM} bf16 | benches={benches} "
          f"| L={lens} nc={ncs} d_qk={dqks} routing={routings} (cells: state<KV only) | RoLA(rla,gla) + attn", flush=True)
    rows = []
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    try:
        sweep(benches, lens, ncs, dqks, rows, routings=routings)
    finally:
        if rows:
            C.write_csv(out / "scaling.csv", rows)
            print(f"\nWROTE {out}/scaling.csv  [{C.gpu_name()}]  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
