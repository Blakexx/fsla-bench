"""Chunked-parallel kernel-efficiency benchmark (forward + backward).

The training-time regime: chunk-parallel attention with a shared content Gram. At MATCHED
total state we compare a set of implementations, for both the RLA (no decay) and scalar-GLA
(per-state decay) inner kernels, plus cross-architecture baselines that hold the SAME recurrent
state. Two axes mirror the MQAR accuracy grid exactly (so perf cells == accuracy cells):

  * state rung: the full nc scan {2,4,8,16,32,64,128,256} (_common.NCS), not just {16,64,256}.
  * shape: HOW a baseline spends its matched state — wide (widen keys), square (d_k=d_v), heads
    (scale head count). The dim solve mirrors rola_bench/mqar/experiments/canonical_baselines.py,
    in the perf base shape (H=8, d_qk=d_v=16): total recurrent floats S = H*nc*d_qk*d_v held
    constant across shapes. heads is capped at nc<=32 and gdn-wide/square at head_dim<=256, exactly
    as the accuracy grid returns None there (no phantom cells the accuracy table doesn't have).

RoLA-family arms (variant in {RLA, GLA}; the PROPOSED method, native shape only):
  rola      - the fla_rola routed `chunk_simple_gla` (shared-Gram Triton, r/w[/g] params).
  vh        - virtual-head on CANONICAL fla: q/k replicated across nc states folded into the head
              axis, v pre-scaled by the write gate, read-combine outside. The honest "what you'd do
              without the routed kernel" baseline.
  kappa     - rola + the per-state-denominator normalization pre-pass (kappa=0.5); isolates the
              den-kernel cost on top of rola.

Baseline arms, swept over wide/square/heads (each its own published kernel):
  monolith  - canonical fla simple_gla, ONE dense state per head (no routing). The RLA/GLA dense
              lower bound; shaped by widening d_k (wide), d_k=d_v (square), or scaling H (heads).
  gdn       - fla `chunk_gated_delta_rule`. head_dim widens the recall axis; kernel-capped at 256.
  based     - fla `fused_chunk_based` (Taylor-exp map). Single head, feature_dim the sole knob ->
              wide-only (square/heads undefined, as in the accuracy grid).
  hedgehog  - fla `chunk_linear_attn` over the canonical `HedgehogFeatureMap`. The map doubles the
              feature dim (cat([2x,-2x]).softmax), so state = (2*d_k)*d_v (fm=2 in the solve).

Reports, per cell: variant, nc, shape, impl, fwd_ms, fwd_iqr, fb_ms (fwd+bwd), fb_iqr, peak_mib,
act_mib, plus a projection-cost table (stage-1 GEMMs + params) and a combined LaTeX table with
stage1+stage2 totals. Outputs to results/perf/ (override with --out).

ROBUSTNESS: each cell is wrapped so a CUDA OOM (or any cell error) writes an 'OOM' sentinel row,
empties the cache, and continues. Cells the shape solve declares infeasible (heads at nc>32, gdn
head_dim>256) are SKIPPED (no row), exactly as the accuracy grid omits them. The CSV is written in
a `finally` so a partial run is never lost.

Run:  python -m rola_bench.perf.chunked [--reps N] [--out DIR]   (idle GPU)
"""
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn

import importlib

import fla.ops.simple_gla as fla_sg
import fla_rola.ops.simple_gla as fork_sg
from fla_rola.ops.rola import chunk_rola
from fla_rola.ops.based import fused_chunk_based
from fla_rola.ops.gated_delta_rule import chunk_gated_delta_rule
from fla_rola.ops.linear_attn import chunk_linear_attn
from fla_rola.modules.feature_map import HedgehogFeatureMap

from zoology.mixers.rola import RoLAMixer
from rola_bench.mqar.instances import rola_instance
import rola_bench.mqar.experiments.canonical_baselines as cb

from . import _common as C


# --------------------------------------------------------------------------------------------
# Shared input maker for the RoLA-family arms (rola / vh / kappa) at the NATIVE shape (H, d_qk, d_v).
# --------------------------------------------------------------------------------------------
def mk(nc, gla, seed=0):
    """Routed inputs at matched total state. Signed q/k/v; routers are softmax simplex weights."""
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    t = dict(
        q=g(C.B, C.L, C.H, C.DQK), k=g(C.B, C.L, C.H, C.DQK), v=g(C.B, C.L, C.H, C.DV),
        r=torch.softmax(g(C.B, C.L, C.H, nc).float(), -1).to(C.DT),
        w=torch.softmax(g(C.B, C.L, C.H, nc).float(), -1).to(C.DT),
    )
    if gla:
        alpha = torch.sigmoid(g(C.B, C.L, C.H, nc).float())
        t["ld"] = (1 - t["w"].float() * (1 - alpha)).clamp(min=1e-6).log().clamp(min=-2.5).to(C.DT)
    return t


def run_rola(t, gla):
    # un-normalized routed readout (numerator only) via the first-class op
    return chunk_rola(t["q"], t["k"], t["v"], r=t["r"], w=t["w"], g=t.get("ld"), norm="raw", scale=1.0)


def run_kappa(t, gla):
    """Per-state-denominator (kappa=0.5) normalization — chunk_rola owns the den pre-pass, gate
    rescale, numerator-only readout, and divide. Times the production kappa path end-to-end."""
    B, L, H, _ = t["q"].shape
    kap = torch.full((B, L, H, 1), 0.5, device=t["q"].device, dtype=t["q"].dtype)
    return chunk_rola(t["q"], t["k"], t["v"], r=t["r"], w=t["w"], g=t.get("ld"),
                      norm="kappa", kappa=kap, scale=1.0)


def run_vh(t, gla):
    nc = t["r"].shape[-1]
    qv = t["q"].unsqueeze(3).expand(C.B, C.L, C.H, nc, C.DQK).reshape(C.B, C.L, C.H * nc, C.DQK)
    kv = t["k"].unsqueeze(3).expand(C.B, C.L, C.H, nc, C.DQK).reshape(C.B, C.L, C.H * nc, C.DQK)
    vv = (t["v"].unsqueeze(3) * t["w"].unsqueeze(-1)).reshape(C.B, C.L, C.H * nc, C.DV)
    g = t["ld"].reshape(C.B, C.L, C.H * nc).float() if gla else None
    o, _ = fla_sg.chunk_simple_gla(qv, kv, vv, g=g, scale=1.0)
    return (o.view(C.B, C.L, C.H, nc, C.DV) * t["r"].unsqueeze(-1)).sum(3)


# Matched-state shaped dims (wide/square/heads) live in _common, shared with the recurrent bench:
# C.shaped_dims(kind, shape, nc) -> {H,d_k,d_v} or None. kind: 'lin1' monolith / 'lin2' hedgehog
# (feature-doubling) / 'gdn' (head_dim<=256). gdn-wide skips nc>=32; heads skips nc>32.
shaped_dims = C.shaped_dims


# --------------------------------------------------------------------------------------------
# Baseline input makers + run_fns. Every maker has signature (shape, nc, gla) and returns the input
# dict, or None when shaped_dims declares the cell infeasible (caller skips it). gla is ignored by
# the 'own' (cross-arch) arms.
# --------------------------------------------------------------------------------------------
def mk_mono(shape, nc, gla, seed=0):
    d = shaped_dims("lin1", shape, nc)
    if d is None:
        return None
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    t = dict(q=g(C.B, C.L, d["H"], d["d_k"]), k=g(C.B, C.L, d["H"], d["d_k"]),
             v=g(C.B, C.L, d["H"], d["d_v"]))
    if gla:                                              # one scalar log-decay per head
        t["mg"] = torch.nn.functional.logsigmoid(g(C.B, C.L, d["H"]).float()).to(C.DT)
    return t


def run_mono(t, gla):
    o, _ = fla_sg.chunk_simple_gla(t["q"], t["k"], t["v"],
                                   g=(t["mg"].float() if gla else None), scale=1.0)
    return o


def mk_gdn(shape, nc, gla=False, seed=0):
    d = shaped_dims("gdn", shape, nc)
    if d is None:
        return None
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    # q AND k are raw (unnormalized): the canonical fla.layers.GatedDeltaNet L2-normalizes BOTH
    # q and k INSIDE the kernel (use_qk_l2norm_in_kernel=True). Pre-normalizing k here (and leaving
    # q raw) measured a DIFFERENT computation than the real layer and hid the in-kernel norm cost.
    # beta in (0,1); g in log space (logsigmoid).
    return dict(
        q=g(C.B, C.L, d["H"], d["d_k"]),
        k=g(C.B, C.L, d["H"], d["d_k"]),
        v=g(C.B, C.L, d["H"], d["d_v"]),
        beta=torch.rand(C.B, C.L, d["H"], device=C.DEV, dtype=C.DT).sigmoid(),
        ldg=torch.nn.functional.logsigmoid(g(C.B, C.L, d["H"]).float()).to(C.DT),
    )


def run_gdn(t, gla):
    # use_qk_l2norm_in_kernel=True == fla.layers.GatedDeltaNet (normalizes q,k in-kernel).
    o, _ = chunk_gated_delta_rule(t["q"], t["k"], t["v"], g=t["ldg"], beta=t["beta"], scale=1.0,
                                  use_qk_l2norm_in_kernel=True)
    return o


def mk_hedgehog(shape, nc, gla=False, seed=0):
    d = shaped_dims("lin2", shape, nc)
    if d is None:
        return None
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    fmap = HedgehogFeatureMap(head_dim=d["d_k"]).to(C.DEV, C.DT)
    return dict(
        q=g(C.B, C.L, d["H"], d["d_k"]), k=g(C.B, C.L, d["H"], d["d_k"]),
        v=g(C.B, C.L, d["H"], d["d_v"]), _fmap=fmap,
    )


def run_hedgehog(t, gla):
    # Canonical Hedgehog: feature map on q AND k, then normalized linear attn (chunk_linear_attn).
    phi_q = t["_fmap"](t["q"])
    phi_k = t["_fmap"](t["k"])
    o, _ = chunk_linear_attn(phi_q, phi_k, t["v"], scale=1.0, normalize=True)
    return o


# Based: `fused_chunk_based` applies the 2nd-order Taylor map INSIDE the kernel, so the recurrent
# state per head is (1 + K + K^2) * V (K = raw feature dim, kernel-capped at 16). Based's identity is
# the rich Taylor map, so we pin K at the kernel cap (16) and let V absorb the state to hit nc*d_qk*d_v
# (mirrors the MQAR 'wide' Based: feature_dim the sole knob). Single head -> square/heads undefined.
def based_dims(nc):
    K = 16  # kernel asserts q.shape[-1] <= 16
    # feat = the fla fused_chunk_based kernel's TRUE expanded feature dim = 1 + K + K*K (FULL outer
    # product — its recurrent kv state is [feat, d_v]; cf. ops/based/naive.py whose quadratic einsum
    # uses INDEPENDENT key-feature indices 'x y', not a triangular dedup). This is the honest state of
    # the kernel we benchmark, so V solves 273*V == nc*d_qk*d_v to hold total state. (NB: the accuracy
    # grid's zoology Based uses the TRIANGULAR taylor expanded_size 1+K+K(K+1)/2=153 — a different,
    # numerically-equivalent kernel that stores each symmetric pair once; each bench correctly holds S
    # for ITS own kernel.)
    feat = 1 + K + K * K  # 273
    V = max(1, round(C.DQK * C.DV * nc / feat))   # per-head matched state nc*d_qk*d_v
    return dict(K=K, d_v=V)


def mk_based(shape, nc, gla=False, seed=0):
    if shape != "wide":
        return None
    d = based_dims(nc)
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    return dict(q=g(C.B, C.L, C.H, d["K"]), k=g(C.B, C.L, C.H, d["K"]), v=g(C.B, C.L, C.H, d["d_v"]))


def run_based(t, gla):
    # scale = feature_dim**-0.5 is the CANONICAL Based scale (fla's own default): it makes this SOTA
    # fla kernel compute the same map as the zoology Based we train accuracy on (verified identical,
    # rel 1.9e-6). Timing is scale-invariant, so the perf number is unchanged.
    return fused_chunk_based(t["q"], t["k"], t["v"], scale=t["q"].shape[-1] ** -0.5,
                             use_norm=True, head_first=False)


# --------------------------------------------------------------------------------------------
# Projection-cost configs: stage-1 q/k/v/router GEMM out-widths at d_model=DM, per (shape, nc).
# Return a list of out-feature*DM products (write_tex divides back by DM), or None when the cell is
# infeasible (shaped_dims None) so bench_projections skips it.
# --------------------------------------------------------------------------------------------
def _rola_proj(shape, nc, DM):
    # routed path: q,k (H*d_qk), v (H*d_v), r,w routers (H*nc). Native shape (no shape axis).
    return [DM * C.H * C.DQK] * 2 + [DM * C.H * C.DV] + [DM * C.H * nc] * 2


def _mono_proj(shape, nc, DM):
    d = shaped_dims("lin1", shape, nc)
    return None if d is None else [DM * d["H"] * d["d_k"]] * 2 + [DM * d["H"] * d["d_v"]]


def _gdn_proj(shape, nc, DM):
    d = shaped_dims("gdn", shape, nc)
    # wide q,k (H*d_k), v (H*d_v), plus scalar beta + scalar g routers (H each).
    return None if d is None else [DM * d["H"] * d["d_k"]] * 2 + [DM * d["H"] * d["d_v"]] + [DM * d["H"]] * 2


def _hedgehog_proj(shape, nc, DM):
    d = shaped_dims("lin2", shape, nc)
    # q,k (H*d_k), v (H*d_v). The HedgehogFeatureMap Linear(d_k,d_k) is applied inside run_hedgehog,
    # so its cost lands in the stage-2 kernel column, not here.
    return None if d is None else [DM * d["H"] * d["d_k"]] * 2 + [DM * d["H"] * d["d_v"]]


def _based_proj(shape, nc, DM):
    if shape != "wide":
        return None
    d = based_dims(nc)
    return [DM * C.H * d["K"]] * 2 + [DM * C.H * d["d_v"]]


# --------------------------------------------------------------------------------------------
# ARM REGISTRY. Each arm:
#   name            : column label (impl)
#   variants        : 'rla_gla' (run for both RLA & GLA inner kernels) or 'own' (cross-arch, once)
#   shapes          : list of shape labels to sweep ('native' = no shape axis; the RoLA-family)
#   make_inputs     : fn(shape, nc, gla) -> input dict, or None to SKIP an infeasible cell
#   run             : fn(inputs, gla) -> output tensor
#   grad_keys       : input dict keys to mark requires_grad
#   projection_cfg  : fn(shape, nc, DM) -> stage-1 GEMM widths, or None to reuse the 'rola' row
# --------------------------------------------------------------------------------------------
ROLA_GRAD_KEYS = ["q", "k", "v", "r", "w", "ld"]
MONO_GRAD_KEYS = ["q", "k", "v"]


def _native(maker):
    """Wrap a (nc, gla) RoLA-family maker as a (shape, nc, gla) maker (shape ignored)."""
    return lambda shape, nc, gla: maker(nc, gla)


ARMS = [
    dict(name="rola", variants="rla_gla", shapes=["native"], make_inputs=_native(mk),
         run=run_rola, grad_keys=ROLA_GRAD_KEYS, projection_cfg=_rola_proj),
    dict(name="vh", variants="rla_gla", shapes=["native"], make_inputs=_native(mk),
         run=run_vh, grad_keys=ROLA_GRAD_KEYS, projection_cfg=None),     # shares rola stage-1
    dict(name="kappa", variants="rla_gla", shapes=["native"], make_inputs=_native(mk),
         run=run_kappa, grad_keys=ROLA_GRAD_KEYS, projection_cfg=None),  # shares rola stage-1
    dict(name="monolith", variants="rla_gla", shapes=["wide", "square", "heads"],
         make_inputs=mk_mono, run=run_mono, grad_keys=MONO_GRAD_KEYS, projection_cfg=_mono_proj),
    dict(name="gdn", variants="own", shapes=["wide", "square", "heads"], make_inputs=mk_gdn,
         run=run_gdn, grad_keys=["q", "k", "v", "beta", "ldg"], projection_cfg=_gdn_proj),
    dict(name="based", variants="own", shapes=["wide"], make_inputs=mk_based,
         run=run_based, grad_keys=["q", "k", "v"], projection_cfg=_based_proj),
    dict(name="hedgehog", variants="own", shapes=["wide", "square", "heads"], make_inputs=mk_hedgehog,
         run=run_hedgehog, grad_keys=["q", "k", "v"], projection_cfg=_hedgehog_proj),
]

# tex column order (per variant block) and the cross-arch block.
TEX_ROLA_IMPLS = ("rola", "kappa", "vh", "monolith")
TEX_OWN_IMPLS = ("gdn", "based", "hedgehog")
# arms with projection_cfg=None reuse this arm's stage-1 row in the tex.
PROJ_FALLBACK = {"vh": "rola", "kappa": "rola"}


# --------------------------------------------------------------------------------------------
# END-TO-END: the FUSED layer cost, timed as ONE region (fwd, and fwd+bwd). This is the real
# per-layer wall-clock for the ACTUAL published module — no hand-assembled op-on-synthetic-tensor
# scaffold. Each arm instantiates its CANONICAL nn.Module at d_model=DM and runs `module(x)`:
#
#   rola      - the zoology RoLAMixer (= fla_rola.layers.RoLA), routed inner kernel RLA or GLA,
#               via rola_instance('rola-{rla,gla}-kappa-asym') at the perf base (H, d_qk, d_v).
#   monolith  - the dense RLA/GLA lower bound, dispatched by the gla flag to the SAME canonical
#               baseline cell the accuracy grid uses: RLA -> fla_rola LinearAttention (elu map),
#               GLA -> zoology GatedLinearAttention. Shaped wide/square/heads.
#   gdn       - fla.layers.GatedDeltaNet (canonical baseline_cell('gdn', ...)).
#   based     - zoology Based (canonical baseline_cell('based', ...), wide only).
#   hedgehog  - fla_rola LinearAttention w/ hedgehog feature map (baseline_cell('hedgehog', ...)).
#
# params_m now comes from sum(p.numel()) over the REAL module's parameters (in/out projections,
# routers, feature maps, norms — everything the layer owns), not a geometry formula. The shaped
# dims, infeasible-cell skips (gdn head_dim>256, head-scaling at nc>32), and matched-state solve
# are owned by canonical_baselines.baseline_cell, so the perf cells are identical to the MQAR grid.
# NOTE: the op-level decomposition bench (bench_kernel / bench_projections, ARMS) is UNCHANGED and
# still uses the synthetic makers — only this e2e path was repointed to the canonical modules.
# --------------------------------------------------------------------------------------------
def _instantiate(kernel_cfg, DM):
    """Build the canonical baseline module from a baseline_cell kernel dict. `name` is a dotted
    import path; kwargs already carry hidden_size for the fla classes, so d_model is injected only
    for the zoology mixers (which take d_model= and have neither key in their kwargs)."""
    mod_path, cls_name = kernel_cfg["name"].rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_path), cls_name)
    kwargs = dict(kernel_cfg["kwargs"])
    if "hidden_size" not in kwargs and "d_model" not in kwargs:
        kwargs["d_model"] = DM
    return cls(**kwargs).to(C.DEV, C.DT)


# E2E arm registry. method: the canonical baseline method ('rla'/'gla'/'gdn'/'based'/'hedgehog'),
# or None for the routed RoLA arm (its own RoLAMixer path). monolith's method is chosen from the
# gla flag at build time. variants: 'rla_gla' runs both inner kernels, 'own' runs once.
E2E_ARMS = [
    dict(name="rola", method=None, variants="rla_gla", shapes=["native"]),
    dict(name="monolith", method="dispatch", variants="rla_gla", shapes=["wide", "square", "heads"]),
    dict(name="gdn", method="gdn", variants="own", shapes=["wide", "square", "heads"]),
    dict(name="based", method="based", variants="own", shapes=["wide"]),
    dict(name="hedgehog", method="hedgehog", variants="own", shapes=["wide", "square", "heads"]),
]


def build_e2e(arm, shape, nc, gla, DM):
    """Return (forward_fn(leaves)->out, module, n_params) or None if the cell is infeasible.

    Instantiates the arm's CANONICAL module at d_model=DM and times `module(x)`; n_params is the
    real parameter count (sum of p.numel()). For baseline arms the shaped dims + matched-state
    solve + infeasible-cell skip are delegated to canonical_baselines.baseline_cell at DMODEL=DM."""
    if arm["name"] == "rola":
        kw = rola_instance("rola-gla-kappa-asym" if gla else "rola-rla-kappa-asym",
                           d_qk=C.DQK, d_v=C.DV, num_chunks=nc, n_heads=C.H)
        module = RoLAMixer(d_model=DM, **kw).to(C.DEV, C.DT)
    else:
        method = ("gla" if gla else "rla") if arm["method"] == "dispatch" else arm["method"]
        cb.DMODEL = DM                                    # baseline_cell solves projections at DM
        cell = cb.baseline_cell(method, shape, nc)
        if cell is None:                                  # shaped_dims declared this cell infeasible
            return None
        kernel_cfg, _state = cell
        module = _instantiate(kernel_cfg, DM)

    def forward(leaves):
        o = module(leaves["x"])
        return o[0] if isinstance(o, tuple) else o        # fla layers return (o, …); unwrap

    return forward, module, sum(p.numel() for p in module.parameters())


def _e2e_rep(forward, leaves, module):
    """One fused fwd+bwd rep -> (fwd_ms, fwd_bwd_ms, peak_mib). Clears x AND param grads per rep."""
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    e0.record(); o = forward(leaves); e1.record(); torch.cuda.synchronize()
    o.float().sum().backward(); e2.record(); torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    leaves["x"].grad = None
    module.zero_grad(set_to_none=True)
    return e0.elapsed_time(e1), e0.elapsed_time(e2), peak


def bench_e2e(reps, rows):
    """Fused end-to-end fwd + fwd+bwd for every CANONICAL arm × shape × nc, at d_model=DM. Same
    in-process run as bench_kernel so Triton autotuning is shared (no extra tuning cost).
    params_m is the real module parameter count (sum of p.numel())."""
    DM = 512
    for arm in E2E_ARMS:
        variants = [(False, "RLA"), (True, "GLA")] if arm["variants"] == "rla_gla" else [(False, "own")]
        for gla, var in variants:
            for shape in arm["shapes"]:
                for nc in C.NCS:
                    name = arm["name"]
                    try:
                        built = build_e2e(arm, shape, nc, gla, DM)
                        if built is None:
                            continue
                        forward, module, npar = built
                        torch.manual_seed(0)
                        x = torch.randn(C.B, C.L, DM, device=C.DEV, dtype=C.DT)
                        leaves = {"x": x.detach().requires_grad_()}
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*format mismatch.*")
                            data = C.round_robin([(name, None)],
                                                 lambda nm, _p: _e2e_rep(forward, leaves, module), reps=reps)
                        f, fb, pk = zip(*data[name])
                        row = dict(variant=var, nc=nc, shape=shape, impl=name,
                                   e2e_fwd_ms=round(C.median(f), 3), e2e_fwd_iqr=round(C.iqr(f), 3),
                                   e2e_fb_ms=round(C.median(fb), 3), e2e_fb_iqr=round(C.iqr(fb), 3),
                                   e2e_peak_mib=round(max(pk)), params_m=round(npar / 1e6, 2))
                        rows.append(row); print(row, flush=True)
                    except torch.cuda.OutOfMemoryError:
                        rows.append(dict(variant=var, nc=nc, shape=shape, impl=name, e2e_fwd_ms="OOM",
                                         e2e_fb_ms="OOM")); torch.cuda.empty_cache()
                        print(f"OOM  e2e {var}/{shape}/nc{nc}/{name}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        rows.append(dict(variant=var, nc=nc, shape=shape, impl=name, e2e_fwd_ms="ERR",
                                         e2e_fb_ms="ERR")); torch.cuda.empty_cache()
                        print(f"ERR  e2e {var}/{shape}/nc{nc}/{name}: {type(e).__name__}: {e}", flush=True)
    return rows


def _leaves(t, keys):
    """Mask requires_grad onto `keys`; pass non-tensor entries (e.g. modules) through untouched."""
    out = {}
    for k, v in t.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
        elif k in keys:
            out[k] = v.detach().requires_grad_()
        else:
            out[k] = v
    return out


def _oom_row(variant, nc, shape, impl):
    return dict(variant=variant, nc=nc, shape=shape, impl=impl, fwd_ms="OOM", fwd_iqr="OOM",
                fb_ms="OOM", fb_iqr="OOM", peak_mib="OOM", act_mib="OOM")


def _bench_cell(arm, variant, shape, nc, gla, reps, rows):
    """Time one (arm, variant, shape, nc) cell. Infeasible cells (make_inputs->None) are SKIPPED;
    on OOM / any cell error write a sentinel row + recover."""
    name = arm["name"]
    try:
        t = arm["make_inputs"](shape, nc, gla)
        if t is None:                                    # shaped_dims declared this cell infeasible
            print(f"skip  variant={variant} nc={nc} shape={shape} impl={name} (infeasible)", flush=True)
            return
        leaves = _leaves(t, arm["grad_keys"])
        data = C.round_robin([(name, None)],
                             lambda nm, _p: C.fwd_bwd_rep(lambda lv: arm["run"](lv, gla), leaves),
                             reps=reps)
        f, fb, pk, ac = zip(*data[name])
        row = dict(
            variant=variant, nc=nc, shape=shape, impl=name,
            fwd_ms=round(C.median(f), 3), fwd_iqr=round(C.iqr(f), 3),
            fb_ms=round(C.median(fb), 3), fb_iqr=round(C.iqr(fb), 3),
            peak_mib=round(max(pk)), act_mib=round(max(ac)),
        )
        rows.append(row)
        print(row, flush=True)
    except torch.cuda.OutOfMemoryError:
        rows.append(_oom_row(variant, nc, shape, name))
        print(f"OOM  variant={variant} nc={nc} shape={shape} impl={name}", flush=True)
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 -- never abort the whole run on one cell
        rows.append(_oom_row(variant, nc, shape, name))
        print(f"ERROR variant={variant} nc={nc} shape={shape} impl={name}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()


def bench_kernel(reps, rows):
    """Drive every arm × shape × nc. RoLA-family arms run for RLA and GLA; 'own' arms once.

    The benign fla 'format mismatch' warning fires for the vh arm at large nc (the folded head axis
    H*nc exceeds L=1024). It is harmless here (we pass [B,L,H,*] deliberately) and is suppressed."""
    for arm in ARMS:
        variants = [(False, "RLA"), (True, "GLA")] if arm["variants"] == "rla_gla" else [(False, "own")]
        for gla, var in variants:
            for shape in arm["shapes"]:
                for nc in C.NCS:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*format mismatch.*")
                        _bench_cell(arm, var, shape, nc, gla, reps, rows)
    return rows


def bench_projections(reps, proj):
    """Stage-1 q/k/v/router GEMMs at d_model=512, one config per (arm, shape) with a projection_cfg.
    Infeasible (shape, nc) cells (cfg -> None) are skipped. Arms whose projection_cfg is None reuse
    another arm's stage-1 row (see PROJ_FALLBACK)."""
    DM = 512
    for nc in C.NCS:
        torch.manual_seed(5)
        x = torch.randn(C.B, C.L, DM, device=C.DEV, dtype=C.DT)
        mods, meta = {}, {}
        for a in ARMS:
            if a["projection_cfg"] is None:
                continue
            for shape in a["shapes"]:
                ws = a["projection_cfg"](shape, nc, DM)
                if ws is None:
                    continue
                key = f"{a['name']}|{shape}"
                mods[key] = nn.ModuleList(
                    [nn.Linear(DM, w // DM, bias=False) for w in ws]).to(C.DEV, C.DT)
                meta[key] = (a["name"], shape)

        def run_one(key, m):
            xi = x.detach().requires_grad_()
            torch.cuda.synchronize()
            e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            e0.record()
            outs = [lin(xi) for lin in m]
            e1.record()
            sum(o.float().sum() for o in outs).backward()
            e2.record()
            torch.cuda.synchronize()
            m.zero_grad(set_to_none=True)
            return e0.elapsed_time(e1), e0.elapsed_time(e2)

        data = C.round_robin(list(mods.items()), run_one, reps=reps)
        for key, m in mods.items():
            name, shape = meta[key]
            f, fb = zip(*data[key])
            proj.append(dict(nc=nc, impl=name, shape=shape,
                             params_m=round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                             proj_fwd_ms=round(C.median(f), 3), proj_fb_ms=round(C.median(fb), 3)))
            print(proj[-1], flush=True)
    return proj


def write_tex(out, rows, proj, reps):
    proj_by = {(p["nc"], p["impl"], p["shape"]): p for p in proj}
    row_by = {(r["variant"], r["nc"], r["impl"], r["shape"]): r for r in rows}

    def numfmt(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)

    def emit(fh, var, nc, impl, shape):
        r = row_by.get((var, nc, impl, shape))
        if r is None:
            return
        proj_impl, proj_shape = PROJ_FALLBACK.get(impl, impl), shape
        if impl in PROJ_FALLBACK:
            proj_shape = "native"
        pj = proj_by.get((nc, proj_impl, proj_shape))
        is_num = isinstance(r["fwd_ms"], (int, float))
        if pj is not None and is_num:
            totf = numfmt(r["fwd_ms"] + pj["proj_fwd_ms"])
            tot = numfmt(r["fb_ms"] + pj["proj_fb_ms"])
            params = f"{pj['params_m']:.2f}"
        else:
            totf = tot = "--" if is_num else "OOM"
            params = "--"
        fh.write(f"{var}/{nc} & {shape} & {impl} & {numfmt(r['fwd_ms'])} & {numfmt(r['fb_ms'])} & "
                 f"{r['peak_mib']} & {r['act_mib']} & {totf} & {tot} & {params} \\\\\n")

    path = out / "kernel_efficiency.tex"
    with open(path, "w") as fh:
        fh.write(f"% generated by rola_bench.perf.chunked — GPU: {C.gpu_name()}\n")
        fh.write(f"%% B={C.B} H={C.H} L={C.L} d_qk={C.DQK} d_v={C.DV} bf16; median of {reps} interleaved reps\n")
        fh.write("\\begin{tabular}{lllrrrrrrr}\n\\toprule\n")
        fh.write("& & & \\multicolumn{2}{c}{kernel (ms)} & \\multicolumn{2}{c}{memory (MiB)} & "
                 "\\multicolumn{3}{c}{incl. projections} \\\\\n")
        fh.write("variant/$n_c$ & shape & impl & fwd & fwd+bwd & peak & activ. & "
                 "total fwd & total f+b & params (M) \\\\\n\\midrule\n")
        for var in ("RLA", "GLA"):
            for nc in C.NCS:
                for impl in TEX_ROLA_IMPLS:
                    shapes = ["native"] if impl in ("rola", "kappa", "vh") else ["wide", "square", "heads"]
                    for shape in shapes:
                        emit(fh, var, nc, impl, shape)
            fh.write("\\midrule\n")
        # cross-architecture block (variant='own').
        for nc in C.NCS:
            for impl in TEX_OWN_IMPLS:
                shapes = ["wide"] if impl == "based" else ["wide", "square", "heads"]
                for shape in shapes:
                    emit(fh, "own", nc, impl, shape)
        fh.write("\\bottomrule\n\\end{tabular}\n")
    return path


def main():
    ap = argparse.ArgumentParser(description="Chunked (fwd+bwd) kernel-efficiency benchmark.")
    ap.add_argument("--reps", type=int, default=C.REPS, help="timing rounds after warmup")
    ap.add_argument("--out", default=str(C.RESULTS_DIR), help="output dir for csv/tex")
    a = ap.parse_args()
    C.require_cuda()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"# chunked fwd+bwd | {C.gpu_name()} | B={C.B} H={C.H} L={C.L} d_qk=d_v={C.DQK} bf16 "
          f"| nc {tuple(C.NCS)} | shapes wide/square/heads | median of {a.reps} interleaved reps")
    rows, proj, e2e = [], [], []
    try:
        bench_kernel(a.reps, rows)        # stage-2 kernel (isolated)
        bench_projections(a.reps, proj)   # stage-1 projection GEMMs (isolated)
        bench_e2e(a.reps, e2e)            # FUSED end-to-end layer (fwd + fwd+bwd) — same process, warm autotune
    finally:
        # Always persist whatever completed, so a partial / OOM-truncated run is not lost.
        if rows:
            C.write_csv(out / "kernel_efficiency.csv", rows)
        if proj:
            C.write_csv(out / "projection_cost.csv", proj)
        if e2e:
            C.write_csv(out / "kernel_e2e.csv", e2e)
        if rows and proj:
            write_tex(out, rows, proj, a.reps)
    print(f"\nWROTE {out}/kernel_efficiency.csv, projection_cost.csv, kernel_e2e.csv, kernel_efficiency.tex  [{C.gpu_name()}]")


if __name__ == "__main__":
    main()
