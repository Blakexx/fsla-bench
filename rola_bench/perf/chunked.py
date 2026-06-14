"""Chunked-parallel kernel-efficiency benchmark (forward + backward).

The training-time regime: chunk-parallel attention with a shared content Gram. At MATCHED
total state (nc states x d_qk x d_v per head) we compare a set of implementations, for both
the RLA (no decay) and scalar-GLA (per-state decay) inner kernels, plus a few cross-architecture
baselines that hold the SAME recurrent-state float count per head.

RoLA-family arms (variant in {RLA, GLA}):
  rola      - the fla_rola routed `chunk_simple_gla` (shared-Gram Triton, r/w[/g] params): the
              kernel the paper proposes.
  vh        - virtual-head on CANONICAL fla: q/k replicated across the nc states folded into the
              head axis, v pre-scaled by the write gate, read-combine outside. The honest "what
              you'd do without the routed kernel" baseline (per-state scalar decay -> simple_gla g).
  monolith  - canonical fla simple_gla, ONE state per head at d_qk_mono = nc*d_qk (same state
              size, no routing). Lower bound on kernel cost for that much dense state.
  kappa     - rola + the per-state-denominator normalization pre-pass (kappa=0.5) + the same
              fork kernel; isolates the den-kernel cost on top of rola.

Cross-architecture arms (variant='own' — each is its own published kernel, no RLA/GLA split):
  gdn       - fla `chunk_gated_delta_rule` (gated delta rule). Matched state by widening KEYS to
              d_k = nc*d_qk with d_v held at 16 (state d_k*d_v = nc*d_qk*d_v). Same sizing
              principle as the MQAR 'wide' GDN baseline (head_dim widens the recall axis).
  based     - fla `fused_chunk_based` (Taylor-exp feature-map linear attn). The kernel applies the
              2nd-order Taylor map internally, so the recurrent state per head is (1 + K + K^2)*V
              (K = raw feature dim, kernel-capped at 16). K,V solved to hit nc*d_qk*d_v.
  hedgehog  - fla `chunk_linear_attn` over the canonical `HedgehogFeatureMap` applied to q/k. The
              map doubles the feature dim (cat([2x,-2x]).softmax), so state = (2*d_k)*d_v; d_k set
              to nc*d_qk/2 with d_v held at 16 to hit nc*d_qk*d_v.

Reports, per cell, the exact columns of the paper table:
  variant, nc, impl, fwd_ms, fwd_iqr, fb_ms (fwd+bwd), fb_iqr, peak_mib, act_mib
plus a projection-cost table (stage-1 q/k/v/router GEMMs + params) and a combined LaTeX table
with stage1+stage2 totals. Outputs to results/perf/ (override with --out).

ROBUSTNESS: each cell is wrapped so a CUDA OOM (or any cell error) writes an 'OOM' sentinel row,
empties the cache, and continues instead of aborting the whole run; the CSV is written in a
`finally` so a partial run is never lost.

Run:  python -m rola_bench.perf.chunked [--reps N] [--out DIR]   (idle GPU)
"""
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn

import fla.ops.simple_gla as fla_sg
import fla_rola.ops.simple_gla as fork_sg
from fla_rola.ops.based import fused_chunk_based
from fla_rola.ops.gated_delta_rule import chunk_gated_delta_rule
from fla_rola.ops.linear_attn import chunk_linear_attn
from fla_rola.modules.feature_map import HedgehogFeatureMap

from . import _common as C


# --------------------------------------------------------------------------------------------
# Shared input maker for the RoLA-family arms (rola / vh / monolith / kappa).
# Byte-identical to the original `mk`: same seed, same draw order, same dtypes.
# --------------------------------------------------------------------------------------------
def mk(nc, gla, seed=0):
    """Inputs at matched total state. Signed q/k/v; routers are softmax simplex weights."""
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    t = dict(
        q=g(C.B, C.L, C.H, C.DQK), k=g(C.B, C.L, C.H, C.DQK), v=g(C.B, C.L, C.H, C.DV),
        r=torch.softmax(g(C.B, C.L, C.H, nc).float(), -1).to(C.DT),
        w=torch.softmax(g(C.B, C.L, C.H, nc).float(), -1).to(C.DT),
        qm=g(C.B, C.L, C.H, nc * C.DQK), km=g(C.B, C.L, C.H, nc * C.DQK),  # monolith matched-state q/k
    )
    if gla:
        alpha = torch.sigmoid(g(C.B, C.L, C.H, nc).float())
        t["ld"] = (1 - t["w"].float() * (1 - alpha)).clamp(min=1e-6).log().clamp(min=-2.5).to(C.DT)
    return t


# --------------------------------------------------------------------------------------------
# RoLA-family run_fns (math byte-identical to the original).
# --------------------------------------------------------------------------------------------
def run_rola(t, gla):
    o, _ = fork_sg.chunk_simple_gla(t["q"], t["k"], t["v"], g=t.get("ld"), scale=1.0, r=t["r"], w=t["w"])
    return o


def run_kappa(t, gla):
    """Per-state-denominator normalization (kappa=0.5) pre-pass + gate rescale + same fork kernel.
    |d| guard keeps timing numerics finite on random (non-positive) bench inputs."""
    Bq, Lq, Hq, _ = t["q"].shape
    fold = lambda x: x.permute(0, 2, 1, 3).reshape(Bq * Hq, Lq, x.shape[-1])
    if gla:
        from fla_rola.ops.simple_gla.rola import rola_perstate_den_gla_triton
        d = rola_perstate_den_gla_triton(fold(t["q"]), fold(t["k"]), fold(t["w"]), fold(t["ld"]))
    else:
        from fla_rola.ops.simple_gla.rola import rola_perstate_den_triton
        d = rola_perstate_den_triton(fold(t["q"]), fold(t["k"]), fold(t["w"]))
    d = d.view(Bq, Hq, Lq, -1).permute(0, 2, 1, 3)
    rk = t["r"] * (d.abs() + 1e-5).pow(-0.5).to(t["r"].dtype)
    o, _ = fork_sg.chunk_simple_gla(t["q"], t["k"], t["v"], g=t.get("ld"), scale=1.0, r=rk, w=t["w"])
    return o


def run_vh(t, gla):
    nc = t["r"].shape[-1]
    qv = t["q"].unsqueeze(3).expand(C.B, C.L, C.H, nc, C.DQK).reshape(C.B, C.L, C.H * nc, C.DQK)
    kv = t["k"].unsqueeze(3).expand(C.B, C.L, C.H, nc, C.DQK).reshape(C.B, C.L, C.H * nc, C.DQK)
    vv = (t["v"].unsqueeze(3) * t["w"].unsqueeze(-1)).reshape(C.B, C.L, C.H * nc, C.DV)
    g = t["ld"].reshape(C.B, C.L, C.H * nc).float() if gla else None
    o, _ = fla_sg.chunk_simple_gla(qv, kv, vv, g=g, scale=1.0)
    return (o.view(C.B, C.L, C.H, nc, C.DV) * t["r"].unsqueeze(-1)).sum(3)


def run_monolith(t, gla):
    g = t["ld"][..., 0].float() if gla else None  # one scalar gate per head
    o, _ = fla_sg.chunk_simple_gla(t["qm"], t["km"], t["v"], g=g, scale=1.0)
    return o


# --------------------------------------------------------------------------------------------
# Cross-architecture arms (variant='own'). Each is its OWN published kernel; matched state holds
# the recurrent-state float count per head equal to the RoLA cell (nc*d_qk*d_v). Sizing principle
# is reused from rola_bench/mqar/experiments/canonical_baselines.py.
# --------------------------------------------------------------------------------------------
TOTAL_STATE = lambda nc: nc * C.DQK * C.DV  # RoLA recurrent floats per head = nc*d_qk*d_v


def gdn_dims(nc):
    """GDN (delta-rule) state = d_k * d_v per head. Hold d_v=DV, widen keys: d_k = nc*d_qk.
    Mirrors the MQAR 'wide' GDN baseline (head_dim widens the recall axis).

    TODO(gpu): d_k = nc*d_qk is 256 / 1024 / 4096 for nc = 16 / 64 / 256. canonical_baselines.py
    notes the GDN chunk kernel's practical head_dim ceiling ~256 (the 'wide' GDN baseline tops out
    near nc<=19 for that reason). nc=64/256 may therefore fail to compile/run or OOM in the GDN
    chunk kernel; that is exactly what the per-cell OOM/error sentinel is for. I could NOT confirm
    the realized ceiling without a GPU run — verify on an idle GPU which nc actually succeed and,
    if the large-nc cells legitimately don't fit GDN's kernel, document them as architectural-limit
    OOMs (do not silently drop). The matched-state float count itself is exact (d_k*d_v == nc*d_qk*d_v)."""
    return dict(d_k=nc * C.DQK, d_v=C.DV)


def hedgehog_dims(nc):
    """Hedgehog: HedgehogFeatureMap doubles the feature dim (cat([2x,-2x]).softmax), so the
    linear-attn recurrent state is (2*d_k) * d_v. Hold d_v=DV, set d_k = nc*d_qk / 2 to hit
    nc*d_qk*d_v. (DQK is even, so this is exact at the perf shape.)"""
    d_k = max(1, (nc * C.DQK) // 2)
    return dict(d_k=d_k, d_v=C.DV)


def based_dims(nc):
    """Based: `fused_chunk_based` applies the 2nd-order Taylor map INSIDE the kernel, so the
    recurrent state per head is (1 + K + K^2) * V  (zero/first/second-order taylor blocks),
    with K = raw feature dim kernel-capped at 16. Based's identity is the rich Taylor feature
    map, so we pin K at its richest feasible value (16, the kernel cap) and let the value dim V
    absorb the state budget to hit nc*d_qk*d_v. (Mirrors the MQAR 'wide' Based reasoning, where
    feature_dim is the sole state knob; here the kernel's max K is the knob and V matches state.)"""
    K = 16  # kernel asserts q.shape[-1] <= 16
    feat = 1 + K + K * K  # 273
    V = max(1, round(TOTAL_STATE(nc) / feat))
    return dict(K=K, d_v=V)


def mk_gdn(nc, seed=0):
    d = gdn_dims(nc)
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    # GDN expects L2-normalized keys; beta in (0,1); g in log space (logsigmoid).
    return dict(
        q=g(C.B, C.L, C.H, d["d_k"]),
        k=torch.nn.functional.normalize(g(C.B, C.L, C.H, d["d_k"]).float(), p=2, dim=-1).to(C.DT),
        v=g(C.B, C.L, C.H, d["d_v"]),
        beta=torch.rand(C.B, C.L, C.H, device=C.DEV, dtype=C.DT).sigmoid(),
        ldg=torch.nn.functional.logsigmoid(g(C.B, C.L, C.H).float()).to(C.DT),
    )


def run_gdn(t, gla):
    # chunk_gated_delta_rule(q, k, v, g, beta): q/k/v [B,T,H,K|V], g [B,T,H] log-space, beta [B,T,H].
    o, _ = chunk_gated_delta_rule(t["q"], t["k"], t["v"], g=t["ldg"], beta=t["beta"], scale=1.0)
    return o


def mk_based(nc, seed=0):
    d = based_dims(nc)
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    return dict(
        q=g(C.B, C.L, C.H, d["K"]), k=g(C.B, C.L, C.H, d["K"]), v=g(C.B, C.L, C.H, d["d_v"]),
    )


def run_based(t, gla):
    # fused_chunk_based(q, k, v, scale, use_norm, head_first): taylor feature map applied inside.
    # scale = feature_dim**-0.5 is the CANONICAL Based scale (fla's own default): it makes this
    # SOTA fla kernel compute the same map as the zoology Based we train accuracy on (verified
    # identical, rel 1.9e-6; scale=1.0 would be an unscaled non-canonical variant). Timing is
    # scale-invariant, so the perf number is unchanged — this keeps the arm faithful to the model.
    return fused_chunk_based(t["q"], t["k"], t["v"], scale=t["q"].shape[-1] ** -0.5,
                             use_norm=True, head_first=False)


def mk_hedgehog(nc, seed=0):
    d = hedgehog_dims(nc)
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=C.DEV, dtype=C.DT)
    fmap = HedgehogFeatureMap(head_dim=d["d_k"]).to(C.DEV, C.DT)
    return dict(
        q=g(C.B, C.L, C.H, d["d_k"]), k=g(C.B, C.L, C.H, d["d_k"]), v=g(C.B, C.L, C.H, d["d_v"]),
        _fmap=fmap,
    )


def run_hedgehog(t, gla):
    # Canonical Hedgehog: feature map on q AND k, then normalized linear attn (chunk_linear_attn).
    phi_q = t["_fmap"](t["q"])
    phi_k = t["_fmap"](t["k"])
    o, _ = chunk_linear_attn(phi_q, phi_k, t["v"], scale=1.0, normalize=True)
    return o


# --------------------------------------------------------------------------------------------
# ARM REGISTRY. Each arm is a record:
#   name            : column label (impl)
#   variants        : which variant axis it lives on -- 'rla_gla' (run for both RLA & GLA) or 'own'
#   make_inputs     : fn(nc, gla) -> input dict (gla ignored for 'own' arms)
#   run             : fn(inputs, gla) -> output tensor
#   grad_keys       : input dict keys to mark requires_grad (others are constants / params)
#   projection_cfg  : fn(nc, DM) -> list of stage-1 GEMM widths (out features), or None to reuse
#                     the shared 'rola' projection row in the tex (cross-arch arms have their own
#                     stage-1 shapes; see projection_cfg notes per arm).
#
# `_leaves` masks grad on the named keys; non-tensor entries (e.g. the Hedgehog feature-map module
# under '_fmap') are passed through untouched.
# --------------------------------------------------------------------------------------------
ROLA_GRAD_KEYS = ["q", "k", "v", "r", "w", "ld"]
MONO_GRAD_KEYS = ["qm", "km", "v"]


def _rola_proj(nc, DM):
    # stage-1 GEMMs for the routed path: q,k (H*d_qk), v (H*d_v), r,w routers (H*nc).
    return [DM * C.H * C.DQK] * 2 + [DM * C.H * C.DV] + [DM * C.H * nc] * 2


def _mono_proj(nc, DM):
    # monolith pays nc-scaling on q,k here too; v unchanged.
    return [DM * C.H * nc * C.DQK] * 2 + [DM * C.H * C.DV]


def _gdn_proj(nc, DM):
    # GDN stage-1: wide q,k (H*d_k), v (H*d_v), plus scalar beta + scalar g routers (H each).
    d = gdn_dims(nc)
    return [DM * C.H * d["d_k"]] * 2 + [DM * C.H * d["d_v"]] + [DM * C.H] * 2


def _based_proj(nc, DM):
    # Based stage-1: q,k at raw feature dim K (H*K), v (H*d_v). Taylor expansion is parameter-free.
    d = based_dims(nc)
    return [DM * C.H * d["K"]] * 2 + [DM * C.H * d["d_v"]]


def _hedgehog_proj(nc, DM):
    # Hedgehog stage-1: q,k (H*d_k), v (H*d_v). The HedgehogFeatureMap Linear(d_k,d_k) is applied
    # inside run_hedgehog, so its cost lands in the stage-2 kernel column, not here.
    d = hedgehog_dims(nc)
    return [DM * C.H * d["d_k"]] * 2 + [DM * C.H * d["d_v"]]


ARMS = [
    dict(name="rola", variants="rla_gla", make_inputs=mk, run=run_rola,
         grad_keys=ROLA_GRAD_KEYS, projection_cfg=_rola_proj),
    dict(name="vh", variants="rla_gla", make_inputs=mk, run=run_vh,
         grad_keys=ROLA_GRAD_KEYS, projection_cfg=None),       # shares rola's stage-1 shapes
    dict(name="monolith", variants="rla_gla", make_inputs=mk, run=run_monolith,
         grad_keys=MONO_GRAD_KEYS, projection_cfg=_mono_proj),
    dict(name="kappa", variants="rla_gla", make_inputs=mk, run=run_kappa,
         grad_keys=ROLA_GRAD_KEYS, projection_cfg=None),       # shares rola's stage-1 shapes
    dict(name="gdn", variants="own", make_inputs=mk_gdn, run=run_gdn,
         grad_keys=["q", "k", "v", "beta", "ldg"], projection_cfg=_gdn_proj),
    dict(name="based", variants="own", make_inputs=mk_based, run=run_based,
         grad_keys=["q", "k", "v"], projection_cfg=_based_proj),
    dict(name="hedgehog", variants="own", make_inputs=mk_hedgehog, run=run_hedgehog,
         grad_keys=["q", "k", "v"], projection_cfg=_hedgehog_proj),
]

# tex column order (per variant block, RoLA-family) and the cross-arch block.
TEX_ROLA_IMPLS = ("rola", "kappa", "vh", "monolith")
TEX_OWN_IMPLS = ("gdn", "based", "hedgehog")


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


def _oom_row(variant, nc, impl):
    return dict(variant=variant, nc=nc, impl=impl, fwd_ms="OOM", fwd_iqr="OOM",
                fb_ms="OOM", fb_iqr="OOM", peak_mib="OOM", act_mib="OOM")


def _bench_cell(arm, variant, nc, gla, reps, rows):
    """Time one (arm, variant, nc) cell. On OOM / any cell error write a sentinel row + recover."""
    name = arm["name"]
    try:
        t = arm["make_inputs"](nc, gla)
        leaves = _leaves(t, arm["grad_keys"])
        data = C.round_robin([(name, None)],
                             lambda nm, _p: C.fwd_bwd_rep(lambda lv: arm["run"](lv, gla), leaves),
                             reps=reps)
        f, fb, pk, ac = zip(*data[name])
        row = dict(
            variant=variant, nc=nc, impl=name,
            fwd_ms=round(C.median(f), 3), fwd_iqr=round(C.iqr(f), 3),
            fb_ms=round(C.median(fb), 3), fb_iqr=round(C.iqr(fb), 3),
            peak_mib=round(max(pk)), act_mib=round(max(ac)),
        )
        rows.append(row)
        print(row, flush=True)
    except torch.cuda.OutOfMemoryError:
        rows.append(_oom_row(variant, nc, name))
        print(f"OOM  variant={variant} nc={nc} impl={name}", flush=True)
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 -- never abort the whole run on one cell
        rows.append(_oom_row(variant, nc, name))
        print(f"ERROR variant={variant} nc={nc} impl={name}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()


def bench_kernel(reps, rows):
    """Drive every arm from the registry. RoLA-family arms run for RLA and GLA; 'own' arms once.

    The benign fla 'format mismatch' warning fires for the vh arm at large nc (the folded head
    axis H*nc exceeds L=1024, so seq_len < num_heads). It is harmless here (we pass [B,L,H,*]
    deliberately) and is suppressed; see README 'Determinism'."""
    # RoLA-family arms, both variants.
    for gla in (False, True):
        var = "GLA" if gla else "RLA"
        for nc in C.NCS:
            for arm in ARMS:
                if arm["variants"] != "rla_gla":
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*format mismatch.*")
                    _bench_cell(arm, var, nc, gla, reps, rows)
    # Cross-architecture arms, variant='own' (no RLA/GLA split).
    for nc in C.NCS:
        for arm in ARMS:
            if arm["variants"] != "own":
                continue
            _bench_cell(arm, "own", nc, False, reps, rows)
    return rows


def bench_projections(reps, proj):
    """Stage-1 q/k/v/router GEMMs at d_model=512, one config per arm with a projection_cfg.
    Arms whose projection_cfg is None reuse another arm's stage-1 row (see write_tex)."""
    DM = 512
    cfg_arms = [a for a in ARMS if a["projection_cfg"] is not None]
    for nc in C.NCS:
        torch.manual_seed(5)
        x = torch.randn(C.B, C.L, DM, device=C.DEV, dtype=C.DT)
        mods = {}
        for a in cfg_arms:
            ws = a["projection_cfg"](nc, DM)
            mods[a["name"]] = nn.ModuleList(
                [nn.Linear(DM, w // DM, bias=False) for w in ws]).to(C.DEV, C.DT)

        def run_one(name, m):
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
        for name, m in mods.items():
            f, fb = zip(*data[name])
            proj.append(dict(nc=nc, impl=name,
                             params_m=round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                             proj_fwd_ms=round(C.median(f), 3), proj_fb_ms=round(C.median(fb), 3)))
            print(proj[-1], flush=True)
    return proj


# arms with projection_cfg=None reuse this arm's stage-1 row in the tex.
PROJ_FALLBACK = {"vh": "rola", "kappa": "rola"}


def write_tex(out, rows, proj, reps):
    proj_by = {(p["nc"], p["impl"]): p for p in proj}
    row_by = {(r["variant"], r["nc"], r["impl"]): r for r in rows}

    def numfmt(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)

    def emit(fh, var, nc, impl):
        r = row_by.get((var, nc, impl))
        if r is None:
            return
        proj_impl = PROJ_FALLBACK.get(impl, impl)
        pj = proj_by.get((nc, proj_impl))
        is_num = isinstance(r["fwd_ms"], (int, float))
        if pj is not None and is_num:
            totf = numfmt(r["fwd_ms"] + pj["proj_fwd_ms"])
            tot = numfmt(r["fb_ms"] + pj["proj_fb_ms"])
            params = f"{pj['params_m']:.2f}"
        else:
            totf = tot = "--" if is_num else "OOM"
            params = "--"
        fh.write(f"{var}/{nc} & {impl} & {numfmt(r['fwd_ms'])} & {numfmt(r['fb_ms'])} & "
                 f"{r['peak_mib']} & {r['act_mib']} & {totf} & {tot} & {params} \\\\\n")

    path = out / "kernel_efficiency.tex"
    with open(path, "w") as fh:
        fh.write(f"% generated by rola_bench.perf.chunked — GPU: {C.gpu_name()}\n")
        fh.write(f"%% B={C.B} H={C.H} L={C.L} d_qk={C.DQK} d_v={C.DV} bf16; median of {reps} interleaved reps\n")
        fh.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
        fh.write("& & \\multicolumn{2}{c}{kernel (ms)} & \\multicolumn{2}{c}{memory (MiB)} & "
                 "\\multicolumn{3}{c}{incl. projections} \\\\\n")
        fh.write("variant/$n_c$ & impl & fwd & fwd+bwd & peak & activ. & total fwd & total f+b & params (M) \\\\\n\\midrule\n")
        for var in ("RLA", "GLA"):
            for nc in C.NCS:
                for impl in TEX_ROLA_IMPLS:
                    emit(fh, var, nc, impl)
            fh.write("\\midrule\n")
        # cross-architecture block (variant='own').
        for nc in C.NCS:
            for impl in TEX_OWN_IMPLS:
                emit(fh, "own", nc, impl)
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
          f"| median of {a.reps} interleaved reps")
    rows, proj = [], []
    try:
        bench_kernel(a.reps, rows)
        bench_projections(a.reps, proj)
    finally:
        # Always persist whatever completed, so a partial / OOM-truncated run is not lost.
        if rows:
            C.write_csv(out / "kernel_efficiency.csv", rows)
        if proj:
            C.write_csv(out / "projection_cost.csv", proj)
        if rows and proj:
            write_tex(out, rows, proj, a.reps)
    print(f"\nWROTE {out}/kernel_efficiency.csv, projection_cost.csv, kernel_efficiency.tex  [{C.gpu_name()}]")


if __name__ == "__main__":
    main()
