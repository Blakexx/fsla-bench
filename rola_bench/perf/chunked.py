"""Chunked-parallel end-to-end layer benchmark (forward + backward).

The training-time regime, measured as the FUSED per-layer cost of the ACTUAL published module:
each arm instantiates its CANONICAL nn.Module at d_model=DM and times `module(x)` (forward, and
forward+backward) as ONE region — no hand-assembled op-on-synthetic-tensor scaffold. At MATCHED
total state we compare the routed RoLA layer against cross-architecture baselines that hold the SAME
recurrent state, for both the RLA (no decay) and scalar-GLA (per-state decay) inner kernels.

Two axes mirror the MQAR accuracy grid exactly (so perf cells == accuracy cells):

  * state rung: the full nc scan {2,4,8,16,32,64,128,256} (_common.NCS), not just {16,64,256}.
  * shape: HOW a baseline spends its matched state — wide (widen keys), square (d_k=d_v), heads
    (scale head count). The dim solve, matched-state solve, and infeasible-cell skips (gdn
    head_dim>256, head-scaling at nc>32) are delegated to canonical_baselines.baseline_cell at
    DMODEL=DM, so the perf cells are identical to the MQAR grid (no phantom cells).

Arms (each instantiates its CANONICAL module via canonical_baselines / rola_instance):
  rola      - the zoology RoLAMixer (= fla_rola.layers.RoLA), routed inner kernel RLA or GLA,
              via rola_instance('rola-{rla,gla}-kappa-asym') at the perf base (H, d_qk, d_v).
  monolith  - the dense RLA/GLA lower bound, dispatched by the gla flag to the SAME canonical
              baseline cell the accuracy grid uses: RLA -> fla_rola LinearAttention (elu map),
              GLA -> zoology GatedLinearAttention. Shaped wide/square/heads.
  gdn       - fla.layers.GatedDeltaNet (canonical baseline_cell('gdn', ...)). head_dim widens the
              recall axis; kernel-capped at 256.
  based     - zoology Based (canonical baseline_cell('based', ...), wide only).
  hedgehog  - fla_rola LinearAttention w/ hedgehog feature map (baseline_cell('hedgehog', ...)).

Reports, per cell: variant, nc, shape, impl, e2e_fwd_ms, e2e_fwd_iqr, e2e_fb_ms (fwd+bwd),
e2e_fb_iqr, e2e_peak_mib, params_m. params_m is the real module parameter count (sum of p.numel()
over the layer's in/out projections, routers, feature maps, norms — everything the layer owns).
Outputs to results/perf/ (override with --out).

ROBUSTNESS: each cell is wrapped so a CUDA OOM (or any cell error) writes an 'OOM'/'ERR' sentinel
row, empties the cache, and continues. Cells the shape solve declares infeasible are SKIPPED (no
row), exactly as the accuracy grid omits them. The CSV is written in a `finally` so a partial run
is never lost.

Run:  python -m rola_bench.perf.chunked [--reps N] [--out DIR]   (idle GPU)
"""
import argparse
import warnings
from pathlib import Path

import importlib

import torch

from zoology.mixers.rola import RoLAMixer
from rola_bench.mqar.instances import rola_instance
import rola_bench.mqar.experiments.canonical_baselines as cb

from . import _common as C


# --------------------------------------------------------------------------------------------
# END-TO-END: the FUSED layer cost, timed as ONE region (fwd, and fwd+bwd), for the ACTUAL
# published module. Each arm instantiates its CANONICAL nn.Module at d_model=DM and runs `module(x)`.
# The shaped dims, infeasible-cell skips, and matched-state solve are owned by
# canonical_baselines.baseline_cell, so the perf cells are identical to the MQAR grid.
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
    """Fused end-to-end fwd + fwd+bwd for every CANONICAL arm × shape × nc, at d_model=DM.
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


def main():
    ap = argparse.ArgumentParser(description="Chunked (fwd+bwd) end-to-end layer benchmark.")
    ap.add_argument("--reps", type=int, default=C.REPS, help="timing rounds after warmup")
    ap.add_argument("--out", default=str(C.RESULTS_DIR), help="output dir for csv")
    a = ap.parse_args()
    C.require_cuda()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"# chunked e2e fwd+bwd | {C.gpu_name()} | B={C.B} H={C.H} L={C.L} d_qk=d_v={C.DQK} bf16 "
          f"| nc {tuple(C.NCS)} | shapes wide/square/heads | median of {a.reps} interleaved reps")
    e2e = []
    try:
        bench_e2e(a.reps, e2e)            # FUSED end-to-end layer (fwd + fwd+bwd)
    finally:
        # Always persist whatever completed, so a partial / OOM-truncated run is not lost.
        if e2e:
            C.write_csv(out / "kernel_e2e.csv", e2e)
    print(f"\nWROTE {out}/kernel_e2e.csv  [{C.gpu_name()}]")


if __name__ == "__main__":
    main()
