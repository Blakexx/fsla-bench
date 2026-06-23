"""RoLA SCALING benchmark — per-architecture characterization, NOT cross-architecture comparison.

Arms: **RoLA** (rla + gla inner kernels), its matched-state **wide-monolith** equivalent (nc=1, all
the state in one wide head — the "what if you didn't route" ablation, RoLA's own control, NOT a zoo
baseline), and **softmax attention**. Rationale for excluding the rest: GDN / GLA / Based / ... are
the SAME complexity class as RoLA (O(L) compute, bounded state), so on an L-axis they are parallel
lines with different constants — "the constants" is a matched-state COMPARISON question (see
perf/comparison), not a scaling-shape one. Attention is the ONLY baseline that scales DIFFERENTLY in
L (O(L^2) compute, O(L) growing KV cache), the entire reason RoLA exists. The wide monolith earns its
place as RoLA's structural ablation (matched state, routed-vs-not), not as a competitor.

IMPORTANT: the numbers here are NOT a fairness verdict. RoLA sitting at X<100% of attention's KV
cache is the bounded-state VALUE PROPOSITION (the whole point), not an unfair tilt — bounded-vs-
unbounded is the legitimate contrast. The output is the SHAPE of each curve and where RoLA crosses
attention; head-to-head verdicts at matched state live in the (separate) comparison benchmark.

Axes:
  * L      sequence length (the primary scaling axis)
  * X%     RoLA content state as a fraction of attention's KV cache -> nc = round(2*L*X/d), d_qk=d_v=d
  * d_qk   feature dim {16,64,128} -- the occupancy / operating-point knob (16 = the starved corner)
  * bench  training (chunked fwd+bwd) | prefill (chunked fwd-only) | decode (recurrent, per-token)

attention depends only on L (no X / d_qk axis) -> timed once per (bench, L) and reused as the
reference curve. Each cell is wrapped so an OOM/error writes a sentinel and the sweep continues;
the CSV is written in a `finally`. Use small PERF_B (e.g. 2) so long L fits.

Run:  python -m rola_bench.perf.scaling [--benches train,prefill,decode]
            [--lens 1024,4096,16384,65536] [--xs 10,25,50,100] [--dqks 16,64,128] [--reps N] [--out DIR]
"""
import argparse
import os
import warnings
from pathlib import Path

import torch

from zoology.mixers.rola import RoLAMixer
from rola_bench.mqar.instances import rola_instance

from . import _common as C
from .chunked import _CausalMHA, _e2e_rep, _instantiate
from .recurrent import decode_routed_step, decode_monolith_step, make_inputs
import rola_bench.mqar.experiments.canonical_baselines as cb

DM = 512  # d_model backbone for the scaling layers (matches chunked.bench_e2e)


def nc_for(L, x_pct, d):
    """RoLA states to hit X% of attention's KV cache at d_qk=d_v=d, from X = nc*d/(2L)."""
    return max(1, round(2 * L * (x_pct / 100.0) / d))


def _mib(*dims):
    n = 1
    for x in dims:
        n *= x
    return round(n * 4 / 2 ** 20, 1)


# ------------------------------------------------------------------ builders
def build_rola(gla, nc, d):
    """Routed RoLA layer at d_qk=d_v=d, nc states, perf backbone. Returns the nn.Module."""
    kw = rola_instance("rola-gla-kappa-asym" if gla else "rola-rla-kappa-asym",
                       d_qk=d, d_v=d, num_chunks=nc, n_heads=C.H)
    return RoLAMixer(d_model=DM, **kw).to(C.DEV, C.DT)


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


def _decode_rola_cell(gla, nc, d):
    """One recurrent decode step (per-token) via the canonical fused path -> (ms, state_mib)."""
    C.set_shape(DQK=d, DV=d)
    q, k, v, r, w, ld = make_inputs(1, nc, gla, seed=1)
    cdt = lambda t: t.to(C.DT) if t is not None else None
    state = torch.zeros(C.B, C.H * nc, d, d + 1, device=C.DEV, dtype=torch.float32)
    step = lambda: decode_routed_step(cdt(q), cdt(k), cdt(v), r, w, cdt(ld), state, nc,
                                      "kappa" if gla else "global")
    ms = C.time_forward(step, steps=100, warmup=20)
    return round(ms, 4), _mib(C.B, C.H * nc, d, d + 1)


def _decode_monolith_cell(gla, nc, d):
    """One decode step of the matched-state wide monolith (H heads, one big state d_k=nc·d, d_v=d)."""
    H, dk, dv = C.H, nc * d, d
    g = torch.Generator(device=C.DEV).manual_seed(2)
    rnd = lambda *s: torch.randn(*s, device=C.DEV, generator=g, dtype=C.DT)
    q, k, v = rnd(C.B, 1, H, dk).abs(), rnd(C.B, 1, H, dk).abs(), rnd(C.B, 1, H, dv)
    gate = torch.nn.functional.logsigmoid(rnd(C.B, 1, H).float()) if gla else None
    state = torch.zeros(C.B, H, dk, dv, device=C.DEV, dtype=torch.float32)
    step = lambda: decode_monolith_step(q, k, v, state, g=gate)
    ms = C.time_forward(step, steps=100, warmup=20)
    return round(ms, 4), _mib(C.B, H, dk, dv)


def _decode_attn_cell(L):
    """One decode step: a single query attending to an L-long cached KV -> (ms, cache_mib)."""
    hd = DM // C.H
    kc = torch.randn(C.B, C.H, L, hd, device=C.DEV, dtype=C.DT)
    vc = torch.randn(C.B, C.H, L, hd, device=C.DEV, dtype=C.DT)
    q = torch.randn(C.B, C.H, 1, hd, device=C.DEV, dtype=C.DT)
    step = lambda: torch.nn.functional.scaled_dot_product_attention(q, kc, vc)
    ms = C.time_forward(step, steps=100, warmup=20)
    return round(ms, 4), _mib(C.B, C.H, L, hd, 2)  # K+V cache


# ------------------------------------------------------------------ sweep
def _row(**kw):
    base = dict(bench=None, L=None, X_pct=None, d_qk=None, nc=None, arm=None,
                fwd_ms=None, fb_ms=None, peak_mib=None, act_mib=None, decode_ms=None, state_mib=None)
    base.update(kw)
    return base


def sweep(benches, lens, xs, dqks, rows):
    for bench in benches:
        for L in lens:
            C.set_shape(L=L)
            # ---- attention reference: depends only on L (no X / d_qk axis) ----
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

            # ---- RoLA + its matched-state wide-monolith ablation: (X%, d_qk) grid, both kernels ----
            for d in dqks:
                for x_pct in xs:
                    nc = nc_for(L, x_pct, d)
                    for kind in ("rola", "monolith"):
                        for gla, var in ((False, "rla"), (True, "gla")):
                            arm = f"{kind}-{var}"
                            try:
                                if bench == "decode":
                                    fn = _decode_rola_cell if kind == "rola" else _decode_monolith_cell
                                    ms, mib = fn(gla, nc, d)
                                    rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc, arm=arm,
                                                     decode_ms=ms, state_mib=mib))
                                else:
                                    C.set_shape(L=L, DQK=d, DV=d)
                                    m = build_rola(gla, nc, d) if kind == "rola" else build_monolith(gla, nc, d)
                                    if m is None:
                                        continue                       # baseline_cell declared it infeasible
                                    if bench == "train":
                                        fwd, fb, pk, ac = _train_cell(m, L)
                                        rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc, arm=arm,
                                                         fwd_ms=fwd, fb_ms=fb, peak_mib=pk, act_mib=ac))
                                    else:
                                        fwd, pk = _prefill_cell(m, L)
                                        rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc, arm=arm,
                                                         fwd_ms=fwd, peak_mib=pk))
                                    del m
                                print(rows[-1], flush=True)
                            except torch.cuda.OutOfMemoryError:
                                rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc, arm=arm, fwd_ms="OOM"))
                                torch.cuda.empty_cache(); print(f"OOM {arm} {bench} L{L} X{x_pct} d{d} nc{nc}", flush=True)
                            except Exception as e:  # noqa: BLE001
                                rows.append(_row(bench=bench, L=L, X_pct=x_pct, d_qk=d, nc=nc, arm=arm, fwd_ms="ERR"))
                                torch.cuda.empty_cache()
                                print(f"ERR {arm} {bench} L{L} X{x_pct} d{d} nc{nc}: {type(e).__name__}: {e}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="RoLA scaling benchmark (RoLA vs attention; per-arch curves).")
    ap.add_argument("--benches", default="train,prefill,decode", help="comma: train,prefill,decode")
    ap.add_argument("--lens", default=os.environ.get("SCALE_LENS", "1024,4096,16384,65536"))
    ap.add_argument("--xs", default=os.environ.get("SCALE_XS", "10,25,50,100"), help="state %% of KV cache")
    ap.add_argument("--dqks", default=os.environ.get("SCALE_DQKS", "16,64,128"))
    ap.add_argument("--reps", type=int, default=C.REPS)
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    C.require_cuda()
    C.REPS = a.reps
    benches = [b for b in a.benches.split(",") if b.strip()]
    lens = [int(v) for v in a.lens.split(",") if v.strip()]
    xs = [float(v) for v in a.xs.split(",") if v.strip()]
    dqks = [int(v) for v in a.dqks.split(",") if v.strip()]
    print(f"# scaling | {C.gpu_name()} | B={C.B} H={C.H} d_model={DM} bf16 | benches={benches} "
          f"| L={lens} X%={xs} d_qk={dqks} | RoLA(rla,gla) + attn", flush=True)
    rows = []
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    try:
        sweep(benches, lens, xs, dqks, rows)
    finally:
        if rows:
            C.write_csv(out / "scaling.csv", rows)
            print(f"\nWROTE {out}/scaling.csv  [{C.gpu_name()}]  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
