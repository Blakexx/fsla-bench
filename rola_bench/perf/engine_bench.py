"""rola_bench.perf.engine_bench — cross-engine RoLA layer bench (#config-refactor surface).

Times the RoLA layer's chunk path — BOTH fwd-only (prefill) and fwd+bwd (training) — on the CURRENT
layer surface (`levels_per_head`/`states_per_level`/`routing_mode`), pointed at a chosen ENGINE
WORKTREE (a checkout of flash-linear-attention-rola-cuda-p0 on some branch — rola / rola-triton /
rola-cute / ...). Point `--worktree` at different checkouts to compare engines head-to-head on
identical shapes; the JSON line records `worktree` for provenance.

CUDA is exclusive on this box — ALWAYS run under the lock:
    flock /tmp/rola_gpu.lock python -m rola_bench.perf.engine_bench --worktree <path> [...]

Methodology (see METHODOLOGY.md) — adopts `scaling.py`'s conventions so the two report comparably:
  * median-of-N (`--reps`, default 20), `--warmup` (default 4) rounds dropped; full IQR/IQR% via
    `rola_bench.perf._common.sample_stats` (the same helper `scaling.py`'s `_summarize_*` use).
  * fwd-only and fwd+bwd are warmed and timed as TWO SEPARATE loops — Triton autotunes per grad-mode
    (the chunk kernel's forward-only inference path and its differentiable-with-backward path are
    different kernel selections), so warming only one and timing the other bakes a cold autotune
    compile into the first timed rep. This is mandatory, not an optimization (mirrors `scaling.py`'s
    "warmup absorbs the cold autotune compile" rule).
  * cuda.Event timing (GPU-side, excludes host dispatch jitter); peak memory via
    `reset_peak_memory_stats` + `max_memory_allocated` delta from a post-warmup baseline.
  * A cell that OOMs (or hits a Triton `OutOfResources`) writes a SENTINEL row (`ok=False,
    label="OOM"`) and returns cleanly rather than crashing — same infeasible-frontier-is-a-finding
    rule as `scaling.py` (reuses its `_infeasible_excs()` classification).
  * One JSON line per cell (`--out` appends; default prints to stdout) — pipe multiple invocations
    into one file for a sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_engine(worktree: str | None) -> str:
    """Insert `worktree` at sys.path[0] BEFORE any `fla_rola` import, so this process's `fla_rola` /
    `fla` resolve from the requested engine checkout rather than whatever's already importable.
    `worktree=None` (unset `--worktree` and no `ROLA_ENGINE_WORKTREE`) is a NO-OP -- `fla_rola`
    resolves from the normal import path, i.e. however the current process/image already provides it
    (pip-installed on the fleet perf image, same as every other `rola_bench/perf/*.py` script). Local
    ad hoc cross-engine comparisons pass `--worktree` explicitly; fleet dispatch (`launch.py perf
    <config>`) leaves it unset and lets the image's baked-in `fla_rola` resolve. Returns the resolved
    absolute path for the JSON provenance field, or `"<import>"` when unset."""
    if not worktree:
        return "<import>"
    wt = str(Path(worktree).resolve())
    sys.path.insert(0, wt)
    return wt


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worktree", default=os.environ.get("ROLA_ENGINE_WORKTREE"),
        help="engine checkout to import fla_rola/fla from; unset (default) = resolve fla_rola from "
             "the normal import path (pip-installed on the fleet perf image)")
    p.add_argument("--L", type=int, default=1024)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--d_qk", type=int, default=16)
    p.add_argument("--d_v", type=int, default=16)
    p.add_argument("--hidden", type=int, default=None, help="default H*d_v (no o_proj bottleneck)")
    # Topology: either --nc + --routing (legacy triple, translated via rola_compat) or --levels/--b
    # directly on the new (D, b) surface.
    p.add_argument("--nc", type=int, default=None, help="states per head (legacy triple w/ --routing)")
    p.add_argument("--routing", default="flat", choices=("flat", "square", "tree"))
    p.add_argument("--levels", type=int, default=None, help="levels_per_head (D); overrides --nc/--routing")
    p.add_argument("--b", type=int, default=None, help="states_per_level (b); overrides --nc/--routing")
    p.add_argument("--kernel", default="rla", choices=("rla", "gla_scalar"))
    p.add_argument("--state_norm", default="kappa")
    p.add_argument("--tied", action="store_true")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--out", default=None, help="append one JSON line here; default stdout only")
    p.add_argument("--smoke", action="store_true",
                    help="tiny shape (L=16,B=1,nc=4), 1 rep, 0 warmup — construct+run only, no timing claim")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    wt = _load_engine(a.worktree)

    import torch
    from rola_bench.perf._common import require_cuda, sample_stats
    from rola_bench.perf.scaling import _INFEASIBLE   # (CUDA OOM, Triton OutOfResources) classification

    require_cuda()
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16

    if a.smoke:
        a.L, a.B, a.reps, a.warmup = 16, 1, 1, 0
        if a.nc is None and a.levels is None:
            a.nc = 4

    if a.levels is not None and a.b is not None:
        D, b = a.levels, a.b
    else:
        nc = a.nc if a.nc is not None else 16
        from rola_bench.rola_compat import topology_from_legacy
        D, b = topology_from_legacy(nc, a.routing)
    nc = b ** D
    hidden = a.hidden or (a.H * a.d_v)

    row = dict(
        bench="engine_bench", worktree=wt, L=a.L, B=a.B, H=a.H, d_qk=a.d_qk, d_v=a.d_v, hidden=hidden,
        levels_per_head=D, states_per_level=b, nc=nc, routing=a.routing, kernel=a.kernel,
        state_norm=a.state_norm, tied=a.tied, reps=a.reps, warmup=a.warmup, smoke=a.smoke,
        gpu=torch.cuda.get_device_name(0), ok=True,
    )

    def _emit(r):
        line = json.dumps(r)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, "a") as fh:
                fh.write(line + "\n")
        print(line)
        return r

    try:
        from fla_rola.layers import RoLA

        layer = RoLA(hidden_size=hidden, num_heads=a.H, head_k_dim=a.d_qk, head_v_dim=a.d_v,
                     levels_per_head=D, states_per_level=b, kernel=a.kernel, state_norm=a.state_norm,
                     routing_mode={"tied": a.tied, "read": "dense", "write": "dense"},
                     layer_idx=0).to(dev, dt)

        def make_x(requires_grad):
            return torch.randn(a.B, a.L, hidden, device=dev, dtype=dt, requires_grad=requires_grad)

        def fwd_only():
            x = make_x(False)
            with torch.no_grad():
                o = layer(x)
            return o[0] if isinstance(o, tuple) else o

        def fwd_bwd():
            x = make_x(True)
            o = layer(x)
            o = o[0] if isinstance(o, tuple) else o
            o.float().sum().backward()
            return o

        def timed(fn, reps, warmup):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            samples = []
            for _ in range(reps):
                e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                samples.append(e0.elapsed_time(e1))
            peak_gb = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
            return samples, peak_gb

        fwd_samples, fwd_peak_gb = timed(fwd_only, a.reps, a.warmup)
        fwdbwd_samples, fwdbwd_peak_gb = timed(fwd_bwd, a.reps, a.warmup)

        fwd_stats = sample_stats(fwd_samples)
        fwdbwd_stats = sample_stats(fwdbwd_samples)
        row.update(
            fwd_ms_median=fwd_stats["median"], fwd_ms_iqr=fwd_stats["iqr"], fwd_ms_iqr_pct=fwd_stats["iqr_pct"],
            fwdbwd_ms_median=fwdbwd_stats["median"], fwdbwd_ms_iqr=fwdbwd_stats["iqr"],
            fwdbwd_ms_iqr_pct=fwdbwd_stats["iqr_pct"],
            fwd_peak_gb=round(fwd_peak_gb, 4), fwdbwd_peak_gb=round(fwdbwd_peak_gb, 4),
        )
    except _INFEASIBLE as e:
        row.update(ok=False, label="OOM", error=f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 -- one cell's failure must not crash the sweep
        row.update(ok=False, label="ERR", error=f"{type(e).__name__}: {e}")

    return _emit(row)


if __name__ == "__main__":
    main()
