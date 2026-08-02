"""Perf suite entry point.

Two DISTINCT things live under `rola_bench/perf/`, and this suite smoke-gates both:

  * **paper_v1** (citable, A100 fleet only, `launch.py perf paper_v1`): FINAL RoLA vs REAL external
    implementations, layer-vs-layer via `scaling.py`'s mature harness — attn/SDPA, the flagship
    (`rola-base`/`rola-base-rla`, `scaling.build_rola_named`), and the canonical `fla_rola.layers`
    baselines (GLA/GatedDeltaNet/DeltaNet/GSA). train/prefill/decode(+graphed) x L-sweep. This suite's
    smoke exercises `build_rola_named` + the FLA baseline builders directly (construct+one-tiny-fwd),
    not a full `scaling.sweep()` (that's the fleet job's job).
  * **engine_regression** (engineering-only, NOT paper material): the internal CuTe-vs-Triton bake-off,
    `rola_bench.perf.engine_bench` at the flagship geometry — per
    `rola_bench.named_configs.PERF_SUITE_CONFIGS` (`RoLA-base-rla`; the gla_scalar/kappa
    `RoLA-base` and `RoLA-tied` cells were retired by campaign-prereqs E2/E3).

Real (non-smoke) results land in `results/perf/` (CSVs from `scaling.py`) and as JSON lines from
`engine_bench.py`/`decode_bench.py` (pipe `--out` to a file under `results/perf/`). See
`rola_bench/perf/METHODOLOGY.md` for the measurement rules (local-vs-fleet, flock, warm-both-paths,
median-N) and `experiments/{paper_v1,engine_regression}.yaml` for the pinned fleet configs.

    flock /tmp/rola_gpu.lock python -m rola_bench.suites.perf --smoke
"""
from __future__ import annotations

import argparse
import subprocess
import sys

_WORKTREE = "/home/blake/flash-linear-attention-rola-cuda-p0"


def pinned_configs():
    from rola_bench.named_configs import PERF_SUITE_CONFIGS
    return list(PERF_SUITE_CONFIGS)


def _run(argv, label):
    """Each perf script runs as its OWN subprocess (matches real usage: `flock ... python -m
    rola_bench.perf.<script>` per cell) -- NOT imported and called in-process. A failed
    `torch.cuda.graph` capture attempt (e.g. decode_bench's generic fallback hitting a host-sync
    breaker) leaves the CUDA context in a bad capture state for the rest of that process; running
    every bench as a subprocess keeps one cell's capture failure from corrupting the next cell."""
    r = subprocess.run([sys.executable, "-m", *argv], capture_output=True, text=True)
    ok = r.returncode == 0
    tail = (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else ""
    print(f"{'OK  ' if ok else 'FAIL'} {label:28s} -> {tail}")
    if not ok:
        print(r.stderr[-2000:])
    return ok


def _paper_v1_smoke():
    """Construct-gate paper_v1's arm builders directly (no PYTHONPATH juggling needed -- this process
    already has the engine importable when invoked normally, unlike engine_bench/decode_bench which
    are meant to run standalone against an arbitrary `--worktree`)."""
    import torch
    import rola_bench.perf.scaling as sc
    import rola_bench.perf._common as C
    C.set_shape(L=16, B=1)
    x = lambda: torch.randn(1, 16, sc.DM, device=C.DEV, dtype=C.DT)
    ok = True
    for kind in sc._NAMED_ROLA_ARM_KINDS:
        try:
            m, used = sc.build_rola_named(kind, nc=16)
            with torch.no_grad():
                o = m(x())
            o = o[0] if isinstance(o, tuple) else o
            print(f"OK   paper_v1 arm {kind:16s} -> {tuple(o.shape)}")
        except Exception as e:
            ok = False
            print(f"FAIL paper_v1 arm {kind:16s} -> {type(e).__name__}: {e}")
    for kind in sc._FLA_BASELINE_KINDS:
        try:
            m = sc._build_fla_baseline(kind)
            with torch.no_grad():
                o = m(x())
            o = o[0] if isinstance(o, tuple) else o
            print(f"OK   paper_v1 arm {kind:16s} -> {tuple(o.shape)}")
        except Exception as e:
            ok = False
            print(f"FAIL paper_v1 arm {kind:16s} -> {type(e).__name__}: {e}")
    return ok


def run_smoke():
    ok = _paper_v1_smoke()
    for name in pinned_configs():
        # engine_bench takes --nc/--routing OR --levels/--b; the flagship named configs are square
        # topology (D=2, b=16 at nc=256) -- drive engine_bench structurally (it benches the UNIFORM
        # per-level knobs any (kernel, state_norm, tied) cell needs; the per-level-heterogeneous write
        # mix itself is exercised by the mqar/lm suites' `named_configs.construct` gate instead). This
        # still exercises the same topology, kernel, and hidden geometry as the flagship at smoke scale.
        argv = [
            "rola_bench.perf.engine_bench", "--worktree", _WORKTREE,
            "--levels", "2", "--b", "16", "--H", "8", "--d_qk", "16", "--d_v", "64",
            # campaign-prereqs E2/E3: 'gla_scalar'/'kappa' exist in no engine checkout.
            "--kernel", "rla", "--state_norm", "global", "--smoke",
        ]
        if "tied" in name.lower():
            argv.append("--tied")
        ok &= _run(argv, f"engine_bench {name}")

    ok &= _run(["rola_bench.perf.decode_bench", "--worktree", _WORKTREE, "--smoke"], "decode_bench")

    try:
        import torch
        import rola_bench.perf.scaling as sc
        m = sc.build_rola(False, 16, 16, routing="square")   # d_qk=16 -- the engine's only compiled prototype
        x = torch.randn(1, 4, sc.DM, device=sc.C.DEV, dtype=sc.C.DT)
        with torch.no_grad():
            o = m(x)
        print(f"OK   scaling.build_rola -> {tuple((o[0] if isinstance(o, tuple) else o).shape)}")
    except Exception as e:
        ok = False
        print(f"FAIL scaling.build_rola -> {type(e).__name__}: {e}")

    print("SMOKE " + ("OK" if ok else "FAILED"))
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    if a.smoke:
        sys.exit(0 if run_smoke() else 1)
    print(__doc__)


if __name__ == "__main__":
    main()
