"""rola_bench.perf.decode_bench — thin named entry point for ONE eager-vs-graphed decode cell.

The actual decode measurement lives in `scaling.py` (the mature architecture-comparison harness —
IQR, sentinel rows, state-MiB, round-robin scheduling): `_decode_layer_sample` (eager) /
`_decode_layer_sample_graphed` (the C9 pattern — a `torch.cuda.graph` capture of the decode call,
`rola-scratch/workflows/dag/dag_tasks/C9-decode-swap-graph.md`). This script does NOT reimplement a
second timing discipline; it just builds one RoLA layer via `scaling.build_rola`, times ONE eager
sample and ONE graphed sample through
those same functions, and reports the pair as a JSON line — for ad hoc / `rola_bench.suites.perf`
use. For a real sweep (multiple nc/L/routing cells, round-robin, sentinels), use
`scaling.py --benches decode` with `SCALE_DECODE_GRAPHED=1` (see METHODOLOGY.md / `paper_v1.yaml`).

    flock /tmp/rola_gpu.lock python -m rola_bench.perf.decode_bench [--worktree <path>] [...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_engine(worktree: str | None) -> str:
    """See `engine_bench._load_engine` — `worktree=None` (unset) is a no-op, `fla_rola` resolves from
    the normal import path (pip-installed on the fleet perf image); `--worktree` overrides for local
    ad hoc cross-engine comparisons."""
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
    p.add_argument("--nc", type=int, default=256)
    p.add_argument("--dqk", type=int, default=16)
    p.add_argument("--routing", default="square", choices=("flat", "square", "tree"))
    p.add_argument("--L", type=int, default=512, help="prefill length used to seed the decode state")
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true",
                    help="tiny shape (nc=4, L=8) — construct+run one eager + one graphed sample only")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    wt = _load_engine(a.worktree)

    from rola_bench.perf._common import require_cuda
    require_cuda()

    if a.smoke:
        a.nc, a.L = 4, 8

    import rola_bench.perf.scaling as sc

    module = sc.build_rola(a.nc, a.dqk, routing=a.routing)

    eager_ms, eager_peak = sc._decode_layer_sample(module, a.L)
    graph_kind, graphed_ms, graphed_peak, graph_error = "cuda_graph", None, None, None
    try:
        graphed_ms, graphed_peak = sc._decode_layer_sample_graphed(module, a.L)
    except Exception as e:
        graph_kind, graph_error = f"unsupported ({graph_kind})", f"{type(e).__name__}: {e}"

    row = dict(
        bench="decode_bench", worktree=wt, nc=a.nc, dqk=a.dqk, routing=a.routing, L=a.L,
        eager_ms=round(eager_ms, 4), eager_peak_mib=eager_peak,
        graphed_ms=(round(graphed_ms, 4) if graphed_ms is not None else None),
        graphed_peak_mib=graphed_peak, graph_kind=graph_kind, graph_error=graph_error,
        uplift_pct=(round((eager_ms / graphed_ms - 1.0) * 100.0, 1)
                    if graphed_ms not in (None, 0) else None),
        smoke=a.smoke,
    )
    import torch
    row["gpu"] = torch.cuda.get_device_name(0)
    line = json.dumps(row)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as fh:
            fh.write(line + "\n")
    print(line)
    return row


if __name__ == "__main__":
    main()
