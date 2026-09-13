"""`python -m rola_bench.suite run|plan|show` -- see rola_bench/suite/README.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import engine
from .modules import Options, nodes_for
from .target import config, parse


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m rola_bench.suite", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "run every incomplete node"), ("plan", "list every node and whether it is complete")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--target", required=True, help="worktree:PATH[,venv:PATH][,label:NAME][,schedule:first|DENSE/SPARSE]")
        p.add_argument("--reference", action="append", default=[], help="an arm timed against the target (repeatable)")
        p.add_argument("--modules", default="all", help="all, or a comma list of modules or prefixes (carry, timing.session)")
        p.add_argument("--cells", default="all", help="all, gate, or a comma list of carry cells")
        p.add_argument("--subjects", default="all", help="all, or a comma list of bench subjects")
        p.add_argument("--reps", type=int, default=5)
        p.add_argument("--warmup", type=int, default=2)
        p.add_argument("--rounds", type=int, default=3)
        if name == "run":
            p.add_argument("--repeat", action="store_true", help="add a sample to complete nodes of repeatable modules")
            p.add_argument("--force", action="store_true", help="run every selected node again")
    p = sub.add_parser("show", help="print the records of a module")
    p.add_argument("module")
    p.add_argument("--store", type=Path, required=True, help="the store root (tools/dev.py get store.root)")
    a = ap.parse_args()

    if a.cmd == "show":
        for path in sorted((a.store / "suite" / a.module).glob("*.json")):
            if path.name.count(".") == 1:
                rec = json.loads(path.read_text())
                print(f"{rec['key'][:12]} {rec['semantics']['unit'] or '-':40s} ok={rec['ok']} samples={len(rec['samples'])}"
                      f" {rec['meta'].get('label', '')} {rec['meta'].get('git_sha', '')[:7]}")
        return 0

    target = parse(a.target)
    opt = Options([parse(r) for r in a.reference], a.cells, a.subjects, a.reps, a.warmup, a.rounds)
    store = engine.Store(Path(config(target, "store.root")))
    nodes = nodes_for(target, opt, a.modules)
    outcomes = engine.run(nodes, store, dry=a.cmd == "plan", repeat=getattr(a, "repeat", False),
                          force=getattr(a, "force", False), log=lambda line: print(line, flush=True))
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    print("suite: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + f"; records under {store.root}")
    return 1 if counts.get("failed") or counts.get("blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
