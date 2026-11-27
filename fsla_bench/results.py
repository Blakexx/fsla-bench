"""THE BENCH'S OWN RECORDS: what its programs store through `fsla_devtools.results`, each at the label of the target that
runs it (a program run outside Bazel names the label of the target it stands for), and the readings over them.

    FLEET        `@fsla_bench//:fleet`            every row a fleet job pulls (`fsla_bench.fleet.jobs`)
    LOCAL_GRID   `@fsla_bench//:mqar_local_grid`  every cell the local MQAR grid trains (`fsla_bench.mqar.local_grid`)

What names a cell is its SEMANTICS -- the bench, the config, the cell, and the code that ran it (the fleet's image, the
local grid's checkouts) -- so a cell's samples group by their key: a cell is DONE for that code when one of them has no
error (`completed`). A reading (`rows`) takes both labels' successful samples of one bench (and config), the newest
output per run id.
"""
from __future__ import annotations

import json
from pathlib import Path

from fsla_devtools.results import Store, key

FLEET = "@fsla_bench//:fleet"
LOCAL_GRID = "@fsla_bench//:mqar_local_grid"


def completed(store: Store, semantics: dict) -> bool:
    """Whether the cell these semantics name has a sample with no error."""
    return any(s["ok"] for s in store.samples(key(semantics)))


def rows(bench: str, config: str | None = None, root: Path | str | None = None) -> list[dict]:
    """Every completed cell of `bench` (of `config` alone when given) the fleet and the local grid stored: the newest
    output per run id, each an `ok` row naming its `run_id`."""
    found = []
    for label in (FLEET, LOCAL_GRID):
        store = Store(label, root)
        for sample in store.samples():
            semantics = sample["semantics"]
            if sample["ok"] and semantics["bench"] == bench and (config is None or semantics["config"] == config):
                found.append((sample["utc"], sample["run"], json.loads(store.output(sample).read_text())))
    newest = {}
    for _utc, _run, row in sorted(found, key=lambda f: (f[0], f[1])):
        if row.get("ok") and row.get("run_id"):
            newest[row["run_id"]] = row
    return list(newest.values())


__all__ = ["FLEET", "LOCAL_GRID", "completed", "rows"]
