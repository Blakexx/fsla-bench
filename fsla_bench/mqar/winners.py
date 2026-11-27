"""THE SEED CONFIRMATION'S WINNERS: the best-max_acc run_id per (kernel, state rung) across every MQAR cell the fleet and
the local grid stored (`fsla_bench.results.rows`), which `fsla_bench.mqar.experiments.seed_confirm` re-runs at extra
seeds. Standard library and the store only: reading the winners trains nothing."""
from __future__ import annotations

import re

from fsla_bench.results import rows


def _kernel(rid):
    if rid.startswith("grid-routed-"):
        return "routed-" + rid.split("-")[2]
    if rid.startswith("grid-mha"):
        return "mha"
    if rid.startswith("grid-sse"):
        return "sse"
    return rid.split("-")[1]


def pick_winners(valid_ids=None, root=None):
    """Best-max_acc run_id per (kernel, nc) across the results (SSE excluded — its faithful impl is
    too slow to seed-confirm). `valid_ids` restricts the search to one spec's own cells, so a
    higher-acc cell from a DIFFERENT experiment can't steal a (kernel, nc) winner slot."""
    best = {}
    for j in rows("mqar", root=root):                  # every stored MQAR cell, the newest sample per run_id
        if valid_ids is not None and j["run_id"] not in valid_ids:
            continue
        k = _kernel(j["run_id"])
        if k == "sse":
            continue
        ncm = re.search(r"nc(\d+)", j["run_id"])
        nc = int(ncm.group(1)) if ncm else -1
        acc = j.get("max_acc") or 0.0
        if acc > best.get((k, nc), (-1.0, None))[0]:
            best[(k, nc)] = (acc, j["run_id"])
    return {rid for _, rid in best.values()}


__all__ = ["pick_winners"]
