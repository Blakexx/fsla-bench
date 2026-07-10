"""Compat translation for the #config-refactor RoLA layer surface change (2026-07-06).

The old `fla_rola.layers.RoLA` construction kwargs `states_per_head` (nc) + `routing` ∈
{'flat','square','tree'} and `tie_routers: bool` were REMOVED. The new surface is structural:
`levels_per_head` (D) / `states_per_level` (b) with `nc = b**D`, and `routing_mode` /
`level_routing_configs` for the tied/sparse-vs-dense knobs.

This module is a PURE construction-boundary translation — it does not change any pinned MQAR/LM
config NUMBERS (nc, tie-ness), only how they're spelled to the new layer/config constructors.
`zoology.mixers.rola.RoLAMixer` itself speaks the new (D, b) / `routing_mode` surface directly (no
shim) — these two functions exist for the rola-bench call sites that still want the convenient
flat/square/tree + tie_routers spelling on their OWN CLI/config surface (`rola_bench.mqar.instances`,
`rola_bench.perf.engine_bench`, ...) and translate it at their own construction boundary before
passing kwargs to `RoLAMixer`/`fla_rola.layers.RoLA`. Every such site should route through these two
functions rather than re-deriving the (D, b) factorization inline.
"""
from __future__ import annotations

import math


def topology_from_legacy(num_chunks: int, routing: str = "flat") -> tuple[int, int]:
    """(states_per_head=nc, routing) -> (levels_per_head=D, states_per_level=b), b**D == nc.

    flat(nc)   -> (1, nc)
    square(nc) -> (2, sqrt(nc))      -- nc must be a perfect square
    tree(nc)   -> (log2(nc), 2)      -- nc must be a power of 2
    """
    if routing == "flat":
        return 1, int(num_chunks)
    if routing == "square":
        b = int(round(math.sqrt(num_chunks)))
        if b * b != num_chunks:
            raise ValueError(f"square routing requires a perfect-square num_chunks, got {num_chunks}")
        return 2, b
    if routing == "tree":
        D = int(round(math.log2(num_chunks)))
        if 2 ** D != num_chunks:
            raise ValueError(f"tree routing requires a power-of-2 num_chunks, got {num_chunks}")
        return D, 2
    raise ValueError(f"unknown legacy routing {routing!r} (expected 'flat'|'square'|'tree')")


def routing_mode_from_tie_routers(tie_routers: bool) -> dict:
    """Old `tie_routers: bool` -> the new `routing_mode` dict. The old router was ALWAYS softmax
    (dense) on both sides -- there was no sparse/entmax knob pre-refactor -- so the translation is
    always {'read': 'dense', 'write': 'dense'}, only `tied` varies. This is numerically exact: dense
    (softmax, no bitmap) is the new layer's strict superset of the old always-dense router."""
    return {"tied": bool(tie_routers), "read": "dense", "write": "dense"}
