"""MQAR suite entry point.

Pinned configs = the 13 named `rola_instance()` presets (`rola_bench.mqar.instances.ROLA_INSTANCES`,
the LOCKED zoo — numbers/semantics unchanged by the #config-refactor, only the construction-boundary
kwargs were translated, see `zoology.mixers.rola.RoLAMixer` / `rola_bench.rola_compat`) crossed with
the three routing topologies {flat, square, tree}, PLUS the new pinned flagship named configs
(`rola_bench.named_configs`) in their MQAR (conv-free) form, including the base, symmetric-coarse,
and union-split ablations. Unsupported production capabilities fail loudly rather than changing a
cell's topology or semantics.

Real runs go through the existing fleet machinery (`rola_bench/mqar/run_rla_sweep.py`,
`rola_bench/mqar/job.py`) and land in the MQAR results tree / zoology run ids — this entry point is
the construct-gate + a thin driver, not a second sweep engine.

    python -m rola_bench.suites.mqar --smoke      # construct every pinned config, one tiny fwd+bwd
    python -m rola_bench.suites.mqar --list        # print the pinned config table (name, cell schema)
"""
from __future__ import annotations

import argparse
import json
import sys

SEEDS = (0, 1, 2)          # MQAR seed convention (matches rola_bench.mqar.experiments.seed_confirm)
ROUTINGS = ("flat", "square", "tree")


def pinned_cells():
    """Yield {name, routing, d_qk, d_v, num_chunks, n_heads} for every LOCKED preset x topology."""
    from rola_bench.mqar.instances import ROLA_INSTANCES
    for name in ROLA_INSTANCES:
        for routing in ROUTINGS:
            yield dict(name=name, routing=routing, d_qk=16, d_v=16, num_chunks=16, n_heads=2)


def named_cells():
    """Yield {name, nc} for the new flagship named configs, MQAR (conv-free) form."""
    from rola_bench.named_configs import MQAR_SUITE_CONFIGS
    for name in MQAR_SUITE_CONFIGS:
        yield dict(name=name, nc=256)


def _construct_pinned(cell, dev="cuda"):
    from rola_bench.mqar.instances import rola_instance
    from zoology.mixers.rola import RoLAMixer
    kw = rola_instance(cell["name"], d_qk=cell["d_qk"], d_v=cell["d_v"], num_chunks=cell["num_chunks"],
                       n_heads=cell["n_heads"], routing=cell["routing"])
    d_model = cell["n_heads"] * cell["d_v"]
    return RoLAMixer(d_model=d_model, layer_idx=0, **kw).to(dev)


def _construct_named(cell, dev="cuda"):
    """MQAR (conv-free) form of a named flagship config via the level-routing passthrough.

    BLOCKED on a zoology-side migration (campaign-prereqs E1, 2026-08-01). `named_configs` now
    speaks the V3 layer surface (`decay`, no `head_k_dim`/`d_qk`/`qk_norm`), but
    `zoology.mixers.rola.RoLAMixer` (zoology @ efe1a0b) still takes a REQUIRED `d_qk` and forwards
    `head_k_dim=d_qk`, `kernel=...`, `qk_norm=...` into `fla_rola.layers.RoLA` — the pre-V3 Q/K
    surface the mainline engine deleted. There is no translation that preserves the cell: `d_qk` has
    no V3 meaning at all. Refusing loudly here rather than silently constructing a different cell
    against a stale engine.
    """
    raise NotImplementedError(
        "MQAR named-config construction is blocked on migrating zoology.mixers.rola.RoLAMixer to "
        "the V3 layer surface (it still requires d_qk and passes head_k_dim/kernel/qk_norm into "
        "fla_rola.layers.RoLA, none of which exist on the mainline V3 layer). See "
        "workflows/20260801-campaign-prereqs/prereqs.md E1. Layer-level construction of the same "
        "named configs is gated by rola_bench/test_named_config_construction.py.")


def run_smoke():
    import torch
    if not torch.cuda.is_available():
        print("SMOKE SKIPPED: no CUDA GPU available")
        return True
    ok = True
    for cell in pinned_cells():
        try:
            m = _construct_pinned(cell)
            x = torch.randn(1, 4, cell["n_heads"] * cell["d_v"], device="cuda")
            with torch.no_grad():
                o = m(x)
            print(f"OK  pinned {cell['name']:24s} {cell['routing']:7s} -> {tuple(o.shape)}")
        except Exception as e:
            ok = False
            print(f"FAIL pinned {cell['name']:24s} {cell['routing']:7s} -> {type(e).__name__}: {e}")
    for cell in named_cells():
        try:
            m, used = _construct_named(cell)
            x = torch.randn(1, 4, 8 * 64, device="cuda")
            with torch.no_grad():
                o = m(x)
            print(f"OK  named  {cell['name']:24s} -> {tuple(o.shape)}")
        except Exception as e:
            ok = False
            print(f"FAIL named  {cell['name']:24s} -> {type(e).__name__}: {e}")
    print("SMOKE " + ("OK" if ok else "FAILED"))
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    if a.list:
        for cell in pinned_cells():
            print(json.dumps(cell))
        for cell in named_cells():
            print(json.dumps(cell))
        return
    if a.smoke:
        ok = run_smoke()
        sys.exit(0 if ok else 1)
    print(__doc__)


if __name__ == "__main__":
    main()
