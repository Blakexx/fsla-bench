"""MQAR suite entry point.

Pinned configs = the 13 named `rola_instance()` presets (`rola_bench.mqar.instances.ROLA_INSTANCES`,
the LOCKED zoo — numbers/semantics unchanged by the #config-refactor, only the construction-boundary
kwargs were translated, see `zoology.mixers.rola.RoLAMixer` / `rola_bench.rola_compat`) crossed with
the three routing topologies {flat, square, tree}, PLUS the new pinned flagship named configs
(`rola_bench.named_configs`) in their MQAR (conv-free) form, including the base, symmetric-coarse,
and union-split ablations. The construction gate records any compatibility fallback rather than
silently swapping a cell; union-split itself has no fallback because that would change the ablation.

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
    """MQAR (conv-free) form of a named flagship config, via the RoLAMixer level_routing_configs
    passthrough (zoology.mixers.rola). Falls back the same way `rola_bench.named_configs.construct`
    does (checks `_routing_resolution_error` post-construction, no forward() needed)."""
    from rola_bench.named_configs import NAMED_CONFIGS
    from zoology.mixers.rola import RoLAMixer
    cfg = NAMED_CONFIGS[cell["name"]]
    kwargs = cfg.build_kwargs(cell["nc"], False)   # use_short_conv=False -- MQAR convs confound recall
    level_routing_configs = kwargs.pop("level_routing_configs")
    D, b = kwargs.pop("levels_per_head"), kwargs.pop("states_per_level")
    kernel = kwargs.pop("kernel")
    d_model = 8 * 64  # H=8, d_v=64 (flagship geometry)
    m = RoLAMixer(d_model=d_model, d_qk=16, d_v=64, levels_per_head=D, states_per_level=b, n_heads=8,
                 kernel=kernel, level_routing_configs=level_routing_configs, layer_idx=0,
                 **kwargs).to(dev)
    err = getattr(m, "_routing_resolution_error", None)
    if err is None:
        return m, cell["name"], False
    if cfg.fallback is None or "per-level heterogeneous routing activation" not in str(err):
        raise err
    fb = NAMED_CONFIGS[cfg.fallback]
    fb_kwargs = fb.build_kwargs(cell["nc"], False)
    fb_lrc = fb_kwargs.pop("level_routing_configs")
    fb_D, fb_b = fb_kwargs.pop("levels_per_head"), fb_kwargs.pop("states_per_level")
    fb_kernel = fb_kwargs.pop("kernel")
    m = RoLAMixer(d_model=d_model, d_qk=16, d_v=64, levels_per_head=fb_D, states_per_level=fb_b, n_heads=8,
                 kernel=fb_kernel, level_routing_configs=fb_lrc, layer_idx=0,
                 **fb_kwargs).to(dev)
    return m, fb.name, True


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
            m, used, requires_c8 = _construct_named(cell)
            x = torch.randn(1, 4, 8 * 64, device="cuda")
            with torch.no_grad():
                o = m(x)
            tag = f"(fallback -> {used}, requires_c8)" if requires_c8 else ""
            print(f"OK  named  {cell['name']:24s} -> {tuple(o.shape)} {tag}")
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
