"""MQAR suite entry point — the construct-gate + a thin smoke driver, not a second sweep engine.

Pinned cells, REBUILT 2026-08-02 (Phase 1.6):

  * the PRESET cells — `rola_bench.mqar.instances.ROLA_INSTANCES`, the V3 registry: the three
    canonical wirings (dense-read/sparse-write, union, level split), the D1 DENSE CONTROL, and the
    variants a grid axis demands (the alpha-2 write split; the tied spellings for the
    coupling-independence cells). The pre-V3 zoo of 13 presets x {flat,square,tree} is gone: those
    presets were keyed on `d_qk`, `kernel='gla_scalar'` and `state_norm in {'kappa','per_state'}`,
    all deleted by V3, and each retires by name in `instances.RETIRED_INSTANCES`. Topology is no
    longer a separate cross — it is part of the preset's own wiring (D=1 for the control, D=2 for
    the wirings), which is what makes the wirings comparable at matched N.

  * the NAMED cells — `rola_bench.models.named_configs.MQAR_SUITE_CONFIGS` in their MQAR (conv-free) form,
    constructed through the migrated `zoology.mixers.rola.RoLAMixer`.

Real runs go through the fleet machinery (`rola_bench/mqar/run.py`, `rola_bench/mqar/
job.py`) and the `experiments/*.yaml` specs.

    python -m rola_bench.mqar.smoke --smoke      # construct every pinned config, one tiny fwd+bwd
    python -m rola_bench.mqar.smoke --list       # print the pinned config table
"""
from __future__ import annotations

import argparse
import json
import sys

SEEDS = (0, 1, 2)          # MQAR seed convention (matches rola_bench.mqar.experiments.seed_confirm)
#: Head geometry for the NAMED cells' smoke construction. `named_configs` pins routing, topology,
#: decay and normalization; `head_v_dim`/`num_heads` are a caller geometry and are stated here.
NAMED_SMOKE_HEADS, NAMED_SMOKE_DV = 8, 64


def pinned_cells():
    """Yield {name, nc, n_heads, d_v} for every preset in the V3 registry.

    `nc=16` is a smoke rung, not a published one: it is the smallest state count that factorizes for
    every preset in the registry (D=1 -> b=[16]; D=2 -> b=[4,4]). The published rungs are the `ncs`
    ladders in `mqar/experiments/*.yaml`."""
    from rola_bench.mqar.instances import ROLA_INSTANCES
    for name in ROLA_INSTANCES:
        yield dict(name=name, nc=16, n_heads=2, d_v=16)


def named_cells():
    """Yield {name, nc} for the new flagship named configs, MQAR (conv-free) form."""
    from rola_bench.models.named_configs import MQAR_SUITE_CONFIGS
    for name in MQAR_SUITE_CONFIGS:
        yield dict(name=name, nc=256)


def _construct_pinned(cell, dev="cuda"):
    from rola_bench.mqar.instances import rola_instance
    from zoology.mixers.rola import RoLAMixer
    kw = rola_instance(cell["name"], cell["nc"], n_heads=cell["n_heads"], d_v=cell["d_v"])
    d_model = cell["n_heads"] * cell["d_v"]
    return RoLAMixer(d_model=d_model, layer_idx=0, **kw).to(dev)


def _construct_named(cell, dev="cuda"):
    """MQAR (conv-free) form of a named flagship config, through the migrated mixer.

    UNBLOCKED 2026-08-02 (Phase 1.6). `named_configs.NAMED_CONFIGS[...].build_kwargs` emits V3 layer
    kwargs and `RoLAMixer` forwards them verbatim (aliasing only the three zoology names), so this
    is the same layer `named_configs.construct()` builds. Convs are forced OFF regardless of the
    registry's default: short convs confound short-offset recall in MQAR, which is the whole task.
    """
    from rola_bench.models.named_configs import NAMED_CONFIGS, RETIRED_CONFIGS
    from zoology.mixers.rola import RoLAMixer
    name = cell["name"]
    if name in RETIRED_CONFIGS:
        raise ValueError(f"named config {name!r} is RETIRED: {RETIRED_CONFIGS[name]}")
    kwargs = NAMED_CONFIGS[name].build_kwargs(cell["nc"], False)     # conv OFF, MQAR form
    mixer = RoLAMixer(d_model=NAMED_SMOKE_HEADS * NAMED_SMOKE_DV, layer_idx=0,
                      n_heads=NAMED_SMOKE_HEADS, d_v=NAMED_SMOKE_DV, **kwargs).to(dev)
    return mixer, name


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
            print(f"OK  pinned {cell['name']:32s} nc={cell['nc']:<4d} -> {tuple(o.shape)}")
        except Exception as e:
            ok = False
            print(f"FAIL pinned {cell['name']:32s} nc={cell['nc']:<4d} -> {type(e).__name__}: {e}")
    for cell in named_cells():
        try:
            m, used = _construct_named(cell)
            x = torch.randn(1, 4, NAMED_SMOKE_HEADS * NAMED_SMOKE_DV, device="cuda")
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
