"""LM suite entry point.

Pinned configs = `rola_bench.lm.arms.ARMS` (the dense-router RoLA comparison variants, the
published-flagship RoLA-base arms, gdn, gla, attn). The kappa / gla_scalar / gla_static arms are
RETIRED, and `gdn-isostate` / `gla-isostate` were deleted as byte-identical duplicates of
`gdn`/`gla` (finding F4) — see `arms.RETIRED_LM_ARMS` and the campaign-prereqs E1-E7 migration.

The published flagship — `rola-base-rla` / `rola-base-union-global` /
`rola-base-hybrid-global-rla` / `rola-base-hybrid-tiedtop-global-rla` — router_bias=True,
two-level topology (required mixed-radix [8,16], N=128, at d_v=128 — d_v re-pinned 2026-08-02 per
finding F1, `arms._rola_base_topology`), per-level heterogeneous write: sparse@coarse +
dense@fine, dense read (asym arms), short conv ON — see `rola_bench.lm.arms.ROLA_BASE_*` — reachable
through the HF `AutoModelForCausalLM` path like every other arm: `RoLAConfig`
(`fla_rola/models/rola/configuration_rola.py`) carries the `level_routing_configs` field,
threaded through `rola_instance()`/`RoLABlock` into the RoLA layer kwargs
(`fla_rola/models/rola/instances.py`, `modeling_rola.py`). Per-level heterogeneous activation support
is required for the production cell and must execute end-to-end.
`rola_bench.named_configs`/`construct()` (the LAYER-direct-construction surface used by the MQAR/perf
suites) is unaffected by this change and still exists as an independent construction path.

Real training runs go through the existing fleet machinery (`rola_bench/lm/run_lm.py` ->
`train_lm.py`) and land in `results/lm/<dataset>/*.json` (see the top-level README's suite table).

    python -m rola_bench.suites.lm --smoke      # build every pinned arm config + one tiny fwd+bwd step
    python -m rola_bench.suites.lm --list        # print the pinned arm list
"""
from __future__ import annotations

import argparse
import sys

SEEDS = (7, 11, 42, 1337, 2024)   # matches results/lm/*/gdn_s*.json seed convention


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)

    from rola_bench.lm import arms

    if a.list:
        for arm in arms.ARMS:
            print(arm)
        from rola_bench.named_configs import LM_SUITE_CONFIGS
        print("(flagship layer configs, see docstring): " + " ".join(LM_SUITE_CONFIGS))
        return

    if not a.smoke:
        print(__doc__)
        return

    import torch
    ok = True
    for arm in arms.ARMS:
        try:
            cfg = arms.arm_config(arm, vocab_size=256)
            model = arms.build_arm(arm, vocab_size=256)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(dev)
            ids = torch.randint(0, 256, (1, 8), device=dev)
            # A real tiny train step (fwd+bwd), not just a no_grad forward -- catches autograd-graph
            # gaps a forward-only smoke would miss (see layer-dispatch-autograd-gap history).
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            n = sum(p.numel() for p in model.parameters())
            print(f"OK   arm {arm:24s} params={n/1e6:.1f}M loss={out.loss.item():.3f}")
        except Exception as e:
            # attn needs flash-attn installed (a separate infra dependency, not a #config-refactor
            # migration gap) -- report it distinctly so a real regression doesn't hide in the noise.
            is_infra = arm == "attn" and "flash" in str(e).lower()
            if not is_infra:
                ok = False
            print(f"{'INFRA' if is_infra else 'FAIL '} arm {arm:24s} -> {type(e).__name__}: {e}")

    from rola_bench.named_configs import construct, LM_SUITE_CONFIGS
    for name in LM_SUITE_CONFIGS:
        try:
            # head_v_dim=128 is the flagship value width (F1 re-pin, matched with GDN/GLA at
            # textbook d_k=d_v=128). nc stays 256 because `named_configs` builds a SQUARE topology
            # (b=16 here) while the LM arms are mixed-radix [8,16] — a pre-existing divergence
            # between the two construction paths, not something this smoke should paper over.
            layer, used = construct(
                name, hidden_size=512, num_heads=8, head_v_dim=128, nc=256,
                use_short_conv=True)
            layer = layer.cuda() if torch.cuda.is_available() else layer
            x = torch.randn(1, 8, 512, device=next(layer.parameters()).device)
            with torch.no_grad():
                o = layer(x)
            print(f"OK   flagship-layer {name:20s} -> {tuple((o[0] if isinstance(o, tuple) else o).shape)}")
        except Exception as e:
            ok = False
            print(f"FAIL flagship-layer {name:20s} -> {type(e).__name__}: {e}")

    print("SMOKE " + ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
