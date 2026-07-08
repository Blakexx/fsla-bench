"""LM suite entry point.

Pinned configs = `rola_bench.lm.arms.ARMS` (the LOCKED 5-arm iso-backbone registry — RoLA kappa
variants, gdn/gdn-isostate, gla/gla-isostate, attn), unchanged by the #config-refactor beyond the
`RoLAConfig(levels_per_head=1, states_per_level=nc, ...)` construction-boundary translation
(`rola_bench.lm.arms.arm_config`, `rola_bench.rola_compat`).

KNOWN GAP (flagged, not silently worked around): the coordinator's new named flagship — `RoLA-base` /
`RoLA-base-rla` / `RoLA-base-global` (the kappa-vs-global ablation row; `rola_bench.named_configs`,
per-level heterogeneous write routing: sparse coarse +
dense fine) — is a `fla_rola.layers.RoLA` DIRECT-construction config (verified constructible with the
`requires_c8` gate, see `rola_bench.named_configs.construct` / `rola_bench.suites.mqar`). It is NOT
yet reachable through the HF LM path: `RoLAConfig` (`fla_rola/models/rola/configuration_rola.py`) has
no `routing_mode`/`level_routing_configs` field, so `modeling_rola.py` cannot build a per-level
heterogeneous layer from a config today. Wiring that is an ENGINE-side change (new RoLAConfig fields +
threading them into `RoLAModel`/`RoLAForCausalLM`), out of this bench-repo migration's scope — tracked
here as the follow-up gap, not implemented. `--smoke` therefore verifies the flagship's LAYER-level
constructibility (via `rola_bench.named_configs`) alongside the existing ARMS registry, so the gap is
gated (never silently regresses) even though it isn't wired into `arm_config()` yet.

Real training runs go through the existing fleet machinery (`rola_bench/lm/run_lm.py` ->
`train_lm.py`) and land in `results/lm/<dataset>/*.json` (see the top-level README's suite table).

    python -m rola_bench.suites.lm --smoke      # build every pinned arm config + one tiny fwd step
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

    sys.path.insert(0, "/home/blake/flash-linear-attention-rola-cuda-p0")
    from rola_bench.lm import arms

    if a.list:
        for arm in arms.ARMS:
            print(arm)
        from rola_bench.named_configs import LM_SUITE_CONFIGS
        print("(flagship, layer-level only, see docstring): " + " ".join(LM_SUITE_CONFIGS))
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
            with torch.no_grad():
                out = model(input_ids=ids)
            n = sum(p.numel() for p in model.parameters())
            print(f"OK   arm {arm:24s} params={n/1e6:.1f}M logits={tuple(out.logits.shape)}")
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
            layer, used, requires_c8 = construct(
                name, hidden_size=512, num_heads=8, head_k_dim=16, head_v_dim=64, nc=256,
                use_short_conv=True)
            layer = layer.cuda() if torch.cuda.is_available() else layer
            x = torch.randn(1, 8, 512, device=next(layer.parameters()).device)
            with torch.no_grad():
                o = layer(x)
            tag = f"(fallback -> {used}, requires_c8)" if requires_c8 else ""
            print(f"OK   flagship-layer {name:20s} -> {tuple((o[0] if isinstance(o, tuple) else o).shape)} {tag}")
        except Exception as e:
            ok = False
            print(f"FAIL flagship-layer {name:20s} -> {type(e).__name__}: {e}")

    print("SMOKE " + ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
