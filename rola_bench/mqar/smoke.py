#!/usr/bin/env python3
"""Lightweight per-box GPU smoke — the fleet boot gate (FLEET_VERIFY_CMD).

This is NOT correctness verification (that's pre-deploy CI: the full fla-rola suite in
rola-scratch). It just exercises the production routed-kernel path once — a tiny RoLA fwd+bwd on
CUDA — so a host with a broken GPU / driver / Triton-compile env fails FAST and gets blacklisted,
instead of silently failing every cell of a batch. Prints "VERIFY OK" (the token box_server greps).
"""
import sys
import torch


def main():
    if not torch.cuda.is_available():
        print("VERIFY FAILED: torch.cuda.is_available() is False")
        return 1
    import fla.ops.simple_gla, fla.layers  # noqa: F401  — load the kernel stack
    from rola_bench import engine
    engine.import_engine()        # the engine package itself (kernel stack side effect)
    from rola_bench.engine import RoLA
    torch.manual_seed(0)
    dev = "cuda"
    # V3 surface (2026-08-02): `head_k_dim` is GONE — the layer owns no Q/K projection, and a state
    # is [N, d_v]. This call carried it until today, which means the fleet boot gate would have
    # TypeError'd on every box the moment the mainline engine was pinned. N = 4 via
    # levels_per_head=1, states_per_level=4 (the D1 factorization).
    m = RoLA(hidden_size=64, head_v_dim=16, levels_per_head=1, states_per_level=4,
             num_heads=2).to(dev)
    x = torch.randn(2, 64, 64, device=dev, requires_grad=True)
    o = m(x)
    if isinstance(o, tuple):
        o = o[0]
    o.float().pow(2).mean().backward()
    ok = (bool(torch.isfinite(o).all()) and x.grad is not None
          and bool(torch.isfinite(x.grad).all()))
    print(f"VERIFY {'OK' if ok else 'FAILED'} | routed RoLA fwd+bwd on {torch.cuda.get_device_name(0)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
