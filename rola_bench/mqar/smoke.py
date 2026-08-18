#!/usr/bin/env python3
"""Lightweight per-box GPU smoke — the fleet boot gate (FLEET_VERIFY_CMD).

This is NOT correctness verification (that's pre-deploy CI: the engine's own suite). It just
exercises the production routed-kernel path once — a tiny RoLA fwd+bwd on CUDA — so a host with a
broken GPU / driver / extension load fails FAST and gets blacklisted, instead of silently failing
every cell of a batch. Prints "VERIFY OK" (the token box_server greps).

The extension is loaded FIRST and on purpose. The engine is closed-world: a compiled arm matrix
ratified per architecture, with a device-side refusal on the first launch for an architecture the
build was never measured for (`check_arch_table`). A rented box of the wrong arch is therefore a
boot-gate failure rather than a mid-batch one, and only a load on the real device can say so.
"""
import sys
import torch


def main():
    if not torch.cuda.is_available():
        print("VERIFY FAILED: torch.cuda.is_available() is False")
        return 1
    from rola_bench import engine
    engine.import_grid_engine("ops._ext").extension()   # the closed-world binary, on this device
    engine.import_engine()                              # the FLA-integration vehicle
    from rola_bench.engine import RoLA
    torch.manual_seed(0)
    dev = "cuda"
    # The layer owns no Q/K projection and a state is [N, d_v], so the geometry is stated as
    # (D, widths, d_v): N = 4 via levels_per_head=1, states_per_level=4 (the D1 factorization).
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
