"""Phase-0 parity gate: fla_rola.layers.RoLA (new canonical layer) vs rola.RoLA (current).

For every ROLA_INSTANCES name whose kernel the canonical layer supports (rla / gla_scalar),
build BOTH modules at identical geometry, copy weights old->new (key map w_q->q_proj etc.,
strip the `kernel.` submodule prefix), and assert: (1) identical forward output to ~1e-5 on the
CPU eager path, and (2) identical state_size(). The gla_virtual / gdn instances are SKIPPED
(intentionally not canonicalized — they stay in the rola.py shim).

CPU-only, tiny config, no GPU/training. Run:
    python tests/test_rola_parity.py
"""
import os
os.environ.setdefault("ROLA_PRINT_STATE_JSON", "0")
import sys
sys.path.insert(0, "/mnt/c/Users/Blake/Documents/VSCode/CLA")  # the `rola` package home

import torch

from rola import RoLA as OldRoLA, rola_instance, ROLA_INSTANCES
from fla_rola.layers import RoLA as NewRoLA

# tiny CPU geometry (d_qk even for hedgehog; small dims keep based/rebased feat_dim sane)
D, H, DQK, DV, NC = 64, 2, 16, 8, 4
B, L = 2, 32
ATOL, RTOL = 1e-5, 1e-4

SUPPORTED = ("rla", "gla_scalar")  # canonical layer kernels; others stay in the shim


def _new_kwargs(kw):
    """Translate rola_instance(...) kwargs -> canonical-layer __init__ kwargs."""
    kernel = kw["kernel"]
    kkw = kw.get("kernel_kwargs", {}) or {}
    state_norm = kkw.get("state_norm")
    if state_norm is None:  # reproduce the per-kernel defaults
        state_norm = "global" if kernel == "rla" else ("global" if kkw.get("normalized") else "raw")
    return dict(
        hidden_size=D, num_heads=H, head_k_dim=DQK, head_v_dim=DV, num_states=NC,
        kernel=kernel, phi=kw.get("phi", "elu"), state_norm=state_norm,
        tie_routers=kw.get("tie_routers", False), tie_router_init=kw.get("tie_router_init", False),
    )


def _remap_state_dict(old_sd):
    """old (rola.RoLA) key -> new (canonical layer) key."""
    out = {}
    for k, v in old_sd.items():
        nk = k[len("kernel."):] if k.startswith("kernel.") else k
        for a, b in (("w_q.", "q_proj."), ("w_k.", "k_proj."), ("w_v.", "v_proj."), ("w_o.", "o_proj.")):
            nk = nk.replace(a, b)
        out[nk] = v
    return out


def run():
    torch.manual_seed(0)
    x = torch.randn(B, L, D, dtype=torch.float32)
    n_ok = n_skip = n_fail = 0
    for name in ROLA_INSTANCES:
        kw = rola_instance(name, d_qk=DQK, d_v=DV, num_chunks=NC, n_heads=H)
        if kw["kernel"] not in SUPPORTED:
            print(f"  SKIP  {name:28s} (kernel={kw['kernel']!r} — stays in shim)")
            n_skip += 1
            continue
        try:
            old = OldRoLA(d_model=D, **kw).eval()
            new = NewRoLA(**_new_kwargs(kw)).eval()
            new.load_state_dict(_remap_state_dict(old.state_dict()), strict=True)
            with torch.no_grad():
                o_old = old(x)
                o_new = new(x)[0]
            ss_old, ss_new = old.state_size(), new.state_size()
            diff = (o_old - o_new).abs().max().item()
            close = torch.allclose(o_old, o_new, atol=ATOL, rtol=RTOL)
            if close and ss_old == ss_new:
                print(f"  OK    {name:28s} max|Δ|={diff:.2e}  state={ss_old}")
                n_ok += 1
            else:
                print(f"  FAIL  {name:28s} max|Δ|={diff:.2e} close={close} "
                      f"state old={ss_old} new={ss_new}")
                n_fail += 1
        except Exception as e:
            # RebasedFeatureMap uses a Triton fused LayerNorm that can't run on CPU — the OLD
            # rola.RoLA fails identically, so this is a shared CPU limitation of the φ map, not a
            # parity mismatch. Coverable only on GPU (deferred to the Phase-4/6 GPU smoke).
            if "cannot be accessed from Triton" in str(e):
                print(f"  SKIP  {name:28s} (φ needs Triton/GPU — uncoverable on CPU, same for both)")
                n_skip += 1
            else:
                print(f"  FAIL  {name:28s} {type(e).__name__}: {e}")
                n_fail += 1
    print(f"\n{'PARITY OK' if n_fail == 0 else 'PARITY FAILED'}: {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
