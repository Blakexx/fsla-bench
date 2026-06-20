"""Phase-4 parity gate: zoology.mixers.rola.RoLAMixer (now over fla_rola.layers.RoLA, with the
legacy-kwarg translation + 3-tuple unwrap) vs rola.RoLA built with equivalent kwargs. Confirms the
kwarg translation (d_qk->head_k_dim, kernel_kwargs->state_norm, absorbed writer/reader/route_on)
and forward-unwrap are correct: weight-copied, forward bit-identical on CPU, state_size matches.

Run:  python tests/test_rola_zoology_mixer.py
"""
import os
os.environ.setdefault("ROLA_PRINT_STATE_JSON", "0")
import sys
sys.path.insert(0, "/mnt/c/Users/Blake/Documents/VSCode/CLA")

import torch
from rola import RoLA as RefRoLA
from zoology.mixers.rola import RoLAMixer

D, H, DQK, DV, NC = 64, 2, 16, 16, 4
B, L = 2, 32

# (label, MQAR-style mixer kwargs, equivalent rola.RoLA kwargs)
CELLS = [
    ("rla-global (plain rola_rla)",
     dict(d_qk=DQK, d_v=DV, num_chunks=NC, n_heads=H, writer="softmax_linear",
          reader="softmax_linear", tie_routers=False, use_short_conv=False, route_on="kq"),
     dict(d_qk=DQK, d_v=DV, num_chunks=NC, n_heads=H, tie_routers=False, kernel="rla")),
    ("rla-kappa-asym",
     dict(d_qk=DQK, d_v=DV, num_chunks=NC, n_heads=H, tie_routers=False, route_on="kq",
          kernel="rla", kernel_kwargs={"state_norm": "kappa"}),
     dict(d_qk=DQK, d_v=DV, num_chunks=NC, n_heads=H, tie_routers=False, kernel="rla",
          kernel_kwargs={"state_norm": "kappa"})),
    ("gla-kappa-sym",
     dict(d_qk=DQK, d_v=DV, num_chunks=NC, n_heads=H, tie_routers=True, route_on="kq",
          kernel="gla_scalar", kernel_kwargs={"state_norm": "kappa"}),
     dict(d_qk=DQK, d_v=DV, num_chunks=NC, n_heads=H, tie_routers=True, kernel="gla_scalar",
          kernel_kwargs={"state_norm": "kappa"})),
]


def remap(k):
    k = k[len("kernel."):] if k.startswith("kernel.") else k
    for a, b in (("w_q.", "q_proj."), ("w_k.", "k_proj."), ("w_v.", "v_proj."), ("w_o.", "o_proj.")):
        k = k.replace(a, b)
    return k


def run():
    torch.manual_seed(0)
    x = torch.randn(B, L, D)
    n_fail = 0
    for label, mix_kw, ref_kw in CELLS:
        try:
            new = RoLAMixer(d_model=D, **mix_kw).eval()
            ref = RefRoLA(d_model=D, **ref_kw).eval()
            new.load_state_dict({remap(k): v for k, v in ref.state_dict().items()}, strict=True)
            with torch.no_grad():
                o_ref = ref(x)              # tensor
                o_new = new(x)              # unwrapped tensor
            diff = (o_ref - o_new).abs().max().item()
            ok = torch.allclose(o_ref, o_new, atol=1e-6) and new.state_size() == ref.state_size()
            print(f"  {'OK  ' if ok else 'FAIL'} {label:30s} max|Δ|={diff:.2e} state={new.state_size()}")
            n_fail += (not ok)
        except Exception as e:
            print(f"  FAIL {label:30s} {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\n{'ZOOLOGY MIXER PARITY OK' if n_fail == 0 else 'FAILED'}: {len(CELLS)-n_fail}/{len(CELLS)}")
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
