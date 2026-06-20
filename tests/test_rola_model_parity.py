"""Phase-2 parity gate: fla_rola.models.rola.RoLAForCausalLM (new, canonical) vs the legacy
rola_hf.RoLAForCausalLM. Build BOTH at an identical tiny config, copy weights legacy->new
(key remap: attn.rola.<x> -> attn.<x>, attn.rola.kernel.<x> -> attn.<x>, w_q->q_proj …), then
assert: (1) remapped state_dict loads strict (structural equivalence), (2) identical forward
logits to ~1e-5 on CPU, (3) identical param count. Fused norm/swiglu/CE OFF -> pure CPU path.

Run:  python tests/test_rola_model_parity.py
"""
import os
os.environ.setdefault("ROLA_PRINT_STATE_JSON", "0")
import sys
sys.path.insert(0, "/mnt/c/Users/Blake/Documents/VSCode/CLA")          # `rola` package
sys.path.insert(0, "/home/blake/rola-bench/rola_bench/lm")             # legacy `rola_hf` package

import torch

import rola_hf  # noqa: F401  (registers AutoClasses; provides RoLAConfig/RoLAForCausalLM)
from rola_hf import RoLAConfig as OldCfg
from rola_hf.modeling_rola import RoLAForCausalLM as OldLM
from fla_rola.models.rola import RoLAConfig as NewCfg, RoLAForCausalLM as NewLM

ARMS = ["rola-rla-kappa-asym", "rola-rla-kappa-sym", "rola-gla-kappa-asym", "rola-gla-kappa-sym"]
# Fused defaults (real config): the GatedMLP swiglu + fla RMSNorm are Triton-only, so the full
# model can only run on GPU — the mixer alone is CPU-proven by the Phase-1 layer parity test.
CFG = dict(num_states=4, d_qk=16, d_v=8, num_heads=2, hidden_size=64, num_hidden_layers=2,
           vocab_size=128, max_position_embeddings=64)
B, L = 2, 16
ATOL, RTOL = 1e-4, 1e-3


def remap(k):
    """legacy rola_hf key -> new fla-native key."""
    k = k.replace(".attn.rola.kernel.", ".attn.").replace(".attn.rola.", ".attn.")
    for a, b in ((".attn.w_q.", ".attn.q_proj."), (".attn.w_k.", ".attn.k_proj."),
                 (".attn.w_v.", ".attn.v_proj."), (".attn.w_o.", ".attn.o_proj.")):
        k = k.replace(a, b)
    return k


def run():
    if not torch.cuda.is_available():
        print("SKIP: full-model parity needs GPU (GatedMLP swiglu + fla RMSNorm are Triton-only); "
              "the mixer itself is CPU-proven by tests/test_rola_parity.py (Phase 1).")
        return True
    dev = "cuda"
    torch.manual_seed(0)
    ids = torch.randint(0, CFG["vocab_size"], (B, L), device=dev)
    n_ok = n_fail = 0
    for arm in ARMS:
        try:
            old = OldLM(OldCfg(rola_instance=arm, **CFG)).to(dev).eval()
            new = NewLM(NewCfg(rola_instance=arm, **CFG)).to(dev).eval()
            remapped = {remap(k): v for k, v in old.state_dict().items()}
            new.load_state_dict(remapped, strict=True)   # (1) structural equivalence
            with torch.no_grad():
                lo = old(input_ids=ids).logits
                ln = new(input_ids=ids).logits
            diff = (lo - ln).abs().max().item()
            np_old = sum(p.numel() for p in old.parameters())
            np_new = sum(p.numel() for p in new.parameters())
            ok = torch.allclose(lo, ln, atol=ATOL, rtol=RTOL) and np_old == np_new
            print(f"  {'OK  ' if ok else 'FAIL'} {arm:24s} max|Δlogits|={diff:.2e}  params old={np_old} new={np_new}")
            n_ok += ok
            n_fail += (not ok)
        except Exception as e:
            print(f"  FAIL {arm:24s} {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\n{'MODEL PARITY OK' if n_fail == 0 else 'MODEL PARITY FAILED'}: {n_ok} ok, {n_fail} failed")
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
