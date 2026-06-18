"""LM arm registry — the 5 architectures of the RoLA LM comparison, all on ONE shared backbone
so only the sequence MIXER differs (true iso-backbone mixer-swap).

Design (see TASKS #102):
  * Backbone identical across arms: hidden_size, num_hidden_layers, hidden_ratio=4 SwiGLU MLP,
    RMSNorm(eps=1e-6), untied embeddings, GPT-2 vocab. RoLAConfig mirrors fla GLAConfig's backbone
    fields byte-for-byte, and the fla baselines (GDN, Transformer) share the same fla backbone, so
    swapping arms changes ONLY the token mixer.
  * The bounded-state arms (3 RoLA variants + GDN) are matched on RECURRENT STATE (the MQAR
    matched-state principle carried into LM). Softmax attention is the unbounded-state CEILING
    (no recurrent state to match) — the reference the bounded models try to approach.

Matched recurrent state (headline rung, Option Y) = 1,048,576 floats/layer, ALL ARMS AT 8 HEADS:
    RoLA  n_h=8, nc=128, d_qk=16, d_v=64  ->  8*128*16*64 = 1048576   (output width n_h*d_v = 512 = d_model)
    GDN   8 heads, head_dim=256, expand_v=2 (->head_v=512)  ->  8*256*512 = 1048576   (EXACT; canonical)
  All three arms share num_heads=8 (head is the same primitive: an independent mixing channel), GDN
  stays at its CANONICAL head shape (head_dim<=256 kernel cap never binds), and attn=8 (head_dim 64)
  is the standard softmax config. State = 1/2 of attn's KV cache at L=2048 (2*L*d_model = 2097152),
  a compression advantage that GROWS linearly with context (constant state vs attn's linear KV).
  Attention has no recurrent state (unbounded-KV ceiling). [Option X = 524288 / GDN-4h, run later.]

  RoLA's d_v=64 (not 16) is deliberate: the mixer readout width is n_h*d_v, and w_o lifts it to
  d_model. At d_v=16 that readout is only 128-wide (d_model/4) -> the mixer could write at most a
  rank-128 update to the 512-d residual, an output BOTTLENECK vs attn (512) and GDN (2048). We set
  n_h*d_v = 512 = d_model so RoLA is not handicapped on output capacity; the state budget is then
  spent on nc=64 routed states (output_width * nc*d_qk = state is the fixed-state tradeoff).

Each arm returns a fresh `AutoModelForCausalLM.from_config(cfg)`. `arm_config(...)` builds the
config only (cheap, for param/state accounting); `build_arm(...)` builds the model.
"""
from __future__ import annotations

# rola_hf is hard-wired to import as a TOP-LEVEL module (its __init__ does `from rola_hf.X import`),
# so put this file's dir (lm/) on sys.path — then `import rola_hf` resolves whether arms.py is run
# from lm/ (train_lm) or imported as rola_bench.lm.arms (job.py). registers RoLA's HF AutoClasses.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import rola_hf  # noqa: F401,E402
from rola_hf import RoLAConfig  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

# --- env-overridable knobs (defaults = Option-Y headline @ d_model=512). The d_model=128 small-
# backbone diagnostic sets LM_HIDDEN/LM_ROLA_*/LM_ATTN_HEADS to run RoLA's narrow-output regime
# (nc=256, d_qk=d_v=16 -> output 128 = d_model=128, no waste) against a matched small attn. -------
_HID     = int(_os.environ.get("LM_HIDDEN", "512"))        # backbone d_model
_NC      = int(_os.environ.get("LM_ROLA_NC", "128"))       # RoLA routed states (Y: 128)
_DQK     = int(_os.environ.get("LM_ROLA_DQK", "16"))
_DV      = int(_os.environ.get("LM_ROLA_DV", "64"))        # Y: 64 -> n_h*d_v = 512 = d_model (no bottleneck)
_NH      = int(_os.environ.get("LM_ROLA_NH", "8"))         # shared head count
_GDN_NH  = int(_os.environ.get("LM_GDN_NH", "8"))          # GDN heads (Y: 8 -> canonical-8h, state matches RoLA)
_ATTN_NH = int(_os.environ.get("LM_ATTN_HEADS", "8"))      # softmax-attn heads (head_dim = _HID/this)
_MAXPOS  = int(_os.environ.get("LM_MAX_POS", "2048"))      # max_position_embeddings; set >= train ctx (4k headline)
_LAYERS  = int(_os.environ.get("LM_LAYERS", "12"))         # num_hidden_layers; shrink for micro (TinyStories/ListOps)

# --- shared backbone (the part that must be IDENTICAL across all arms) -------------------------
BACKBONE = dict(
    hidden_size=_HID,         # small/cheap compute-optimal scale (~108M @512). hidden % ATTN_HEADS == 0.
    num_hidden_layers=_LAYERS,
    hidden_ratio=4,           # SwiGLU MLP expansion (fla convention)
    norm_eps=1e-6,
    max_position_embeddings=_MAXPOS,   # >= train ctx; 4k headline. NIAH may eval beyond (extrapolation).
    tie_word_embeddings=False,
    fuse_cross_entropy=True,
)
# --- matched recurrent-state geometry (Option Y: all 8 heads, state = 1,048,576) ---------------
# RoLA  n_h=8 * nc=128 * d_qk=16 * d_v=64 = 1048576 floats/layer; n_h*d_v = 512 = d_model (no output
#   bottleneck — see module docstring). GDN matches EXACTLY at its CANONICAL 8-head shape
#   (head_dim=256, expand_v=2 -> head_v=512): 8*256*512 = 1048576. expand_v=2 is the canonical value
#   (flame gated_deltanet_*.json) — NOT a reverse-engineered fraction; head_dim<=256 cap never binds.
# RoLA additionally stores a per-state DENOMINATOR accumulator (n_h*nc*d_qk = 16384, ~1.5%) for its
# normalized readout — RoLA-specific bookkeeping the delta rule doesn't need. It is NOT part of the
# matched content axis; reported separately (rola_den_overhead) so the matched quantity is like-for-like.
ROLA_GEOM = dict(num_states=_NC, d_qk=_DQK, d_v=_DV, num_heads=_NH)
GDN_GEOM = dict(num_heads=_GDN_NH, head_dim=256, expand_v=2.0)   # 8*256*512 = 1048576 (canonical expand_v)
ATTN_HEADS = _ATTN_NH                                           # head_dim _HID/_ATTN_NH (default 64 @512)
MATCHED_STATE = ROLA_GEOM["num_heads"] * ROLA_GEOM["num_states"] * ROLA_GEOM["d_qk"] * ROLA_GEOM["d_v"]
ROLA_DEN_OVERHEAD = ROLA_GEOM["num_heads"] * ROLA_GEOM["num_states"] * ROLA_GEOM["d_qk"]  # 8192, asym/kappa only

# Uniform component policy across the sub-quadratic arms: short conv OFF for ALL (isolate the mixer;
# attention has no conv, so conv-off keeps the only difference the mixer itself). This runs GDN
# slightly below its canonical (conv-on) recipe — a conservative, RoLA-UNfavorable choice, stated
# openly. Flip USE_SHORT_CONV to True to give RoLA+GDN their canonical conv (attention stays off).
USE_SHORT_CONV = False

# arm name -> the three RoLA variants share geometry + backbone, differ only in rola_instance.
# All three use KAPPA normalization (state_norm='kappa') for consistency with the paper-canonical
# RLA cell: kappa-sym/kappa-asym (RLA) + gla-kappa-asym (scalar GLA, decayed per-state mass — same
# kappa construction). NOT the 'global'-norm scalar-norm variant. (MQAR routed-GLA is being re-run
# on the kappa variant too, #112.)
ROLA_ARMS = ("rola-rla-kappa-sym", "rola-rla-kappa-asym", "rola-gla-kappa-asym")
ARMS = ROLA_ARMS + ("gdn", "attn")


def arm_config(arm: str, vocab_size: int, bos_token_id: int = 50256, eos_token_id: int = 50256):
    """Build the HF config for `arm` on the shared backbone. Raises on an unknown arm (fail loud)."""
    common = dict(vocab_size=vocab_size, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **BACKBONE)
    if arm in ROLA_ARMS:
        return RoLAConfig(rola_instance=arm, **ROLA_GEOM, **common)
    if arm == "gdn":
        from fla.models.gated_deltanet import GatedDeltaNetConfig
        return GatedDeltaNetConfig(use_short_conv=USE_SHORT_CONV, use_gate=True, **GDN_GEOM, **common)
    if arm == "attn":
        from fla.models.transformer import TransformerConfig
        # Softmax-attention ceiling: full causal attention (window_size=None), RoPE positions.
        return TransformerConfig(num_heads=ATTN_HEADS, window_size=None, **common)
    raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")


def build_arm(arm: str, vocab_size: int, **kw):
    """Instantiate the model for `arm` (fresh weights) on the shared backbone."""
    return AutoModelForCausalLM.from_config(arm_config(arm, vocab_size, **kw))


def recurrent_state_floats(arm: str) -> int | None:
    """Nominal recurrent-state floats/layer for the bounded arms (matched target); None for attn
    (unbounded KV cache). Realized state is read back from the built layer in the result row."""
    if arm in ROLA_ARMS:
        return MATCHED_STATE
    if arm == "gdn":
        return GDN_GEOM["num_heads"] * GDN_GEOM["head_dim"] * int(GDN_GEOM["head_dim"] * GDN_GEOM["expand_v"])
    return None  # attn: unbounded
