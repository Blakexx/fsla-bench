"""LM arm registry — the 5 architectures of the RoLA LM comparison, all on ONE shared backbone
so only the sequence MIXER differs (true iso-backbone mixer-swap).

Design (see TASKS #102):
  * Backbone identical across arms: hidden_size, num_hidden_layers, hidden_ratio=4 SwiGLU MLP,
    RMSNorm(eps=1e-6), untied embeddings, GPT-2 vocab. RoLAConfig mirrors fla GLAConfig's backbone
    fields byte-for-byte, and the fla baselines (GDN, Transformer) share the same fla backbone, so
    swapping arms changes ONLY the token mixer.
  * The bounded-state arms are matched on RECURRENT STATE. Each active LM spec sets bounded-state
    floats/layer to half of the training-context attention KV cache:
        target = block_size * hidden_size = 0.5 * (2 * block_size * hidden_size).
    For RoLA this is the X=1 rule: states_per_head * d_qk = block_size, with num_heads * d_v =
    hidden_size. GDN/GLA are sized to the same target in the YAML geometry.
  * Softmax attention is the unbounded-state CEILING (no recurrent state to match) — the reference the
    bounded models try to approach.

Default recurrent state (headline rung) = 1,048,576 floats/layer at block_size=2048, hidden=512:
    RoLA  n_h=8, nc=128, d_qk=16, d_v=64  ->  8*128*16*64 = 1048576
    GDN   n_h=8, head_dim=256, expand_v=2 ->  8*256*512 = 1048576
    GLA   n_h=4, expand_k=2, expand_v=8   ->  (512*2)*(512*8)/4 = 1048576

  RoLA's d_v=64 (not 16) is deliberate: the mixer readout width is n_h*d_v, and w_o lifts it to
  d_model. At d_v=16 that readout is only 128-wide (d_model/4) -> the mixer could write at most a
  rank-128 update to the 512-d residual, an output BOTTLENECK vs attn (512) and GDN (2048). We set
  n_h*d_v = 512 = d_model so RoLA is not handicapped on output capacity; the state budget is then
  spent on nc=64 routed states (output_width * nc*d_qk = state is the fixed-state tradeoff).

Each arm returns a fresh `AutoModelForCausalLM.from_config(cfg)`. `arm_config(...)` builds the
config only (cheap, for param/state accounting); `build_arm(...)` builds the model.
"""
from __future__ import annotations

# RoLA is now a first-class FLA model: importing the package registers its HF AutoClasses.
import os as _os
import fla_rola.models.rola  # noqa: F401  (registers RoLA AutoClasses)
from fla_rola.models.rola import RoLAConfig
from transformers import AutoModelForCausalLM

# --- env-overridable knobs (defaults = Option-Y headline @ d_model=512). The d_model=128 small-
# backbone diagnostic sets LM_HIDDEN/LM_ROLA_*/LM_ATTN_HEADS to run RoLA's narrow-output regime
# (nc=256, d_qk=d_v=16 -> output 128 = d_model=128, no waste) against a matched small attn. -------
_HID     = int(_os.environ.get("LM_HIDDEN", "512"))        # backbone d_model
_NC      = int(_os.environ.get("LM_ROLA_NC", "128"))       # RoLA routed states (Y: 128)
_DQK     = int(_os.environ.get("LM_ROLA_DQK", "16"))
_DV      = int(_os.environ.get("LM_ROLA_DV", "64"))        # Y: 64 -> n_h*d_v = 512 = d_model (no bottleneck)
_NH      = int(_os.environ.get("LM_ROLA_NH", "8"))         # shared head count
_GDN_NH  = int(_os.environ.get("LM_GDN_NH", "8"))          # GDN heads (Y@512: 8 -> canonical-8h, state matches RoLA)
_GDN_HEAD_DIM = int(_os.environ.get("LM_GDN_HEAD_DIM", "256"))    # GDN key head_dim (Y@512: 256; small backbone -> 64)
_GDN_EXPAND_V = float(_os.environ.get("LM_GDN_EXPAND_V", "2.0"))  # GDN value expansion (canonical 2)
_GLA_NH  = int(_os.environ.get("LM_GLA_NH", "4"))          # GLA heads
_GLA_EK  = float(_os.environ.get("LM_GLA_EK", "2.0"))      # default state matches half-KV at L=2048, hidden=512
_GLA_EV  = float(_os.environ.get("LM_GLA_EV", "8.0"))
# Backward-compatible aliases for older specs; active LM YAMLs size gdn/gla directly to half-KV.
_GDNISO_NH = int(_os.environ.get("LM_GDNISO_NH", "8"))
_GDNISO_HEAD_DIM = int(_os.environ.get("LM_GDNISO_HEAD_DIM", "256"))
_GDNISO_EXPAND_V = float(_os.environ.get("LM_GDNISO_EXPAND_V", "2.0"))
_GLAISO_NH = int(_os.environ.get("LM_GLAISO_NH", "4"))
_GLAISO_EK = float(_os.environ.get("LM_GLAISO_EK", "2.0"))
_GLAISO_EV = float(_os.environ.get("LM_GLAISO_EV", "8.0"))
_ATTN_NH = int(_os.environ.get("LM_ATTN_HEADS", "8"))      # softmax-attn heads (head_dim = _HID/this)
_MAXPOS  = int(_os.environ.get("LM_MAX_POS", "2048"))      # max_position_embeddings; set >= train ctx (4k headline)
_LAYERS  = int(_os.environ.get("LM_LAYERS", "12"))         # num_hidden_layers; shrink for micro (e.g. TinyStories)
_QKNORM  = str(_os.environ.get("LM_QK_NORM", "0")).lower() in ("1", "true", "yes")  # qk-rmsnorm (RoLA + attn)
_ZLOSS   = float(_os.environ.get("LM_ROUTER_ZLOSS_COEF", "0"))      # ST-MoE router z-loss (RoLA; 0 = off)

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
# --- matched recurrent-state geometry (default: half of attention KV at L=2048) ----------------
# RoLA  n_h=8 * nc=128 * d_qk=16 * d_v=64 = 1048576 floats/layer; n_h*d_v = 512 = d_model (no output
#   bottleneck -- see module docstring). Active YAML specs override these per context length so
#   every bounded arm lands on block_size * hidden_size.
# RoLA additionally stores a per-state DENOMINATOR accumulator (n_h*nc*d_qk = 16384, ~1.5%) for its
# normalized readout — RoLA-specific bookkeeping the delta rule doesn't need. It is NOT part of the
# matched content axis; reported separately (rola_den_overhead) so the matched quantity is like-for-like.
ROLA_GEOM = dict(states_per_head=_NC, d_qk=_DQK, d_v=_DV, num_heads=_NH, qk_norm=_QKNORM,
                 router_zloss_coef=_ZLOSS)
GDN_GEOM = dict(num_heads=_GDN_NH, head_dim=_GDN_HEAD_DIM, expand_v=_GDN_EXPAND_V)
GLA_GEOM = dict(num_heads=_GLA_NH, expand_k=_GLA_EK, expand_v=_GLA_EV, gate_logit_normalizer=16)
# Alias geometry for older configs that still name gdn-isostate/gla-isostate.
GDNISO_GEOM = dict(num_heads=_GDNISO_NH, head_dim=_GDNISO_HEAD_DIM, expand_v=_GDNISO_EXPAND_V)
GLAISO_GEOM = dict(num_heads=_GLAISO_NH, expand_k=_GLAISO_EK, expand_v=_GLAISO_EV, gate_logit_normalizer=16)
ATTN_HEADS = _ATTN_NH                                           # head_dim _HID/_ATTN_NH (default 64 @512)
MATCHED_STATE = ROLA_GEOM["num_heads"] * ROLA_GEOM["states_per_head"] * ROLA_GEOM["d_qk"] * ROLA_GEOM["d_v"]
ROLA_DEN_OVERHEAD = ROLA_GEOM["num_heads"] * ROLA_GEOM["states_per_head"] * ROLA_GEOM["d_qk"]  # 8192, asym/kappa only

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
ROLA_ARMS = ("rola-rla-kappa-sym", "rola-rla-kappa-asym",
             "rola-gla-kappa-asym", "rola-gla-kappa-sym")   # both kernels' asym/sym pair (asym>sym test)
ARMS = ROLA_ARMS + ("gdn", "gdn-isostate", "gla", "gla-isostate", "attn")


# Optional RoLA STRUCTURE override via arm-name suffix "<base>__nhX_dvY_ncZ_dqkW" (structure-ablation
# spec): overrides num_heads/d_v/states_per_head/d_qk on top of ROLA_GEOM; base must be in ROLA_ARMS.
# e.g. "rola-rla-kappa-asym__nh4_dv64" -> nh4/dv64 (head vs d_v split sweep at fixed nh*d_v=out_dim).
_STRUCT_KEYS = {"nh": "num_heads", "dv": "d_v", "dqk": "d_qk", "nc": "states_per_head"}


def _parse_rola(arm: str):
    if "__" not in arm:
        return arm, {}
    base, ov = arm.split("__", 1)
    over = {}
    for tok in ov.split("_"):
        for k, full in _STRUCT_KEYS.items():
            if tok.startswith(k) and tok[len(k):].isdigit():
                over[full] = int(tok[len(k):]); break
    return base, over


def arm_config(arm: str, vocab_size: int, bos_token_id: int = 50256, eos_token_id: int = 50256):
    """Build the HF config for `arm` on the shared backbone. Raises on an unknown arm (fail loud)."""
    common = dict(vocab_size=vocab_size, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **BACKBONE)
    _base, _struct = _parse_rola(arm)
    if _base in ROLA_ARMS:
        return RoLAConfig(rola_instance=_base, **{**ROLA_GEOM, **_struct}, **common)
    if arm == "gdn":
        from fla.models.gated_deltanet import GatedDeltaNetConfig
        return GatedDeltaNetConfig(use_short_conv=USE_SHORT_CONV, use_gate=True, **GDN_GEOM, **common)
    if arm == "gla":
        from fla.models.gla import GLAConfig
        return GLAConfig(use_short_conv=USE_SHORT_CONV, **GLA_GEOM, **common)
    if arm == "gdn-isostate":
        from fla.models.gated_deltanet import GatedDeltaNetConfig
        return GatedDeltaNetConfig(use_short_conv=USE_SHORT_CONV, use_gate=True, **GDNISO_GEOM, **common)
    if arm == "gla-isostate":
        from fla.models.gla import GLAConfig
        return GLAConfig(use_short_conv=USE_SHORT_CONV, **GLAISO_GEOM, **common)
    if arm == "attn":
        from fla.models.transformer import TransformerConfig
        # Softmax-attention ceiling: full causal attention (window_size=None), RoPE positions.
        # qk_norm matched to RoLA's (fair stabilizer parity — fla.layers.attn supports it natively).
        return TransformerConfig(num_heads=ATTN_HEADS, window_size=None, qk_norm=_QKNORM, **common)
    raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")


def build_arm(arm: str, vocab_size: int, **kw):
    """Instantiate the model for `arm` (fresh weights) on the shared backbone."""
    return AutoModelForCausalLM.from_config(arm_config(arm, vocab_size, **kw))


def recurrent_state_floats(arm: str) -> int | None:
    """Nominal recurrent-state floats/layer for the bounded arms (matched target); None for attn
    (unbounded KV cache). Realized state is read back from the built layer in the result row."""
    _base, _struct = _parse_rola(arm)
    if _base in ROLA_ARMS:
        g = {**ROLA_GEOM, **_struct}
        return g["num_heads"] * g["states_per_head"] * g["d_qk"] * g["d_v"]
    if arm == "gdn":
        return GDN_GEOM["num_heads"] * GDN_GEOM["head_dim"] * int(GDN_GEOM["head_dim"] * GDN_GEOM["expand_v"])
    if arm == "gla":  # nh * key_head * value_head = key_dim*value_dim/nh (canonical, NOT matched to RoLA)
        return int(_HID * GLA_GEOM["expand_k"]) * int(_HID * GLA_GEOM["expand_v"]) // GLA_GEOM["num_heads"]
    if arm == "gdn-isostate":
        return GDNISO_GEOM["num_heads"] * GDNISO_GEOM["head_dim"] * int(GDNISO_GEOM["head_dim"] * GDNISO_GEOM["expand_v"])
    if arm == "gla-isostate":
        return int(_HID * GLAISO_GEOM["expand_k"]) * int(_HID * GLAISO_GEOM["expand_v"]) // GLAISO_GEOM["num_heads"]
    return None  # attn: unbounded
