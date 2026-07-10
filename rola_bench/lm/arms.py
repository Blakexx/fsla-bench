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
_ROUTER_BIAS = str(_os.environ.get("LM_ROUTER_BIAS", "0")).lower() in ("1", "true", "yes")

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
# `RoLAConfig` (#config-refactor 2026-07-06) takes `levels_per_head`/`states_per_level` (D, b; nc=b**D)
# instead of `states_per_head` — the geometry here is flat (D=1, b=nc), the strict equivalent of the
# old dense per-head router (rola_bench.rola_compat.topology_from_legacy). `states_per_head` is kept
# as a KEY in ROLA_GEOM (nc, the locked pinned number) for every OTHER call site in this module
# (MATCHED_STATE, ROLA_DEN_OVERHEAD, recurrent_state_floats, the `nc` structure-override key) — only
# `arm_config`'s RoLAConfig(**ROLA_GEOM) construction call translates it to levels_per_head/states_per_level.
ROLA_GEOM = dict(states_per_head=_NC, d_qk=_DQK, d_v=_DV, num_heads=_NH, qk_norm=_QKNORM,
                 router_bias=_ROUTER_BIAS)
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
# This uniform conv-off policy applies ONLY to the LEGACY comparison cells (ROLA_ARMS + the fla
# baselines) below. The published-flagship arms (ROLA_BASE_ARMS, just below) are NOT part of that
# iso-conv comparison — they carry their OWN canonical conv setting (ON) per the published
# architecture (`rola_bench.named_configs`), independent of USE_SHORT_CONV.
USE_SHORT_CONV = False

# --- published-flagship arms (F3, pre-E1 gap fix) -----------------------------------------------
# RoLA-base / RoLA-base-rla / RoLA-base-global (`rola_bench.named_configs`): router_bias=True,
# two-level topology (levels_per_head=2; branching per the GEOMETRY block below), UNTIED, per-level
# heterogeneous write (sparse@coarse level0, dense@fine level1), dense read at both levels, conv ON. These
# are the PUBLISHED cells, reachable through the HF `AutoModelForCausalLM` path now that
# `RoLAConfig` carries `routing_mode`/`level_routing_configs`
# (`fla_rola/models/rola/configuration_rola.py`) and `rola_instance()`/`RoLABlock` thread them into
# the layer kwargs (`fla_rola/models/rola/instances.py`, `modeling_rola.py`). The C8 per-level
# heterogeneous activation shim-lift (engine-side, `_resolve_uniform_activation` in
# `fla_rola/layers/rola.py`) is what makes the sparse/dense write mix actually EXECUTE (not just
# construct) — verified 2026-07-08 against the rola-cute working tree.
#
# qk_norm (user decision, 2026-07-09): the canonical/published model runs qk_norm=True (RMSNorm on
# q/k pre-feature-map). Pinned EXPLICITLY per flagship arm (`ROLA_BASE_QKNORM`, below the instance
# table), not via the `LM_QK_NORM` env default -- that env stays for the legacy `ROLA_ARMS` cells.
# `rola-base-noqknorm` is the one deliberate exception: rola-base with qk_norm=False, a one-seed
# ablation probing whether historical LM instability traced to the projections (needing norm) or to
# old-kernel numerics (now fixed — the per_state/rt bf16 defect, 3c6567f2).
#
# GEOMETRY (user decision, 2026-07-08, REVISED same day): the flagship LM arms TARGET the
# MIXED-RADIX topology states_per_level=[8, 16] (coarse-first: 8 coarse × 16 fine, nc=128) —
# state = 8·128·16·64 = 1,048,576 floats/layer = EXACT match with the GDN/GLA X=1 iso arms (all at
# 50% of attention's reference KV cache at block_size=2048). Mixed-radix `states_per_level` is the
# F4 engine feature (in flight on the rola-cute tree as of this writing) — so the topology is
# CONSTRUCT-GATED (`_rola_base_topology()`, the `requires_c8` pattern): probe the layer with the
# real [8,16] config once; if the engine tree doesn't accept the list yet, fall back to square
# nc=64 (levels_per_head=2, states_per_level=8) with `requires_f4=True`.  # pending F4 mixed-radix
#
# Both rationales, recorded per the user decision:
#   * PREFERRED [8,16] (matched-1M): iso-state with GDN/GLA — the clean like-for-like comparison,
#     every bounded arm at exactly half of attention's KV.
#   * FALLBACK nc=64 square (undersized-0.5M): state = 8·64·16·64 = 524,288 floats/layer = 50% of
#     the baselines — RoLA claims the win under a COMPRESSION HANDICAP (half the baselines' state,
#     a quarter of attention's KV), a stronger claim than matched, acceptable until F4 lands.
# GDN/GLA stay at their canonical 1M-float iso geometries in BOTH cases (never rescaled). The PERF
# suite keeps nc=256 (paper_v1 unchanged): the decode/systems numbers use the bigger state — a
# different claim under a different config, both documented.
_ROLA_BASE_SPL = [int(t) for t in _os.environ.get("LM_ROLA_BASE_SPL", "8,16").split(",")]
_ROLA_BASE_FALLBACK_B = int(_os.environ.get("LM_ROLA_BASE_FALLBACK_B", "8"))   # square fallback: D=2, b=8, nc=64
_ROLA_BASE_DQK = int(_os.environ.get("LM_ROLA_BASE_DQK", "16"))
_ROLA_BASE_DV  = int(_os.environ.get("LM_ROLA_BASE_DV", "64"))
_ROLA_BASE_NH  = int(_os.environ.get("LM_ROLA_BASE_NH", "8"))
_ROLA_BASE_D   = len(_ROLA_BASE_SPL)

_rola_base_f4_gate: bool | None = None    # None = not yet probed


def _rola_base_topology() -> tuple[int, "int | list[int]", int, bool]:
    """Return (levels_per_head, states_per_level, nc, requires_f4) for the flagship arms —
    construct-gated on the F4 mixed-radix engine feature (probe cached per process). The probe
    constructs one tiny REAL `fla_rola.layers.RoLA` with the actual [8,16]-shaped topology + the
    published write mix; a rejection of the list form (the current pre-F4 tree raises at
    construction) selects the undersized square fallback.  # pending F4 mixed-radix"""
    global _rola_base_f4_gate
    import math
    if _rola_base_f4_gate is None:
        try:
            from fla_rola.layers.rola import RoLA
            RoLA(hidden_size=32, num_heads=2, head_k_dim=8, head_v_dim=8,
                 levels_per_head=_ROLA_BASE_D, states_per_level=list(_ROLA_BASE_SPL),
                 kernel='gla_scalar', state_norm='kappa', router_bias=True,
                 level_routing_configs=_convert_write_mix_for_probe())
            _rola_base_f4_gate = False
        except (TypeError, ValueError, NotImplementedError):
            _rola_base_f4_gate = True   # pending F4 mixed-radix — fall back to square nc=64
    if _rola_base_f4_gate:
        return 2, _ROLA_BASE_FALLBACK_B, _ROLA_BASE_FALLBACK_B ** 2, True
    return _ROLA_BASE_D, list(_ROLA_BASE_SPL), math.prod(_ROLA_BASE_SPL), False


def _convert_write_mix_for_probe():
    from fla_rola.layers.rola import LevelRoutingConfig
    return [LevelRoutingConfig(tied=c['tied'], read=c['read'], write=c['write']) for c in ROLA_BASE_WRITE_MIX]

# The published write mix: sparse (entmax, fixed alpha=1.5) at the coarse level, dense (softmax) at
# the fine level; read stays dense at both levels (dead-end-free by construction, ROLA_SEMANTICS.md
# §5). Serializable dict form -- `RoLAConfig.level_routing_configs` converts these to
# `LevelRoutingConfig` at the `instances.py` boundary.
ROLA_BASE_WRITE_MIX = [
    {'tied': False, 'read': 'dense', 'write': 'sparse'},   # level 0 (coarse)
    {'tied': False, 'read': 'dense', 'write': 'dense'},    # level 1 (fine)
]

# rola-base-sym (user decision, 2026-07-08): the SYMMETRIC-COARSE ablation — the needle-threading
# ablation pair with rola-base:
#   * rola-base     = ASYMMETRIC coarse routing + dead-end-free dense read (write-only block skip).
#   * rola-base-sym = SHARED coarse support: TIED sparse/sparse at the coarse level (read region ≡
#     write region — still dead-end-safe: identical logits + identical activation ⇒ identical
#     support, the trainability-safe double-sparse form; fine level stays dense/dense untied) =
#     the MAX-SKIP cell: shared-support state blocks are never loaded on EITHER side.
# Quality delta comes from LM/MQAR, perf delta from the skip planes. Note the mixed per-level tied
# config means the layer's read_W holds ONLY the fine (untied) level — construction verified in the
# suite smoke. Everything else (kernel=gla_scalar, kappa, bias, conv-on, geometry incl. the
# [8,16]-with-fallback probe) is identical to rola-base; recurrent_state_floats unchanged (same nc).
ROLA_BASE_SYM_MIX = [
    {'tied': True,  'read': 'sparse', 'write': 'sparse'},   # level 0 (coarse): shared support
    {'tied': False, 'read': 'dense', 'write': 'dense'},     # level 1 (fine)
]

# rola-base-sym-global (user decision, 2026-07-09): rola-base-sym with state_norm='global' (kappa≡0).
# THE decisive cell for the destroy-kappa hypothesis (user 2026-07-09): kappa/per_state were added to
# close the sym<->asym gap by repairing read-mass imbalance; tied-sparse-coarse prevents the
# imbalance structurally, so sym+global should match base-kappa if the hypothesis holds -- then the
# RESCALED norms (and all their precision/perf machinery) can be deleted.

# rola-base-union-global (candidate C, 2026-07-10): the UNION-SPLIT routing cell — coarse level
# UNTIED sparse/sparse. NO new vocabulary (settled 2026-07-10, workflows/union-split/findings.txt):
# union-split IS the engine's realization of the existing per-level combo tied=False + read='sparse'
# + write='sparse' (previously rejected in `LevelRoutingConfig.__post_init__` as the ill-posed
# "untied double-sparse"; the rejection is LIFTED — the engine executes the combo as union-split:
# p = entmax_1.5(z_r+z_w) shared support, sigma = sigmoid(z_r−z_w) role split, read = p·sigma
# unnormalized, write = p·(1−sigma) renormalized; dag_tasks/C-union-split.md). The coarse-level
# A⊂C ablation triple: rola-base = untied sparse-write/dense-read; rola-base-sym = TIED
# sparse/sparse (the sigma≡1/2 special case); rola-base-union-global = UNTIED sparse/sparse
# (asymmetry earned per-state, not assumed away). Global norm (kappa≡0) is the primary cell per the
# C acceptance criteria. Everything else (kernel=gla_scalar, router_bias, conv-on, qk_norm=True,
# the [8,16]-with-fallback geometry probe) is identical to the other flagship arms;
# recurrent_state_floats UNCHANGED — union changes routing, not state shape (same nc/topology).
ROLA_BASE_UNION_MIX = [
    {'tied': False, 'read': 'sparse', 'write': 'sparse'},   # level 0 (coarse): union-split
    {'tied': False, 'read': 'dense', 'write': 'dense'},     # level 1 (fine)
]

# arm -> (rola_instance name, per-level routing mix). The instance name picks (kernel, state_norm,
# conv) only — its own routing_mode default is fully SUPERSEDED by the explicit
# `level_routing_configs` passthrough (so the '-asym' suffix on rola-base-sym's instance is inert).
# Topology/router_bias are shared across all four (see _ROLA_BASE_* above).
ROLA_BASE_INSTANCE = {
    'rola-base':        ('rola-gla-kappa-asym-conv', ROLA_BASE_WRITE_MIX),       # gla_scalar, kappa
    'rola-base-rla':    ('rola-rla-kappa-asym-conv', ROLA_BASE_WRITE_MIX),        # rla,        kappa
    'rola-base-global': ('rola-gla-scalar-norm-asym-conv', ROLA_BASE_WRITE_MIX),  # gla_scalar, global
    'rola-base-sym':    ('rola-gla-kappa-asym-conv', ROLA_BASE_SYM_MIX),          # gla_scalar, kappa, tied-sparse coarse
    'rola-base-sym-global': ('rola-gla-scalar-norm-asym-conv', ROLA_BASE_SYM_MIX),  # gla_scalar, global, tied-sparse coarse
    'rola-base-union-global': ('rola-gla-scalar-norm-asym-conv', ROLA_BASE_UNION_MIX),  # gla_scalar, global, union-split coarse
    # rola-base-noqknorm (user decision, 2026-07-09): rola-base with qk_norm=False. Tests whether the
    # historical LM instability was the projections (needs norm) or old-kernel numerics (now fixed,
    # per_state/rt bf16 defect 3c6567f2) — one-seed probe with the fp64 canary.
    'rola-base-noqknorm': ('rola-gla-kappa-asym-conv', ROLA_BASE_WRITE_MIX),
}
ROLA_BASE_ARMS = tuple(ROLA_BASE_INSTANCE)
# qk_norm is PINNED explicitly per flagship arm (user decision, 2026-07-09): the canonical/published
# model runs with qk_norm=True (RMSNorm on q/k pre-feature-map) — pinned here, NOT via the LM_QK_NORM
# env default (which stays env-overridable for the legacy ROLA_ARMS comparison cells below).
# rola-base-noqknorm is the one deliberate exception (see comment above).
ROLA_BASE_QKNORM = {arm: True for arm in ROLA_BASE_ARMS}
ROLA_BASE_QKNORM['rola-base-noqknorm'] = False

# arm name -> RoLA variants share geometry + backbone, differ only in rola_instance. The "-conv"
# variants are the asym RLA/GLA cells with RoLA's short convolution enabled.
# All use KAPPA normalization (state_norm='kappa') for consistency with the paper-canonical RLA cell:
# kappa-sym/kappa-asym (RLA) + gla-kappa-asym (scalar GLA, decayed per-state mass).
ROLA_ARMS = ("rola-rla-kappa-sym", "rola-rla-kappa-asym",
             "rola-gla-kappa-asym", "rola-gla-kappa-sym",
             "rola-rla-kappa-asym-conv", "rola-gla-kappa-asym-conv")
ARMS = ROLA_ARMS + ROLA_BASE_ARMS + ("gdn", "gdn-isostate", "gla", "gla-isostate", "attn")


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
    if arm in ROLA_BASE_ARMS:
        D, spl, _nc, _requires_f4 = _rola_base_topology()   # construct-gated: [8,16] or nc=64 fallback
        instance, mix = ROLA_BASE_INSTANCE[arm]
        return RoLAConfig(
            rola_instance=instance, levels_per_head=D, states_per_level=spl,
            d_qk=_ROLA_BASE_DQK, d_v=_ROLA_BASE_DV, num_heads=_ROLA_BASE_NH, router_bias=True,
            qk_norm=ROLA_BASE_QKNORM[arm],
            level_routing_configs=mix, **common)
    _base, _struct = _parse_rola(arm)
    if _base in ROLA_ARMS:
        g = {**ROLA_GEOM, **_struct}
        nc = g.pop("states_per_head")   # -> flat (D=1, b=nc); RoLAConfig has no states_per_head kwarg.
        return RoLAConfig(rola_instance=_base, levels_per_head=1, states_per_level=nc, **g, **common)
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
        # qk_norm mirrors _QKNORM (LM_QK_NORM env, default off) -- fair stabilizer parity for the
        # legacy iso-conv comparison (ROLA_ARMS + fla baselines), which is still env-controlled.
        # fla.layers.attn supports qk_norm natively. NOTE: the published-flagship arms
        # (ROLA_BASE_ARMS) now PIN qk_norm=True explicitly (ROLA_BASE_QKNORM, see above) independent
        # of this env knob -- attn's qk_norm no longer auto-tracks the flagship arms' setting; if a
        # flagship-vs-attn qk_norm-matched comparison is needed, set LM_QK_NORM=1 explicitly.
        return TransformerConfig(num_heads=ATTN_HEADS, window_size=None, qk_norm=_QKNORM, **common)
    raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")


def build_arm(arm: str, vocab_size: int, **kw):
    """Instantiate the model for `arm` (fresh weights) on the shared backbone."""
    return AutoModelForCausalLM.from_config(arm_config(arm, vocab_size, **kw))


def recurrent_state_floats(arm: str) -> int | None:
    """Nominal recurrent-state floats/layer for the bounded arms (matched target); None for attn
    (unbounded KV cache). Realized state is read back from the built layer in the result row."""
    if arm in ROLA_BASE_ARMS:
        _D, _spl, nc, _requires_f4 = _rola_base_topology()   # [8,16]->1,048,576; fallback nc=64->524,288
        return _ROLA_BASE_NH * nc * _ROLA_BASE_DQK * _ROLA_BASE_DV
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
