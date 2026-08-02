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
from rola_bench import engine as _engine
_engine.register_hf_models()   # registers RoLA AutoClasses
from rola_bench.engine import RoLAConfig
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
ROLA_DEN_OVERHEAD = ROLA_GEOM["num_heads"] * ROLA_GEOM["states_per_head"] * ROLA_GEOM["d_qk"]

# Uniform component policy across the sub-quadratic arms: short conv OFF for ALL (isolate the mixer;
# attention has no conv, so conv-off keeps the only difference the mixer itself). This runs GDN
# slightly below its canonical (conv-on) recipe — a conservative, RoLA-UNfavorable choice, stated
# openly. Flip USE_SHORT_CONV to True to give RoLA+GDN their canonical conv (attention stays off).
# This uniform conv-off policy applies ONLY to the LEGACY comparison cells (ROLA_ARMS + the fla
# baselines) below. The published-flagship arms (ROLA_BASE_ARMS, just below) are NOT part of that
# iso-conv comparison — they carry their OWN canonical conv setting (ON) per the published
# architecture (`rola_bench.named_configs`), independent of USE_SHORT_CONV.
USE_SHORT_CONV = False

# --- published-flagship arms --------------------------------------------------------------------
# The flagship cells (`rola_bench.named_configs`; RoLA-base / RoLA-base-global were retired by
# campaign-prereqs E3, see RETIRED_LM_ARMS): router_bias=True,
# two-level topology (levels_per_head=2; branching per the GEOMETRY block below), UNTIED, per-level
# heterogeneous write (sparse@coarse level0, dense@fine level1), dense read at both levels, conv ON. These
# are the PUBLISHED cells, reachable through the HF `AutoModelForCausalLM` path now that
# `RoLAConfig` carries `routing_mode`/`level_routing_configs`
# (`fla_rola/models/rola/configuration_rola.py`) and `rola_instance()`/`RoLABlock` thread them into
# the layer kwargs (`fla_rola/models/rola/instances.py`, `modeling_rola.py`). Per-level heterogeneous
# activation is a required production capability for the sparse/dense write mix.
#
# qk_norm (RETIRED, campaign-prereqs E1, 2026-08-01): the V3 layer owns no Q/K surface at all, so
# there is no q/k RMSNorm to pin and no `rola-base-noqknorm` ablation to run. `LM_QK_NORM` survives
# only as the softmax-attention baseline's own stabilizer knob (see the `attn` arm below).
#
# GEOMETRY (user decision, 2026-07-08, REVISED same day): the flagship LM arms TARGET the
# MIXED-RADIX topology states_per_level=[8, 16] (coarse-first: 8 coarse × 16 fine, nc=128) —
# state = 8·128·16·64 = 1,048,576 floats/layer = EXACT match with the GDN/GLA X=1 iso arms (all at
# 50% of attention's reference KV cache at block_size=2048). Mixed-radix `states_per_level` is the
# The production flagship topology is mixed-radix [8,16] (coarse-first, nc=128). It is required
# capability, not a construction probe or a compatibility choice. GDN/GLA retain their canonical
# 1M-float iso geometries; the RoLA arms use the same matched content state.
_ROLA_BASE_SPL = [8, 16]
_ROLA_BASE_DQK = int(_os.environ.get("LM_ROLA_BASE_DQK", "16"))
_ROLA_BASE_DV  = int(_os.environ.get("LM_ROLA_BASE_DV", "64"))
_ROLA_BASE_NH  = int(_os.environ.get("LM_ROLA_BASE_NH", "8"))
_ROLA_BASE_D   = len(_ROLA_BASE_SPL)

def _rola_base_topology() -> tuple[int, list[int], int]:
    """Return the required production flagship topology: D=2, b=[8,16], nc=128."""
    return _ROLA_BASE_D, list(_ROLA_BASE_SPL), 128

# The published write mix: sparse (entmax, fixed alpha=1.5) at the coarse level, dense (softmax) at
# the fine level; read stays dense at both levels (dead-end-free by construction, ROLA_SEMANTICS.md
# §5). Serializable dict form -- `RoLAConfig.level_routing_configs` converts these to
# `LevelRoutingConfig` at the `instances.py` boundary.
ROLA_BASE_WRITE_MIX = [
    {'tied': False, 'read': 'dense', 'write': 'sparse'},   # level 0 (coarse)
    {'tied': False, 'read': 'dense', 'write': 'dense'},    # level 1 (fine)
]

# The symmetric-coarse cells rola-base-sym / rola-base-sym-global (2026-07-08/09) are RETIRED
# (campaign-prereqs E3, see RETIRED_LM_ARMS): both were gla_scalar cells, and the destroy-kappa
# hypothesis rola-base-sym-global existed to test is moot now that E2 retired kappa outright. Their
# tied-sparse-coarse mix survives, in the depth-hybrid family, as ROLA_BASE_HYBRID_TIEDTOP_MIX.

# rola-base-union-global (candidate C, 2026-07-10): the UNION-SPLIT routing cell — coarse level
# UNTIED sparse/sparse. NO new vocabulary (settled 2026-07-10, workflows/union-split/findings.txt):
# union-split IS the engine's realization of the existing per-level combo tied=False + read='sparse'
# + write='sparse' (previously rejected in `LevelRoutingConfig.__post_init__` as the ill-posed
# "untied double-sparse"; the rejection is LIFTED — the engine executes the combo as union-split:
# p = entmax_1.5(z_r+z_w) shared support, sigma = sigmoid(z_r−z_w) role split, read = p·sigma
# unnormalized, write = p·(1−sigma) renormalized; dag_tasks/C-union-split.md). The coarse-level
# A⊂C ablation triple: rola-base-rla = untied sparse-write/dense-read; the tied-sparse/sparse
# sigma≡1/2 special case (now carried by the hybrid-tiedtop cell); rola-base-union-global = UNTIED
# sparse/sparse (asymmetry earned per-state, not assumed away). Everything else (kernel='rla',
# state_norm='global', router_bias, conv-on, the required [8,16] geometry) is identical to the
# other flagship arms;
# recurrent_state_floats UNCHANGED — union changes routing, not state shape (same nc/topology).
ROLA_BASE_UNION_MIX = [
    {'tied': False, 'read': 'sparse', 'write': 'sparse'},   # level 0 (coarse): union-split
    {'tied': False, 'read': 'dense', 'write': 'dense'},     # level 1 (fine)
]

# rola-base-hybrid-global (candidate D, 2026-07-11, dag_tasks/D-and-E-tail.md CANDIDATE D): the
# DEPTH-HYBRID cell — union at block level, decoupled commitment in-block. Coarse level is UNTIED
# sparse/sparse (candidate C's union-split: total block removal + read/write alignment via sigma).
# Fine level is UNTIED dense-read/sparse-write (candidate B's decoupled combo: sharp fine writes =
# sparse x sparse folded events, dense in-block reads = no fine dead-ends). Everything else
# (state_norm='global', router_bias, conv-on, the required [8,16]
# geometry) is identical to the other flagship arms; recurrent_state_floats UNCHANGED — the hybrid
# mix changes routing only, not state shape (same nc/topology).
#
# GATED vs RLA (campaign-prereqs E4, 2026-08-01, supersedes the 2026-07-11 kernel-selector
# correction): "gated" for this family meant kernel='gla_static', which the V3 engine refuses by
# name — its defining non-detached clock has no V3 expression. The gla_static cells
# rola-base-hybrid-global / rola-base-hybrid-tiedtop-global are RETIRED; only their kernel='rla'
# (ungated, no decay) siblings survive. If a gated depth-hybrid cell is wanted again it must be
# kernel='gla_mass_detached' (detached leaf-mass decay), which is a DIFFERENT experiment and a
# coordinator decision.
#
# E1 descope (D-and-E-tail.md, 2026-07-11): published LM/MQAR data = hybrid-global {gated, rla}
# (both kernels); hybrid-tiedtop-global is a FIXED-SIZE ablation only (no full sweep).
ROLA_BASE_HYBRID_MIX = [
    {'tied': False, 'read': 'sparse', 'write': 'sparse'},   # level 0 (coarse): union-split combo
    {'tied': False, 'read': 'dense',  'write': 'sparse'},   # level 1 (fine): sparse-write/dense-read (B-combo)
]

# rola-base-hybrid-tiedtop-global (user decision, 2026-07-11): the canonical D-vs-tied-top ablation
# pair with rola-base-hybrid-global. Coarse level is TIED sparse/sparse (candidate A's shared-support
# combo, the sigma≡1/2 special case of union-split) instead of hybrid's UNTIED sparse/sparse
# (union-split, asymmetry earned per-state). Fine level is IDENTICAL to the hybrid arm (untied
# dense-read/sparse-write). This isolates whether block-level sigma-untying earns its parameters --
# everything else (state_norm='global', router_bias, conv-on, qk_norm=True, the
# required [8,16] geometry) held fixed; recurrent_state_floats UNCHANGED (same nc/topology). Gated vs
# rla kernel selection mirrors rola-base-hybrid-global (see the note above ROLA_BASE_HYBRID_MIX).
# E1 descope: this is a FIXED-SIZE ablation cell only (no full sweep).
ROLA_BASE_HYBRID_TIEDTOP_MIX = [
    {'tied': True,  'read': 'sparse', 'write': 'sparse'},   # level 0 (coarse): TIED sym combo (the ablation axis)
    {'tied': False, 'read': 'dense',  'write': 'sparse'},   # level 1 (fine): identical to the hybrid arm
]

# rola-base-levelsplit-global (ablation ARM 3, promoted to an LM arm 2026-08-02): the LEVEL-SPLIT
# DUTY cell — see `rola_bench.named_configs._levelsplit_kwargs` for the full rationale. Each level
# carries exactly ONE sparse duty and one dense duty, and the two levels carry OPPOSITE duties:
# coarse commits narrowly and reads broadly, fine writes broadly and addresses narrowly. Both rows
# are untied with at most one sparse side, so each resolves to `IndependentRouting` and every duty
# solves the EXACT entmax/softmax on its OWN projected logits — the deliberate contrast with
# ROLA_BASE_UNION_MIX, whose untied sparse/sparse coarse level resolves to `UnionRouting` and splits
# ONE shared support over averaged logits.
#
# This completes the paper's three structural wirings at IDENTICAL N and L and identical router
# parameter shape (all three arms are untied on both levels => matched addressing entropy; only the
# per-side activation assignment differs): ARM 1 base = the (N,1) O(N) point, ARM 2 union = off the
# hyperbola O(1), ARM 3 level-split = the equal-width (sqrt N, sqrt N) O(sqrt N) point.
# recurrent_state_floats UNCHANGED — duty assignment changes routing, not state shape.
ROLA_BASE_LEVELSPLIT_MIX = [
    {'tied': False, 'read': 'dense',  'write': 'sparse'},   # level 0 (coarse): commit narrowly, read broadly
    {'tied': False, 'read': 'sparse', 'write': 'dense'},    # level 1 (fine): write broadly, address narrowly
]

# rola-base-levelsplit-a2w-global (user decision, 2026-08-02): ARM 3's ALPHA-CHOICE ablation.
# `alpha` is a PER-LEVEL entmax order applying to whichever side of that level is 'sparse' (a
# 'dense' side is softmax and ignores it). Because the level-split arm gives each level exactly one
# sparse duty, and opposite duties across levels, one per-level alpha IS a per-duty dial here:
# every sparse WRITE duty becomes sparsemax (alpha=2, exact point mass on the demanded corner — what
# maps 1:1 onto skippable state blocks) while every sparse READ duty stays at the softer alpha=1.5.
# 'Commit hard, address softly.' Identical to ROLA_BASE_LEVELSPLIT_MIX in every other respect, so
# the pair isolates the alpha choice. recurrent_state_floats UNCHANGED.
ROLA_BASE_LEVELSPLIT_A2W_MIX = [
    {'tied': False, 'read': 'dense',  'write': 'sparse', 'alpha': 2.0},   # coarse: SPARSEMAX write, softmax read
    {'tied': False, 'read': 'sparse', 'write': 'dense',  'alpha': 1.5},   # fine: alpha-1.5 entmax read, softmax write
]

# rola-base-massdecay-global (user decision, 2026-08-02): the DECAY-ON arm, and the only one on the
# slate. Its routing is ROLA_BASE_WRITE_MIX byte-for-byte — i.e. it is rola-base-rla with the clock
# turned on and NOTHING else changed, which is exactly what makes the paper's "no-decay ablation"
# an ablation rather than an isolated cell. Until now every surviving arm was kernel='rla'
# (decay='none') and the decay-ON counterpart existed nowhere.
#
# Semantics are normative in `development/plans/GLA_MASS_DECAY_SPEC.md`:
#   * compact AND rate     rate_s = prod_l delta_l[d_l(s)],  delta_l = sigmoid(theta_l) in (0,1).
#     Parameters are COMPACT, O(H * sum_l b_l) — per level-branch dials, never an [H,nc] table.
#     All levels must agree to forget a leaf; one zero dial vetoes forgetting (§3.1).
#   * DETACHED mass clock  c_s[t] = stop_gradient(w_s[t]) on the folded stored write amplitude (§3.2)
#     — V3's clock is unconditionally detached, which is precisely why 'gla_static' (a non-detached
#     EVENT clock) has no V3 expression and is retired by E4.
#   * exponential survival A[t,s] = (1 - rate_s)^c_s[t]; carried state decays, the current write is
#     deposited UNDAMPED, the inclusive readout observes the post-write state (§3.3). The superseded
#     linear clock `1 - rate*w` is NOT normative.
#   * routing trains through deposits and reads; the dials train through the affine rate sensitivity
#     (§3.4) — no state snapshots.
# The dial init target rho_init is the V3 layer's own `decay_target_leaf_rate` (default 2**-8).
# NOTE (hygiene): `fla_rola.models.rola.instances.rola_instance` does not thread that kwarg, so this
# arm runs at the layer default and rho_init is not sweepable through the HF path today.
ROLA_BASE_MASSDECAY_MIX = ROLA_BASE_WRITE_MIX

# arm -> (rola_instance name, per-level routing mix, kernel). The instance name picks (state_norm,
# conv) only — its own routing default is fully SUPERSEDED by the explicit `level_routing_configs`
# passthrough (so the '-asym' suffix is inert). `kernel` is the config-level decay-family selector
# ('rla' -> decay='none', 'gla_mass_detached' -> decay='leaf_mass'), resolved at the
# `rola_instance` boundary; it is PER-ARM, not a module constant, since 2026-08-02 added the first
# decay-ON cell. Topology/router_bias are shared across all arms (see _ROLA_BASE_* above).
#
# RECONCILED onto the V3 mainline engine, 2026-08-01 (campaign-prereqs E1/E3/E4/E5): every arm now
# names one of mainline's own `fla_rola.models.rola.instances.ROLA_INSTANCES`
# ('rola-rla-asym', 'rola-rla-sym', 'rola-rla-asym-conv'). The flagship cells are conv-ON, so they
# all take 'rola-rla-asym-conv' (state_norm='global', conv=True, dense default routing that the
# explicit mix supersedes). See RETIRED_LM_ARMS below for what E3/E4 removed and why.
ROLA_BASE_INSTANCE = {
    'rola-base-rla':    ('rola-rla-asym-conv', ROLA_BASE_WRITE_MIX, 'rla'),        # rla, global — ablation ARM 1
    'rola-base-union-global': ('rola-rla-asym-conv', ROLA_BASE_UNION_MIX, 'rla'),  # rla, global, union-split coarse — ARM 2
    # ARM 3, promoted to an LM arm 2026-08-02: the level-split duty cell (see ROLA_BASE_LEVELSPLIT_MIX)
    # and its alpha-choice ablation (sparsemax writes, softer reads).
    'rola-base-levelsplit-global': ('rola-rla-asym-conv', ROLA_BASE_LEVELSPLIT_MIX, 'rla'),
    'rola-base-levelsplit-a2w-global': ('rola-rla-asym-conv', ROLA_BASE_LEVELSPLIT_A2W_MIX, 'rla'),
    # The decay-ON counterpart of ARM 1 — identical routing, detached leaf-mass clock.
    'rola-base-massdecay-global': ('rola-rla-asym-conv', ROLA_BASE_MASSDECAY_MIX, 'gla_mass_detached'),
    # The candidate-D depth-hybrid cell and its D-vs-tied-top ablation. Only the ungated (kernel=
    # 'rla') spellings survive: their 'gla_static' siblings are retired by E4 (V3's clock is
    # unconditionally detached, so the engine refuses 'gla_static' by name).
    'rola-base-hybrid-global-rla': ('rola-rla-asym-conv', ROLA_BASE_HYBRID_MIX, 'rla'),  # rla (ungated), global, depth-hybrid (union coarse + B-combo fine)
    'rola-base-hybrid-tiedtop-global-rla': ('rola-rla-asym-conv', ROLA_BASE_HYBRID_TIEDTOP_MIX, 'rla'),  # rla (ungated), global, depth-hybrid ablation: TIED-sparse coarse + B-combo fine
}
ROLA_BASE_ARMS = tuple(ROLA_BASE_INSTANCE)

# RETIRED LM arms (campaign-prereqs E1/E3/E4, 2026-08-01) — arm name -> reason. Each was DEFINED by
# a knob the V3 architecture no longer has, so re-pinning it to a live knob would publish a
# different cell under an old name. `arm_config` refuses these by name.
RETIRED_LM_ARMS = {
    'rola-base': "E3: kernel='gla_scalar' (+ E2 state_norm='kappa'). Its mix survives as 'rola-base-rla'.",
    'rola-base-global': "E3: kernel='gla_scalar'. Its global-norm base mix survives as 'rola-base-rla'.",
    'rola-base-sym': "E3: kernel='gla_scalar' (+ E2 state_norm='kappa').",
    'rola-base-sym-global': "E3: kernel='gla_scalar'; and E2 retired kappa outright, so the destroy-kappa cell is moot.",
    'rola-base-hybrid-global': "E4: the GATED sibling via kernel='gla_static', which V3 refuses by name. Ungated sibling: 'rola-base-hybrid-global-rla'.",
    'rola-base-hybrid-tiedtop-global': "E4: kernel='gla_static'. Ungated sibling: 'rola-base-hybrid-tiedtop-global-rla'.",
    'rola-multiscale': "E4: kernel='gla_static' was its entire definition and V3 has no non-detached clock.",
    'rola-base-noqknorm': "E1: defined solely by qk_norm=False, and the V3 layer owns no Q/K surface at all.",
    # Legacy iso-conv comparison cells: every one of them is a kappa and/or gla_scalar instance
    # string, and mainline's ROLA_INSTANCES contains neither (E5).
    'rola-rla-kappa-sym': "E2/E5: state_norm='kappa' instance string; not in mainline ROLA_INSTANCES.",
    'rola-rla-kappa-asym': "E2/E5: state_norm='kappa' instance string; not in mainline ROLA_INSTANCES.",
    'rola-gla-kappa-asym': "E2/E3/E5: kappa + gla_scalar instance string; not in mainline ROLA_INSTANCES.",
    'rola-gla-kappa-sym': "E2/E3/E5: kappa + gla_scalar instance string; not in mainline ROLA_INSTANCES.",
    'rola-rla-kappa-asym-conv': "E2/E5: state_norm='kappa' instance string; not in mainline ROLA_INSTANCES.",
    'rola-gla-kappa-asym-conv': "E2/E3/E5: kappa + gla_scalar instance string; not in mainline ROLA_INSTANCES.",
}

# Legacy (non-flagship) RoLA comparison cells: the dense-router arms, named by the mainline engine's
# own instance strings (E5 — mainline `ROLA_INSTANCES` is exactly this set). The historical
# `rola-*-kappa-*` / `rola-gla-*` spellings of these cells are retired above.
ROLA_ARMS = ("rola-rla-sym", "rola-rla-asym", "rola-rla-asym-conv")
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
    if arm in RETIRED_LM_ARMS:
        raise ValueError(f"LM arm {arm!r} is RETIRED: {RETIRED_LM_ARMS[arm]}")
    if arm in ROLA_BASE_ARMS:
        D, spl, _nc = _rola_base_topology()
        instance, mix, kernel = ROLA_BASE_INSTANCE[arm]
        # E1: no d_qk / qk_norm — the V3 layer owns no Q/K surface. `kernel` is PINNED per arm here
        # rather than left to the instance string's default, so the decay family a published cell
        # ran on is explicit in the config (and so the decay-ON arm cannot silently drift to 'rla').
        return RoLAConfig(
            rola_instance=instance, levels_per_head=D, states_per_level=spl,
            d_v=_ROLA_BASE_DV, num_heads=_ROLA_BASE_NH, router_bias=True,
            kernel=kernel,
            level_routing_configs=mix, **common)
    _base, _struct = _parse_rola(arm)
    if _base in ROLA_ARMS:
        g = {**ROLA_GEOM, **_struct}
        nc = g.pop("states_per_head")   # -> flat (D=1, b=nc); RoLAConfig has no states_per_head kwarg.
        g.pop("d_qk"); g.pop("qk_norm")  # E1: state-accounting keys only, not construction kwargs.
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
        # qk_norm mirrors _QKNORM (LM_QK_NORM env, default off); fla.layers.attn supports it
        # natively. It is attention's own stabilizer knob only -- the RoLA arms have no Q/K surface
        # to normalize any more (campaign-prereqs E1), so there is nothing to match it against.
        return TransformerConfig(num_heads=ATTN_HEADS, window_size=None, qk_norm=_QKNORM, **common)
    raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")


def build_arm(arm: str, vocab_size: int, **kw):
    """Instantiate the model for `arm` (fresh weights) on the shared backbone."""
    return AutoModelForCausalLM.from_config(arm_config(arm, vocab_size, **kw))


def recurrent_state_floats(arm: str) -> int | None:
    """Nominal recurrent-state floats/layer for the bounded arms (matched target); None for attn
    (unbounded KV cache). Realized state is read back from the built layer in the result row."""
    if arm in ROLA_BASE_ARMS:
        _D, _spl, nc = _rola_base_topology()
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


def recurrent_state_overhead_floats(arm: str) -> int | None:
    """Physical denominator-sidecar floats/layer, reported separately from matched content state."""
    _base, _struct = _parse_rola(arm)
    if arm in ROLA_BASE_ARMS:
        _D, _spl, nc = _rola_base_topology()
        return _ROLA_BASE_NH * nc * _ROLA_BASE_DQK
    if _base in ROLA_ARMS:
        g = {**ROLA_GEOM, **_struct}
        return g["num_heads"] * g["states_per_head"] * g["d_qk"]
    return None


def static_decay_parameter_floats(arm: str) -> int:
    """Additional fp32 theta parameters owned by a DECAY arm, per layer.

    Kept under its historical name (it is a `train_lm` result-row field, so renaming it would break
    the result schema); "static" is now a misnomer — V3's only clock is the detached leaf-mass one.
    The gla_static arm this counted for (`rola-multiscale`) was retired by campaign-prereqs E4.

    Since 2026-08-02 the count is real again for `rola-base-massdecay-global`
    (kernel='gla_mass_detached' -> the layer's `decay='leaf_mass'`). Per
    `development/plans/GLA_MASS_DECAY_SPEC.md` §3.1/§5 the dials are COMPACT — one theta per level
    BRANCH, `O(H * sum_l b_l)`, explicitly never an `[H, nc]` table — and the layer materializes
    them as `decay_theta`, a ParameterList of per-level `[H, b_l]` fp32 tensors. So:

        floats = num_heads * sum_l b_l                      (= 8 * (8+16) = 192 at the flagship
                                                             topology, vs nc=128 states/head)

    which is the number that makes the decay-on/decay-off comparison honest: the clock costs three
    orders less than the state it gates. Every other arm owns no decay parameters and returns 0.
    """
    if arm in ROLA_BASE_ARMS:
        _instance, _mix, kernel = ROLA_BASE_INSTANCE[arm]
        if kernel == 'gla_mass_detached':
            _D, spl, _nc = _rola_base_topology()
            return _ROLA_BASE_NH * sum(spl)
    return 0
