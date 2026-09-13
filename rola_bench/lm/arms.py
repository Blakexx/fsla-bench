"""LM arm registry — the 5 architectures of the RoLA LM comparison, all on ONE shared backbone
so only the sequence MIXER differs (true iso-backbone mixer-swap).

Design (see TASKS #102):
  * Backbone identical across arms: hidden_size, num_hidden_layers, hidden_ratio=4 SwiGLU MLP,
    RMSNorm(eps=1e-6), untied embeddings, GPT-2 vocab. RoLAConfig mirrors fla GLAConfig's backbone
    fields byte-for-byte, and the fla baselines (GDN, Transformer) share the same fla backbone, so
    swapping arms changes ONLY the token mixer.
  * The bounded-state arms are matched on RECURRENT STATE at TEXTBOOK head shape — `d_k = d_v = 128`
    at `H = 8`, the standard linear-attention head geometry — so the arms agree on state size, on
    head shape, and on addressing capacity (RoLA's `d_qk`-equivalent is likewise 128) rather than
    only on a total.
  * Softmax attention is the unbounded-state CEILING (no recurrent state to match) — the reference the
    bounded models try to approach.

**WHAT THIS TIER IS FOR (user ruling, 2026-08-02).** The LM tier's role is TRAINS-STABLY-AT-STANDARD-
SCALE, nothing more. CAPACITY evidence lives in the MQAR `N ∝ L` grid, not here. So the flagship is
NOT sized to a capacity target; it is sized to the standard head geometry a reader expects, and the
comparison is honest at that scale.

Recurrent state (headline rung), hidden=512:
    RoLA  n_h=8, N=128 (b=[8,16]), d_v=128 ->  8*128*128 = 131,072   (re-pinned 2026-08-02, F1)
    GDN   n_h=8, head_dim=128, expand_v=1  ->  8*128*128 = 131,072
    GLA   n_h=8, expand_k=2, expand_v=2    ->  (512*2)*(512*2)/8 = 131,072

The baselines' previous 1,048,576-float sizing (GDN head_dim=256/expand_v=2, GLA H=4/ev=8) was 8x
the standard head geometry — chosen to hit a "half of attention's KV cache at L=2048" target, which
is a capacity argument this tier no longer makes. It is simply re-sized to textbook here.

**F1 (2026-08-02) — the state accounting is READ BACK FROM THE ENGINE, never computed here.**
Every RoLA state number in this module comes from `fla_rola.layers.RoLA.get_stats()` on a layer
built at the arm's own geometry (`_rola_stats`). The formula this module used to carry —
`num_heads * N * d_qk * d_v` — was the V2 shape, where one state was a `[d_qk, d_v]` MATRIX. V3's
state is `[N, d_v]` per head and the architecture deleted `d_qk` outright, so that formula
overstated RoLA's state by exactly `d_qk` = 16x and every RoLA-vs-GDN/GLA row published against it
was matched to a number the layer never allocated. A formula cannot be wrong in that way twice if
there is no formula: the engine is the authority.

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

# --- env-overridable knobs (defaults = the headline rung @ d_model=512). A small-backbone
# diagnostic overrides LM_HIDDEN/LM_ROLA_*/LM_ATTN_HEADS to run a narrower regime against a matched
# small attn. ------------------------------------------------------------------------------------
_HID     = int(_os.environ.get("LM_HIDDEN", "512"))        # backbone d_model
_NC      = int(_os.environ.get("LM_ROLA_NC", "128"))       # LEGACY dense-router cells' routed states
_DV      = int(_os.environ.get("LM_ROLA_DV", "64"))        # legacy cells' value width
_NH      = int(_os.environ.get("LM_ROLA_NH", "8"))         # shared head count
# TEXTBOOK baseline geometry (re-sized 2026-08-02): d_k = d_v = 128 at H = 8 for BOTH baselines.
#   GDN: head_dim IS head_k_dim and head_v_dim = head_dim * expand_v (fla.layers.gated_deltanet),
#        so 128 / 1.0 gives 128x128 per head -> 8*128*128 = 131,072 floats/layer.
#   GLA: head_k = hidden*expand_k/H and head_v = hidden*expand_v/H (fla.layers.gla), so at
#        hidden=512, H=8, ek=ev=2.0 both are 128 -> (512*2)*(512*2)/8 = 131,072 floats/layer.
_GDN_NH  = int(_os.environ.get("LM_GDN_NH", "8"))          # GDN heads
_GDN_HEAD_DIM = int(_os.environ.get("LM_GDN_HEAD_DIM", "128"))    # GDN key head_dim = d_k (textbook 128)
_GDN_EXPAND_V = float(_os.environ.get("LM_GDN_EXPAND_V", "1.0"))  # d_v = head_dim * expand_v = 128
_GLA_NH  = int(_os.environ.get("LM_GLA_NH", "8"))          # GLA heads (8, matched with GDN/RoLA)
_GLA_EK  = float(_os.environ.get("LM_GLA_EK", "2.0"))      # d_k = hidden*ek/H = 128 at hidden=512
_GLA_EV  = float(_os.environ.get("LM_GLA_EV", "2.0"))      # d_v = hidden*ev/H = 128 at hidden=512
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
# --- matched recurrent-state geometry (textbook d_k = d_v = 128 at H = 8) ----------------------
# The LEGACY dense-router comparison cells (ROLA_ARMS below) live at flat D=1, b=nc=128, d_v=64,
# H=8 -> a realized 8*128*64 = 65,536 content floats/layer. They are NOT iso-state with GDN/GLA and
# never were: the number that used to claim they were came from the deleted `d_qk` (see the module
# docstring, F1). They are kept as the pre-flagship routed-vs-dense-router control, and their state
# is now REPORTED, not asserted — `recurrent_state_floats` reads it off the engine.
# `RoLAConfig` takes `levels_per_head`/`states_per_level` (D, b; nc = prod b), so `states_per_head`
# survives as a KEY in ROLA_GEOM only; `arm_config` translates it at the construction call.
ROLA_GEOM = dict(states_per_head=_NC, d_v=_DV, num_heads=_NH, router_bias=_ROUTER_BIAS)
GDN_GEOM = dict(num_heads=_GDN_NH, head_dim=_GDN_HEAD_DIM, expand_v=_GDN_EXPAND_V)
GLA_GEOM = dict(num_heads=_GLA_NH, expand_k=_GLA_EK, expand_v=_GLA_EV, gate_logit_normalizer=16)
# `gdn-isostate` / `gla-isostate` were DELETED 2026-08-02 (finding F4): they were byte-identical
# aliases of `gdn`/`gla` at every current env default, so running both spent budget twice on one
# cell and any table listing both implied a distinction that did not exist. With the flagship RoLA
# geometry re-pinned to the same 131,072 floats/layer (F1), `gdn`/`gla` ARE the iso-state
# baselines exactly — an "-isostate" variant would have to be either a duplicate or a fake.
# See RETIRED_LM_ARMS.
ATTN_HEADS = _ATTN_NH                                           # head_dim _HID/_ATTN_NH (default 64 @512)

# Uniform component policy across the sub-quadratic arms: short conv OFF for ALL (isolate the mixer;
# attention has no conv, so conv-off keeps the only difference the mixer itself). This runs GDN
# slightly below its canonical (conv-on) recipe — a conservative, RoLA-UNfavorable choice, stated
# openly. Flip USE_SHORT_CONV to True to give RoLA+GDN their canonical conv (attention stays off).
# This uniform conv-off policy applies ONLY to the LEGACY comparison cells (ROLA_ARMS + the fla
# baselines) below. The published-flagship arms (ROLA_BASE_ARMS, just below) are NOT part of that
# iso-conv comparison — they carry their OWN canonical conv setting (ON) per the published
# architecture (`rola_bench.models.named_configs`), independent of USE_SHORT_CONV.
USE_SHORT_CONV = False

# --- published-flagship arms --------------------------------------------------------------------
# The flagship cells (`rola_bench.models.named_configs`; RoLA-base / RoLA-base-global were retired by
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
# GEOMETRY (user decision 2026-07-08; d_v RE-PINNED 2026-08-02 per finding F1 + the same day's
# scope ruling). The flagship LM arms run the MIXED-RADIX topology states_per_level=[8, 16]
# (8 coarse × 16 fine, N = 128) at d_v = 128, H = 8:
#
#     realized content state = H * N * d_v = 8 * 128 * 128 = 131,072 floats/layer
#
# matched against GDN (8 * 128 * 128) and GLA ((512*2)*(512*2)/8) at TEXTBOOK d_k = d_v = 128, H = 8.
# The three arms now agree on state size, on head shape, AND on addressing capacity (RoLA's
# d_qk-equivalent width is likewise 128).
#
# WHAT MOVED, AND WHAT DID NOT. Only `d_v` (64 -> 128) and the baselines' sizing moved. The topology
# stays [8,16], N stays 128. What was WRONG was never the topology: this module claimed
# 8*128*16*64 = 1,048,576 floats/layer via the V2 `d_qk` factor while the layer REALIZED
# 8*128*64 = 65,536 (F1 / arms-ledger H1), and the baselines were sized to the claim — 8x the
# standard head geometry — so the "matched-state" comparison was neither matched nor standard.
#
# CAPACITY IS NOT THIS TIER'S JOB (user ruling 2026-08-02). An N-scaled LM row was considered and
# REJECTED: the LM tier exists to show the architecture trains stably at standard scale, and the
# capacity claim (the N vs L boundary) is carried by the MQAR `N ∝ L` grid, where it can be swept
# properly instead of asserted by one over-sized cell.
#
# Readout width: H * d_v = 1024 = 2 x d_model(512). The old H*d_v = 512 = d_model was chosen to
# avoid an output BOTTLENECK; 2x d_model is on the safe side of that constraint and is where the
# baselines sit too (GDN's value dim is 8*128 = 1024, identical).
#
# VERIFIED against the engine, not assumed (2026-08-02, mainline b71608f9): a RoLA layer at
# H=8, d_v=128, states_per_level=[8,16] constructs and runs a CPU forward, and `get_stats()` reports
# d_v=128, num_chunks=128, state_floats=132,096 = 131,072 content + 1,024 denominator.
# Gated in `test_flagship_geometry_and_raw_arms.py`.
_ROLA_BASE_SPL = [8, 16]
_ROLA_BASE_DV  = int(_os.environ.get("LM_ROLA_BASE_DV", "128"))
_ROLA_BASE_NH  = int(_os.environ.get("LM_ROLA_BASE_NH", "8"))
_ROLA_BASE_D   = len(_ROLA_BASE_SPL)
_ROLA_BASE_NC  = _ROLA_BASE_SPL[0] * _ROLA_BASE_SPL[1]


def _rola_base_topology() -> tuple[int, list[int], int]:
    """Return the required production flagship topology: D=2, b=[8,16], N=128."""
    return _ROLA_BASE_D, list(_ROLA_BASE_SPL), _ROLA_BASE_NC

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
# DUTY cell — see `rola_bench.models.named_configs._levelsplit_kwargs` for the full rationale. Each level
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
    # F4 (2026-08-02): these were byte-identical to 'gdn'/'gla' at every current env default —
    # GDNISO_GEOM == GDN_GEOM and GLAISO_GEOM == GLA_GEOM, same class, same kwargs — so a slate
    # carrying both spent budget twice on one cell and implied a distinction that did not exist.
    # They were kept as back-compat aliases for older YAML specs; every one of those specs also
    # names arms retired above, so nothing runnable depended on them. Deleted rather than made
    # "real": with the flagship RoLA geometry re-pinned to 1,048,576 floats/layer (F1), 'gdn' and
    # 'gla' ARE the exact iso-state baselines, so a distinct -isostate variant could only be a
    # duplicate under another name or a differently-sized cell pretending to be the matched one.
    'gdn-isostate': "F4: byte-identical duplicate of 'gdn'. Use 'gdn' — it is the exact iso-state "
                    "baseline for the re-pinned flagship geometry (131,072 floats/layer).",
    'gla-isostate': "F4: byte-identical duplicate of 'gla'. Use 'gla' — it is the exact iso-state "
                    "baseline for the re-pinned flagship geometry (131,072 floats/layer).",
}

# Legacy (non-flagship) RoLA comparison cells: the dense-router arms, named by the mainline engine's
# own instance strings (E5 — mainline `ROLA_INSTANCES` is exactly this set). The historical
# `rola-*-kappa-*` / `rola-gla-*` spellings of these cells are retired above.
ROLA_ARMS = ("rola-rla-sym", "rola-rla-asym", "rola-rla-asym-conv")
ARMS = ROLA_ARMS + ROLA_BASE_ARMS + ("gdn", "gla", "attn")

# NO `-raw` LM ARM EXISTS, and it is an ENGINE gap, not an omission here (2026-08-02). The
# normalization axis's other end is registered on the LAYER path as `RoLA-base-rla-raw` /
# `RoLA-base-levelsplit-raw` (`rola_bench.models.named_configs`), because `state_norm` cannot reach the
# layer through the HF path at all: `fla_rola.models.rola.configuration_rola.RoLAConfig` has no
# `state_norm` field and `fla_rola.models.rola.instances.rola_instance` HARD-CODES
# `state_norm='global'` for every instance in `ROLA_INSTANCES`. Passing it on `RoLAConfig` would be
# silently absorbed by HF and do nothing — exactly the F3 failure mode — so no such arm is
# registered. Threading `state_norm` (and the both-sides gain it implies) through `RoLAConfig` +
# `rola_instance` is an engine-side change; until it lands the raw ablation is layer-path only.

# Optional RoLA STRUCTURE override via arm-name suffix "<base>__nhX_dvY_ncZ" (structure-ablation
# spec): overrides num_heads/d_v/states_per_head on top of ROLA_GEOM; base must be in ROLA_ARMS.
# e.g. "rola-rla-asym__nh4_dv64" -> nh4/dv64 (head vs d_v split sweep at fixed nh*d_v=out_dim).
# The `dqk` key was DELETED with the rest of the V2 Q/K accounting (F1): the V3 layer has no d_qk,
# so a `__dqk16` token no longer names anything. An unrecognized token RAISES rather than being
# silently dropped — a silently-ignored geometry override is how a cell gets published at a
# geometry nobody asked for (the F3 failure mode again).
_STRUCT_KEYS = {"nh": "num_heads", "dv": "d_v", "nc": "states_per_head"}


def _parse_rola(arm: str):
    if "__" not in arm:
        return arm, {}
    base, ov = arm.split("__", 1)
    over = {}
    for tok in ov.split("_"):
        for k, full in _STRUCT_KEYS.items():
            if tok.startswith(k) and tok[len(k):].isdigit():
                over[full] = int(tok[len(k):]); break
        else:
            raise ValueError(
                f"arm {arm!r}: unrecognized structure-override token {tok!r}; known prefixes are "
                f"{sorted(_STRUCT_KEYS)} followed by an integer. ('dqk' was deleted with the V2 "
                "Q/K accounting — the V3 layer owns no d_qk.)")
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
        return RoLAConfig(rola_instance=_base, levels_per_head=1, states_per_level=nc, **g, **common)
    if arm == "gdn":
        from fla.models.gated_deltanet import GatedDeltaNetConfig
        return GatedDeltaNetConfig(use_short_conv=USE_SHORT_CONV, use_gate=True, **GDN_GEOM, **common)
    if arm == "gla":
        from fla.models.gla import GLAConfig
        return GLAConfig(use_short_conv=USE_SHORT_CONV, **GLA_GEOM, **common)
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


_ROLA_STATS_CACHE: dict[str, dict] = {}


def _rola_layer_kwargs(arm: str) -> dict | None:
    """The RoLA LAYER geometry an arm runs at, or None if `arm` is not a RoLA arm.

    One place that answers "what shape does this arm actually allocate", so the state accounting and
    the construction path cannot disagree. Conv is OFF here: the short-conv cache is a separate
    per-layer cache, not part of the matched recurrent-state axis, and `get_stats()` folds it in.
    """
    if arm in ROLA_BASE_ARMS:
        D, spl, _nc = _rola_base_topology()
        return dict(num_heads=_ROLA_BASE_NH, head_v_dim=_ROLA_BASE_DV,
                    levels_per_head=D, states_per_level=spl)
    base, struct = _parse_rola(arm)
    if base in ROLA_ARMS:
        g = {**ROLA_GEOM, **struct}
        return dict(num_heads=g["num_heads"], head_v_dim=g["d_v"],
                    levels_per_head=1, states_per_level=g["states_per_head"])
    return None


def rola_realized_stats(arm: str) -> dict | None:
    """`RoLA.get_stats()` for `arm`, read off a layer the ENGINE built at the arm's geometry.

    F1 (2026-08-02): this is the authority for every RoLA state number in this module. There is no
    state formula here any more — the previous one (`H * nc * d_qk * d_v`) was the V2 matrix-state
    shape and overstated V3's `[N, d_v]` state by exactly the deleted `d_qk` (16x), which is how the
    published rows came to be "matched" against a number no layer ever allocated.

    Returns the engine's own dict — `{'d_v', 'n_heads', 'num_chunks', 'state_floats'}` — where
    `state_floats` is the TOTAL (content + the per-state denominator sidecar that `state_norm`
    'global' carries and 'raw' does not). Cached per arm: constructing the accounting layer is
    cheap but not free, and the geometry is fixed for the process.
    """
    if arm in _ROLA_STATS_CACHE:
        return _ROLA_STATS_CACHE[arm]
    kwargs = _rola_layer_kwargs(arm)
    if kwargs is None:
        return None
    from rola_bench.engine import RoLA
    layer = RoLA(hidden_size=_HID, use_short_conv=False, layer_idx=0, **kwargs)
    err = getattr(layer, '_routing_resolution_error', None)
    if err is not None:
        raise err
    stats = dict(layer.get_stats())
    _ROLA_STATS_CACHE[arm] = stats
    return stats


def recurrent_state_floats(arm: str) -> int | None:
    """REALIZED content recurrent-state floats/layer (the matched axis); None for attn (unbounded).

    RoLA arms: read off `RoLA.get_stats()` (see `rola_realized_stats`) as
    `n_heads * num_chunks * d_v` — the three fields the ENGINE reports for the layer it built, not a
    geometry formula of ours. GDN/GLA keep their canonical closed forms: those are the baselines'
    own published state definitions and neither has ever been in doubt (they were correct at
    1,048,576 all along; F1's error was entirely on the RoLA side).
    """
    stats = rola_realized_stats(arm)
    if stats is not None:
        return stats['n_heads'] * stats['num_chunks'] * stats['d_v']
    if arm == "gdn":
        return GDN_GEOM["num_heads"] * GDN_GEOM["head_dim"] * int(GDN_GEOM["head_dim"] * GDN_GEOM["expand_v"])
    if arm == "gla":  # nh * key_head * value_head = key_dim*value_dim/nh (canonical, NOT matched to RoLA)
        return int(_HID * GLA_GEOM["expand_k"]) * int(_HID * GLA_GEOM["expand_v"]) // GLA_GEOM["num_heads"]
    return None  # attn: unbounded


def recurrent_state_overhead_floats(arm: str) -> int | None:
    """REALIZED denominator-sidecar floats/layer, reported separately from the matched content axis.

    Also read off the engine: `get_stats()['state_floats']` is content PLUS the sidecar (the layer's
    own `d_v + int(state_norm != 'raw')` column), so the difference is the sidecar, measured rather
    than asserted. It is `H * N` under `state_norm='global'` and exactly 0 under `'raw'` — which is
    one of the things the raw arms ablate, so it must be reported, not assumed.
    """
    stats = rola_realized_stats(arm)
    if stats is None:
        return None
    return stats['state_floats'] - stats['n_heads'] * stats['num_chunks'] * stats['d_v']


def realized_state_floats_from_model(model) -> tuple[int | None, int | None]:
    """`(content, overhead)` recurrent-state floats/layer read off a BUILT model's first mixer.

    This is what a result row records: the state the trained object actually allocated, taken from
    the mixer instance itself, so a row can never inherit a number from a registry that has drifted
    from the model. Returns `(None, None)` for a mixer with no `get_stats()` (GDN/GLA/attention go
    through the registry's closed forms instead).

    `overhead` here is everything the layer counts as state beyond the matched content axis: the
    denominator sidecar AND — unlike `recurrent_state_overhead_floats`, whose accounting layer is
    built conv-off — the short-conv cache when the arm runs conv ON (every flagship arm does). That
    is deliberate: the registry number is the comparable one, the model number is the true one, and
    a row carrying both is a row that can be audited.
    """
    for module in model.modules():
        stats = getattr(module, 'get_stats', None)
        if stats is None or not hasattr(module, 'topology'):
            continue
        s = stats()
        content = s['n_heads'] * s['num_chunks'] * s['d_v']
        return content, s['state_floats'] - content
    return None, None


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

        floats = num_heads * sum_l b_l                      (= 8 * (8+16) = 192 at the
                                                             flagship topology, vs N=128 states/head)

    which is the number that makes the decay-on/decay-off comparison honest: the clock costs three
    orders less than the state it gates. Every other arm owns no decay parameters and returns 0.
    """
    if arm in ROLA_BASE_ARMS:
        _instance, _mix, kernel = ROLA_BASE_INSTANCE[arm]
        if kernel == 'gla_mass_detached':
            _D, spl, _nc = _rola_base_topology()
            return _ROLA_BASE_NH * sum(spl)
    return 0
