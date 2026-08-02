"""Named RoLA configs — the pinned, PUBLISHED-flagship architecture the suites default to (user
decision, 2026-07-08), RECONCILED onto the V3 mainline engine
(`/home/blake/flash-linear-attention-rola` @ b71608f9) on 2026-08-01 by the campaign-prereqs
migration E1-E7 (`rola-scratch/workflows/20260801-campaign-prereqs/prereqs.md` §Q3).

These are `fla_rola.layers.RoLA` DIRECT construction kwargs (levels_per_head / states_per_level /
level_routing_configs), not the legacy zoology-mixer vocabulary.

**The V3 construction surface (what the migration changed).**
  * E1 — the V3 layer owns NO Q/K surface: `head_k_dim`, `d_qk` and `qk_norm` are gone from
    `construct()` and from every builder here.
  * E2 — `state_norm='kappa'` is retired; `'global'` is the only published normalization
    (`[[rola-self-normalizing-dead-leaves-inert]]`; the live set is {'raw','global'}).
  * E3 — `kernel='gla_scalar'` is retired (zero occurrences in any current engine checkout).
  * E4 — `kernel='gla_static'` is retired: V3's clock is unconditionally detached
    (ROLA_V3_SPEC.md §2 stop_gradient), so the engine REFUSES 'gla_static' by name. The surviving
    decay vocabulary is `'rla'` (no decay) and `'gla_mass_detached'` (detached leaf-mass decay).
  * The config-level `kernel` string is resolved to the V3 layer's `decay` kwarg here, at the
    construction boundary, exactly as `fla_rola.models.rola.instances.rola_instance` does it
    ('rla' -> decay='none', 'gla_mass_detached' -> decay='leaf_mass').

**RoLA-base-rla** (the surviving flagship base cell): router_bias=True, kernel='rla',
state_norm='global', SQUARE topology (levels_per_head=2, states_per_level=sqrt(nc), nc=256 default),
UNTIED, per-LEVEL heterogeneous routing — write is SPARSE at level 0 (coarse) + DENSE at level 1
(fine), read is DENSE at both levels:
    level_routing_configs=[LevelRoutingConfig(tied=False, read='dense', write='sparse'),
                            LevelRoutingConfig(tied=False, read='dense', write='dense')]
Rationale: coarse write-zeros map 1:1 onto skippable state-blocks (all the §7 skip perf lives here),
dense-fine keeps within-region blending, dense read is dead-end-free by construction. This is the
campaign's ablation ARM 1 (prereqs Q3 §S1).

**RoLA-base-union-global**: the candidate-C UNION-SPLIT cell (ablation ARM 2) — coarse level UNTIED
sparse/sparse. Verified EQUIVALENT to the V3 reference under the producer's own `_split_level_mass`
mass mapping (prereqs Q1); no semantics edit was needed (E7).

**RoLA-base-levelsplit-global**: the LEVEL-SPLIT DUTY cell (ablation ARM 3) — see
`_levelsplit_kwargs`.

**RoLA-base-levelsplit-a2w-global**: ARM 3's ALPHA-CHOICE ablation — same duty assignment, sparsemax
(alpha=2) on every sparse WRITE duty and the softer alpha=1.5 on every sparse READ duty
(user decision, 2026-08-02). See `_levelsplit_a2w_kwargs`.

**RoLA-base-massdecay-global**: the DECAY-ON arm (user decision, 2026-08-02) — ARM 1's routing
byte-for-byte with `kernel='gla_mass_detached'` instead of `'rla'`, i.e. the V3 layer's
`decay='leaf_mass'`. This is the paper's decay-ON counterpart to the no-decay ablation; before it,
every arm on the slate was `decay='none'` and the ablation had nothing to ablate against.
Semantics per `development/plans/GLA_MASS_DECAY_SPEC.md` (normative): compact AND rate
`rate_s = prod_l delta_l[d_l(s)]` with `delta_l = sigmoid(theta_l)` over per-level branch dials
(O(H * sum_l b_l) parameters, never [H,nc]); DETACHED mass clock
`c_s[t] = stop_gradient(w_s[t])` on the folded stored write amplitude; exponential survival
`A[t,s] = (1 - rate_s)^c_s[t]`, deposits undamped, inclusive readout. Routing trains through
deposits and reads; the dials train through the affine rate sensitivity. The dial init target
`rho_init` is the layer's own `decay_target_leaf_rate` (default 2**-8) — NOT currently overridable
through the HF `RoLAConfig` path (`rola_instance` does not thread it), so this arm runs at that
default.

**MQAR variants**: the same cells minus short conv (`use_short_conv=False`) — convs confound
short-offset recall in MQAR; everything else identical.

Suite wiring (survivors only; every retired name was dropped from these tuples by E2/E3/E4):
  * LM suite     : RoLA-base-rla, RoLA-base-union-global, RoLA-base-levelsplit-global,
                   RoLA-base-levelsplit-a2w-global, RoLA-base-massdecay-global. (2026-08-02: ARM 3
                   joined so all three structural wirings of the central ablation are LM-runnable,
                   together with the alpha and decay axes.)
  * MQAR suite   : RoLA-base-rla, RoLA-base-union-global, RoLA-base-hybrid-global-rla,
                   RoLA-base-hybrid-tiedtop-global-rla. (The hybrid pair is the candidate-D
                   depth-hybrid cell and its D-vs-tied-top ablation; only their kernel='rla'
                   spellings survive — the 'gla_static' siblings are retired by E4.) The MQAR tier
                   is blocked on the zoology mixer migration, so this tuple was left unchanged.
  * Perf suite   : RoLA-base-rla. `perf/scaling.py` DERIVES its named arms from this tuple.

See `RETIRED_CONFIGS` for every retired name, its E-number and the knob that no longer exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


# Config-level decay vocabulary -> the V3 layer's `decay` kwarg. Mirrors
# `fla_rola.models.rola.instances.rola_instance`'s own mapping byte-for-byte, including its refusal
# of 'gla_static': V3's clock is unconditionally detached (ROLA_V3_SPEC.md §2 stop_gradient), which
# is exactly the 'gla_mass_detached' contract, NOT 'gla_static''s defining non-detached clock, so a
# lingering 'gla_static' spelling must raise rather than silently resolve to 'leaf_mass'.
_KERNEL_TO_DECAY = {'rla': 'none', 'gla_mass_detached': 'leaf_mass'}


def _decay_from_kernel(kernel: str) -> str:
    if kernel in _KERNEL_TO_DECAY:
        return _KERNEL_TO_DECAY[kernel]
    if kernel == 'gla_static':
        raise ValueError(
            "kernel='gla_static' has no V3 layer equivalent: V3's clock is unconditionally detached "
            "(ROLA_V3_SPEC.md §2 stop_gradient), which is exactly the 'gla_mass_detached' contract, "
            "not 'gla_static''s defining non-detached clock. Use 'gla_mass_detached' for a detached "
            "leaf-mass decay, or 'rla' for no decay (campaign-prereqs E4).")
    if kernel == 'gla_scalar':
        raise ValueError(
            "kernel='gla_scalar' is retired vocabulary — it exists in no current fla_rola checkout "
            "(campaign-prereqs E3). Use 'rla' or 'gla_mass_detached'.")
    raise ValueError(f"unknown RoLA kernel {kernel!r}; expected one of {sorted(_KERNEL_TO_DECAY)}")


def _square_topology(nc: int) -> tuple[int, int]:
    b = int(round(math.sqrt(nc)))
    if b * b != nc:
        raise ValueError(f"square topology requires a perfect-square nc, got {nc}")
    return 2, b


@dataclass(frozen=True)
class NamedConfig:
    name: str
    build_kwargs: Callable[[int, bool], dict]   # (nc, use_short_conv) -> RoLA(**kwargs) (minus hidden/heads/dims)


def _base_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'global') -> dict:
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, decay=_decay_from_kernel(kernel), state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='dense', write='sparse'),
            LevelRoutingConfig(tied=False, read='dense', write='dense'),
        ],
    )


def _union_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'global') -> dict:
    """RoLA-base-union-global: the candidate-C UNION-SPLIT cell (mirrors `rola_bench.lm.arms`
    ROLA_BASE_UNION_MIX byte-for-byte). Coarse level is UNTIED sparse/sparse — NO new vocabulary
    (settled 2026-07-10, workflows/union-split/findings.txt): union-split IS the engine's
    realization of the existing combo tied=False + read='sparse' + write='sparse', whose
    `LevelRoutingConfig.__post_init__` rejection ("untied double-sparse") is LIFTED. The engine
    executes it as union-split (p = entmax_1.5(z_r+z_w) shared support — no read/write dead-ends;
    sigma = sigmoid(z_r−z_w) role split; read = p·sigma unnormalized, write = p·(1−sigma)
    renormalized; dag_tasks/C-union-split.md). Fine level stays untied dense/dense, same as the base
    cell.

    The unnormalized union READ is NOT a V3 semantics gap (campaign-prereqs Q1, 2026-08-01): the V3
    producer's own `producer_v3.py::_split_level_mass` folds the per-level mass into the side gain
    before the read channel sees it, and under `state_norm='global'` the read mass cancels exactly in
    num/den. Measured EQUIVALENT at <=8.9e-16 vs a 1e-14 budget on both alphas and both
    normalizations. Hence E7: no semantics edit here."""
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, decay=_decay_from_kernel(kernel), state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='sparse', write='sparse'),
            LevelRoutingConfig(tied=False, read='dense', write='dense'),
        ],
    )


def _levelsplit_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'global') -> dict:
    """RoLA-base-levelsplit-global: the LEVEL-SPLIT DUTY cell — the ablation's third arm against
    RoLA-base-rla (untied sparse-write/dense-read coarse) and RoLA-base-union-global (union-split
    coarse), 2026-08-01.

    **The duty assignment.** In this D=2 hierarchy each level carries ONE sparse duty and one dense
    duty, and the two levels carry OPPOSITE duties:

      * coarse level: sparse WRITE / dense READ  (commit narrowly, read broadly)
      * fine level:   dense  WRITE / sparse READ (write broadly, address narrowly)

    **Why this needs no new engine vocabulary and no new flag.** Both rows are untied with at most
    one sparse side, so `fla_rola.layers.rola._resolve_level_routing` maps each to
    `IndependentRouting(tied=False, read_activation=..., write_activation=...)` — every duty solves
    the EXACT entmax/softmax on its OWN projected logits (`routing.reference.routing_factor_levels`
    -> `_activation_factors`). Nothing is shared, averaged, or renormalized across duties. This is
    the deliberate contrast with `_union_kwargs`, whose untied sparse/sparse coarse level resolves
    to `UnionRouting` and therefore runs `_union_split_with_support` on the AVERAGED logits
    `0.5*z_r + 0.5*z_w`, splitting one shared entmax support by sigma = sigmoid(z_r - z_w).

    **Corner purity — the theoretical property union structurally lacks.** Because the sparse write
    duty here owns its own entmax, a write logit vector that demands a point mass gets an EXACT
    point mass (delta), whatever the read logits say. Under union-split the write is confined to,
    and renormalized over, the shared support of the AVERAGED logits, so a read logit pointing
    elsewhere keeps the write off its corner. `test_levelsplit_configs.py` tests exactly this.

    **Matched addressing entropy vs the union arm.** Identical topology (`_square_topology(nc)`, so
    identical D, b, nc and identical state geometry), identical `router_bias`/`state_norm`/conv, and
    identical ROUTER PARAMETER SHAPE: both arms are untied on both levels, i.e. two independent
    logit sets per level in the packed projection. Only the per-side activation assignment differs.

    Note on `kernel`: this arm was first registered on `'gla_static'` (2026-08-01) only because
    `'gla_scalar'` was already dead; E4 moved it to the ungated `'rla'`, which is what the mainline
    V3 engine can actually execute and what the ablation's other two arms use.

    Note on `alpha`: this arm leaves `LevelRoutingConfig.alpha` at the engine default 1.5 on both
    levels. `_levelsplit_a2w_kwargs` is its alpha-choice sibling (sparsemax writes, softer reads);
    the two differ in that one knob and nothing else.
    """
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, decay=_decay_from_kernel(kernel), state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='dense', write='sparse'),
            LevelRoutingConfig(tied=False, read='sparse', write='dense'),
        ],
    )


def _levelsplit_a2w_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'global') -> dict:
    """RoLA-base-levelsplit-a2w-global: the level-split duty cell at SPARSEMAX WRITE / softer read
    (user decision, 2026-08-02). The ALPHA-CHOICE ablation axis against `_levelsplit_kwargs`.

    **What differs from `_levelsplit_kwargs` — one knob, per level.** `LevelRoutingConfig.alpha` is
    a PER-LEVEL entmax order that applies to whichever side of that level is `'sparse'` (a `'dense'`
    side is softmax and ignores alpha). The level-split arm gives each level exactly ONE sparse
    duty, and the two levels carry OPPOSITE duties — which is what makes alpha a clean per-DUTY dial
    here without any per-(side,level) vocabulary:

      * coarse level: sparse WRITE, dense read -> alpha=2.0  => SPARSEMAX write (hard, exactly-zero
        off-support), softmax read (the softest read there is).
      * fine level:   sparse READ, dense write -> alpha=1.5  => alpha-1.5 entmax read, i.e. the
        SOFTER of the two exact-sparse orders, against a softmax write.

    So every sparse WRITE duty in this arm is sparsemax and every sparse READ duty is the softer
    alpha=1.5 — 'commit hard, address softly'. `_levelsplit_kwargs` holds alpha=1.5 on both levels
    (the engine default), so the pair isolates the alpha choice with the duty assignment, topology,
    bias, norm, decay and conv all held fixed.

    **Why the write side is the one to harden.** alpha=2 (sparsemax) writes give an EXACT point mass
    on the demanded corner, which is what maps 1:1 onto skippable state blocks; a hard READ instead
    risks addressing a leaf the write never touched. Reads stay soft for the dead-end-free property
    the base arm's dense read gives by construction.

    Both rows remain untied with at most one sparse side, so each resolves to `IndependentRouting`
    with its own exact solve — same structural path as `_levelsplit_kwargs`, no union anywhere, and
    identical router PARAMETER shape (untied on both levels => matched addressing entropy).
    """
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, decay=_decay_from_kernel(kernel), state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='dense', write='sparse', alpha=2.0),
            LevelRoutingConfig(tied=False, read='sparse', write='dense', alpha=1.5),
        ],
    )


def _levelsplit_dense_degenerate_kwargs(nc: int, use_short_conv: bool, kernel: str,
                                        state_norm: str = 'global') -> dict:
    """The level-split arm's DEGENERATE spelling: both duties on both levels dense.

    Not a published cell — it exists so the equivalence gate in `test_levelsplit_configs.py` can
    assert that turning every duty dense collapses the level-split arm onto the plain untied
    dense/dense routing, with nothing left over from the duty machinery. Same topology, bias, norm
    and kernel as `_levelsplit_kwargs`; only the activations differ.
    """
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, decay=_decay_from_kernel(kernel), state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='dense', write='dense'),
            LevelRoutingConfig(tied=False, read='dense', write='dense'),
        ],
    )


def _hybrid_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'global') -> dict:
    """RoLA-base-hybrid-global-rla: the candidate-D DEPTH-HYBRID cell (mirrors `rola_bench.lm.arms`
    ROLA_BASE_HYBRID_MIX byte-for-byte, dag_tasks/D-and-E-tail.md CANDIDATE D, 2026-07-11). Coarse
    level is UNTIED sparse/sparse — candidate C's union-split (total block removal + read/write
    alignment via sigma). Fine level is UNTIED dense-read/sparse-write — candidate B's decoupled
    commitment in-block (sharp fine writes = sparse x sparse folded events; dense in-block reads =
    no fine dead-ends). Only the ungated (kernel='rla') spelling survives E4."""
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, decay=_decay_from_kernel(kernel), state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='sparse', write='sparse'),
            LevelRoutingConfig(tied=False, read='dense', write='sparse'),
        ],
    )


def _hybrid_tiedtop_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'global') -> dict:
    """RoLA-base-hybrid-tiedtop-global-rla: the canonical D-vs-tied-top ablation of
    RoLA-base-hybrid-global-rla (mirrors `rola_bench.lm.arms` ROLA_BASE_HYBRID_TIEDTOP_MIX
    byte-for-byte, user decision 2026-07-11). Coarse level is TIED sparse/sparse (candidate A's
    shared-support combo, the sigma≡1/2 special case of union-split) instead of the hybrid arm's
    UNTIED sparse/sparse (union-split). Fine level is identical to the hybrid arm (untied
    dense-read/sparse-write). This isolates whether block-level sigma-untying earns its parameters."""
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, decay=_decay_from_kernel(kernel), state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=True, read='sparse', write='sparse'),
            LevelRoutingConfig(tied=False, read='dense', write='sparse'),
        ],
    )


NAMED_CONFIGS: dict[str, NamedConfig] = {
    # ARM 1 of the campaign's three-arm ablation (prereqs Q3 §S1): the flagship base mix on the live
    # vocabulary. E2 moved it off state_norm='kappa'; kernel='rla' was always its defining knob.
    'RoLA-base-rla': NamedConfig(
        'RoLA-base-rla', build_kwargs=lambda nc, conv: _base_kwargs(nc, conv, 'rla', 'global')),
    # ARM 2: the candidate-C UNION-SPLIT cell — see `_union_kwargs`. Completes the coarse A⊂C
    # ablation triple (base = untied sparse-write/dense-read; union = untied sparse/sparse with the
    # sum/diff bijection — asymmetry earned per-state, not assumed). E3 moved it off 'gla_scalar' to
    # the ungated 'rla', matching the other two arms; E7 confirmed its routing needs no edit.
    'RoLA-base-union-global': NamedConfig(
        'RoLA-base-union-global', build_kwargs=lambda nc, conv: _union_kwargs(nc, conv, 'rla', 'global')),
    # ARM 3: the LEVEL-SPLIT DUTY cell — see `_levelsplit_kwargs`. Each level carries one sparse duty
    # and one dense duty, opposite across levels (coarse sparse-write/dense-read, fine
    # dense-write/sparse-read), every duty solving the exact entmax/softmax on its OWN logits.
    # E4 moved it off the dead 'gla_static' onto 'rla'. NOT in any *_SUITE_CONFIGS tuple — the
    # published slate is a coordinator decision.
    'RoLA-base-levelsplit-global': NamedConfig(
        'RoLA-base-levelsplit-global',
        build_kwargs=lambda nc, conv: _levelsplit_kwargs(nc, conv, 'rla', 'global')),
    # The alpha-choice ablation of ARM 3: identical level-split duty assignment, sparsemax (alpha=2)
    # on every sparse WRITE duty and the softer alpha=1.5 on every sparse READ duty. See
    # `_levelsplit_a2w_kwargs`.
    'RoLA-base-levelsplit-a2w-global': NamedConfig(
        'RoLA-base-levelsplit-a2w-global',
        build_kwargs=lambda nc, conv: _levelsplit_a2w_kwargs(nc, conv, 'rla', 'global')),
    # The DECAY-ON counterpart of ARM 1 — see `_base_kwargs` for the routing (it is byte-identical to
    # 'RoLA-base-rla'; only the clock differs). This is the arm the paper's no-decay ablation is an
    # ablation OF, and it is the ONLY decay-on cell on the slate.
    'RoLA-base-massdecay-global': NamedConfig(
        'RoLA-base-massdecay-global',
        build_kwargs=lambda nc, conv: _base_kwargs(nc, conv, 'gla_mass_detached', 'global')),
    # The candidate-D depth-hybrid cell and its D-vs-tied-top ablation. Only the ungated ('rla')
    # spellings survive: their 'gla_static' siblings are retired by E4.
    'RoLA-base-hybrid-global-rla': NamedConfig(
        'RoLA-base-hybrid-global-rla', build_kwargs=lambda nc, conv: _hybrid_kwargs(nc, conv, 'rla', 'global')),
    'RoLA-base-hybrid-tiedtop-global-rla': NamedConfig(
        'RoLA-base-hybrid-tiedtop-global-rla', build_kwargs=lambda nc, conv: _hybrid_tiedtop_kwargs(nc, conv, 'rla', 'global')),
}

# RETIRED named configs (campaign-prereqs E2/E3/E4, 2026-08-01). These are not "needs edit" — the
# knob that DEFINED each of them does not exist in the V3 architecture any more, so re-pinning them
# to a live knob would silently publish a different cell under an old name. `construct()` refuses
# them by name with the reason below.
RETIRED_CONFIGS: dict[str, str] = {
    'RoLA-base':
        "E3: defined by kernel='gla_scalar' (per-token scalar decay), retired vocabulary in every "
        "engine checkout. Its routing mix survives as 'RoLA-base-rla'.",
    'RoLA-base-global':
        "E3: kernel='gla_scalar'. Its state_norm='global' + base mix survives as 'RoLA-base-rla'.",
    'RoLA-tied':
        "E3: kernel='gla_scalar' (and E2: state_norm='kappa'). The tied all-sparse perf cell has no "
        "live-vocabulary replacement pinned; re-registering one is a coordinator decision.",
    'RoLA-base-sym':
        "E3: kernel='gla_scalar' (and E2: state_norm='kappa').",
    'RoLA-base-sym-global':
        "E3: kernel='gla_scalar'. Its destroy-kappa hypothesis is moot — E2 retired kappa outright.",
    'RoLA-base-hybrid-global':
        "E4: defined as the GATED sibling via kernel='gla_static', which the V3 engine refuses by "
        "name (non-detached clock, no V3 expression). Ungated sibling: "
        "'RoLA-base-hybrid-global-rla'.",
    'RoLA-base-hybrid-tiedtop-global':
        "E4: kernel='gla_static'. Ungated sibling: 'RoLA-base-hybrid-tiedtop-global-rla'.",
    'RoLA-multiscale':
        "E4: the production STATIC-decay cell — kernel='gla_static' was its entire definition, and "
        "V3 has no non-detached clock. A detached-clock cell IS a different experiment; it is now "
        "registered separately as 'RoLA-base-massdecay-global' (kernel='gla_mass_detached'), on the "
        "base routing mix rather than multiscale's.",
}

# Suite wiring, survivors only (every retired name removed by E2/E3/E4, 2026-08-01).
# 2026-08-02: ARM 3 (level-split) joins the LM slate so all THREE structural wirings of the paper's
# central ablation are LM-runnable, together with its alpha ablation and the decay-ON arm.
LM_SUITE_CONFIGS = ('RoLA-base-rla', 'RoLA-base-union-global', 'RoLA-base-levelsplit-global',
                    'RoLA-base-levelsplit-a2w-global', 'RoLA-base-massdecay-global')
MQAR_SUITE_CONFIGS = ('RoLA-base-rla', 'RoLA-base-union-global', 'RoLA-base-hybrid-global-rla',
                      'RoLA-base-hybrid-tiedtop-global-rla')   # use_short_conv forced False regardless of preset
PERF_SUITE_CONFIGS = ('RoLA-base-rla',)


def construct(name: str, *, hidden_size: int, num_heads: int, head_v_dim: int,
              nc: int = 256, use_short_conv: bool | None = None, layer_idx: int | None = 0, **extra):
    """Build one named config and fail loudly if the engine cannot resolve its production routing.

    E1: there is no `head_k_dim` / `d_qk` / `qk_norm` parameter — the V3 layer owns no Q/K surface.
    """
    from fla_rola.layers import RoLA
    if name in RETIRED_CONFIGS:
        raise ValueError(f"named config {name!r} is RETIRED: {RETIRED_CONFIGS[name]}")
    cfg = NAMED_CONFIGS[name]
    conv = True if use_short_conv is None else use_short_conv
    kwargs = cfg.build_kwargs(nc, conv)
    layer = RoLA(hidden_size=hidden_size, num_heads=num_heads,
                 head_v_dim=head_v_dim, layer_idx=layer_idx, **kwargs, **extra)
    err = getattr(layer, '_routing_resolution_error', None)
    if err is not None:
        raise err
    return layer, name
