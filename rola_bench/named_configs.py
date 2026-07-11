"""Named RoLA configs — the pinned, PUBLISHED-flagship architecture the suites default to (user
decision, 2026-07-08). These are `fla_rola.layers.RoLA` DIRECT construction kwargs (levels_per_head /
states_per_level / level_routing_configs — the current #config-refactor surface), not the legacy
zoology-mixer vocabulary.

**RoLA-base** (the flagship): router_bias=True, kernel='gla_scalar', use_short_conv=True, SQUARE
topology (levels_per_head=2, states_per_level=sqrt(nc), nc=256 default), UNTIED, per-LEVEL
heterogeneous routing — write is SPARSE at level 0 (coarse) + DENSE at level 1 (fine), read is DENSE
at both levels:
    level_routing_configs=[LevelRoutingConfig(tied=False, read='dense', write='sparse'),
                            LevelRoutingConfig(tied=False, read='dense', write='dense')]
Rationale: coarse write-zeros map 1:1 onto skippable state-blocks (all the §7 skip perf lives here),
dense-fine keeps within-region blending, dense read is dead-end-free by construction (no untied
double-sparse gate to trip, ROLA_SEMANTICS.md §5 — dead-leaf reads are inert anyway).

**RoLA-base-rla**: RoLA-base with kernel='rla' (the LM ablation variant — no decay).

All published variants use **state_norm='kappa'** EXPLICITLY (never the layer's own default — set
here so a future default change can't silently drift the pinned configs). Rationale: κ starts ≈global
(init bias −4, ROLA_SEMANTICS.md §4) and learns the global↔per-state interpolation per token; RoLA-base's
sparse WRITE routing concentrates state mass onto fewer leaves, which is exactly the regime κ>0 pays
for (recall from lightly-written states that a pure global pooled readout would drown out).

**RoLA-base-global**: RoLA-base with `state_norm='global'` (κ≡0 fixed) — the kappa-vs-global ablation
row for the paper, otherwise identical to RoLA-base.

**MQAR variants**: RoLA-base(-rla) minus short conv (`use_short_conv=False`) — convs confound
short-offset recall in MQAR; everything else identical.

**RoLA-tied** (perf suite's max-perf variant): tied=True, read=write='sparse' at every level — the

Suite wiring (per the coordinator's addition):
  * LM suite     : RoLA-base, RoLA-base-rla, RoLA-base-global, RoLA-base-union-global, and
                   RoLA-multiscale (static hard-support decay, [8,16], global norm).
  * MQAR suite   : the conv-free variants of RoLA-base, RoLA-base-rla, plus RoLA-base-sym (the
                    symmetric-coarse ablation, MQAR flagship addition 2026-07-09 -- sym-on-MQAR is
                    the informative cell for coarse tying: recall is where read/write routing
                    asymmetry should matter most -- sym≈base ⇒ max-skip is free; sym≪base ⇒ tying
                    costs recall specifically), plus RoLA-base-sym-global (destroy-kappa hypothesis
                    cell, 2026-07-09 -- sym+global vs base-kappa isolates whether kappa/per_state
                    were only compensating for read-mass imbalance that tied-sparse-coarse already
                    removes structurally), plus RoLA-base-union-global (candidate-C union-split
                    cell, 2026-07-10 -- coarse level UNTIED sparse/sparse, the engine's union-split
                    realization of that previously-rejected combo: shared entmax support from
                    z_r+z_w, sigmoid role split -- completes the coarse A⊂C ablation triple with
                    RoLA-base (untied sparse-write/dense-read) and RoLA-base-sym (tied sparse/sparse
                    = the sigma≡1/2 special case)).
  * Perf suite   : RoLA-base, RoLA-tied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


def _square_topology(nc: int) -> tuple[int, int]:
    b = int(round(math.sqrt(nc)))
    if b * b != nc:
        raise ValueError(f"square topology requires a perfect-square nc, got {nc}")
    return 2, b


@dataclass(frozen=True)
class NamedConfig:
    name: str
    build_kwargs: Callable[[int, bool], dict]   # (nc, use_short_conv) -> RoLA(**kwargs) (minus hidden/heads/dims)


def _base_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'kappa') -> dict:
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, kernel=kernel, state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='dense', write='sparse'),
            LevelRoutingConfig(tied=False, read='dense', write='dense'),
        ],
    )


def _tied_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'kappa') -> dict:
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, kernel=kernel, state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=LevelRoutingConfig(tied=True, read='sparse', write='sparse'),
    )


def _sym_kwargs(nc: int, use_short_conv: bool, kernel: str, state_norm: str = 'kappa') -> dict:
    """RoLA-base-sym: the SYMMETRIC-COARSE ablation of RoLA-base (mirrors `rola_bench.lm.arms`
    ROLA_BASE_SYM_MIX byte-for-byte). Coarse level is TIED sparse/sparse (shared read/write support
    -- still dead-end-safe: identical logits + identical activation ⇒ identical support, the
    trainability-safe double-sparse form) = the MAX-SKIP cell: shared-support state blocks are never
    loaded on either side. Fine level stays untied dense/dense, same as RoLA-base."""
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, kernel=kernel, state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=True, read='sparse', write='sparse'),
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
    renormalized; dag_tasks/C-union-split.md). Fine level stays untied dense/dense, same as
    RoLA-base."""
    from fla_rola.layers.rola import LevelRoutingConfig
    D, b = _square_topology(nc)
    return dict(
        levels_per_head=D, states_per_level=b, kernel=kernel, state_norm=state_norm,
        router_bias=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='sparse', write='sparse'),
            LevelRoutingConfig(tied=False, read='dense', write='dense'),
        ],
    )


def _multiscale_kwargs(nc: int, use_short_conv: bool) -> dict:
    """Production static-decay cell: required [8,16] topology, with no substitute."""
    from fla_rola.layers.rola import LevelRoutingConfig
    if nc != 128:
        raise ValueError(f"RoLA-multiscale requires mixed-radix nc=128 ([8,16]), got nc={nc}")
    return dict(
        levels_per_head=2, states_per_level=[8, 16], kernel='gla_static', state_norm='global',
        router_bias=True, qk_norm=True, use_short_conv=use_short_conv,
        level_routing_configs=[
            LevelRoutingConfig(tied=False, read='dense', write='sparse'),
            LevelRoutingConfig(tied=False, read='dense', write='dense'),
        ],
    )


NAMED_CONFIGS: dict[str, NamedConfig] = {
    'RoLA-base': NamedConfig(
        'RoLA-base', build_kwargs=lambda nc, conv: _base_kwargs(nc, conv, 'gla_scalar', 'kappa')),
    'RoLA-base-rla': NamedConfig(
        'RoLA-base-rla', build_kwargs=lambda nc, conv: _base_kwargs(nc, conv, 'rla', 'kappa')),
    'RoLA-base-global': NamedConfig(
        # kappa-vs-global ablation row: identical to RoLA-base except state_norm='global' (kappa≡0).
        'RoLA-base-global', build_kwargs=lambda nc, conv: _base_kwargs(nc, conv, 'gla_scalar', 'global')),
    'RoLA-tied': NamedConfig(
        'RoLA-tied', build_kwargs=lambda nc, conv: _tied_kwargs(nc, conv, 'gla_scalar', 'kappa')),
    # RoLA-base-sym (MQAR flagship addition, 2026-07-09): the symmetric-coarse ablation pair with
    # RoLA-base -- see `_sym_kwargs` and `rola_bench.lm.arms.ROLA_BASE_SYM_MIX` (same mix, LM side).
    # sym-on-MQAR is the informative cell for coarse tying: recall is where read/write routing
    # asymmetry should matter most -- sym≈base ⇒ max-skip is free; sym≪base ⇒ tying costs recall
    # specifically. Per-level heterogeneous activation is a required production capability.
    'RoLA-base-sym': NamedConfig(
        'RoLA-base-sym', build_kwargs=lambda nc, conv: _sym_kwargs(nc, conv, 'gla_scalar', 'kappa')),
    # RoLA-base-sym-global (MQAR addition, 2026-07-09): RoLA-base-sym with state_norm='global'
    # (kappa≡0). THE decisive cell for the destroy-kappa hypothesis (user 2026-07-09): kappa/per_state
    # were added to close the sym<->asym gap by repairing read-mass imbalance; tied-sparse-coarse
    # prevents the imbalance structurally, so sym+global should match base-kappa if the hypothesis
    # holds -- then the RESCALED norms (and all their precision/perf machinery) can be deleted.
    'RoLA-base-sym-global': NamedConfig(
        'RoLA-base-sym-global', build_kwargs=lambda nc, conv: _sym_kwargs(nc, conv, 'gla_scalar', 'global')),
    # RoLA-base-union-global (MQAR addition, 2026-07-10): the candidate-C UNION-SPLIT cell — see
    # `_union_kwargs`. Completes the coarse A⊂C ablation triple on MQAR (base = untied
    # sparse-write/dense-read; sym = tied sparse/sparse, the sigma≡1/2 special case; union = untied
    # sparse/sparse with the sum/diff bijection — asymmetry earned per-state, not assumed). Global
    # norm (kappa≡0) is the primary cell per the C acceptance criteria.
    'RoLA-base-union-global': NamedConfig(
        'RoLA-base-union-global', build_kwargs=lambda nc, conv: _union_kwargs(nc, conv, 'gla_scalar', 'global')),
    'RoLA-multiscale': NamedConfig(
        'RoLA-multiscale', build_kwargs=_multiscale_kwargs),
}

# Suite wiring (coordinator addition, 2026-07-08; RoLA-base-sym added to MQAR 2026-07-09;
# RoLA-base-sym-global added to MQAR 2026-07-09, destroy-kappa hypothesis cell;
# RoLA-base-union-global added to MQAR 2026-07-10, candidate-C union-split cell).
LM_SUITE_CONFIGS = ('RoLA-base', 'RoLA-base-rla', 'RoLA-base-global', 'RoLA-base-union-global', 'RoLA-multiscale')
MQAR_SUITE_CONFIGS = ('RoLA-base', 'RoLA-base-rla', 'RoLA-base-sym', 'RoLA-base-sym-global', 'RoLA-base-union-global')   # use_short_conv forced False regardless of preset
PERF_SUITE_CONFIGS = ('RoLA-base', 'RoLA-tied')


def construct(name: str, *, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
              nc: int = 256, use_short_conv: bool | None = None, layer_idx: int | None = 0, **extra):
    """Build one named config and fail loudly if the engine cannot resolve its production routing."""
    from fla_rola.layers import RoLA
    cfg = NAMED_CONFIGS[name]
    conv = True if use_short_conv is None else use_short_conv
    kwargs = cfg.build_kwargs(nc, conv)
    layer = RoLA(hidden_size=hidden_size, num_heads=num_heads, head_k_dim=head_k_dim,
                 head_v_dim=head_v_dim, layer_idx=layer_idx, **kwargs, **extra)
    err = getattr(layer, '_routing_resolution_error', None)
    if err is not None:
        raise err
    return layer, name
