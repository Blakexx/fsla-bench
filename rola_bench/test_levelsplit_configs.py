"""Gates for the LEVEL-SPLIT DUTY arm (`RoLA-base-levelsplit-global`), 2026-08-01.

The level-split arm gives every level one sparse duty and one dense duty, opposite across levels
(coarse sparse-write/dense-read, fine dense-write/sparse-read), each duty solving the EXACT
entmax/softmax on its OWN projected logits. See `rola_bench.named_configs._levelsplit_kwargs`.

Everything here runs on CPU in fp64 against the ENGINE's own canonical references
(`fla_rola.ops.rola.routing.reference.routing_factor_levels` and
`fla_rola.ops.rola.naive.naive_rola_routed`) — never against a freshly written reimplementation
(memory: `matching-the-naive-antipattern`). The engine's CUDA consumer is forward-only
(`fla_rola/ops/rola/cuda/chunk.py::_RoLASharedGramForwardOnlyFn.backward` raises), so the
forward+backward gate necessarily exercises the differentiable canonical oracle, which is the only
implemented training path for this engine.

Declared tolerances and their derivation:

* ``ENTMAX_SOLVE_TOL = 1e-12``. entmax's exact solve is a bisection on the threshold tau run for
  ``_ROUTING_REFERENCE_N_ITER = 80`` iterations; the bracket halves each iteration, so the residual
  is O(2**-80) of the initial logit range — many orders below fp64 eps (2.2e-16) scaled by the
  O(1) logit ranges used here. 1e-12 is therefore a slack-loaded bound, not a fitted one.
* Off-support entries are asserted EXACTLY zero: entmax clamps below the threshold to zero, so
  these are representable zeros, not small numbers.
* The dense-degenerate equivalence is asserted at ``atol=0.0`` (bit-identical): both spellings
  reach the same ``torch.softmax`` call on the same logits, so any nonzero difference is a defect,
  not numerics.
"""

import pytest
import torch

from rola_bench import named_configs

ENTMAX_SOLVE_TOL = 1e-12


def _mix_rows(configs):
    return [(cfg.tied, cfg.read, cfg.write) for cfg in configs]


def _resolved_routing(kwargs):
    """Resolve a named-config kwargs dict to the ops-internal ResolvedRouting, exactly as the
    layer's own constructor does (no reimplementation of the mapping)."""
    from fla_rola.layers.rola import _resolve_level_routing
    from fla_rola.ops.rola.routing.topology import resolve_b_spec
    from fla_rola.ops.rola.routing.types import ResolvedRouting

    branches = resolve_b_spec(kwargs['states_per_level'], kwargs['levels_per_head'])
    return ResolvedRouting(
        levels=tuple(_resolve_level_routing(cfg) for cfg in kwargs['level_routing_configs']),
        branches=tuple(branches),
    )


# --------------------------------------------------------------------------------------------
# config surface
# --------------------------------------------------------------------------------------------

def test_levelsplit_named_config_is_registered_with_opposite_per_level_duties():
    name = 'RoLA-base-levelsplit-global'
    assert name in named_configs.NAMED_CONFIGS
    kwargs = named_configs.NAMED_CONFIGS[name].build_kwargs(256, False)
    assert kwargs['state_norm'] == 'global'
    assert kwargs['router_bias'] is True
    assert kwargs['use_short_conv'] is False
    # coarse: sparse WRITE / dense READ. fine: dense WRITE / sparse READ. Untied everywhere.
    assert _mix_rows(kwargs['level_routing_configs']) == [
        (False, 'dense', 'sparse'),
        (False, 'sparse', 'dense'),
    ]


def test_levelsplit_kernel_is_live_vocabulary():
    """'gla_scalar' is retired in every current fla_rola checkout; this arm must be constructible."""
    kwargs = named_configs.NAMED_CONFIGS['RoLA-base-levelsplit-global'].build_kwargs(256, False)
    from fla_rola.ops.rola.internal.decay import PUBLIC_DECAY_KERNELS
    assert kwargs['kernel'] in PUBLIC_DECAY_KERNELS


def test_levelsplit_matches_the_union_arm_shape_and_router_parameter_count():
    """Matched addressing entropy: same topology, same bias/norm/conv, same untied-everywhere
    router parameter shape. Only the per-side activation assignment differs."""
    union = named_configs._union_kwargs(256, False, 'gla_static', 'global')
    split = named_configs._levelsplit_kwargs(256, False, 'gla_static', 'global')
    for key in ('levels_per_head', 'states_per_level', 'router_bias', 'state_norm',
                'use_short_conv', 'kernel'):
        assert union[key] == split[key], key
    # untied on every level in both arms => identical packed-projection logit-set count
    assert [cfg.tied for cfg in union['level_routing_configs']] == [False, False]
    assert [cfg.tied for cfg in split['level_routing_configs']] == [False, False]


def test_levelsplit_resolves_to_independent_routing_never_union():
    """The engine flag the arm drives: every duty is its own exact solve on its own logits.
    No level may resolve to UnionRouting (shared/averaged logits) — the whole point of the arm."""
    from fla_rola.ops.rola.routing.types import (
        EntmaxActivation, IndependentRouting, SoftmaxActivation, UnionRouting)

    routing = _resolved_routing(named_configs._levelsplit_kwargs(256, False, 'gla_static', 'global'))
    coarse, fine = routing.levels
    assert type(coarse) is IndependentRouting and type(fine) is IndependentRouting
    assert not isinstance(coarse, UnionRouting) and not isinstance(fine, UnionRouting)
    assert type(coarse.read_activation) is SoftmaxActivation
    assert type(coarse.write_activation) is EntmaxActivation
    assert type(fine.read_activation) is EntmaxActivation
    assert type(fine.write_activation) is SoftmaxActivation
    # untied + not shares_solve => read and write are solved separately, from separate logits
    assert coarse.shares_solve is False and fine.shares_solve is False

    union = _resolved_routing(named_configs._union_kwargs(256, False, 'gla_static', 'global'))
    assert type(union.levels[0]) is UnionRouting   # the contrast case


# --------------------------------------------------------------------------------------------
# (a) CORNER PURITY -- the theoretical property union-split structurally lacks
# --------------------------------------------------------------------------------------------

def _corner_logits(width, *, write_corner, read_corner, gap=3.0, device='cpu'):
    """One token's [1,1,D,width] read/write logits with each side peaked on a different index."""
    read = torch.zeros(1, 1, 2, width, dtype=torch.float64, device=device)
    write = torch.zeros(1, 1, 2, width, dtype=torch.float64, device=device)
    for level in range(2):
        write[0, 0, level, write_corner] = gap
        read[0, 0, level, read_corner] = gap
    return read, write


def test_levelsplit_sparse_write_duty_is_an_exact_point_mass_at_the_corner():
    """The write duty owns its own entmax, so a corner write logit gives an EXACT delta —
    regardless of where the read logits point."""
    from fla_rola.ops.rola.routing.reference import routing_factor_levels

    width = 16
    kwargs = named_configs._levelsplit_kwargs(width * width, False, 'gla_static', 'global')
    routing = _resolved_routing(kwargs)
    read_logits, write_logits = _corner_logits(width, write_corner=3, read_corner=11)
    reads, writes = routing_factor_levels(read_logits, write_logits, routing)

    # coarse level (index 0) carries the SPARSE WRITE duty
    w = writes[0][0, 0]
    assert w[3] == pytest.approx(1.0, abs=ENTMAX_SOLVE_TOL)
    off = torch.cat((w[:3], w[4:]))
    assert torch.equal(off, torch.zeros_like(off)), f"off-support write mass {off.abs().max()}"

    # fine level (index 1) carries the SPARSE READ duty -> exact delta on the READ corner
    r = reads[1][0, 0]
    assert r[11] == pytest.approx(1.0, abs=ENTMAX_SOLVE_TOL)
    off_r = torch.cat((r[:11], r[12:]))
    assert torch.equal(off_r, torch.zeros_like(off_r))


def test_union_split_write_cannot_reach_the_corner_the_level_split_write_reaches():
    """THE contrast. On identical logits, union-split's coarse write is confined to (and
    renormalized over) the shared support of the AVERAGED logits 0.5*z_r+0.5*z_w, so a read logit
    pointing elsewhere keeps the write off its own corner. Level-split's write is exactly there."""
    from fla_rola.ops.rola.routing.reference import routing_factor_levels

    width = 16
    read_logits, write_logits = _corner_logits(width, write_corner=3, read_corner=11)

    split = _resolved_routing(named_configs._levelsplit_kwargs(width * width, False, 'gla_static', 'global'))
    union = _resolved_routing(named_configs._union_kwargs(width * width, False, 'gla_static', 'global'))

    _, split_writes = routing_factor_levels(read_logits, write_logits, split)
    _, union_writes = routing_factor_levels(read_logits, write_logits, union)

    w_split = split_writes[0][0, 0]
    w_union = union_writes[0][0, 0]

    # level-split: exact point mass -- one nonzero, equal to 1, every other entry a hard zero.
    assert int((w_split > 0).sum()) == 1
    assert w_split[3] == pytest.approx(1.0, abs=ENTMAX_SOLVE_TOL)
    assert w_split[11] == 0.0

    # union-split: the shared support of 0.5*z_r+0.5*z_w is {3, 11}, and the write is a softmax
    # renormalized OVER that support, so it is strictly positive on the read corner and is not a
    # point mass. Read back at gap=3.0: w_union[3] ~ 0.9526, w_union[11] ~ 4.74e-2.
    assert int((w_union > 0).sum()) >= 2, "union write should spread over the shared support"
    assert 0.9 < float(w_union[3]) < 0.99
    assert float(w_union[11]) > 1e-3

    # The contamination is structural, not a small-gap artifact: no matter how hard the write
    # logits push, union keeps strictly positive mass on the read corner while level-split holds
    # an exact zero there.
    for gap in (3.0, 5.0, 10.0, 40.0):
        r, w = _corner_logits(width, write_corner=3, read_corner=11, gap=gap)
        _, sw = routing_factor_levels(r, w, split)
        _, uw = routing_factor_levels(r, w, union)
        assert float(uw[0][0, 0][11]) > 0.0, gap
        assert float(sw[0][0, 0][11]) == 0.0, gap


def test_levelsplit_write_purity_is_independent_of_the_read_logits():
    """Sweep the read corner across the level width: the sparse write duty stays an exact delta."""
    from fla_rola.ops.rola.routing.reference import routing_factor_levels

    width = 8
    routing = _resolved_routing(
        named_configs._levelsplit_kwargs(width * width, False, 'gla_static', 'global'))
    for read_corner in range(width):
        read_logits, write_logits = _corner_logits(width, write_corner=5, read_corner=read_corner)
        _, writes = routing_factor_levels(read_logits, write_logits, routing)
        w = writes[0][0, 0]
        assert w[5] == pytest.approx(1.0, abs=ENTMAX_SOLVE_TOL), read_corner
        off = torch.cat((w[:5], w[6:]))
        assert torch.equal(off, torch.zeros_like(off)), read_corner


# --------------------------------------------------------------------------------------------
# (c) DEGENERATE CASE: both duties dense == plain untied dense/dense
# --------------------------------------------------------------------------------------------

def test_levelsplit_dense_degenerate_equals_plain_dense_routing_bit_for_bit():
    from fla_rola.ops.rola.routing.reference import routing_factor_levels
    from fla_rola.layers.rola import LevelRoutingConfig
    from fla_rola.ops.rola.routing.types import ResolvedRouting, IndependentRouting, SoftmaxActivation

    width = 16
    degenerate = named_configs._levelsplit_dense_degenerate_kwargs(
        width * width, False, 'gla_static', 'global')
    assert _mix_rows(degenerate['level_routing_configs']) == [
        (False, 'dense', 'dense'), (False, 'dense', 'dense')]
    degenerate_routing = _resolved_routing(degenerate)

    # the reference "base dense arm": plain untied dense/dense on the same topology, built from the
    # ops vocabulary directly rather than from the level-split builder.
    dense_routing = ResolvedRouting(
        levels=tuple(
            IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                               write_activation=SoftmaxActivation())
            for _ in range(2)),
        branches=(width, width),
    )
    assert degenerate_routing == dense_routing

    torch.manual_seed(0)
    read_logits = torch.randn(2, 7, 2, width, dtype=torch.float64)
    write_logits = torch.randn(2, 7, 2, width, dtype=torch.float64)
    got_r, got_w = routing_factor_levels(read_logits, write_logits, degenerate_routing)
    ref_r, ref_w = routing_factor_levels(read_logits, write_logits, dense_routing)
    for level in range(2):
        assert torch.equal(got_r[level], ref_r[level])   # atol = 0.0, see module docstring
        assert torch.equal(got_w[level], ref_w[level])
    _ = LevelRoutingConfig  # imported for the vocabulary assertion above


# --------------------------------------------------------------------------------------------
# (b) forward + backward step, against the engine's differentiable canonical oracle
# --------------------------------------------------------------------------------------------

def _run_reference_step(kwargs, *, B=1, T=6, H=2, K=4, V=4, dm=8, seed=0):
    """One forward+backward through `naive_rola_routed`, the engine's canonical fp64 oracle and
    its only differentiable path (the CUDA consumer is forward-only)."""
    from fla_rola.ops.rola.naive import naive_rola_routed

    routing = _resolved_routing(kwargs)
    torch.manual_seed(seed)
    q = torch.rand(B, T, H, K, dtype=torch.float64).add_(0.5).requires_grad_(True)
    k = torch.rand(B, T, H, K, dtype=torch.float64).add_(0.5).requires_grad_(True)
    v = torch.randn(B, T, H, V, dtype=torch.float64, requires_grad=True)
    h = torch.randn(B, T, dm, dtype=torch.float64, requires_grad=True)
    route_W = torch.randn(
        H, routing.packed_slot_count, dm, routing.real_width,
        dtype=torch.float64).mul_(0.5).requires_grad_(True)
    theta = None
    if kwargs['kernel'] != 'rla':
        theta = torch.zeros(H, sum(routing.branches), dtype=torch.float64).requires_grad_(True)
    out, _ = naive_rola_routed(
        q, k, v, h, route_W, routing, norm=kwargs['state_norm'], kernel=kwargs['kernel'],
        decay_theta=theta)
    assert torch.isfinite(out).all()
    out.sum().backward()
    return out, dict(q=q, k=k, v=v, h=h, route_W=route_W, theta=theta)


def test_levelsplit_forward_and_backward_run_without_error():
    kwargs = named_configs.NAMED_CONFIGS['RoLA-base-levelsplit-global'].build_kwargs(16, False)
    out, tensors = _run_reference_step(kwargs)
    assert out.shape == (1, 6, 2, 4)
    for name, t in tensors.items():
        if t is None:
            continue
        assert t.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(t.grad).all(), f"{name} gradient is not finite"
    # the sparse duties must actually train: routing gradient must be nonzero somewhere
    assert tensors['route_W'].grad.abs().max() > 0


def test_levelsplit_runs_at_the_union_arms_pinned_routing_width():
    """The union arm's pinned MQAR shape is nc=256 -> (D=2, b=16). The recurrence oracle is a
    Python loop over nc, so the RECURRENCE gate above runs narrow; this gate runs the arm's real
    routing width through the same producer reference the layer uses."""
    from fla_rola.ops.rola.routing.reference import routing_factor_levels

    kwargs = named_configs.NAMED_CONFIGS['RoLA-base-levelsplit-global'].build_kwargs(256, False)
    assert (kwargs['levels_per_head'], kwargs['states_per_level']) == (2, 16)
    routing = _resolved_routing(kwargs)
    torch.manual_seed(1)
    read_logits = torch.randn(2, 64, 2, 16, dtype=torch.float64, requires_grad=True)
    write_logits = torch.randn(2, 64, 2, 16, dtype=torch.float64, requires_grad=True)
    reads, writes = routing_factor_levels(read_logits, write_logits, routing)
    for level in range(2):
        for side in (reads[level], writes[level]):
            assert torch.allclose(side.sum(-1), torch.ones_like(side.sum(-1)), atol=ENTMAX_SOLVE_TOL)
            assert (side >= 0).all()
    (sum(r.sum() for r in reads) + sum(w.sum() for w in writes)).backward()
    assert read_logits.grad is not None and write_logits.grad is not None
    assert torch.isfinite(read_logits.grad).all() and torch.isfinite(write_logits.grad).all()
