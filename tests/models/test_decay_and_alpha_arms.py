"""Gates for the three arms added 2026-08-02: the DECAY-ON arm, the ALPHA-CHOICE arm, and ARM 3's
promotion onto the LM slate.

  * `RoLA-base-massdecay-global` / `rola-base-massdecay-global` — `kernel='gla_mass_detached'`
    (the V3 layer's `decay='leaf_mass'`) on ARM 1's routing byte-for-byte. Before it, every arm on
    the slate was `decay='none'` and the paper's no-decay ablation had no decay-ON counterpart.
    Semantics are normative in `development/plans/GLA_MASS_DECAY_SPEC.md`: compact AND rate over
    per-level branch dials (§3.1), DETACHED mass clock `c_s[t] = stop_gradient(w_s[t])` (§3.2),
    exponential survival `(1-rate_s)^c_s[t]` with undamped deposit (§3.3).
  * `RoLA-base-levelsplit-a2w-global` / `rola-base-levelsplit-a2w-global` — ARM 3's duty assignment
    with alpha=2 (sparsemax) on the sparse WRITE duty and the softer alpha=1.5 on the sparse READ
    duty. Construction-gated the same way `test_levelsplit_configs.py` gates ARM 3.
  * ARM 3 (`RoLA-base-levelsplit-global`) is now in `LM_SUITE_CONFIGS` and in the `lm/arms.py`
    registry, so all three structural wirings are LM-runnable.

Everything here is CPU: config-registry assertions, `LevelRoutingConfig` resolution against the
ENGINE's own `_resolve_level_routing` / `routing_factor_levels` (never a reimplementation, per
`matching-the-naive-antipattern`), and layer construction on CPU. No CUDA, no timing.

TRANSLATED onto the standalone engine's routing vocabulary on 2026-08-17 (the `fla_rola/ops/rola/`
hollow-out). The alpha cells keep their claims and restate them in the engine's terms: the routing
reference takes PACKED per-side logits `[BH, T, packed_logit_width]` sliced by
`ResolvedRouting.level_logit_slice` (the legacy one took a padded `[B, T, D, real_width]`), a level
object carries its own `width` and spells its duties `read`/`write`, and `ResolvedRouting` takes no
`branches=` argument. The DECAY cells above are unaffected: they were already producer/source-level
(registry surface, dial shape, dial dtype) rather than operator-level, and the operator refuses
decay on both the forward and the backward, so operator-level decay behaviour is not gateable here
in any vocabulary.

`ENTMAX_SOLVE_TOL = 1e-12`: the exact entmax solve is an 80-iteration bisection on the threshold,
residual O(2**-80) of the logit range — orders below fp64 eps. Off-support entries are asserted
EXACTLY zero (entmax clamps to a representable zero), and mix/alpha equality is asserted exactly
(these are declared config values, not computed ones).
"""

import pytest
import torch

from rola_bench import named_configs
from rola_bench.lm import arms

ENTMAX_SOLVE_TOL = 1e-12

DECAY_CONFIG = 'RoLA-base-massdecay-global'
DECAY_ARM = 'rola-base-massdecay-global'
A2W_CONFIG = 'RoLA-base-levelsplit-a2w-global'
A2W_ARM = 'rola-base-levelsplit-a2w-global'
SPLIT_CONFIG = 'RoLA-base-levelsplit-global'
SPLIT_ARM = 'rola-base-levelsplit-global'


def _mix_rows(configs):
    return [(cfg.tied, cfg.read, cfg.write) for cfg in configs]


def _dict_rows(mix):
    """The lm/arms.py serializable dict form, normalized to the same row tuple."""
    return [(m['tied'], m['read'], m['write']) for m in mix]


def _resolved_routing(kwargs):
    """The layer's own config -> engine mapping. The engine's level object carries its own
    `width`, so the widths are zipped INTO the levels rather than passed beside them."""
    from rola_bench.engine import _resolve_level_routing
    from rola_bench.engine import resolve_b_spec
    from rola_bench.engine import ResolvedRouting

    branches = resolve_b_spec(kwargs['states_per_level'], kwargs['levels_per_head'])
    return ResolvedRouting(levels=tuple(
        _resolve_level_routing(cfg, int(width))
        for width, cfg in zip(branches, kwargs['level_routing_configs'])))


def _packed_logits(routing, per_level):
    """Pack one per-level row into the reference's `[BH=1, T=1, packed_logit_width]` layout.

    `per_level` is a 1-D tensor of length `width` applied to EVERY level's own column span; the
    span comes from `ResolvedRouting.level_logit_slice`, the engine's single definition of the
    packed layout, so no offset arithmetic is restated here.
    """
    packed = torch.zeros(1, 1, routing.packed_logit_width, dtype=torch.float64)
    for level in range(routing.D):
        packed[0, 0, routing.level_logit_slice(level)] = per_level
    return packed


# =============================================================================================
# THE DECAY-ON ARM
# =============================================================================================

def test_decay_arm_is_registered_and_speaks_the_v3_decay_surface():
    assert DECAY_CONFIG in named_configs.NAMED_CONFIGS
    kwargs = named_configs.NAMED_CONFIGS[DECAY_CONFIG].build_kwargs(256, False)
    assert 'kernel' not in kwargs                      # E3/E4: resolved at the boundary
    assert kwargs['decay'] == 'leaf_mass'              # gla_mass_detached -> the V3 layer's kwarg
    assert kwargs['state_norm'] == 'global'


def test_decay_arm_differs_from_arm_1_in_the_CLOCK_AND_NOTHING_ELSE():
    """This is what makes the paper's no-decay ablation an ABLATION: same routing, same topology,
    same bias/norm/conv — one knob."""
    base = named_configs.NAMED_CONFIGS['RoLA-base-rla'].build_kwargs(256, True)
    decay = named_configs.NAMED_CONFIGS[DECAY_CONFIG].build_kwargs(256, True)
    assert set(base) == set(decay)
    for key in base:
        if key == 'decay':
            continue
        assert base[key] == decay[key], key
    assert base['decay'] == 'none' and decay['decay'] == 'leaf_mass'


def test_decay_arm_constructs_with_COMPACT_dials_not_a_per_state_table():
    """GLA_MASS_DECAY_SPEC §3.1/§5.4: decay parameters are `O(H * sum_l b_l)` — one theta per level
    BRANCH — and explicitly NEVER an `[H, nc]` table. Read back off the constructed layer.
    """
    layer, _ = named_configs.construct(
        DECAY_CONFIG, hidden_size=256, num_heads=8, head_v_dim=32, nc=16, use_short_conv=False)
    assert layer.decay_kind == 'leaf_mass'
    widths = tuple(layer.topology.widths)
    assert [tuple(t.shape) for t in layer.decay_theta] == [(8, b) for b in widths]
    n_dials = sum(t.numel() for t in layer.decay_theta)
    assert n_dials == 8 * sum(widths)                 # O(H * sum_l b_l)
    # compactness is the CLAIM, so state it as one: the dials must be strictly cheaper than the
    # [H, nc] per-state table the spec forbids. (nc = prod_l b_l = 16 here, sum_l b_l = 8.)
    nc = 1
    for b in widths:
        nc *= b
    assert n_dials < 8 * nc, f"{n_dials} dials is not compact against an [H={8}, nc={nc}] table"


def test_decay_arm_dials_are_learnable_fp32_and_the_no_decay_arm_owns_none():
    """The dials train (through the affine rate sensitivity, §3.4); the clock itself is detached."""
    layer, _ = named_configs.construct(
        DECAY_CONFIG, hidden_size=256, num_heads=8, head_v_dim=32, nc=16, use_short_conv=False)
    for theta in layer.decay_theta:
        assert theta.requires_grad
        assert theta.dtype is torch.float32     # kept fp32 through module transforms by design
    ungated, _ = named_configs.construct(
        'RoLA-base-rla', hidden_size=256, num_heads=8, head_v_dim=32, nc=16, use_short_conv=False)
    assert ungated.decay_kind == 'none'
    assert getattr(ungated, 'decay_theta', None) is None


def test_gla_static_is_still_refused_and_is_not_what_the_decay_arm_resolves_to():
    """E4 stands: V3's clock is unconditionally detached, so the non-detached EVENT clock has no V3
    expression. The decay arm must NOT be a quiet re-registration of gla_static."""
    with pytest.raises(ValueError, match='gla_static'):
        named_configs._decay_from_kernel('gla_static')
    assert named_configs._decay_from_kernel('gla_mass_detached') == 'leaf_mass'


def test_decay_arm_lm_registry_pins_the_kernel_per_arm():
    assert DECAY_ARM in arms.ROLA_BASE_ARMS and DECAY_ARM in arms.ARMS
    _instance, mix, kernel = arms.ROLA_BASE_INSTANCE[DECAY_ARM]
    assert kernel == 'gla_mass_detached'
    assert mix == arms.ROLA_BASE_WRITE_MIX          # ARM 1's routing, byte-for-byte
    cfg = arms.arm_config(DECAY_ARM, vocab_size=50257)
    assert cfg.kernel == 'gla_mass_detached'
    # the other flagship arms must not have drifted off 'rla' when kernel became per-arm
    for arm in arms.ROLA_BASE_ARMS:
        if arm != DECAY_ARM:
            assert arms.arm_config(arm, vocab_size=50257).kernel == 'rla', arm


def test_decay_parameter_accounting_is_real_again():
    """`static_decay_parameter_floats` returned a hollow 0 for every arm while no decay arm
    existed. It must now report the decay arm's compact dial cost and only that arm's."""
    _D, spl, _nc = arms._rola_base_topology()
    assert arms.static_decay_parameter_floats(DECAY_ARM) == arms._ROLA_BASE_NH * sum(spl)
    for arm in arms.ARMS:
        if arm != DECAY_ARM:
            assert arms.static_decay_parameter_floats(arm) == 0, arm


# =============================================================================================
# THE ALPHA-CHOICE ARM (sparsemax write, softer read)
# =============================================================================================

def test_a2w_arm_is_registered_with_the_level_split_duties_and_the_alpha_split():
    assert A2W_CONFIG in named_configs.NAMED_CONFIGS
    kwargs = named_configs.NAMED_CONFIGS[A2W_CONFIG].build_kwargs(256, False)
    assert _mix_rows(kwargs['level_routing_configs']) == [
        (False, 'dense', 'sparse'),      # coarse: sparse WRITE
        (False, 'sparse', 'dense'),      # fine:   sparse READ
    ]
    coarse, fine = kwargs['level_routing_configs']
    assert coarse.alpha == 2.0, "the sparse WRITE duty must be sparsemax"
    assert fine.alpha == 1.5, "the sparse READ duty must stay the softer exact-sparse order"
    assert kwargs['state_norm'] == 'global' and kwargs['decay'] == 'none'


def test_a2w_differs_from_arm_3_in_ALPHA_AND_NOTHING_ELSE():
    """The pair isolates the alpha choice: same duty assignment, topology, bias, norm, decay, conv,
    and the same untied-everywhere router parameter shape (matched addressing entropy)."""
    split = named_configs._levelsplit_kwargs(256, False, 'rla', 'global')
    a2w = named_configs._levelsplit_a2w_kwargs(256, False, 'rla', 'global')
    for key in ('levels_per_head', 'states_per_level', 'router_bias', 'state_norm',
                'use_short_conv', 'decay'):
        assert split[key] == a2w[key], key
    assert _mix_rows(split['level_routing_configs']) == _mix_rows(a2w['level_routing_configs'])
    assert [c.alpha for c in split['level_routing_configs']] == [1.5, 1.5]
    assert [c.alpha for c in a2w['level_routing_configs']] == [2.0, 1.5]


def test_a2w_resolves_to_independent_routing_with_the_declared_entmax_orders():
    """Every duty is its own exact solve on its own logits — no level may resolve to UnionRouting
    (that is the whole point of the level-split family), and the entmax ORDER must survive the
    resolution rather than being flattened to the 1.5 default."""
    from rola_bench.engine import (
        EntmaxActivation, IndependentRouting, SoftmaxActivation, UnionRouting)

    routing = _resolved_routing(named_configs._levelsplit_a2w_kwargs(256, False, 'rla', 'global'))
    coarse, fine = routing.levels
    assert type(coarse) is IndependentRouting and type(fine) is IndependentRouting
    assert not isinstance(coarse, UnionRouting) and not isinstance(fine, UnionRouting)
    # the engine spells the per-duty activations `read`/`write` on the level itself
    assert type(coarse.read) is SoftmaxActivation
    assert type(coarse.write) is EntmaxActivation
    assert coarse.write.alpha == 2.0
    assert type(fine.read) is EntmaxActivation
    assert fine.read.alpha == 1.5
    assert type(fine.write) is SoftmaxActivation
    assert coarse.shares_solve is False and fine.shares_solve is False
    # the alpha must be a RATIFIED registry row, not merely a stored float: an order with no row
    # is refused at construction, so surviving resolution is itself the ratification statement.
    assert coarse.write.properties.exact_zeros and fine.read.properties.exact_zeros
    assert routing.branches == (16, 16)


def test_a2w_sparsemax_write_is_an_exact_point_mass_and_is_STRICTLY_sparser_than_alpha_1_5():
    """The behavioural content of alpha=2 on the write duty. At a corner logit both orders give an
    exact delta; the discriminating case is a FLATTER logit vector, where sparsemax's support is a
    subset of alpha-1.5's."""
    from rola_bench.engine import routing_factor_levels

    width = 16
    a2w = _resolved_routing(named_configs._levelsplit_a2w_kwargs(width * width, False, 'rla', 'global'))
    split = _resolved_routing(named_configs._levelsplit_kwargs(width * width, False, 'rla', 'global'))

    # a graded write logit ramp: several entries are competitive, so the entmax order decides how
    # many survive. Read side held flat so it cannot be the cause of any difference.
    assert a2w.branches == split.branches == (width, width)   # one packed layout, one logit pair
    ramp = torch.linspace(1.0, 0.0, width, dtype=torch.float64)
    read = _packed_logits(a2w, torch.zeros(width, dtype=torch.float64))
    write = _packed_logits(a2w, ramp)

    _, w_a2w = routing_factor_levels(read, write, a2w)
    _, w_split = routing_factor_levels(read, write, split)
    hard = w_a2w[0][0, 0]
    soft = w_split[0][0, 0]

    assert float(hard.sum()) == pytest.approx(1.0, abs=ENTMAX_SOLVE_TOL)
    assert float(soft.sum()) == pytest.approx(1.0, abs=ENTMAX_SOLVE_TOL)
    n_hard, n_soft = int((hard > 0).sum()), int((soft > 0).sum())
    assert n_hard < n_soft, f"sparsemax write should be strictly sparser: {n_hard} vs {n_soft}"
    # support containment: sparsemax keeps a subset of what alpha=1.5 keeps.
    assert torch.all(soft[hard > 0] > 0)

    # and the corner case still gives an exact delta with hard zeros everywhere else.
    corner = torch.zeros(width, dtype=torch.float64)
    corner[3] = 3.0
    corner_w = _packed_logits(a2w, corner)
    _, w_corner = routing_factor_levels(read, corner_w, a2w)
    c = w_corner[0][0, 0]
    assert c[3] == pytest.approx(1.0, abs=ENTMAX_SOLVE_TOL)
    off = torch.cat((c[:3], c[4:]))
    assert torch.equal(off, torch.zeros_like(off))


def test_a2w_read_stays_softer_than_its_write():
    """'Commit hard, address softly': on identical logits the fine level's alpha-1.5 READ must keep
    a WIDER support than the coarse level's sparsemax WRITE."""
    from rola_bench.engine import routing_factor_levels

    width = 16
    routing = _resolved_routing(named_configs._levelsplit_a2w_kwargs(width * width, False, 'rla', 'global'))
    ramp = torch.linspace(1.0, 0.0, width, dtype=torch.float64)
    read = _packed_logits(routing, ramp)
    write = _packed_logits(routing, ramp)

    reads, writes = routing_factor_levels(read, write, routing)
    n_write = int((writes[0][0, 0] > 0).sum())      # coarse, sparsemax
    n_read = int((reads[1][0, 0] > 0).sum())        # fine, alpha=1.5
    assert n_read > n_write, f"read support {n_read} should exceed write support {n_write}"


def test_a2w_arm_is_in_the_lm_registry():
    assert A2W_ARM in arms.ROLA_BASE_ARMS and A2W_ARM in arms.ARMS
    _instance, mix, kernel = arms.ROLA_BASE_INSTANCE[A2W_ARM]
    assert kernel == 'rla'
    assert mix == arms.ROLA_BASE_LEVELSPLIT_A2W_MIX
    assert [m.get('alpha') for m in mix] == [2.0, 1.5]
    assert arms.arm_config(A2W_ARM, vocab_size=50257).level_routing_configs == mix


# =============================================================================================
# ARM 3's PROMOTION ONTO THE LM SLATE
# =============================================================================================

def test_arm3_is_on_the_lm_suite_slate():
    assert SPLIT_CONFIG in named_configs.LM_SUITE_CONFIGS
    # all three structural wirings of the central ablation are LM-runnable
    for name in ('RoLA-base-rla', 'RoLA-base-union-global', SPLIT_CONFIG):
        assert name in named_configs.LM_SUITE_CONFIGS, name


def test_arm3_is_in_the_lm_arm_registry():
    assert SPLIT_ARM in arms.ROLA_BASE_ARMS and SPLIT_ARM in arms.ARMS
    _instance, mix, kernel = arms.ROLA_BASE_INSTANCE[SPLIT_ARM]
    assert kernel == 'rla'
    assert mix == arms.ROLA_BASE_LEVELSPLIT_MIX
    assert arms.arm_config(SPLIT_ARM, vocab_size=50257).kernel == 'rla'


@pytest.mark.parametrize("arm,config", [
    (SPLIT_ARM, SPLIT_CONFIG), (A2W_ARM, A2W_CONFIG), (DECAY_ARM, DECAY_CONFIG),
    ('rola-base-rla', 'RoLA-base-rla'), ('rola-base-union-global', 'RoLA-base-union-global'),
])
def test_lm_arm_mix_mirrors_the_named_config_builder(arm, config):
    """`lm/arms.py` (dict form, HF path) and `named_configs.py` (LevelRoutingConfig form, layer path)
    each carry their own copy of every routing mix. They MUST agree, or the LM cell and the
    MQAR/perf cell of the same published name are different experiments. Nothing else checks this.
    """
    _instance, mix, _kernel = arms.ROLA_BASE_INSTANCE[arm]
    kwargs = named_configs.NAMED_CONFIGS[config].build_kwargs(256, True)
    assert _dict_rows(mix) == _mix_rows(kwargs['level_routing_configs'])
    # alpha too: a default on one side and an explicit value on the other would silently diverge.
    assert [m.get('alpha', 1.5) for m in mix] == [c.alpha for c in kwargs['level_routing_configs']]


@pytest.mark.parametrize("name", list(named_configs.LM_SUITE_CONFIGS))
def test_every_lm_suite_config_constructs_on_the_v3_layer(name):
    layer, used = named_configs.construct(
        name, hidden_size=256, num_heads=8, head_v_dim=32, nc=16, use_short_conv=False)
    assert used == name
    assert layer.decay_kind in ('none', 'leaf_mass')
