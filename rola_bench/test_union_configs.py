"""Config-only gates for the candidate-C union-split LM and MQAR benchmark arms.

Reconciled onto the V3 mainline engine by campaign-prereqs E1-E7 (2026-08-01):
  * E1 — no `d_qk` / `qk_norm` / `head_k_dim` anywhere on the construction surface.
  * E3 — the union arm moved off the retired `kernel='gla_scalar'` onto the ungated `'rla'`.
  * E4 — the static-decay arm `rola-multiscale` / `RoLA-multiscale` is RETIRED: `kernel='gla_static'`
    was its entire definition and V3's clock is unconditionally detached, so the engine refuses it
    by name. The retirement itself is gated here.
  * E7 — union SEMANTICS are unchanged. The unnormalized union read is not a V3 gap: the producer's
    own `_split_level_mass` folds the level mass into the side gain, and under `state_norm='global'`
    it cancels exactly in num/den (measured EQUIVALENT at <=8.9e-16 vs a 1e-14 budget).
"""

import pytest

from rola_bench import named_configs
from rola_bench.lm import arms


def _mix_rows(configs):
    return [(cfg.tied, cfg.read, cfg.write) for cfg in configs]


def test_lm_union_arm_is_flagship_global_conv_with_unchanged_state():
    name = 'rola-base-union-global'
    assert name in arms.ROLA_BASE_ARMS
    assert 'RoLA-base-union-global' in named_configs.LM_SUITE_CONFIGS
    instance, mix, kernel = arms.ROLA_BASE_INSTANCE[name]
    assert kernel == 'rla'
    # E5: every LM arm names one of mainline's own ROLA_INSTANCES.
    assert instance == 'rola-rla-asym-conv'
    assert mix == [
        {'tied': False, 'read': 'sparse', 'write': 'sparse'},
        {'tied': False, 'read': 'dense', 'write': 'dense'},
    ]

    assert arms._rola_base_topology() == (2, [8, 16], 128)
    assert arms.recurrent_state_floats(name) == arms.recurrent_state_floats('rola-base-rla')
    cfg = arms.arm_config(name, vocab_size=256)
    assert cfg.states_per_level == [8, 16]
    assert cfg.router_bias is True
    assert cfg.kernel == 'rla'          # E3/E4: the whole surviving LM slate is ungated
    assert cfg.level_routing_configs == mix
    # E1: the V3 layer owns no Q/K surface, so no arm may pin one.
    assert not hasattr(cfg, 'qk_norm') and not hasattr(cfg, 'd_qk')


def test_mqar_union_arm_is_conv_free_global_and_in_the_pinned_suite():
    name = 'RoLA-base-union-global'
    assert name in named_configs.MQAR_SUITE_CONFIGS
    named = named_configs.NAMED_CONFIGS[name]
    kwargs = named.build_kwargs(256, False)
    assert 'kernel' not in kwargs and kwargs['decay'] == 'none'
    assert kwargs['state_norm'] == 'global'
    assert kwargs['use_short_conv'] is False
    assert kwargs['router_bias'] is True
    assert _mix_rows(kwargs['level_routing_configs']) == [
        (False, 'sparse', 'sparse'),
        (False, 'dense', 'dense'),
    ]


def test_static_decay_arms_are_retired_on_both_registries():
    """E4: `gla_static` has no V3 expression, so its arms are removed rather than re-pinned."""
    assert 'rola-multiscale' not in arms.ROLA_BASE_ARMS
    assert 'RoLA-multiscale' not in named_configs.NAMED_CONFIGS
    with pytest.raises(ValueError, match='RETIRED'):
        arms.arm_config('rola-multiscale', vocab_size=256)
    with pytest.raises(ValueError, match='RETIRED'):
        named_configs.construct('RoLA-multiscale', hidden_size=32, num_heads=2, head_v_dim=16, nc=16)
    # No live arm owns per-state STATIC-decay parameters any more. (The decay-ON arm added
    # 2026-08-02 owns COMPACT detached-clock dials instead; gated in test_decay_and_alpha_arms.py.)
    assert arms.static_decay_parameter_floats('rola-base-union-global') == 0


def test_union_named_config_constructs_on_the_v3_layer():
    """The E3 fix read back where it matters: the arm that used to fail on 'gla_scalar' now builds.

    CPU-only — construction and routing resolution need no GPU.
    """
    layer, used = named_configs.construct(
        'RoLA-base-union-global', hidden_size=32, num_heads=2, head_v_dim=16,
        nc=16, use_short_conv=False, layer_idx=0)
    from fla_rola.ops.rola.routing.types import IndependentRouting, UnionRouting
    assert used == 'RoLA-base-union-global'
    assert getattr(layer, '_routing_resolution_error', None) is None
    # the coarse level really did resolve to union-split, the fine level did not
    coarse, fine = layer.topology.levels
    assert type(coarse.routing) is UnionRouting
    assert type(fine.routing) is IndependentRouting
