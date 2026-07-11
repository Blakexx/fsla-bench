"""Config-only gates for the candidate-C LM and MQAR benchmark arms."""

import pytest
import torch

from rola_bench import named_configs
from rola_bench.lm import arms


def _mix_rows(configs):
    return [(cfg.tied, cfg.read, cfg.write) for cfg in configs]


def test_lm_union_arm_is_flagship_global_conv_qknorm_with_unchanged_state():
    name = 'rola-base-union-global'
    assert name in arms.ROLA_BASE_ARMS
    assert 'RoLA-base-union-global' in named_configs.LM_SUITE_CONFIGS
    instance, mix = arms.ROLA_BASE_INSTANCE[name]
    assert instance == 'rola-gla-scalar-norm-asym-conv'
    assert mix == [
        {'tied': False, 'read': 'sparse', 'write': 'sparse'},
        {'tied': False, 'read': 'dense', 'write': 'dense'},
    ]
    assert arms.ROLA_BASE_QKNORM[name] is True

    assert arms._rola_base_topology() == (2, [8, 16], 128)
    assert arms.recurrent_state_floats(name) == arms.recurrent_state_floats('rola-base')
    cfg = arms.arm_config(name, vocab_size=256)
    assert cfg.states_per_level == [8, 16]
    assert cfg.qk_norm is True and cfg.router_bias is True
    assert cfg.level_routing_configs == mix


def test_mqar_union_arm_is_conv_free_global_and_in_the_pinned_suite():
    name = 'RoLA-base-union-global'
    assert name in named_configs.MQAR_SUITE_CONFIGS
    named = named_configs.NAMED_CONFIGS[name]
    kwargs = named.build_kwargs(256, False)
    assert kwargs['kernel'] == 'gla_scalar'
    assert kwargs['state_norm'] == 'global'
    assert kwargs['use_short_conv'] is False
    assert kwargs['router_bias'] is True
    assert _mix_rows(kwargs['level_routing_configs']) == [
        (False, 'sparse', 'sparse'),
        (False, 'dense', 'dense'),
    ]


def test_lm_multiscale_is_static_global_and_owns_honest_overheads():
    name = 'rola-multiscale'
    assert name in arms.ROLA_BASE_ARMS
    instance, mix = arms.ROLA_BASE_INSTANCE[name]
    assert instance == 'rola-gla-scalar-norm-asym-conv'
    assert arms.ROLA_BASE_QKNORM[name] is True
    assert arms._rola_base_topology() == (2, [8, 16], 128)
    cfg = arms.arm_config(name, vocab_size=256)
    assert cfg.kernel == 'gla_static'
    assert cfg.states_per_level == [8, 16]
    assert cfg.qk_norm is True and cfg.router_bias is True
    assert cfg.level_routing_configs == mix
    assert arms.recurrent_state_floats(name) == 8 * 128 * 16 * 64
    assert arms.recurrent_state_overhead_floats(name) == 8 * 128 * 16
    assert arms.static_decay_parameter_floats(name) == 8 * 128


def test_multiscale_named_config_has_no_compatibility_substitution_surface():
    named = named_configs.NAMED_CONFIGS['RoLA-multiscale']
    assert set(named.__dataclass_fields__) == {'name', 'build_kwargs'}
    kwargs = named.build_kwargs(128, True)
    assert kwargs['kernel'] == 'gla_static'
    assert kwargs['state_norm'] == 'global'
    assert kwargs['qk_norm'] is True
    assert kwargs['use_short_conv'] is True
    with pytest.raises(ValueError, match='nc=128'):
        named.build_kwargs(64, True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='RoLA kernels require CUDA')
def test_named_union_config_constructs_without_substitution():
    layer, used = named_configs.construct(
        'RoLA-base-union-global', hidden_size=32, num_heads=2, head_k_dim=16, head_v_dim=16,
        nc=16, use_short_conv=False, layer_idx=0)
    assert used == 'RoLA-base-union-global'
    assert layer._union_levels == (0,)
    layer = layer.cuda()
    with torch.no_grad():
        out = layer(torch.randn(1, 33, 32, device='cuda'))
    out = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(out).all()
