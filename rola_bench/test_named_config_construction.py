"""Construction gate for the SURVIVING named-config slate against the V3 mainline engine.

This is the acceptance test for the campaign-prereqs reconciliation E1-E7
(`rola-scratch/workflows/20260801-campaign-prereqs/prereqs.md` §Q3): every config still registered
in `named_configs.NAMED_CONFIGS` must construct against `fla_rola.layers.RoLA` as the mainline V3
engine defines it — no `head_k_dim`/`d_qk`/`qk_norm` (E1), no `state_norm='kappa'` (E2), no
`kernel='gla_scalar'` (E3), no `kernel='gla_static'` (E4) — and every RETIRED name must refuse.

CPU-only: construction and routing resolution need no GPU. Run with PYTHONPATH pinned at the
mainline checkout (`/home/blake/flash-linear-attention-rola`).
"""

import pytest

from rola_bench import named_configs

# Small but structurally faithful: nc=16 is the smallest perfect square that keeps the square
# topology (D=2, b=4) real, and num_heads * head_v_dim == hidden_size is the no-output-bottleneck
# rule the flagship geometry obeys.
_HIDDEN, _HEADS, _DV, _NC = 32, 2, 16, 16


def _construct(name, nc=_NC):
    return named_configs.construct(
        name, hidden_size=_HIDDEN, num_heads=_HEADS, head_v_dim=_DV, nc=nc,
        use_short_conv=False, layer_idx=0)


@pytest.mark.parametrize('name', sorted(named_configs.NAMED_CONFIGS))
def test_surviving_named_config_constructs_on_the_v3_layer(name):
    """The whole surviving slate constructs, with its routing actually resolved by the engine."""
    layer, used = _construct(name)
    assert used == name
    assert getattr(layer, '_routing_resolution_error', None) is None


@pytest.mark.parametrize('name', sorted(named_configs.NAMED_CONFIGS))
def test_surviving_named_config_speaks_only_live_layer_vocabulary(name):
    """E1-E4 read back on the kwargs themselves, not just on the fact that construction passed."""
    kwargs = named_configs.NAMED_CONFIGS[name].build_kwargs(_NC, False)
    # E1: no Q/K surface anywhere in the construction kwargs.
    assert not {'head_k_dim', 'd_qk', 'qk_norm'} & set(kwargs)
    # E2: kappa is gone; 'global' is the published normalization.
    assert kwargs['state_norm'] == 'global'
    # E3/E4: the layer takes `decay`, not `kernel`, and only V3's two live decay values exist.
    # 'gla_scalar'/'gla_static' have no resolution at all (see the refusal test below); the one
    # decay-ON arm added 2026-08-02 resolves to 'leaf_mass' (kernel='gla_mass_detached').
    assert 'kernel' not in kwargs
    assert kwargs['decay'] in ('none', 'leaf_mass')
    expected = 'leaf_mass' if name == 'RoLA-base-massdecay-global' else 'none'
    assert kwargs['decay'] == expected, f"{name} must be {expected}, got {kwargs['decay']!r}"


@pytest.mark.parametrize('name', sorted(named_configs.RETIRED_CONFIGS))
def test_retired_named_config_refuses_by_name(name):
    assert name not in named_configs.NAMED_CONFIGS
    with pytest.raises(ValueError, match='RETIRED'):
        _construct(name)


@pytest.mark.parametrize('kernel', ['gla_scalar', 'gla_static'])
def test_retired_kernel_vocabulary_raises_rather_than_resolving(kernel):
    """A lingering dead kernel spelling must fail loudly, never silently map onto a live decay."""
    with pytest.raises(ValueError, match=kernel):
        named_configs._decay_from_kernel(kernel)


def test_live_kernel_vocabulary_maps_to_the_v3_decay_names():
    assert named_configs._decay_from_kernel('rla') == 'none'
    assert named_configs._decay_from_kernel('gla_mass_detached') == 'leaf_mass'


def test_detached_clock_spelling_also_constructs():
    """The one live gated kernel must construct too, so E4's replacement is real and not notional."""
    from rola_bench.engine import RoLA
    kwargs = named_configs._base_kwargs(_NC, False, 'gla_mass_detached', 'global')
    assert kwargs['decay'] == 'leaf_mass'
    layer = RoLA(hidden_size=_HIDDEN, num_heads=_HEADS, head_v_dim=_DV, layer_idx=0, **kwargs)
    assert getattr(layer, '_routing_resolution_error', None) is None


def test_every_suite_tuple_names_a_surviving_config():
    for tup in (named_configs.LM_SUITE_CONFIGS, named_configs.MQAR_SUITE_CONFIGS,
                named_configs.PERF_SUITE_CONFIGS):
        assert tup, 'suite tuples must not be emptied by the retirement'
        for name in tup:
            assert name in named_configs.NAMED_CONFIGS


def test_the_three_ablation_arms_are_registered_and_distinct():
    """prereqs Q3 §S1: the campaign's central experiment is these three routing mixes."""
    def rows(name):
        cfgs = named_configs.NAMED_CONFIGS[name].build_kwargs(_NC, False)['level_routing_configs']
        return [(c.tied, c.read, c.write) for c in cfgs]

    assert rows('RoLA-base-rla') == [(False, 'dense', 'sparse'), (False, 'dense', 'dense')]
    assert rows('RoLA-base-union-global') == [(False, 'sparse', 'sparse'), (False, 'dense', 'dense')]
    assert rows('RoLA-base-levelsplit-global') == [(False, 'dense', 'sparse'), (False, 'sparse', 'dense')]
