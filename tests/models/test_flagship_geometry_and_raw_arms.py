"""Gates for the 2026-08-02 findings pass: the re-pinned flagship geometry + realized state
readback (F1), the raw-normalization arms (F2), the deleted dead `d_qk` kwarg (F3), the deleted
isostate duplicates (F4), and the gain-channel pin (F5).

Everything here is CPU: layer construction, `RoLA.get_stats()` readback, config-registry assertions
and ONE tiny CPU forward at the flagship value width. No CUDA, no timing, no training.

The load-bearing property this file exists for: **the RoLA arms' state is what the ENGINE
allocates**, and the baselines are sized to that same number. Before F1 the registry computed
`H * nc * d_qk * d_v` — the V2 matrix-state shape — against a V3 layer whose state is `[N, d_v]`,
so every published RoLA-vs-GDN/GLA row was matched to a number 16x larger than the layer ever held
and nothing in the tree noticed. Assertions here read the number back off the layer; a formula
appears only as the thing being checked AGAINST, never as the source of truth
([[matching-the-naive-antipattern]]).

Scope ruling (user, 2026-08-02): the LM tier's job is trains-stably-at-standard-scale. Capacity
evidence lives in the MQAR `N ∝ L` grid, so these gates pin the STANDARD head geometry
(`d_k = d_v = 128`, `H = 8`) and deliberately do not pin any capacity target.
"""

import pytest
import torch

from rola_bench import named_configs
from rola_bench.lm import arms

# The published flagship rung, stated once so a drift shows up as one failure and not fifteen.
FLAGSHIP_HEADS = 8
FLAGSHIP_DV = 128
FLAGSHIP_SPL = [8, 16]
FLAGSHIP_N = 128
FLAGSHIP_STATE = FLAGSHIP_HEADS * FLAGSHIP_N * FLAGSHIP_DV      # 131,072 content floats/layer
FLAGSHIP_DEN = FLAGSHIP_HEADS * FLAGSHIP_N                      # 1,024 denominator floats/layer

RAW_CONFIGS = ('RoLA-base-rla-raw', 'RoLA-base-levelsplit-raw')
RAW_PARENTS = {'RoLA-base-rla-raw': 'RoLA-base-rla',
               'RoLA-base-levelsplit-raw': 'RoLA-base-levelsplit-global'}


# =============================================================================================
# F1 — the re-pinned flagship geometry, VERIFIED against the engine
# =============================================================================================

def test_the_engine_accepts_d_v_128_at_the_flagship_topology():
    """The verification the re-pin rests on: construct at `d_v=128, b=[8,16], H=8` AND run a
    forward, so 'the engine accepts it' means executed, not merely constructed."""
    from rola_bench.engine import LevelRoutingConfig, RoLA

    layer = RoLA(hidden_size=512, num_heads=FLAGSHIP_HEADS, head_v_dim=FLAGSHIP_DV,
                 levels_per_head=len(FLAGSHIP_SPL), states_per_level=list(FLAGSHIP_SPL),
                 decay='none', state_norm='global', router_bias=True, use_short_conv=False,
                 layer_idx=0,
                 level_routing_configs=[
                     LevelRoutingConfig(tied=False, read='dense', write='sparse'),
                     LevelRoutingConfig(tied=False, read='dense', write='dense')])
    assert getattr(layer, '_routing_resolution_error', None) is None
    assert layer.topology.N == FLAGSHIP_N

    x = torch.randn(1, 8, 512)
    with torch.no_grad():
        out = layer(hidden_states=x)
    y = out[0] if isinstance(out, tuple) else out
    assert y.shape == (1, 8, 512)
    assert torch.isfinite(y).all()


def test_flagship_topology_is_the_pinned_one():
    assert arms._rola_base_topology() == (len(FLAGSHIP_SPL), FLAGSHIP_SPL, FLAGSHIP_N)
    assert arms._ROLA_BASE_DV == FLAGSHIP_DV
    assert arms._ROLA_BASE_NH == FLAGSHIP_HEADS


@pytest.mark.parametrize('arm', sorted(arms.ROLA_BASE_ARMS))
def test_every_flagship_arm_realizes_the_pinned_state(arm):
    """Read off `RoLA.get_stats()`, not off the registry's own inputs."""
    stats = arms.rola_realized_stats(arm)
    assert stats['d_v'] == FLAGSHIP_DV
    assert stats['n_heads'] == FLAGSHIP_HEADS
    assert stats['num_chunks'] == FLAGSHIP_N
    assert arms.recurrent_state_floats(arm) == FLAGSHIP_STATE
    # the denominator sidecar is ONE column per state under state_norm='global' — H*N, not H*N*d_qk
    assert arms.recurrent_state_overhead_floats(arm) == FLAGSHIP_DEN
    assert stats['state_floats'] == FLAGSHIP_STATE + FLAGSHIP_DEN


def test_the_baselines_are_iso_state_with_the_flagship_at_textbook_head_shape():
    """F1's whole point. GDN and GLA must land on the SAME number the RoLA layer realizes, at the
    standard `d_k = d_v = 128`, `H = 8` head shape — not at a capacity target, and not against a
    RoLA number computed with the deleted `d_qk`."""
    assert arms.recurrent_state_floats('gdn') == FLAGSHIP_STATE
    assert arms.recurrent_state_floats('gla') == FLAGSHIP_STATE
    # and the head shape itself, not just the total
    assert arms.GDN_GEOM['num_heads'] == FLAGSHIP_HEADS
    assert arms.GDN_GEOM['head_dim'] == FLAGSHIP_DV                       # GDN: head_dim IS d_k
    assert int(arms.GDN_GEOM['head_dim'] * arms.GDN_GEOM['expand_v']) == FLAGSHIP_DV   # and d_v
    assert arms.GLA_GEOM['num_heads'] == FLAGSHIP_HEADS
    hid = arms.BACKBONE['hidden_size']
    assert int(hid * arms.GLA_GEOM['expand_k']) // FLAGSHIP_HEADS == FLAGSHIP_DV       # GLA d_k
    assert int(hid * arms.GLA_GEOM['expand_v']) // FLAGSHIP_HEADS == FLAGSHIP_DV       # GLA d_v


def test_attention_is_still_the_unbounded_ceiling_and_reports_no_state():
    assert arms.recurrent_state_floats('attn') is None
    assert arms.recurrent_state_overhead_floats('attn') is None


def test_the_arm_config_geometry_matches_what_the_state_readback_measured():
    """The registry's HF config and the layer the accounting reads must be the same geometry —
    otherwise the reported state belongs to a different object than the one that trains."""
    for arm in arms.ROLA_BASE_ARMS:
        cfg = arms.arm_config(arm, vocab_size=256)
        assert cfg.states_per_level == FLAGSHIP_SPL
        assert cfg.levels_per_head == len(FLAGSHIP_SPL)
        assert cfg.d_v == FLAGSHIP_DV
        assert cfg.num_heads == FLAGSHIP_HEADS


def test_realized_readback_off_a_BUILT_model_agrees_with_the_registry():
    """The end-to-end version: build the actual HF model (one layer, tiny vocab — the geometry that
    matters is the mixer's) and read the state off the mixer instance, which is what a result row
    records. Also gates that `head_v_dim=128` survives the whole HF construction path."""
    from transformers import AutoModelForCausalLM

    arm = 'rola-base-rla'
    cfg = arms.arm_config(arm, vocab_size=256)
    cfg.num_hidden_layers = 1                 # the mixer geometry is per-layer; one is enough
    model = AutoModelForCausalLM.from_config(cfg)
    content, overhead = arms.realized_state_floats_from_model(model)
    assert content == FLAGSHIP_STATE == arms.recurrent_state_floats(arm)
    # the flagship arms are conv-ON, so the model-side overhead carries the conv cache too and is
    # therefore >= the conv-free registry number rather than equal to it.
    assert overhead >= arms.recurrent_state_overhead_floats(arm)


def test_no_state_formula_survives_in_the_lm_registry():
    """F1 as a REGRESSION gate: nothing in `lm/arms.py` may multiply a d_qk back in. The registry
    exposes no d_qk key at all any more, and the struct-override grammar refuses one loudly."""
    assert 'd_qk' not in arms.ROLA_GEOM and 'qk_norm' not in arms.ROLA_GEOM
    assert 'dqk' not in arms._STRUCT_KEYS
    with pytest.raises(ValueError, match='dqk'):
        arms._parse_rola('rola-rla-asym__dqk16')
    # a legitimate override still parses
    assert arms._parse_rola('rola-rla-asym__nh4_dv64') == ('rola-rla-asym', {'num_heads': 4, 'd_v': 64})


def test_legacy_dense_cells_report_their_real_state_rather_than_a_matched_claim():
    """The legacy flat-router cells were never iso-state; with the formula gone they simply report
    what they allocate (H * nc * d_v), which is the honest number."""
    for arm in arms.ROLA_ARMS:
        stats = arms.rola_realized_stats(arm)
        assert stats['num_chunks'] == arms.ROLA_GEOM['states_per_head']
        assert stats['d_v'] == arms.ROLA_GEOM['d_v']
        assert arms.recurrent_state_floats(arm) == (
            arms.ROLA_GEOM['num_heads'] * arms.ROLA_GEOM['states_per_head'] * arms.ROLA_GEOM['d_v'])
        assert arms.recurrent_state_floats(arm) != FLAGSHIP_STATE


# =============================================================================================
# F2 — the raw-normalization arms
# =============================================================================================

@pytest.mark.parametrize('name', RAW_CONFIGS)
def test_raw_arm_is_registered_and_pins_raw(name):
    assert name in named_configs.NAMED_CONFIGS
    kwargs = named_configs.NAMED_CONFIGS[name].build_kwargs(256, False)
    assert kwargs['state_norm'] == 'raw'
    assert kwargs['decay'] == 'none'


@pytest.mark.parametrize('name', RAW_CONFIGS)
def test_raw_arm_differs_from_its_parent_in_the_NORMALIZATION_AND_NOTHING_ELSE(name):
    """What makes the pair an ablation of the normalization axis rather than two unrelated cells."""
    raw = named_configs.NAMED_CONFIGS[name].build_kwargs(256, True)
    parent = named_configs.NAMED_CONFIGS[RAW_PARENTS[name]].build_kwargs(256, True)
    assert set(raw) == set(parent)
    for key in parent:
        if key == 'state_norm':
            continue
        if key == 'level_routing_configs':
            assert [(c.tied, c.read, c.write, c.alpha) for c in raw[key]] == \
                   [(c.tied, c.read, c.write, c.alpha) for c in parent[key]]
            continue
        assert raw[key] == parent[key], key
    assert parent['state_norm'] == 'global' and raw['state_norm'] == 'raw'


@pytest.mark.parametrize('name', RAW_CONFIGS)
def test_raw_arm_realizes_BOTH_SIDES_gain_and_no_denominator_sidecar(name):
    """The two things `raw` actually changes on the engine, read back off the constructed layer:
      * the gain channel becomes read AND write (`gain_W` gets 2 sides) — the both-sides mapping;
      * the per-state denominator sidecar disappears entirely, so `state_floats` is exactly
        `H * N * d_v` with nothing added.
    """
    layer, _ = named_configs.construct(
        name, hidden_size=256, num_heads=8, head_v_dim=32, nc=16, use_short_conv=False)
    assert layer.state_norm == 'raw'
    assert tuple(layer.gain_W.shape) == (8, 2, 256)
    assert layer.gain_bias.shape == (8, 2)
    stats = layer.get_stats()
    assert stats['state_floats'] == stats['n_heads'] * stats['num_chunks'] * stats['d_v']

    glob, _ = named_configs.construct(
        RAW_PARENTS[name], hidden_size=256, num_heads=8, head_v_dim=32, nc=16, use_short_conv=False)
    assert tuple(glob.gain_W.shape) == (8, 1, 256)        # global: write gain only, g_read := 1
    gstats = glob.get_stats()
    assert gstats['state_floats'] == stats['state_floats'] + gstats['n_heads'] * gstats['num_chunks']


@pytest.mark.parametrize('name', RAW_CONFIGS)
def test_raw_arm_runs_a_forward(name):
    layer, _ = named_configs.construct(
        name, hidden_size=256, num_heads=8, head_v_dim=32, nc=16, use_short_conv=False)
    with torch.no_grad():
        out = layer(torch.randn(1, 8, 256))
    y = out[0] if isinstance(out, tuple) else out
    assert y.shape == (1, 8, 256) and torch.isfinite(y).all()


def test_no_raw_union_arm_is_registered():
    """Deliberate, and recorded: under `raw` the union read mass no longer cancels in num/den, so
    `g_read *= read_mass` becomes required of the PRODUCER (prereqs Q1). That is an engine change,
    not a registry entry, and a union-raw cell registered before it would be quietly wrong."""
    for name in named_configs.NAMED_CONFIGS:
        assert not ('union' in name and name.endswith('-raw')), name


def test_no_raw_LM_ARM_is_registered_because_the_hf_path_cannot_express_it():
    """The HF path hard-codes `state_norm='global'` in `rola_instance`, and `RoLAConfig` has no
    such field — so a `-raw` LM arm would be a cell whose defining knob was silently dropped (the
    F3 failure mode). The raw ablation is layer-path only until the engine threads it."""
    for arm in arms.ARMS:
        assert not arm.endswith('-raw'), arm
    from rola_bench.engine import RoLAConfig
    assert not hasattr(RoLAConfig(), 'state_norm')


# =============================================================================================
# F4 — the isostate duplicates are gone
# =============================================================================================

@pytest.mark.parametrize('arm', ['gdn-isostate', 'gla-isostate'])
def test_isostate_duplicates_are_deleted_and_refuse_by_name(arm):
    assert arm not in arms.ARMS
    assert arm in arms.RETIRED_LM_ARMS
    with pytest.raises(ValueError, match='RETIRED'):
        arms.arm_config(arm, vocab_size=256)


def test_the_surviving_baseline_slate_is_exactly_one_cell_per_family():
    assert arms.ARMS == arms.ROLA_ARMS + arms.ROLA_BASE_ARMS + ('gdn', 'gla', 'attn')


# =============================================================================================
# F5 — the gain-channel pin
# =============================================================================================

@pytest.mark.parametrize('name', sorted(named_configs.NAMED_CONFIGS))
def test_every_published_config_pins_the_gain_bias_explicitly(name):
    """Never engine-default-inherited: the key must be PRESENT in the construction kwargs."""
    kwargs = named_configs.NAMED_CONFIGS[name].build_kwargs(16, False)
    assert 'gain_bias_init' in kwargs
    assert kwargs['gain_bias_init'] == named_configs.GAIN_BIAS_INIT


@pytest.mark.parametrize('name', sorted(named_configs.NAMED_CONFIGS))
def test_construct_verifies_the_realized_gain_side_count(name):
    """The pin is only worth something if it is checked against the LAYER. `construct()` asserts
    the realized side count; this gate asserts `construct()` actually does so, both ways."""
    layer, _ = named_configs.construct(
        name, hidden_size=32, num_heads=2, head_v_dim=16, nc=16, use_short_conv=False)
    norm = named_configs.NAMED_CONFIGS[name].build_kwargs(16, False)['state_norm']
    assert tuple(layer.gain_W.shape)[1] == named_configs.GAIN_SIDES_BY_STATE_NORM[norm]


def test_a_gain_contract_change_would_FAIL_construction_rather_than_publish(monkeypatch):
    """The guard has to bite. Flip the published contract and construction must refuse."""
    monkeypatch.setitem(named_configs.GAIN_SIDES_BY_STATE_NORM, 'global', 2)
    with pytest.raises(ValueError, match='gain side'):
        named_configs.construct('RoLA-base-rla', hidden_size=32, num_heads=2, head_v_dim=16,
                                nc=16, use_short_conv=False)
