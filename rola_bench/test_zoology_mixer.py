"""GATES for the migrated `zoology.mixers.rola.RoLAMixer` (Phase 1, 2026-08-02).

Three properties, in order of how much they matter:

1. **THINNESS, BIT-EXACTLY.** The mixer must be an adapter and nothing else. Given the same seed and
   the same configuration, `RoLAMixer(...)` and `fla_rola.layers.RoLA(...)` must produce identical
   parameters and identical forward outputs, to the last bit. This is the property that lets a perf
   row, an MQAR row and a layer-path row of one published name be the SAME cell instead of three
   experiments sharing a string ([[matching-the-naive-antipattern]]: a wrapper validated against a
   fresh reference proves only self-consistency; here the reference is the canonical layer itself).

2. **RETIRED VOCABULARY RAISES.** Every kwarg V3 deleted (`d_qk`, `head_k_dim`, `kernel`, `phi`,
   `qk_norm`, `kernel_kwargs`, ...) must raise by name. The pre-V3 mixer absorbed `writer`,
   `reader`, `route_on` and `router_zloss_coef` into `**kwargs` and forwarded four dead Q/K kwargs
   into the layer; that silence is how `d_qk` outlived the architecture that defined it (rola-bench
   F3).

3. **THE V3 SURFACE IS DERIVED, NOT LISTED.** The accepted keyword set comes from
   `inspect.signature(RoLA.__init__)`, so it cannot go stale.

CPU only. A visible GPU is needed only because pytest collection imports Triton.
"""
from __future__ import annotations

import pytest
import torch


def _mixer_cls():
    from zoology.mixers.rola import RoLAMixer
    return RoLAMixer


#: One configuration per structurally distinct routing resolution the registry can produce:
#: IndependentRouting with a sparse write, UnionRouting (untied sparse/sparse), the level split,
#: sparsemax alphas, the D1 dense control, a tied level, and the leaf-mass decay clock.
_CASES = {
    "arm1": dict(levels_per_head=2, states_per_level=4, decay="none", state_norm="global",
                 level_routing_configs=[dict(tied=False, read="dense", write="sparse"),
                                        dict(tied=False, read="dense", write="dense")]),
    "arm2_union": dict(levels_per_head=2, states_per_level=4, decay="none", state_norm="global",
                       level_routing_configs=[dict(tied=False, read="sparse", write="sparse"),
                                              dict(tied=False, read="dense", write="dense")]),
    "arm3_levelsplit": dict(levels_per_head=2, states_per_level=4, decay="none", state_norm="global",
                            level_routing_configs=[dict(tied=False, read="dense", write="sparse"),
                                                   dict(tied=False, read="sparse", write="dense")]),
    "arm3_a2w": dict(levels_per_head=2, states_per_level=4, decay="none", state_norm="global",
                     level_routing_configs=[dict(tied=False, read="dense", write="sparse", alpha=2.0),
                                            dict(tied=False, read="sparse", write="dense", alpha=1.5)]),
    "d1_dense": dict(levels_per_head=1, states_per_level=16, decay="none", state_norm="global",
                     level_routing_configs=[dict(tied=False, read="dense", write="dense")]),
    "d1_dense_tied": dict(levels_per_head=1, states_per_level=16, decay="none", state_norm="global",
                          level_routing_configs=[dict(tied=True, read="dense", write="dense")]),
    "raw_norm": dict(levels_per_head=2, states_per_level=4, decay="none", state_norm="raw",
                     level_routing_configs=[dict(tied=False, read="dense", write="sparse"),
                                            dict(tied=False, read="dense", write="dense")]),
    "massdecay": dict(levels_per_head=2, states_per_level=4, decay="leaf_mass", state_norm="global",
                      level_routing_configs=[dict(tied=False, read="dense", write="sparse"),
                                             dict(tied=False, read="dense", write="dense")]),
    "mixed_radix": dict(levels_per_head=2, states_per_level=[2, 8], decay="none", state_norm="global",
                        level_routing_configs=[dict(tied=False, read="dense", write="sparse"),
                                               dict(tied=False, read="sparse", write="dense")]),
}

_H, _DV = 2, 16
_DMODEL = _H * _DV


def _materialize(rows):
    from rola_bench.engine import LevelRoutingConfig
    return [LevelRoutingConfig(**r) for r in rows]


def _build_pair(case):
    """(mixer, layer) built from the SAME kwargs under the SAME seed."""
    from rola_bench.engine import RoLA
    kwargs = {k: v for k, v in case.items() if k != "level_routing_configs"}
    rows = case["level_routing_configs"]

    torch.manual_seed(1234)
    mixer = _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                         router_bias=True, use_short_conv=False, gain_bias_init=None,
                         level_routing_configs=[dict(r) for r in rows], **kwargs)
    torch.manual_seed(1234)
    layer = RoLA(hidden_size=_DMODEL, layer_idx=0, num_heads=_H, head_v_dim=_DV,
                 router_bias=True, use_short_conv=False, gain_bias_init=None,
                 level_routing_configs=_materialize(rows), **kwargs)
    return mixer, layer


# --------------------------------------------------------------------------- 1. thinness
@pytest.mark.parametrize("name", sorted(_CASES))
def test_mixer_parameters_are_bit_identical_to_the_layer(name):
    mixer, layer = _build_pair(_CASES[name])
    mp = dict(mixer.named_parameters())
    lp = dict(layer.named_parameters())
    assert set(mp) == set(lp), (
        f"{name}: the mixer and the layer own different parameters "
        f"(mixer-only {sorted(set(mp) - set(lp))}, layer-only {sorted(set(lp) - set(mp))}). A thin "
        "adapter allocates nothing of its own.")
    for key in sorted(mp):
        assert torch.equal(mp[key], lp[key]), (
            f"{name}: parameter {key!r} differs between the mixer and the layer under one seed — the "
            "adapter is not thin, or it changed an initialization-affecting kwarg.")


@pytest.mark.parametrize("name", sorted(_CASES))
def test_mixer_forward_is_bit_identical_to_the_layer(name):
    mixer, layer = _build_pair(_CASES[name])
    mixer.eval(); layer.eval()
    torch.manual_seed(7)
    x = torch.randn(2, 8, _DMODEL)
    with torch.no_grad():
        got = mixer(x)
        want = layer(hidden_states=x)[0]
    assert isinstance(got, torch.Tensor), (
        f"{name}: the mixer must return a plain tensor (zoology's TransformerBlock calls mixer(x) "
        f"and indexes no tuple), got {type(got).__name__}")
    assert got.shape == want.shape
    assert torch.equal(got, want), (
        f"{name}: mixer forward differs from the layer forward at max |delta| = "
        f"{(got - want).abs().max().item():.3e}. BIT-exact is the contract, not close: any tolerance "
        "here would let a wrapper quietly become a second implementation.")


def test_mixer_threads_the_cache_when_asked():
    """`use_cache=True` must return the FULL 3-tuple. Swallowing it once made the perf harness's
    decode arm measure a COLD, un-seeded step while reporting it as a warm continuation."""
    from rola_bench.engine import Cache
    mixer, _ = _build_pair(_CASES["arm1"])
    mixer.eval()
    x = torch.randn(1, 4, _DMODEL)
    cache = Cache()
    with torch.no_grad():
        out = mixer(x, past_key_values=cache, use_cache=True)
    assert isinstance(out, tuple) and len(out) == 3, (
        f"use_cache=True must return the layer's (o, None, past_key_values) tuple, got "
        f"{type(out).__name__}")
    assert out[2] is cache and len(cache) > 0, (
        "the cache came back empty: the mixer swallowed the cache kwargs instead of threading them, "
        "which is what once made the decode arm measure a cold step")
    # and without use_cache it must still unwrap to a bare tensor
    with torch.no_grad():
        assert isinstance(mixer(x), torch.Tensor)


# --------------------------------------------------------------------------- 2. retired vocabulary
@pytest.mark.parametrize("kwarg,value", [
    ("d_qk", 16), ("head_k_dim", 16), ("qk_norm", True), ("kernel", "rla"),
    ("kernel_kwargs", {"state_norm": "kappa"}), ("phi", "elu"), ("routing_mode", {}),
    ("tie_router_init", True), ("states_per_head", 16), ("routing", "flat"),
    ("writer", "softmax_linear"), ("reader", "softmax_linear"), ("route_on", "x"),
    ("router_zloss_coef", 0.1),
])
def test_retired_kwargs_raise_by_name(kwarg, value):
    from zoology.mixers.rola import RETIRED_KWARGS
    assert kwarg in RETIRED_KWARGS, f"{kwarg!r} must carry a retirement reason, not just fail"
    with pytest.raises(TypeError) as excinfo:
        _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                     levels_per_head=1, states_per_level=16, **{kwarg: value})
    assert kwarg in str(excinfo.value) and "RETIRED" in str(excinfo.value)


def test_unknown_kwarg_raises_rather_than_being_absorbed():
    with pytest.raises(TypeError, match="unknown kwarg"):
        _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                     levels_per_head=1, states_per_level=16, not_a_real_kwarg=1)


def test_unknown_level_routing_field_raises():
    with pytest.raises(TypeError, match="unknown LevelRoutingConfig field"):
        _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                     levels_per_head=1, states_per_level=16,
                     level_routing_configs=[dict(tied=False, read="dense", writer="dense")])


# --------------------------------------------------------------------------- 3. derived surface
def test_accepted_kwargs_are_derived_from_the_engine_signature():
    """The guard against this file's own predecessor: a hand-listed surface that outlived the engine.

    Asserts the accepted set IS the layer signature (so it tracks the engine automatically), and
    that the set is non-vacuous and contains the V3 kwargs the registry actually emits.
    """
    import inspect
    from zoology.mixers.rola import _LAYER_PARAMS
    from rola_bench.engine import RoLA
    expected = frozenset(inspect.signature(RoLA.__init__).parameters) - {"self"}
    assert _LAYER_PARAMS == expected
    for required in ("levels_per_head", "states_per_level", "decay", "state_norm", "router_bias",
                     "level_routing_configs", "gain_bias_init", "head_v_dim", "num_heads"):
        assert required in _LAYER_PARAMS, (
            f"{required!r} is emitted by the preset registry but is not a kwarg of the pinned "
            "engine's RoLA.__init__ — the registry and the engine have diverged.")
    for deleted in ("head_k_dim", "qk_norm", "kernel", "phi"):
        assert deleted not in _LAYER_PARAMS, (
            f"{deleted!r} reappeared on the engine's layer signature; the V3 deletion inventory "
            "(ROLA_V3_SPEC.md §1.2) no longer holds and every retirement reason must be re-ratified.")


def test_resolved_config_reports_the_realized_cell():
    """Provenance must be READ BACK, not re-derived from the kwargs. The pre-V3 STATE_FLOATS_JSON
    reported `d_qk`, `qk_norm` and `states_per_head`, three fields that stopped existing without the
    log noticing."""
    mixer, _ = _build_pair(_CASES["mixed_radix"])
    cfg = mixer.resolved_config()
    assert cfg["states_per_level"] == [2, 8] and cfg["N"] == 16
    assert cfg["addressing_entropy"] == 10          # S = sum_l b_l
    assert cfg["levels_per_head"] == 2
    assert cfg["decay"] == "none" and cfg["state_norm"] == "global"
    assert cfg["gain_sides"] == 1                   # 'global' => write gain only (F5's contract)
    assert not any(k in cfg for k in ("d_qk", "qk_norm", "kernel", "phi", "states_per_head"))
    raw, _ = _build_pair(_CASES["raw_norm"])
    assert raw.resolved_config()["gain_sides"] == 2  # 'raw' => read AND write gain
