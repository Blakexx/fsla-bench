"""GATES for `zoology.mixers.rola.RoLAMixer` on the STANDALONE engine (repoint, 2026-08-15).

Three properties, in order of how much they matter:

1. **THINNESS, BIT-EXACTLY.** The mixer must be an adapter and nothing else. Given the same seed and
   the same configuration, `RoLAMixer(...)` and `rola.RoLA(rola.RouteProducer(...))` must produce
   identical parameters and identical forward outputs, to the last bit. This is the property that
   lets a perf row, an MQAR row and a layer-path row of one published name be the SAME cell instead
   of three experiments sharing a string ([[matching-the-naive-antipattern]]: a wrapper validated
   against a fresh reference proves only self-consistency; here the reference is the canonical layer
   itself, constructed the way the engine's own docs construct it).
2. **RETIRED VOCABULARY RAISES.** Every kwarg a deletion removed — the V3 Q/K surface, and now
   `state_block` (BC is the operator's), the decay dials (the tier is deferred) and the HF cache
   kwargs (the layer takes a `state` pair) — must raise BY NAME. The pre-V3 mixer absorbed `writer`,
   `reader`, `route_on` and `router_zloss_coef` into `**kwargs`; that silence is how `d_qk` outlived
   the architecture that defined it (rola-bench F3).
3. **THE FOUR WIRINGS LAND.** Each row spelling must resolve to the engine level config the grid's
   arm tag claims, `UnionRouting` included — the one wiring whose resolution differs structurally.

Construction is CPU. The forward comparison is CUDA: the standalone engine has no CPU arm.
"""
from __future__ import annotations

import pytest
import torch


def _mixer_cls():
    from zoology.mixers.rola import RoLAMixer
    return RoLAMixer


def _engine():
    from rola_bench.engine import import_grid_engine
    return import_grid_engine()


#: One configuration per structurally distinct routing resolution the registry can produce, at the
#: grid's own geometry (H=4, d_v=64) and a rung with a built arm: the sparse-write split, the union
#: level, the level split, the sparsemax alpha, the D1 dense control and the tied spellings.
_ROWS = {
    "arm1": [dict(tied=False, read="dense", write="sparse"),
             dict(tied=False, read="dense", write="dense")],
    "arm2_union": [dict(tied=False, read="sparse", write="sparse"),
                   dict(tied=False, read="dense", write="dense")],
    "arm3_levelsplit": [dict(tied=False, read="dense", write="sparse"),
                        dict(tied=False, read="sparse", write="dense")],
    "arm3_a2w": [dict(tied=False, read="dense", write="sparse", alpha=2.0),
                 dict(tied=False, read="sparse", write="dense", alpha=1.5)],
    "d1_dense": [dict(tied=False, read="dense", write="dense")],
    "d1_dense_tied": [dict(tied=True, read="dense", write="dense")],
    "arm1_tied": [dict(tied=True, read="dense", write="sparse"),
                  dict(tied=True, read="dense", write="dense")],
}
_WIDTHS = {"d1_dense": [64], "d1_dense_tied": [64]}
#: `state_norm` is a ONE-VALUE axis on the standalone engine: `'raw'` is a dead arm
#: (`rola.routing.types.validate_normalization` refuses it by name — the self-normalizing ratio IS
#: the design), and every MQAR cell pins `'global'` already. There is no second normalization case
#: to gate, and a mixer asking for one gets the engine's own refusal.
_NORM = {}
_CASES = {k: k for k in _ROWS}

_H, _DV = 4, 64
_DMODEL = _H * _DV


def _levels(rows, widths):
    """The reference construction, spelled the way the engine's own docs spell it."""
    rola = _engine()
    out = []
    for row, width in zip(rows, widths):
        alpha = float(row.get("alpha", 1.5))
        sparse = {duty for duty in ("read", "write") if row.get(duty) == "sparse"}
        if sparse == {"read", "write"} and not row.get("tied", False):
            out.append(rola.UnionRouting(width=width, alpha=alpha))
            continue
        out.append(rola.IndependentRouting(
            width=width,
            read=rola.entmax(alpha) if "read" in sparse else rola.softmax(),
            write=rola.entmax(alpha) if "write" in sparse else rola.softmax(),
            tied=bool(row.get("tied", False))))
    return out


def _build_pair(case):
    """(mixer, layer) built from the SAME configuration under the SAME seed."""
    rola = _engine()
    rows = _ROWS[_CASES[case]]
    widths = _WIDTHS.get(_CASES[case], [8, 8])
    norm = _NORM.get(case, "global")

    torch.manual_seed(1234)
    mixer = _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                         levels_per_head=len(widths), states_per_level=widths,
                         level_routing_configs=[dict(r) for r in rows],
                         state_norm=norm, router_bias=True, gain_bias_init=None)
    torch.manual_seed(1234)
    layer = rola.RoLA(
        rola.RouteProducer(_levels(rows, widths), hidden_size=_DMODEL, num_heads=_H,
                           normalization=norm, bias=True, gain=True, gain_bias_init=None),
        d_v=_DV, layer_idx=0)
    return mixer, layer


# --------------------------------------------------------------------------- 1. thinness
@pytest.mark.parametrize("name", sorted(_CASES))
def test_mixer_parameters_are_bit_identical_to_the_layer(name):
    mixer, layer = _build_pair(name)
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


@pytest.mark.parametrize("name", ["arm1", "arm2_union", "arm3_levelsplit", "d1_dense"])
def test_mixer_forward_is_bit_identical_to_the_layer(name):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; the standalone engine has no CPU arm")
    mixer, layer = _build_pair(name)
    mixer = mixer.cuda().eval()
    layer = layer.cuda().eval()
    torch.manual_seed(7)
    x = torch.randn(2, 64, _DMODEL, device="cuda")
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


def test_mixer_returns_the_continuation_pair_only_when_asked():
    """The layer's contract is `(y, (state, conv_rings))` — BOTH halves together. zoology's block
    wants the tensor; a caller continuing a sequence wants the pair, and never one half of it."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; the standalone engine has no CPU arm")
    mixer, _ = _build_pair("arm1")
    mixer = mixer.cuda().eval()
    x = torch.randn(1, 64, _DMODEL, device="cuda")
    with torch.no_grad():
        y, carried = mixer(x, return_state=True)
        assert isinstance(y, torch.Tensor)
        assert isinstance(carried, tuple) and len(carried) == 2
        assert isinstance(mixer(x), torch.Tensor)


# --------------------------------------------------------------------------- 2. retired vocabulary
@pytest.mark.parametrize("kwarg,value", [
    ("d_qk", 16), ("head_k_dim", 16), ("qk_norm", True), ("kernel", "rla"),
    ("kernel_kwargs", {"state_norm": "kappa"}), ("phi", "elu"), ("routing_mode", {}),
    ("tie_router_init", True), ("states_per_head", 16), ("routing", "flat"),
    ("writer", "softmax_linear"), ("reader", "softmax_linear"), ("route_on", "x"),
    ("router_zloss_coef", 0.1),
    ("state_block", 16), ("decay_target_leaf_rate", 0.5), ("decay_inclusion_tick", 1.0),
    ("use_cache", True), ("past_key_values", None),
])
def test_retired_kwargs_raise_by_name(kwarg, value):
    from zoology.mixers.rola import RETIRED_KWARGS
    assert kwarg in RETIRED_KWARGS, f"{kwarg!r} must carry a retirement reason, not just fail"
    with pytest.raises(TypeError) as excinfo:
        _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                     levels_per_head=1, states_per_level=64, **{kwarg: value})
    assert kwarg in str(excinfo.value) and "RETIRED" in str(excinfo.value)


def test_unknown_kwarg_raises_rather_than_being_absorbed():
    with pytest.raises(TypeError, match="unknown kwarg"):
        _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                     levels_per_head=1, states_per_level=64, not_a_real_kwarg=1)


def test_unknown_level_routing_field_raises():
    with pytest.raises(TypeError, match="unknown LevelRoutingConfig field"):
        _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                     levels_per_head=1, states_per_level=64,
                     level_routing_configs=[dict(tied=False, read="dense", writer="dense")])


@pytest.mark.parametrize("decay", ["leaf_mass", "inclusion_gated"])
def test_decay_is_refused_by_name_while_the_tier_is_deferred(decay):
    """Decay is retired from the operator forward AND backward, and `inclusion_gated` has no
    standalone spelling at all. A decay cell must refuse at CONSTRUCTION rather than be silently
    rebuilt without its clock and published under a decay run_id."""
    with pytest.raises(ValueError, match="DEFERRED"):
        _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                     levels_per_head=1, states_per_level=64, decay=decay)


# --------------------------------------------------------------------------- 3. the wirings land
@pytest.mark.parametrize("name,union_levels", [
    ("arm1", ()), ("arm2_union", (0,)), ("arm3_levelsplit", ()), ("d1_dense", ()),
])
def test_rows_resolve_to_the_engine_level_config_the_arm_claims(name, union_levels):
    rola = _engine()
    mixer, _ = _build_pair(name)
    for index, level in enumerate(mixer.routes.levels):
        want = rola.UnionRouting if index in union_levels else rola.IndependentRouting
        assert isinstance(level, want), (
            f"{name}: level {index} resolved to {type(level).__name__}, expected {want.__name__}")


def test_alpha_lands_on_the_sparse_duty_only():
    mixer, _ = _build_pair("arm3_a2w")
    cfg = mixer.resolved_config()["level_routing_configs"]
    assert cfg[0]["write"] == "sparse" and cfg[0]["alpha"] == 2.0
    assert cfg[1]["read"] == "sparse" and cfg[1]["alpha"] == 1.5
    with pytest.raises(ValueError, match="dead knob"):
        _mixer_cls()(d_model=_DMODEL, layer_idx=0, n_heads=_H, d_v=_DV,
                     levels_per_head=1, states_per_level=64,
                     level_routing_configs=[dict(read="dense", write="dense", alpha=2.0)])


def test_resolved_config_reports_the_realized_cell():
    """Provenance must be READ BACK, not re-derived from the kwargs. The pre-V3 STATE_FLOATS_JSON
    reported `d_qk`, `qk_norm` and `states_per_head`, three fields that stopped existing without the
    log noticing."""
    mixer, _ = _build_pair("arm3_levelsplit")
    cfg = mixer.resolved_config()
    assert cfg["states_per_level"] == [8, 8] and cfg["N"] == 64
    assert cfg["addressing_entropy"] == 16          # S = sum_l b_l
    assert cfg["levels_per_head"] == 2 and cfg["engine"] == "rola"
    assert cfg["decay"] == "none" and cfg["state_norm"] == "global"
    assert cfg["gain_sides"] == 1                   # 'global' => write gain only (F5's contract)
    assert not any(k in cfg for k in ("d_qk", "qk_norm", "kernel", "phi", "states_per_head"))
