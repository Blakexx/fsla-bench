"""Gate for the paper_v1 kernel harness's arm registry (`rola_bench.perf.scaling`), 2026-08-02.

**Why this file exists.** `perf/scaling.py` is where the paper's E5 kernel numbers come from, and
until today NOTHING tested it. It hardcoded its named-config arms as
``("rola-base", "rola-base-rla")``; the campaign-prereqs E1-E7 migration retired ``RoLA-base`` and
rewrote the builders to emit the V3 layer's ``decay`` kwarg instead of the config-level ``kernel``
string. `build_rola_named` therefore rotted into ``KeyError 'RoLA-base'`` followed by
``KeyError 'kernel'`` — invisible, because the migration's own 38-test gate does not touch this
file. This is that missing gate: it constructs EVERY named config `scaling.py` references and
asserts the arm registry is derived from, and consistent with, `named_configs`.

**Scope: CPU, construction only.** Everything here is registry work plus
`fla_rola.layers.rola.LevelRoutingConfig` dataclasses — no engine layer is instantiated, no CUDA is
touched, no timing is done. The RoLA arms of `scaling.py` cannot construct at all today (they are
blocked on the zoology `RoLAMixer` migration, rola-bench-audit §3); the point of this gate is that
when that block lifts, the arm NAMES and KWARGS are already known-good, and that a future
retirement can never again leave a dangling arm name behind.
"""

import pytest

from rola_bench import named_configs
from rola_bench.perf import _common as C
from rola_bench.perf import scaling


# ---------------------------------------------------------------- the arm registry itself

def test_named_arms_are_derived_from_the_registry_not_hardcoded():
    """The exact anti-rot property: scaling.py's arm names ARE `PERF_SUITE_CONFIGS`, by derivation.

    If someone re-hardcodes a list here, this fails the moment the two disagree — which is the
    failure mode that produced the silent `KeyError 'RoLA-base'`.
    """
    assert scaling.named_rola_arm_names() == tuple(named_configs.PERF_SUITE_CONFIGS)
    assert scaling.named_rola_arm_kinds() == tuple(
        n.lower() for n in named_configs.PERF_SUITE_CONFIGS)
    # The module-level tuple the sweep/CLI actually reads must agree with the derivation.
    assert scaling._NAMED_ROLA_ARM_KINDS == scaling.named_rola_arm_kinds()


def test_perf_slate_is_non_empty():
    """A perf slate that quietly emptied would make every RoLA scaling row silently disappear
    rather than fail — the same class of invisible breakage this file exists to catch."""
    assert scaling.named_rola_arm_names(), "PERF_SUITE_CONFIGS is empty: no RoLA scaling arm at all"


@pytest.mark.parametrize("name", list(named_configs.PERF_SUITE_CONFIGS))
def test_every_referenced_config_is_live(name):
    """Every name `scaling.py` references exists in NAMED_CONFIGS and is not retired."""
    assert name not in named_configs.RETIRED_CONFIGS, (
        f"{name!r} is on the perf slate but RETIRED: "
        f"{named_configs.RETIRED_CONFIGS.get(name)}")
    assert name in named_configs.NAMED_CONFIGS, (
        f"{name!r} is on the perf slate but absent from NAMED_CONFIGS")


# ---------------------------------------------------------------- construction of each arm

@pytest.mark.parametrize("kind", list(scaling.named_rola_arm_kinds()))
@pytest.mark.parametrize("nc", [256, 1024])
def test_resolve_named_arm_builds_v3_kwargs(kind, nc):
    """CONSTRUCT every named config scaling.py references, at the nc values the harness sweeps.

    Asserts the V3 construction surface: `decay` (not `kernel`), no Q/K keys, and the topology
    factorization the square-topology builders promise (nc = states_per_level ** levels_per_head).
    """
    name, kwargs = scaling.resolve_named_arm(kind, nc=nc)
    assert name in named_configs.NAMED_CONFIGS

    # E1: the V3 layer owns no Q/K surface.
    for dead in ("head_k_dim", "d_qk", "qk_norm", "phi"):
        assert dead not in kwargs, f"{name}: dead V2 Q/K key {dead!r} survived into perf kwargs"
    # E3/E4: the config-level `kernel` string is resolved to `decay` at the construction boundary.
    assert "kernel" not in kwargs, f"{name}: emits removed 'kernel' key (the second scaling.py bug)"
    assert kwargs["decay"] in ("none", "leaf_mass"), f"{name}: {kwargs['decay']!r}"
    # E2: kappa is not a normalization in any engine checkout.
    assert kwargs["state_norm"] in ("raw", "global"), f"{name}: {kwargs['state_norm']!r}"

    assert kwargs["states_per_level"] ** kwargs["levels_per_head"] == nc
    assert len(kwargs["level_routing_configs"]) == kwargs["levels_per_head"]
    assert kwargs["use_short_conv"] is True   # the paper/LM geometry this harness pins


@pytest.mark.parametrize("kind", list(scaling.named_rola_arm_kinds()))
def test_arm_kind_round_trips_to_its_config_name(kind):
    name, _ = scaling.resolve_named_arm(kind)
    assert scaling._named_arm_kind(name) == kind


def test_unknown_arm_kind_refuses_by_name():
    with pytest.raises(ValueError, match="unknown named RoLA arm"):
        scaling.resolve_named_arm("rola-base")          # the retired name that broke this harness


def test_arm_kinds_are_accepted_by_the_SCALE_ARMS_validator(monkeypatch):
    """The CLI allowlist must admit exactly the derived arm kinds — a name that resolves but that
    `_arm_kinds()` rejects is just as broken as a dangling one."""
    kinds = scaling.named_rola_arm_kinds()
    monkeypatch.setenv("SCALE_ARMS", ",".join(kinds))
    assert scaling._arm_kinds() == kinds


# ---------------------------------------------------------------- construction (Phase 1.6)
# These two tests replaced the "the builder is blocked" pair when the zoology mixer was migrated to
# the V3 layer surface, DELIBERATELY: the old ones asserted a NotImplementedError and would have
# gone on passing forever if the migration had never happened, so they named themselves as the
# thing to rewrite. What they now assert is that the RoLA arms build the cell they claim.
#
# CPU-only. The builders move the module to `C.DEV`/`C.DT` (cuda/bf16), so the assertions run on the
# same kwargs through the mixer directly; a GPU-only end-to-end build is the `--smoke` cell's job.

@pytest.mark.parametrize("kind", list(scaling.named_rola_arm_kinds()))
def test_named_arm_builds_the_registry_cell(kind):
    """The perf arm and the layer-path arm of one published name must be ONE cell."""
    from zoology.mixers.rola import RoLAMixer
    from rola_bench.named_configs import NAMED_CONFIGS
    name, kwargs = scaling.resolve_named_arm(kind, nc=256)
    assert kwargs == dict(NAMED_CONFIGS[name].build_kwargs(256, True))
    mixer = RoLAMixer(d_model=scaling.DM, layer_idx=0, n_heads=C.H,
                      d_v=scaling.PERF_NAMED_DV, **kwargs)
    cfg = mixer.resolved_config()
    assert cfg["N"] == 256 and cfg["head_v_dim"] == scaling.PERF_NAMED_DV
    assert cfg["num_heads"] == C.H and cfg["hidden_size"] == scaling.DM
    assert cfg["decay"] == kwargs["decay"] and cfg["state_norm"] == kwargs["state_norm"]


@pytest.mark.parametrize("routing,nc,expect", [
    ("flat", 256, [256]), ("square", 256, [16, 16]), ("tree", 256, [2] * 8)])
def test_sweep_arm_topology_axis(routing, nc, expect):
    """The (d, nc, routing) sweep arm's STRUCTURE axis. `d_qk` is gone from it — a V3 state is
    [N, d_v] — and `gla` now selects `decay`, not the retired `kernel='gla_scalar'`."""
    from zoology.mixers.rola import RoLAMixer
    from rola_bench.rola_compat import level_routing_config_from_tie_routers, topology_from_legacy
    from rola_bench.engine import MAX_ROUTING_DEPTH
    D, b = topology_from_legacy(nc, routing)
    if D > MAX_ROUTING_DEPTH:
        pytest.skip(f"routing={routing!r} at nc={nc} needs depth {D} > engine cap {MAX_ROUTING_DEPTH}")
    mixer = RoLAMixer(d_model=scaling.DM, layer_idx=0, n_heads=C.H, d_v=16,
                      levels_per_head=D, states_per_level=b,
                      level_routing_configs=[level_routing_config_from_tie_routers(False)] * D,
                      decay="none", state_norm="global", router_bias=True,
                      use_short_conv=False, gain_bias_init=None)
    cfg = mixer.resolved_config()
    assert cfg["states_per_level"] == expect and cfg["N"] == nc
    assert cfg["decay"] == "none"


def test_sweep_arm_gla_flag_selects_the_leaf_mass_clock():
    from zoology.mixers.rola import RoLAMixer
    from rola_bench.rola_compat import level_routing_config_from_tie_routers
    mixer = RoLAMixer(d_model=scaling.DM, layer_idx=0, n_heads=C.H, d_v=16,
                      levels_per_head=2, states_per_level=16,
                      level_routing_configs=[level_routing_config_from_tie_routers(False)] * 2,
                      decay="leaf_mass", state_norm="global", router_bias=True,
                      use_short_conv=False, gain_bias_init=None)
    assert mixer.resolved_config()["decay"] == "leaf_mass"
