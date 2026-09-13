"""GATES for the rebuilt MQAR grid (Phase 1, 2026-08-02): the preset registry, the contender
ladder, the realized-state readback, and every experiment spec.

The MQAR tier is now the paper's ENTIRE training evidence (v1 has no LM tier), so the bar here is
publication-grade: a preset that constructs is not enough — it must construct the cell its NAME
claims, its capacity coordinate must be measured rather than computed, and its spec must expand to
the cells its header says it does.

Mostly CPU. The standalone engine (2026-08-15 repoint) has no CPU forward arm at all, so the one
gate that exercises a forward is CUDA-marked and skips without a device.
"""
from __future__ import annotations

import os

import pytest

_EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mqar", "experiments")
_SPECS = ("law_grid", "separation", "ood_recall", "canonical", "dmodel64",
          "cue_consistency", "graph_grid", "presence_grid")
#: `router_bias` is PARKED by the standalone repoint and is deliberately NOT in `_SPECS`: its whole
#: routed ladder is pinned at N = 128, which has no uniform D=2 spelling (128 = 2^7) and no built
#: D=1 arm, so every one of its cells drops with the engine's reason and `build_configs` refuses the
#: resulting empty grid by design. It needs its own squares amendment before it is gated again
#: (campaign-dispatch b2-3_build.md, finding F-2).

#: The smallest state rung with a built arm at BOTH depths the slate uses (D=1 as [64], D=2 as
#: [8,8]). Read the derivation off `instances._check_built_arm`: the operator's manifest is what
#: decides, and 64 is where the ladder starts today.
_RUNG = 64


# =============================================================== 1. the preset registry
def test_slate_is_the_three_wirings_plus_the_d1_control():
    from rola_bench.mqar.instances import CANONICAL_WIRINGS, ROLA_INSTANCES
    assert CANONICAL_WIRINGS == ("rola-arm1-densread-sparsewrite", "rola-arm2-union",
                                 "rola-arm3-levelsplit", "rola-d1-dense")
    for name in CANONICAL_WIRINGS:
        assert name in ROLA_INSTANCES
    # variants exist only where a grid axis demands one; nothing else may creep onto the slate.
    assert set(ROLA_INSTANCES) - set(CANONICAL_WIRINGS) == {
        "rola-arm3-levelsplit-a2w",      # the alpha axis
        "rola-arm1-tied", "rola-d1-dense-tied",   # the coupling-independence pair
    }


@pytest.mark.parametrize("name", [
    "rola-arm1-densread-sparsewrite", "rola-arm2-union", "rola-arm3-levelsplit",
    "rola-arm3-levelsplit-a2w", "rola-d1-dense", "rola-arm1-tied", "rola-d1-dense-tied",
])
def test_every_preset_constructs(name):
    """CPU CONSTRUCTION, at the smallest rung with a built arm.

    The forward is CUDA-only in the standalone engine (there is no CPU arm and no fallback), so a
    forward gate lives in `test_canonical_wirings_train_on_the_kernel` below and this one asserts
    what CPU can assert: the preset's kwargs build the module the registry says they do."""
    from rola_bench.mqar.instances import rola_instance
    from zoology.mixers.rola import RoLAMixer
    kw = rola_instance(name, _RUNG)
    mixer = RoLAMixer(d_model=64, layer_idx=0, **kw)
    assert mixer.topology.N == _RUNG and mixer.d_v == 64


@pytest.mark.parametrize("name", ["rola-arm1-densread-sparsewrite", "rola-arm2-union",
                                  "rola-arm3-levelsplit", "rola-d1-dense"])
def test_canonical_wirings_train_on_the_kernel(name):
    """fwd + bwd on the CUDA arm, at the grid's own geometry — the property the grid's cells need
    and the one a CPU construction gate cannot see (the layer-dispatch autograd gap)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from rola_bench.mqar.instances import MQAR_GEOM, rola_instance
    from zoology.mixers.rola import RoLAMixer
    d_model = MQAR_GEOM["n_heads"] * MQAR_GEOM["d_v"]
    mixer = RoLAMixer(d_model=d_model, layer_idx=0, **rola_instance(name, _RUNG)).cuda().train()
    x = torch.randn(2, 64, d_model, device="cuda", requires_grad=True)
    out = mixer(x)
    out.float().pow(2).mean().backward()
    assert out.shape == (2, 64, d_model) and bool(torch.isfinite(out).all())
    assert mixer.last_execution_backend == "cuda"
    assert x.grad is not None and float(mixer.routes.route_W.grad.abs().sum()) > 0


@pytest.mark.parametrize("name,reason_contains", [
    ("rola-rla-kappa-asym", "kappa"),
    ("rola-gla-scalar-sym", "gla_scalar"),
    ("rola-rla-asym-ps", "per_state"),
    ("rola-rla-asym-tieinit", "tie_router_init"),
    ("rola-gla-kappa-asym", "doubly retired"),
])
def test_retired_presets_refuse_by_name(name, reason_contains):
    from rola_bench.mqar.instances import RETIRED_INSTANCES, rola_instance
    assert name in RETIRED_INSTANCES
    assert reason_contains in RETIRED_INSTANCES[name]
    with pytest.raises(ValueError, match="RETIRED"):
        rola_instance(name, 16)


def test_all_thirteen_pre_v3_presets_have_a_disposition():
    """The retirement ledger must be COMPLETE — a preset that simply vanished would leave the 683
    pre-V3 result rows in results/mqar/ uninterpretable."""
    from rola_bench.mqar.instances import RETIRED_INSTANCES
    pre_v3 = {'rola-rla-asym', 'rola-rla-sym', 'rola-rla-asym-ps', 'rola-rla-sym-ps',
              'rola-rla-kappa-asym', 'rola-rla-kappa-sym', 'rola-rla-asym-tieinit',
              'rola-gla-scalar-sym', 'rola-gla-scalar-norm-sym', 'rola-gla-scalar-asym',
              'rola-gla-scalar-norm-asym', 'rola-gla-kappa-sym', 'rola-gla-kappa-asym'}
    assert pre_v3 <= set(RETIRED_INSTANCES)
    for name, reason in RETIRED_INSTANCES.items():
        assert len(reason) > 60, f"{name} has a stub reason; a disposition must be readable"


def test_presets_emit_no_retired_kwarg():
    from zoology.mixers.rola import RETIRED_KWARGS
    from rola_bench.mqar.instances import ROLA_INSTANCES, rola_instance
    for name in ROLA_INSTANCES:
        kw = rola_instance(name, _RUNG)
        assert not (set(kw) & set(RETIRED_KWARGS)), f"{name} emits retired vocabulary: {sorted(set(kw) & set(RETIRED_KWARGS))}"


# =============================================================== 2. wirings mean what they claim
_UNION_EXPECTED = {"rola-arm2-union": (0,)}          # preset -> level indices that must be UnionRouting


@pytest.mark.parametrize("name", ["rola-arm1-densread-sparsewrite", "rola-arm2-union",
                                  "rola-arm3-levelsplit", "rola-d1-dense"])
def test_routing_resolves_to_the_intended_engine_structure(name):
    """UNION is the one wiring whose engine resolution differs structurally from the others: an
    untied sparse/sparse level becomes `UnionRouting` (one shared entmax support split by sigma),
    everything else becomes `IndependentRouting` (each duty solves its own exact activation). The
    slate is only a three-way ablation if that distinction actually lands."""
    from rola_bench.engine import import_grid_engine
    from rola_bench.mqar.instances import rola_instance
    IndependentRouting = import_grid_engine().IndependentRouting
    UnionRouting = import_grid_engine().UnionRouting
    from zoology.mixers.rola import RoLAMixer
    mixer = RoLAMixer(d_model=64, layer_idx=0, **rola_instance(name, _RUNG))
    union_levels = _UNION_EXPECTED.get(name, ())
    for index, level in enumerate(mixer.routes.levels):
        want = UnionRouting if index in union_levels else IndependentRouting
        assert isinstance(level, want), (
            f"{name}: level {index} resolved to {type(level).__name__}, expected {want.__name__}")


def test_d1_dense_control_is_a_single_dense_level():
    """The map-family control: one dense softmax level, no factorization. If this ever gains a level
    or a sparse duty it stops being the control the structured wirings are measured against."""
    from rola_bench.engine import import_grid_engine
    from rola_bench.mqar.instances import rola_instance
    SoftmaxActivation = import_grid_engine().SoftmaxActivation
    from zoology.mixers.rola import RoLAMixer
    mixer = RoLAMixer(d_model=64, layer_idx=0, **rola_instance("rola-d1-dense", _RUNG))
    assert len(mixer.routes.levels) == 1 and mixer.topology.N == _RUNG
    level = mixer.routes.levels[0]
    assert isinstance(level.read, SoftmaxActivation)
    assert isinstance(level.write, SoftmaxActivation)


def test_a2w_variant_differs_from_arm3_in_alpha_alone():
    from rola_bench.mqar.instances import rola_instance
    base = rola_instance("rola-arm3-levelsplit", _RUNG)
    a2w = rola_instance("rola-arm3-levelsplit-a2w", _RUNG)
    for key in set(base) | set(a2w):
        if key == "level_routing_configs":
            continue
        assert base[key] == a2w[key], f"the alpha variant also moved {key!r}; it must move alpha alone"
    for b, a in zip(base["level_routing_configs"], a2w["level_routing_configs"]):
        assert (b["tied"], b["read"], b["write"]) == (a["tied"], a["read"], a["write"])
    assert a2w["level_routing_configs"][0].get("alpha") == 2.0   # sparse WRITE duty -> sparsemax
    assert a2w["level_routing_configs"][1].get("alpha") == 1.5   # sparse READ duty  -> entmax-1.5


def test_mqar_presets_agree_with_the_named_config_registry():
    """The MQAR spelling and the layer/perf spelling of one published wiring must be the SAME cell.
    If these drift, an MQAR row and a perf row of the same arm are different experiments
    ([[matching-the-naive-antipattern]])."""
    from rola_bench.models.named_configs import NAMED_CONFIGS
    from rola_bench.mqar.instances import rola_instance
    pairs = {
        "rola-arm1-densread-sparsewrite": "RoLA-base-rla",
        "rola-arm2-union": "RoLA-base-union-global",
        "rola-arm3-levelsplit": "RoLA-base-levelsplit-global",
        "rola-arm3-levelsplit-a2w": "RoLA-base-levelsplit-a2w-global",
    }
    for mqar_name, named in pairs.items():
        mqar_rows = rola_instance(mqar_name, 256)["level_routing_configs"]
        layer_rows = NAMED_CONFIGS[named].build_kwargs(256, False)["level_routing_configs"]
        assert len(mqar_rows) == len(layer_rows), f"{mqar_name} vs {named}: different depth"
        for m, l in zip(mqar_rows, layer_rows):
            assert (m["tied"], m["read"], m["write"]) == (l.tied, l.read, l.write), (
                f"{mqar_name} vs {named}: routing row differs")
            assert float(m.get("alpha", 1.5)) == float(l.alpha), (
                f"{mqar_name} vs {named}: alpha differs")


# =============================================================== 3. the capacity coordinate
def test_realized_state_is_read_back_not_computed():
    """H*N*d_v content + H*N denominator under 'global', MEASURED off the constructed layer."""
    from rola_bench.mqar.instances import MQAR_GEOM, realized_state_floats
    H, dv = MQAR_GEOM["n_heads"], MQAR_GEOM["d_v"]
    for nc in (64, 256):                       # the rungs with a built arm; see `_RUNG`
        stats = realized_state_floats("rola-arm1-densread-sparsewrite", nc)
        assert stats["num_chunks"] == nc and stats["d_v"] == dv and stats["n_heads"] == H
        assert stats["content"] == H * nc * dv
        assert stats["denominator"] == H * nc      # the 'global' sidecar, observed


def test_ref_is_derived_from_the_pinned_geometry():
    """The `624*nc` constant was written in the DELETED d_qk and over-sized every baseline 12x.
    REF must now MOVE when the geometry moves — that property is the fix, not the number."""
    from rola_bench.mqar import instances
    import rola_bench.mqar.experiments.canonical_baselines as cb
    for nc in (2, 16, 256):
        assert cb.REF(nc) == instances.content_state_floats(nc)
    assert cb.REF(1) != 624, "REF is still the pre-V3 d_qk-based constant"


def test_readback_covers_every_arm_the_grid_builds():
    from rola_bench.mqar.realized_state import state_floats
    import rola_bench.mqar.experiments.canonical_baselines as cb
    from rola_bench.mqar import common as C
    for method in cb.METHODS:
        for shape in ("wide", "square"):
            cell = cb.baseline_cell(method, shape, 64)
            if cell is None:
                continue
            content, overhead = state_floats(cell[0], cb.DMODEL)
            assert isinstance(content, int) and content > 0 and overhead >= 0
    assert state_floats(C.mha, cb.DMODEL) == (None, 0)   # attention: unbounded, no rung


def test_baseline_readback_is_within_a_rounding_step_of_the_target():
    """The matched-state comparison only holds if a cell INSTANTIATES at its nominal target; an
    `int(expand_v * d_model)` truncation could silently under-size a baseline and flatter RoLA."""
    from rola_bench.mqar.realized_state import state_floats
    import rola_bench.mqar.experiments.canonical_baselines as cb
    for method in cb.METHODS:
        for shape in cb.SHAPES:
            for nc in (16, 64, 256):
                cell = cb.baseline_cell(method, shape, nc)
                if cell is None:
                    continue
                kernel, target = cell
                content, _ = state_floats(kernel, cb.DMODEL)
                assert abs(content - target) / target <= 0.12, (
                    f"{method}/{shape}/nc{nc}: realized {content} vs target {target}")


# =============================================================== 4. the contender ladder
def test_contender_ladder_is_one_occupant_per_rung():
    import rola_bench.mqar.experiments.canonical_baselines as cb
    assert cb.METHODS == ["rla", "gla", "gdn"]
    for retired in ("hedgehog", "based", "sse"):
        assert retired in cb.RETIRED_METHODS
        with pytest.raises(ValueError, match="RETIRED"):
            cb.baseline_cell(retired, "wide", 16)


def test_gdn_structural_exit_is_reported_not_hidden():
    """GDN's `head_dim <= 256` kernel cap is a COST COLUMN. The gate pins where it bites, so a
    paper row can state the exit rung instead of showing a gap."""
    import rola_bench.mqar.experiments.canonical_baselines as cb
    assert cb.baseline_cell("gdn", "wide", 256) is not None
    assert cb.baseline_cell("gdn", "wide", 512) is None       # head_dim = N > 256
    assert cb.baseline_cell("gdn", "square", 1024) is not None
    assert cb.baseline_cell("gdn", "square", 2048) is None    # head_dim = 8*sqrt(N) > 256
    # ... and it is a GDN-specific bound, not a grid-wide one: the other rungs still reach.
    assert cb.baseline_cell("gla", "wide", 2048) is not None
    assert cb.baseline_cell("rla", "wide", 2048) is not None


def test_sse_builder_is_retired():
    from rola_bench.mqar.build_configs import RETIRED_BUILDERS, build_configs
    assert "sse_ladder" in RETIRED_BUILDERS
    spec = _load("law_grid")
    spec["tiers"] = {"x": [{"build": "sse_ladder"}]}
    with pytest.raises(ValueError, match="RETIRED"):
        build_configs(spec)


# =============================================================== 5. the specs
def _load(name):
    from rola_bench.mqar.build_configs import load_spec
    return load_spec(os.path.join(_EXP_DIR, f"{name}.yaml"))


@pytest.mark.parametrize("name", _SPECS)
def test_every_spec_expands(name):
    from rola_bench.mqar.build_configs import build_configs
    configs, envs = build_configs(_load(name))
    assert configs and len(configs) == len(envs)
    ids = [c.run_id for c in configs]
    assert len(ids) == len(set(ids)), (
        f"{name}: {len(ids) - len(set(ids))} duplicate run_id(s) — result rows would collide and "
        "silently overwrite each other")


@pytest.mark.parametrize("name", _SPECS)
def test_every_spec_carries_at_least_three_seeds(name):
    spec = _load(name)
    seeds = spec["train"].get("seeds") or [spec["train"]["seed"]]
    assert len(seeds) >= 3 or spec.get("single_seed_rationale"), (
        f"{name}: {len(seeds)} seed(s) and no written rationale")


def test_seed_floor_actually_bites():
    from rola_bench.mqar.build_configs import build_configs
    spec = _load("law_grid")
    spec["train"]["seeds"] = [0]
    with pytest.raises(ValueError, match="seed"):
        build_configs(spec)


def test_retired_specs_refuse_by_name():
    from rola_bench.mqar.run import RETIRED_SPECS, load_configs
    assert set(RETIRED_SPECS) == {"qknorm_ablation", "normsym"}
    for name in RETIRED_SPECS:
        assert not os.path.exists(os.path.join(_EXP_DIR, f"{name}.yaml"))
        with pytest.raises(ValueError, match="RETIRED"):
            load_configs(name)


def test_no_spec_names_retired_vocabulary():
    from rola_bench.mqar.instances import RETIRED_INSTANCES
    import rola_bench.mqar.experiments.canonical_baselines as cb
    for name in _SPECS:
        spec = _load(name)
        for tier in spec["tiers"].values():
            for arm in tier:
                assert "d_qk" not in arm, f"{name}: an arm still carries d_qk"
                assert arm.get("instance") not in RETIRED_INSTANCES, f"{name}: retired instance"
                for method in arm.get("methods", []):
                    assert method not in cb.RETIRED_METHODS, f"{name}: retired method {method}"


# --- the law grid's own design properties --------------------------------------------------------
def test_law_grid_walks_alpha_cap_through_one_in_both_directions():
    """The boundary claim is about alpha_cap = N/L crossing 1. If the ladders never straddle it, the
    grid cannot see the boundary no matter how many cells it runs."""
    spec = _load("law_grid")
    ratios = [nc / L for nc in spec["ncs"] for L in spec["seq_lens"]]
    assert min(ratios) < 1 < max(ratios)
    diagonal = sorted(set(spec["ncs"]) & set(spec["seq_lens"]))
    assert len(diagonal) >= 3, f"the N = L diagonal has only {diagonal}"


def test_law_grid_unconfounds_items_from_length():
    """The SAME item ladder must be evaluated at every length; `ext` tied kv to L, which is exactly
    what made its curves un-interpretable."""
    spec = _load("law_grid")
    proto = spec["data"]["protocol"]
    ladders = {L: set(kvs) for L, kvs in proto["test_kv_by_len"].items()}
    longest = max(ladders)
    for L, kvs in ladders.items():
        # every length carries the whole ladder that FITS in it (zoology's 4*kv <= L)
        assert kvs == {kv for kv in ladders[longest] if 4 * kv <= L}, f"L={L} ladder is {sorted(kvs)}"


def test_law_grid_has_extrapolation_cells():
    spec = _load("law_grid")
    assert max(spec["data"]["protocol"]["test_seq_lens"]) > max(spec["seq_lens"]), (
        "no test length exceeds the longest train length — there is no extrapolation axis")
    assert "input_seq_len" in spec["train"]["slice_keys"], (
        "extrapolation slices are not separable without slicing on input_seq_len")


def test_separation_cells_sit_on_the_diagonal():
    """N = L exactly, at 16K-64K. A cell off the diagonal here would not be a separation cell."""
    spec = _load("separation")
    for tier, arms in spec["tiers"].items():
        if not tier.startswith("diag"):
            continue
        for arm in arms:
            if arm["build"] != "routed":
                continue
            assert arm["ncs"] == arm["seq_lens"], f"{tier}: {arm['tag']} is off the N = L diagonal"
    rungs = sorted({arm["ncs"][0] for t, arms in spec["tiers"].items() if t.startswith("diag")
                    for arm in arms if arm["build"] == "routed"})
    assert rungs == [16384, 32768, 65536]


def test_separation_gdn_is_capped_at_its_reachable_rungs():
    """GDN's presence in this spec must be its MEASURED ceiling, not a cell it cannot build."""
    import rola_bench.mqar.experiments.canonical_baselines as cb
    spec = _load("separation")
    for arm in spec["tiers"]["gdn_ceiling"]:
        if "gdn" not in arm["methods"]:
            continue
        for shape in arm["shapes"]:
            for nc in arm["ncs"]:
                assert cb.baseline_cell("gdn", shape, nc) is not None, (
                    f"separation names gdn/{shape}/nc{nc}, which is above the head_dim cap")


def test_ood_recall_holds_the_ratio_and_sweeps_absolute_length():
    spec = _load("ood_recall")
    proto = spec["data"]["protocol"]
    train_L, train_kv = spec["seq_lens"][0], proto["train_kv"][0]
    ratio = train_kv / train_L
    for L, kvs in proto["test_kv_by_len"].items():
        assert kvs[0] / L == ratio, f"L={L}: the fixed-ratio ladder point is {kvs[0]}, not {L * ratio:g}"
    fixed_item = [kvs[1] for L, kvs in proto["test_kv_by_len"].items() if len(kvs) > 1]
    assert fixed_item and len(set(fixed_item)) == 1, (
        "the fixed-ITEM ladder is missing or not constant; without it the capacity story and the "
        "scale-freeness story cannot be separated")
    assert max(proto["test_seq_lens"]) >= 16 * train_L


def test_attention_dqk_ladder_expands():
    """`d_qk` survives on the attention baseline only — its own capacity dimension, and the fitted
    exchange rate against RoLA's N is a paper row."""
    from rola_bench.mqar.build_configs import build_configs
    spec = _load("canonical")
    spec["tiers"] = {"attn": [{"build": "attn", "head_dims": [2, 4, 8, 16, 32]}]}
    configs, _ = build_configs(spec)
    tags = {c.run_id.split("_")[0] for c in configs}
    assert tags == {f"grid-mha-dqk{d}" for d in (2, 4, 8, 16, 32)}


def test_d1_control_ceiling_is_pinned_and_reported():
    """The unfactorized control cannot exist above the engine's `MAX_BRANCH_WIDTH`, because a D=1
    router is ONE softmax over N logits. That is a reported structural result — the argument for
    factorization as a buildability fact — so it is pinned here rather than discovered on a rented
    box, and the separation spec must not name a d1 arm on the diagonal."""
    from rola_bench.engine import MAX_BRANCH_WIDTH
    from rola_bench.mqar.instances import PresetRungUnavailable, rola_instance
    assert rola_instance("rola-d1-dense", MAX_BRANCH_WIDTH)["states_per_level"] == [MAX_BRANCH_WIDTH]
    with pytest.raises(PresetRungUnavailable, match="MAX_BRANCH_WIDTH"):
        rola_instance("rola-d1-dense", 2 * MAX_BRANCH_WIDTH)
    # the factorized wirings still reach far past it
    assert rola_instance("rola-arm1-densread-sparsewrite", 65536)["states_per_level"] == [256, 256]
    # law_grid's N ladder stops exactly where the control stops, so every law cell has its control
    assert max(_load("law_grid")["ncs"]) <= MAX_BRANCH_WIDTH
    for tier, arms in _load("separation")["tiers"].items():
        if tier.startswith("diag"):
            assert all(a.get("instance") != "rola-d1-dense" for a in arms)


def test_the_grid_refuses_a_rung_with_no_built_arm():
    """THE READOUT TILE IS NO LONGER A CELL PARAMETER (standalone repoint, 2026-08-15). BC is the
    operator's decision, so the grid's buildability question moved from "does the pinned tile divide
    N" to "is this (D, B) in the built arm matrix, forward AND backward" — and it is asked of the
    compiled binary's own census. This gate pins the CONSEQUENCE: no preset emits a `state_block`,
    every rung on the presence grid's squares ladder resolves to a TRAINABLE arm, and a rung below
    the built floor drops with the engine's own reason instead of failing at step 1 on a rented box.

    (The perf tier still pins its own `PERF_STATE_BLOCK` against the LEGACY engine and is NOT
    repointed — campaign-dispatch b2-3_build.md, finding F-3.)"""
    from rola_bench.engine import chunk_arm_for, chunk_scan_arms
    from rola_bench.mqar.instances import (CANONICAL_WIRINGS, PresetRungUnavailable, rola_instance)
    for name in CANONICAL_WIRINGS:
        for nc in (64, 256):
            kw = rola_instance(name, nc)
            assert "state_block" not in kw
            b = kw["states_per_level"]
            arm = chunk_arm_for(len(b), b[0])
            assert arm is not None and (*arm, len(b), b[0]) in chunk_scan_arms(), (
                f"{name}@nc{nc} spells widths {b}, which is not a trainable arm")
        with pytest.raises(PresetRungUnavailable, match="no built forward arm|no UNIFORM"):
            rola_instance(name, 16)
