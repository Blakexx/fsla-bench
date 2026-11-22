"""GATES for `chromatic_recall.data.presence_recall` — v1's theory-confirmation instrument (2026-08-03).

The task: a frozen map `pi` over the key vocabulary with `pi(x) != x`, a sequence of key tokens, and
one bit at every key position — has `pi(x_t)` appeared STRICTLY BEFORE t? Everything the instrument
claims rests on five properties, and these gates check those and nothing decorative:

  §1 THE LABEL IS THE TASK. Every emitted label equals a brute-force prefix membership check written
     independently of the emission machinery ([[matching-the-naive-antipattern]] — a second copy of
     the fast path would only ever prove itself self-consistent), and the labels land on the key
     positions, in the reserved token pair, with the first event of each sequence UNLABELLED.
  §2 pi IS FROZEN AND IS A DERANGEMENT. `pi(x) != x` everywhere, drawn from `pi_seed` and never from
     the segment seed, so a cell's train and eval segments share one map; a different `pi_seed` is a
     different map, and the map is stamped.
  §3 THE CLASS BALANCE IS PLANTED, not hoped for: ~50/50 overall AND flat in position (the stamp's
     `presence_position_bias`), which is what makes the 0.5 chance line sharp and what stops a
     position-only model from scoring above it.
  §4 THE DEMAND IS chi AND IT IS SOLVED. The realized union graph is the complete graph on the pool,
     solved by `graph_chi` on the `disjoint_cliques` rung and gated against the closed form; an
     under-sampled segment is REFUSED rather than shipped with the wrong x-coordinate.
  §5 DETERMINISM. Same pattern as `test_graph_recall.py` / `test_mqar_determinism.py`: every draw
     through `np.random.default_rng(seed)`, gated behaviourally AND by walking the module's parsed
     call graph for global-RNG reads.

Symbols: `pi` = the frozen read map; `K` = `demand` = key-pool size = this task's chi; `s` =
`keys_per_sequence`; `L` = `input_seq_len` in tokens; `chi` = the graph's chromatic number.

CPU only. A visible GPU is needed only because pytest collection imports Triton.
"""
from __future__ import annotations

import pytest

BASE = {"vocab_size": 1024, "num_examples": 2000, "input_seq_len": 64}


# =============================================================== 6. the grid around the generator
def _spec():
    import os

    from fsla_bench.mqar.build_configs import load_spec
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    here = os.path.join(repo, "fsla_bench", "mqar", "experiments")
    return load_spec(os.path.join(here, "presence_grid.yaml"))


@pytest.mark.parametrize("tier,cells", [("presence16", 48), ("presence64", 48),
                                        ("attn_capacity", 60), ("presence_decay16", 36)])
def test_every_tier_is_the_size_its_header_states(tier, cells):
    """The header's arithmetic is a schedule commitment (each tier is an overnight unit on the local
    card). If a tier silently doubles, the night it was planned for is not the night it takes."""
    import os

    from fsla_bench.mqar.build_configs import build_configs
    old = os.environ.get("GRID_TIERS")
    os.environ["GRID_TIERS"] = tier
    try:
        configs, _ = build_configs(_spec())
    finally:
        os.environ.pop("GRID_TIERS") if old is None else os.environ.update(GRID_TIERS=old)
    assert len(configs) == cells


def test_the_slate_is_the_three_wirings_plus_the_control_plus_an_attention_column():
    """The fixed-state class baselines (rla / gla / gdn) are established on the law grid and no claim
    in this file is about them; a stray `baselines` arm would silently triple a tier. Attention is
    admitted in EXACTLY ONE tier — on a capacity tier it would be read as a contender on the N
    ladder, which it cannot be (its state is a growing KV cache)."""
    from fsla_bench.models.fsla import CANONICAL_WIRINGS
    for tier, arms in _spec()["tiers"].items():
        kinds = {a["build"] for a in arms}
        if tier.startswith("attn"):
            assert kinds == {"attn"}
            continue
        assert kinds == {"routed"}, f"{tier} carries a non-routed arm"
        #: the decay column carries the three structured wirings only (d1's dense write would decay everything every step)
        slate = set(CANONICAL_WIRINGS) - ({"fsla-d1-dense"} if "decay" in tier else set())
        assert {a["instance"] for a in arms} == slate, f"{tier} is not the slate"


def test_the_N_ladders_are_pointed_at_the_solved_demand():
    """chi = the pool size, so a ladder entirely above (or below) it cannot see a collapse threshold.
    The routed floor N = 16 (the pre-K31 engine's readout tile) means the chi = 16 tier can only run AT the demand
    and upward, which is stated in the file; the chi = 64 tier must genuinely bracket, since it is
    where the crossing is measured."""
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    for tier, arms in spec["tiers"].items():
        if tier.startswith("attn"):
            continue
        for arm in arms:
            ncs = arm.get("ncs", spec["ncs"])
            for tag in arm["conditions"]:
                chi = conds[tag]["demand"]
                assert max(ncs) >= chi, f"{tier}/{tag}: ladder {ncs} ends below chi={chi}"
                assert min(ncs) <= chi, f"{tier}/{tag}: ladder {ncs} starts above chi={chi}"
    k64 = list(spec["tiers"]["presence64"])
    assert all(min(a["ncs"]) < 64 for a in k64), "the resize tier must bracket chi from BELOW"


def test_the_attention_axis_sweeps_d_qk_over_three_distinct_demands():
    """A slope needs three points (`mqar.analysis.graph.fit_packing_constant` refuses two by name), and
    the d_qk ladder must be a divisor ladder of d_model, since zoology's MHA derives
    head_dim = d_model // num_heads."""
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    d_model = spec["backbone"]["d_model"]
    demands = set()
    for a in spec["tiers"]["attn_capacity"]:
        assert a["head_dims"], "the capacity tier must sweep d_qk"
        for d in a["head_dims"]:
            assert d_model % d == 0, f"d_qk={d} does not divide d_model={d_model}"
        for t in a["conditions"]:
            demands.add(conds[t]["demand"])
    assert len(demands) >= 3, f"the packing-constant fit needs three demands, got {sorted(demands)}"


def test_the_attention_column_runs_each_condition_at_its_routed_length():
    """The floor must be the floor of the cells it calibrates: attention at a different L would be
    answering a different question about the same tag."""
    spec = _spec()
    routed_len = {}
    for tier, arms in spec["tiers"].items():
        if tier.startswith("attn"):
            continue
        for a in arms:
            for t in a["conditions"]:
                routed_len[t] = a.get("seq_lens", spec["seq_lens"])[0]
    for a in spec["tiers"]["attn_capacity"]:
        for t in a["conditions"]:
            if t in routed_len:
                assert a.get("seq_lens", spec["seq_lens"])[0] == routed_len[t]


def test_every_run_id_parses_for_the_analysis():
    """The analysis keys its rows off the run_id, so a template change the parser does not know
    about would silently produce an empty headline table rather than an error."""
    import os

    from fsla_bench.mqar.analysis.graph import parse_run_id
    from fsla_bench.mqar.build_configs import build_configs
    os.environ["GRID_TIERS"] = "all"
    configs, _ = build_configs(_spec())
    tags = {c["tag"] for c in _spec()["conditions"]}
    for cfg in configs:
        got = parse_run_id(cfg.run_id)
        assert got is not None, f"analysis cannot parse {cfg.run_id}"
        assert got[4] in tags, f"{cfg.run_id} parsed to unknown condition {got[4]!r}"


def test_the_grid_stamps_the_realized_demand_as_a_slice_key():
    from chromatic_recall.data.presence_recall import STAMP_FIELDS
    keys = _spec()["train"]["slice_keys"]
    assert "graph_demand_upper" in keys and "graph_construction" in keys
    assert set(keys) <= set(STAMP_FIELDS) | {"num_kv_pairs", "input_seq_len"}


def test_the_spec_conditions_build_the_data_they_name():
    """The knob set is validated by `common.make_presence_data`, so a renamed knob refuses here
    rather than being silently dropped and reported as a condition that was never run."""
    from fsla_bench.mqar import common as C
    spec = _spec()
    for cond in spec["conditions"]:
        knobs = {k: v for k, v in cond.items() if k != "tag"}
        assert set(knobs) <= set(C.PRESENCE_KNOBS), cond["tag"]
    with pytest.raises(ValueError, match="unknown presence-recall knob"):
        C.make_presence_data(1024, 8, 8, None, train_seq_len=64, test_seq_lens=[64],
                             train_examples=8, test_examples=8, construction="complete")


# =============================================================== 7. the analysis reads it correctly
def test_the_analysis_knows_this_task_and_its_chance_line():
    """A presence accuracy of 0.55 is noise off a coin flip and a graph-recall accuracy of 0.55 is
    most of the task, so the criterion cannot be shared between them. The task is read off the
    spec's own `data.protocol.kind`, and a criterion at or below the chance line is refused —
    otherwise the reported crossing would be the threshold of GUESSING."""
    from fsla_bench.mqar.analysis.graph import TASKS, analyse, task_of
    spec = _spec()
    task = task_of(spec)
    assert (task.chance, task.criterion) == (0.5, 0.75)
    assert TASKS["graph_recall"].criterion == 0.5, "the graph grid's historical criterion must hold"
    with pytest.raises(ValueError, match="chance line"):
        analyse("presence_grid", criterion=0.5)


def test_the_analysis_resolves_the_demand_with_the_presence_generator():
    """`predicted_chi` REBUILDS each condition's episodes rather than reading a knob (the stamp is
    absent on a segment-cache hit), so it must dispatch to the generator the spec names — solving a
    presence condition with the graph-recall builder would raise or, worse, answer."""
    from fsla_bench.mqar.analysis.graph import predicted_chi
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    for tag, L in (("presence-k16", 64), ("presence-k64", 256)):
        chi = predicted_chi(spec, conds[tag], L, 2000)
        assert chi["lower"] == chi["upper"] == conds[tag]["demand"], tag
