"""GATES for `chromatic_recall.data.graph_recall` — the graph-structured recall generator (2026-08-02).

The generator's whole value is that the sequence distribution has a KNOWN co-occurrence graph, so
these gates check the three things that claim rests on, and nothing decorative:

  §1 KNOWN DEMAND. For the three constructed families the realized graph's demand (its chromatic
     number, measured off the arrays the renderer actually consumes) equals the construction's
     formula: `complete(k) -> k`, `clustered(k, m) -> k` for every m, `overlap(w, ...) -> w` for
     every birth count and both lifetime profiles. For `reuse` there is no formula and the gate is
     that the reported INTERVAL is well-formed and that the cell carries no nominal claim.
  §2 COVERAGE. The model sees the UNION of the sampled patches, a subgraph of the target, so the
     stamp must MOVE with the sampling rather than echo the knob.
  §3 THE CONTRACT. Every written key is queried, after it was written, with the value it was written
     with, and nothing else in the sequence is labelled — which is what makes the evaluation land on
     REALIZED EDGES (pairs some sequence forced apart) rather than on pairs that never conflicted.
  §4 DETERMINISM. Same pattern as `test_mqar_determinism.py` / `test_cue_consistency.py`: every draw
     through `np.random.default_rng(seed)`, gated behaviourally AND by walking the module's parsed
     call graph for global-RNG reads.

Symbols: `k` = keys a sequence forces apart; `m` = keys per cluster / per birth slot; `w` = maximum
simultaneously live keys; `B` = birth slots per sequence; `Lg` = long-lived slots (the mixed cell);
`chi` = the graph's chromatic number = the demand; `|V|` = realized node count.

CPU only. A visible GPU is needed only because pytest collection imports Triton.
"""
from __future__ import annotations

import pytest

BASE = {"vocab_size": 1024, "num_examples": 2000, "input_seq_len": 128}


def _episodes(**kw):
    from chromatic_recall.data.graph_recall import build_episodes
    return build_episodes(**{**BASE, "seed": 0, **kw})


def _stamp(**kw):
    from chromatic_recall.data.graph_recall import stamp
    return stamp(_episodes(**kw))


def test_predicted_N_is_stamped_for_every_construction():
    """The comparison the grid's headline statistic is built on (`fsla_bench.mqar.analysis.graph`): the cell
    carries the demand the theory predicts it needs, as its own field."""
    assert _stamp(construction="complete", demand=8)["graph_predicted_N"] == "8"
    assert _stamp(construction="overlap", demand=8, births=16)["graph_predicted_N"] == "8"
    reuse = _stamp(num_examples=500, construction="reuse", demand=8, pool_size=128, zipf_a=1.5)
    lo, hi = reuse["graph_demand_lower"], reuse["graph_demand_upper"]
    assert reuse["graph_predicted_N"] == (str(lo) if lo == hi else f"{lo}-{hi}")


# =============================================================== 6. the grid around the generator
def _spec():
    import os

    from fsla_bench.mqar.build_configs import load_spec
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    here = os.path.join(repo, "fsla_bench", "mqar", "experiments")
    return load_spec(os.path.join(here, "graph_grid.yaml"))


#: the header's sizes net of the rungs a D=2 wiring has no uniform spelling at (N = 32 and N = 128: squares only)
@pytest.mark.parametrize("tier,cells", [("decoupling", 36), ("resize", 28), ("overlap", 32),
                                        ("decay", 32), ("reuse", 20), ("contract", 36),
                                        ("cue_map", 32), ("attn_reference", 10),
                                        ("attn_capacity", 28)])
def test_every_tier_is_the_size_its_header_states(tier, cells):
    """The header's arithmetic is a schedule commitment (each tier is one overnight unit on the local
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


def test_the_slate_is_the_three_wirings_plus_the_control_plus_an_attention_reference():
    """The fixed-state class baselines (rla / gla / gdn) are established on the standard law grid and
    no claim in this file is about them; a stray `baselines` arm would silently triple a tier. The
    attention arm is admitted in EXACTLY ONE tier, as the instrument floor — if it ever appeared on a
    capacity tier it would be read as a contender on the N ladder, which it cannot be."""
    from fsla_bench.models.fsla import CANONICAL_WIRINGS
    for tier, arms in _spec()["tiers"].items():
        kinds = {a["build"] for a in arms}
        if tier.startswith("attn"):
            assert kinds == {"attn"}
            continue
        assert kinds == {"routed"}, f"{tier} carries a non-routed arm"
        assert {a["instance"] for a in arms} == set(CANONICAL_WIRINGS), f"{tier} is not the slate"


def test_the_attention_reference_covers_every_family_and_is_never_on_the_n_ladder():
    """One capacity-adequate cell per FAMILY (construction), so no family's routed failure can be
    read as a capacity result without a solvability check behind it. And no `ncs`: attention has no
    rung on the state axis."""
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    arms = spec["tiers"]["attn_reference"]
    covered = {conds[t]["construction"] for a in arms for t in a["conditions"]}
    assert covered == {"complete", "clustered", "overlap", "reuse"}
    assert all("ncs" not in a for a in arms)
    # ... and every attention cell's L matches the length its condition's routed cells train at
    routed_len = {}
    for tier, tarms in spec["tiers"].items():
        if tier == "attn_reference":
            continue
        for a in tarms:
            for t in a["conditions"]:
                routed_len[t] = a.get("seq_lens", spec["seq_lens"])[0]
    for a in arms:
        for t in a["conditions"]:
            assert a.get("seq_lens", spec["seq_lens"])[0] == routed_len[t], (
                f"attention runs {t} at a different L than the routed arms do; the floor would not "
                "be the floor of the cells it is supposed to calibrate")


def test_every_run_id_parses_for_the_analysis():
    """The analysis keys its rows off the run_id, so a template change that the parser does not know
    about would silently produce an empty headline table rather than an error."""
    import os

    from fsla_bench.mqar.analysis.graph import parse_run_id
    from fsla_bench.mqar.build_configs import build_configs
    os.environ["GRID_TIERS"] = "all"
    configs, _ = build_configs(_spec())
    conds = {c["tag"] for c in _spec()["conditions"]}
    for cfg in configs:
        got = parse_run_id(cfg.run_id)
        assert got is not None, f"analysis cannot parse {cfg.run_id}"
        assert got[4] in conds, f"{cfg.run_id} parsed to unknown condition {got[4]!r}"


def test_the_attention_capacity_tier_sweeps_d_qk_over_three_distinct_demands():
    """A slope needs three points. The d_qk ladder must also be a divisor ladder of d_model, since
    zoology's MHA derives head_dim = d_model // num_heads."""
    spec = _spec()
    arms = spec["tiers"]["attn_capacity"]
    d_model = spec["backbone"]["d_model"]
    demands = set()
    for a in arms:
        assert a["head_dims"], "the capacity tier must sweep d_qk"
        for d in a["head_dims"]:
            assert d_model % d == 0, f"d_qk={d} does not divide d_model={d_model}"
        for t in a["conditions"]:
            demands.add(t)
    # the three demand levels the fit needs: chi = 16 (complete-k16 / clustered), 33 (the contract
    # cell) and 64 (complete-k64)
    assert {"complete-k16", "complete-k64", "complete-k64-contract50"} <= demands


def test_the_N_ladders_bracket_the_demand_they_are_pointed_at():
    """A tier whose N ladder sits entirely above (or below) its condition's chi cannot see a collapse
    threshold at all, which would make the headline statistic unmeasurable for that cell."""
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    for tier, arms in spec["tiers"].items():
        for arm in arms:
            ncs = arm.get("ncs", spec["ncs"])
            for tag in arm["conditions"]:
                demand = conds[tag].get("demand")
                if conds[tag]["construction"] == "reuse":
                    continue                     # its chi is emergent (~125); checked by measurement
                if conds[tag].get("contract_fraction", 1.0) < 1.0:
                    # the demanded-apart graph is K_cov joined to an independent rest, so its chi is
                    # cov + 1, not `demand` — the ladder is checked against the SOLVED number
                    demand = int(round(conds[tag]["contract_fraction"] * demand)) + 1
                assert min(ncs) <= demand, f"{tier}/{tag}: ladder {ncs} starts above chi={demand}"
                assert max(ncs) >= demand, f"{tier}/{tag}: ladder {ncs} ends below chi={demand}"


def test_the_grid_stamps_the_realized_demand_as_a_slice_key():
    from chromatic_recall.data.graph_recall import STAMP_FIELDS
    keys = _spec()["train"]["slice_keys"]
    assert "graph_demand_upper" in keys and "graph_construction" in keys
    assert set(keys) <= set(STAMP_FIELDS) | {"num_kv_pairs", "input_seq_len"}
