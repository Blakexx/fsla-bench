"""GATES for the cue-consistency instrument (Phase 2, 2026-08-02).

`chromatic_recall.data.cue_consistency` is the data generator for the paper's SECOND named boundary
(Sec. 7.1): address agreement, Eq. (7), `addr_r(u) = addr_w(pi(u))`. Nothing downstream can detect a
knob that does not do what its name says, because the whole experiment IS the knob: a cell reports
"accuracy at cue_digits = 1" and there is no independent witness that the cue actually carried one
digit's worth of address. So these gates measure the MECHANISM, not the plumbing:

  * knob (a) is gated in NATS. Measured mutual information between the query's cue and the target's
    write address, normalized by the address entropy, must equal `q / D`, and the conditional entropy
    must equal `(D - q) * log P` — literally the count of open digits the read marginalizes over.
  * knob (b) is gated by the two properties it is supposed to move (is token -> slot a function; is
    the write's slot order stable across episodes) AND by the property it must NOT move (the knob (a)
    measurement above).
  * knob (c) is gated by the REALIZED match count and by the contract's target, both recomputed from
    the episode arrays rather than trusted from the config.

Symbols: D = key_digits; q = cue_digits; P = digit_vocab; kv = num_kv_pairs; m = matches_per_cue;
G = kv/m groups (= queries per episode); L = input_seq_len.

CPU only. A visible GPU is needed only because pytest collection imports Triton.
"""
from __future__ import annotations

import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXP_DIR = os.path.join(_REPO, "fsla_bench", "mqar", "experiments")

# A small, statistically clean instance: P = 4 keeps the plug-in entropy estimator's bias far below
# the tolerance (16 address states over 80k samples), and kv = 4 keeps G <= P as the generator
# requires. The knob semantics are P-independent; the arithmetic is not, which is why P is small here
# and 64 in the grid.
BASE = {"vocab_size": 200, "input_seq_len": 48, "num_kv_pairs": 4, "key_digits": 2, "digit_vocab": 4}


def test_unknown_knob_is_refused_at_the_spec_surface():
    from fsla_bench.mqar.common import make_cue_data
    with pytest.raises(ValueError, match="unknown cue-consistency knob"):
        make_cue_data(8192, 8, 8, None, train_seq_len=48, test_seq_lens=[48], train_examples=8,
                      test_examples=8, cue_digit=1)


# ============================================================== 6. the grid spec
def _spec():
    from fsla_bench.mqar.build_configs import load_spec
    return load_spec(os.path.join(_EXP_DIR, "cue_consistency.yaml"))


def test_spec_expands_to_the_cell_count_its_header_states():
    from fsla_bench.mqar.build_configs import build_configs
    cfgs, _ = build_configs(_spec())
    assert len(cfgs) == 450, "the header's arithmetic (378 + 72) is a load-bearing claim"
    assert len({c.run_id for c in cfgs}) == len(cfgs), "condition tags must separate the result rows"


def test_every_condition_is_reachable_and_carries_its_knobs():
    from fsla_bench.mqar.build_configs import build_configs
    spec = _spec()
    tags = [c["tag"] for c in spec["conditions"]]
    assert tags == ["agree", "agree-shuffled", "partial-m1", "partial-m2", "partial-m4",
                    "partial-m4-consensus", "nocue"]
    cfgs, _ = build_configs(spec)
    for tag in tags:
        cells = [c for c in cfgs if c.run_id.endswith(f"-{tag}")]
        assert cells, f"condition {tag} expanded to zero cells"
        seg = cells[0].data.train_configs[0]
        want = next(c for c in spec["conditions"] if c["tag"] == tag)
        for k, v in want.items():
            if k != "tag":
                assert getattr(seg, k) == v


def test_the_knob_axes_are_actually_crossed_not_confounded():
    """Every condition shares L, kv, vocab and geometry with every other; only the cue knobs move."""
    from fsla_bench.mqar.build_configs import build_configs
    cfgs, _ = build_configs(_spec())
    segs = {c.run_id.rsplit("-s", 1)[-1]: c.data.train_configs[0] for c in cfgs}
    fixed = {(s.input_seq_len, s.num_kv_pairs, s.vocab_size, s.key_digits, s.digit_vocab)
             for s in segs.values()}
    assert len(fixed) == 1, f"a condition moved something other than the cue knobs: {fixed}"


def test_capacity_is_held_ample_so_the_boundary_is_not_capacity():
    """PREDICTED-7's precondition: beta = N / kv >= 8 at every rung, so no cell is starved."""
    spec = _spec()
    kv = spec["data"]["protocol"]["num_kv_pairs"]
    assert min(spec["ncs"]) / kv >= 8


def test_full_cue_and_no_cue_are_both_present():
    spec = _spec()
    D = spec["data"]["protocol"]["key_digits"]
    qs = {c["cue_digits"] for c in spec["conditions"]}
    assert D in qs and 0 in qs, "knob (a) must span full agreement to none (§7.1's 'at any N')"


def test_the_multi_match_ladder_has_at_least_two_points():
    """PREDICTED-3 is a 1/m LAW; one point cannot show a law."""
    spec = _spec()
    ms = sorted({c["matches_per_cue"] for c in spec["conditions"]
                 if c["cue_digits"] == 1 and c.get("match_resolution", "recency") == "recency"})
    assert ms == [1, 2, 4]


def test_the_harmless_aliasing_control_matches_a_harmful_cell_exactly():
    spec = _spec()
    by_tag = {c["tag"]: c for c in spec["conditions"]}
    a, b = by_tag["partial-m4"], by_tag["partial-m4-consensus"]
    assert {k: v for k, v in a.items() if k not in ("tag", "match_resolution")} == \
           {k: v for k, v in b.items() if k not in ("tag", "match_resolution")}
    assert b["match_resolution"] == "consensus"


def test_slate_is_the_three_wirings_plus_d1_plus_the_four_contenders():
    from fsla_bench.models.fsla import CANONICAL_WIRINGS
    spec = _spec()
    routed = [a["instance"] for a in spec["tiers"]["knobs"] if a["build"] == "routed"]
    assert set(routed) == set(CANONICAL_WIRINGS)
    assert any(a["build"] == "attn" for a in spec["tiers"]["knobs"])
    assert spec["tiers"]["contenders"][0]["methods"] == ["rla", "gla", "gdn"]


def test_predictions_are_stated_in_the_header_and_marked():
    with open(os.path.join(_EXP_DIR, "cue_consistency.yaml")) as fh:
        head = fh.read().split("project_name:")[0]
    assert "ADVANCE PREDICTIONS" in head
    for n in range(1, 8):
        assert f"PREDICTED-{n} " in head, f"prediction {n} is missing from the spec header"
    assert head.count("FALSIFIED BY") >= 5


# =========================================================== 7. the axis must not leak into old specs
@pytest.mark.parametrize("name", ["law_grid", "ood_recall", "canonical", "router_bias"])
def test_specs_without_conditions_are_untouched_by_the_new_axis(name):
    from fsla_bench.mqar.build_configs import _conditions, load_spec
    spec = load_spec(os.path.join(_EXP_DIR, f"{name}.yaml"))
    assert _conditions(spec) == [(None, {})]
    assert "cond" not in str(spec)


def test_a_condition_axis_without_cue_protocol_refuses():
    from fsla_bench.mqar.build_configs import build_configs
    spec = _spec()
    spec["data"]["protocol"] = {"train_kv": [4], "test_seq_lens": [128],
                                "test_kv_by_len": {128: [4]}, "train_examples": 8, "test_examples": 8}
    with pytest.raises(ValueError, match="silently ignored"):
        build_configs(spec)


def test_an_explicit_run_id_without_cond_refuses():
    from fsla_bench.mqar.build_configs import build_configs
    spec = _spec()
    spec["tiers"] = {"knobs": [{"build": "routed", "instance": "fsla-d1-dense", "tag": "d1",
                                "run_id": "cell-{tag}-{nc}-{seed}"}]}
    with pytest.raises(ValueError, match=r"must contain \{cond\}"):
        build_configs(spec)


def test_an_arm_naming_an_undeclared_condition_refuses():
    from fsla_bench.mqar.build_configs import build_configs
    spec = _spec()
    spec["tiers"]["contenders"][0]["conditions"] = ["agree", "typo-tag"]
    with pytest.raises(ValueError, match="does not declare"):
        build_configs(spec)


def test_duplicate_condition_tags_refuse():
    from fsla_bench.mqar.build_configs import build_configs
    spec = _spec()
    spec["conditions"].append(dict(spec["conditions"][0]))
    with pytest.raises(ValueError, match="duplicate condition tag"):
        build_configs(spec)
