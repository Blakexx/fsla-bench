"""Shared MQAR config primitives — the building blocks every MQAR spec uses, in one place
(previously scattered across rla_sweep / rola_router_width_v2). The per-method baseline solve lives
in canonical_baselines.py (baseline_cell / NCS / REF), which this module re-exports for convenience.

Backbone/data sizes are NOT hardcoded here — they belong in the experiment spec. `DEFAULTS` is just
the fallback a spec's `backbone:` block overrides.
"""
from zoology.config import ModuleConfig, DataConfig
from zoology.data.multiquery_ar import MQARConfig

from rola_bench.mqar.experiments.canonical_baselines import baseline_cell, NCS, REF  # noqa: F401

# Fallbacks; any of these can be overridden per-experiment via the spec's `backbone:` block.
DEFAULTS = {"d_model": 128, "vocab": 8192, "n_layers": 2, "l_max": 1024}


# --- MQAR data presets: name -> (vocab) -> (train_configs, test_configs) ---------------------------
def _ext(vocab):
    """Canonical multi-task MQAR with an EXTENDED test set (kv up to 1024 / seq up to 4096)."""
    train = [
        MQARConfig(vocab_size=vocab, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
        MQARConfig(vocab_size=vocab, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
        MQARConfig(vocab_size=vocab, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
        MQARConfig(vocab_size=vocab, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
        MQARConfig(vocab_size=vocab, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
    ]
    test = [
        MQARConfig(vocab_size=vocab, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4),
        MQARConfig(vocab_size=vocab, input_seq_len=64,   num_examples=1_000, num_kv_pairs=8),
        MQARConfig(vocab_size=vocab, input_seq_len=64,   num_examples=1_000, num_kv_pairs=16),
        MQARConfig(vocab_size=vocab, input_seq_len=128,  num_examples=1_000, num_kv_pairs=32),
        MQARConfig(vocab_size=vocab, input_seq_len=256,  num_examples=1_000, num_kv_pairs=64),
        MQARConfig(vocab_size=vocab, input_seq_len=512,  num_examples=1_000, num_kv_pairs=128),
        MQARConfig(vocab_size=vocab, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
        MQARConfig(vocab_size=vocab, input_seq_len=2048, num_examples=500,   num_kv_pairs=512),
        MQARConfig(vocab_size=vocab, input_seq_len=4096, num_examples=250,   num_kv_pairs=1024),
    ]
    return train, test


DATA_PRESETS = {"ext": _ext}


def make_data(preset, vocab, train_batch, test_batch, cache_dir):
    train, test = DATA_PRESETS[preset](vocab)
    return DataConfig(train_configs=train, test_configs=test,
                      batch_size=(train_batch, test_batch), cache_dir=cache_dir)


# --- THE LAW-GRID DATA PROTOCOL (added 2026-08-02, Phase 1.3) -------------------------------------
# The named presets above are FIXED difficulty ladders: `ext` bakes its own train/test (L, kv) pairs
# in, and `L` is therefore not a sweepable axis. The capacity theory needs it to be one — the N = L
# boundary is a statement about the RATIO alpha_cap = N/L, and nothing in the pre-V3 grid ever moved
# L at fixed everything-else (audit §4.1 item 1). `make_law_data` builds one DataConfig per TRAIN
# sequence length, so `build_configs` can cross the `ncs` ladder against a `seq_lens` ladder and walk
# alpha_cap through 1 from both directions.
#
# Two axes, kept separate on purpose:
#   * the ITEMS/CAPACITY ratio, kv/N, swept at FIXED L. This costs nothing extra: MQAR reports
#     per-slice accuracy keyed on `num_kv_pairs` (`train.slice_keys: [num_kv_pairs]`), so one trained
#     cell yields the whole kv ladder as evaluation slices. This is the existing fixed-L
#     sweep-capacity protocol, generalized to any L rather than hard-coded to `ext`'s.
#   * L-EXTRAPOLATION, test L > train L. A first-class axis, not a by-product: every test length
#     strictly greater than the cell's train length is an extrapolation slice, and the retrospective's
#     H3 (loss of scale-freeness under a delta rule) is a statement about exactly those slices.
def make_law_data(vocab, train_batch, test_batch, cache_dir, *, train_seq_len, train_kv,
                  test_seq_lens, test_kv_by_len, train_examples, test_examples):
    """One `DataConfig` for a single TRAIN sequence length `train_seq_len`.

    `train_kv` is the list of item counts trained on at that length (mixed in one training set, the
    MQAR multi-task convention). `test_seq_lens` lists every evaluation length — including lengths
    greater than `train_seq_len`, which are the extrapolation slices — and `test_kv_by_len` maps each
    evaluation length to the item counts evaluated at it. Both zoology-side structural constraints
    are checked HERE, by name, rather than surfacing as a bare `assert` from inside the generator:
    `4*kv <= L` (`zoology.data.multiquery_ar`: `2*kv` tokens of key/value context plus a `2*kv`
    query block) and `vocab > L`.
    """
    def _cells(pairs, n_examples):
        out = []
        for seq_len, kvs in pairs:
            if seq_len % 2:
                raise ValueError(f"input_seq_len={seq_len} must be even (zoology MQAR).")
            if vocab <= seq_len:
                raise ValueError(
                    f"vocab_size={vocab} must exceed input_seq_len={seq_len} (zoology MQAR draws "
                    "distinct keys from the vocabulary). Raise `backbone.vocab` for this ladder.")
            for kv in kvs:
                if 4 * kv > seq_len:
                    raise ValueError(
                        f"num_kv_pairs={kv} does not fit in input_seq_len={seq_len}: zoology's MQAR "
                        "spends 2*kv tokens on the key/value context and 2*kv more on the query "
                        "block, so 4*kv <= L is structural. Drop the rung from the ladder rather "
                        "than letting the generator assert deep in the data build.")
                out.append(MQARConfig(vocab_size=vocab, input_seq_len=seq_len,
                                      num_examples=n_examples, num_kv_pairs=kv))
        return out

    train = _cells([(train_seq_len, list(train_kv))], train_examples)
    test = _cells([(L, list(test_kv_by_len[L])) for L in test_seq_lens], test_examples)
    return DataConfig(train_configs=train, test_configs=test,
                      batch_size=(train_batch, test_batch), cache_dir=cache_dir)


# --- THE CUE-CONSISTENCY PROTOCOL (added 2026-08-02, Phase 2) -------------------------------------
# The law protocol above sweeps CAPACITY (N, L, kv). It cannot move the paper's SECOND named boundary
# (Sec. 7.1), address agreement: standard MQAR's key IS its cue, so Eq. (7) holds by construction at
# every capacity and the axis is structurally absent. `zoology.data.cue_consistency` makes the cue
# structure the independent variable; this builder is the spec-side surface for it. See that module's
# docstring for the knob->mechanism table.
#
# One DataConfig per (train length, CONDITION), where a condition is one setting of the cue knobs.
# Conditions are TRAINED separately rather than mixed and sliced apart: the claim is about whether a
# given retrieval contract is servable, and a model trained on a mixture of contracts is answering a
# different question (it may learn a union strategy that no single contract demanded).
CUE_KNOBS = ("key_digits", "cue_digits", "digit_vocab", "slot_consistency", "matches_per_cue",
             "match_resolution", "num_kv_pairs", "power_a")


def make_cue_data(vocab, train_batch, test_batch, cache_dir, *, train_seq_len, test_seq_lens,
                  train_examples, test_examples, **knobs):
    """One `DataConfig` for a single (train length, cue condition) pair.

    `knobs` is any subset of `CUE_KNOBS`; the generator owns every structural check (m | kv, the
    q = D => m = 1 law, pool sizing, episode geometry) and raises by name, so this function does not
    re-implement them -- two copies of a validation rule is how the matched axis and the reported
    axis drift apart (Phase 1 finding P6).
    """
    from zoology.data.cue_consistency import CueConsistencyConfig

    unknown = set(knobs) - set(CUE_KNOBS)
    if unknown:
        raise ValueError(
            f"unknown cue-consistency knob(s) {sorted(unknown)}; the axis set is {list(CUE_KNOBS)}. "
            "An unrecognised key would otherwise be dropped silently and the cell would report a "
            "condition it did not run.")
    def _cell(seq_len, n_examples):
        return CueConsistencyConfig(vocab_size=vocab, input_seq_len=seq_len,
                                    num_examples=n_examples, **knobs)
    train = [_cell(train_seq_len, train_examples)]
    test = [_cell(L, test_examples) for L in test_seq_lens]
    return DataConfig(train_configs=train, test_configs=test,
                      batch_size=(train_batch, test_batch), cache_dir=cache_dir)


# --- hybrid backbone (short conv + the sequence mixer) --------------------------------------------
def base_conv_mixer(l_max):
    return dict(name="zoology.mixers.base_conv.BaseConv",
                kwargs={"l_max": l_max, "kernel_size": 3, "implicit_long_conv": True})


def wrap_hybrid(kernel_kwargs, l_max):
    return ModuleConfig(name="zoology.mixers.hybrid.Hybrid",
                        kwargs={"configs": [base_conv_mixer(l_max), kernel_kwargs]})


# MHA ceiling, canonical kwargs (matches add_attention in models_repo.py).
mha = dict(name="zoology.mixers.attention.MHA", kwargs={"num_heads": 2, "dropout": 0.1})
