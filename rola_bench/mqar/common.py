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


# --- hybrid backbone (short conv + the sequence mixer) --------------------------------------------
def base_conv_mixer(l_max):
    return dict(name="zoology.mixers.base_conv.BaseConv",
                kwargs={"l_max": l_max, "kernel_size": 3, "implicit_long_conv": True})


def wrap_hybrid(kernel_kwargs, l_max):
    return ModuleConfig(name="zoology.mixers.hybrid.Hybrid",
                        kwargs={"configs": [base_conv_mixer(l_max), kernel_kwargs]})


# MHA ceiling, canonical kwargs (matches add_attention in models_repo.py).
mha = dict(name="zoology.mixers.attention.MHA", kwargs={"num_heads": 2, "dropout": 0.1})
