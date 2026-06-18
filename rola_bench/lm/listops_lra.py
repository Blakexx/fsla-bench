"""Canonical LRA ListOps (Long-Range Arena) as a non-reciprocal asym-vs-sym benchmark.

Uses the REAL LRA ListOps-1000 splits (HF mirror `fengyang0317/listops-1000`: train 96k / val 2k /
test 2k, `Source` expression + integer `Target` 0-9) — NOT a self-generated dataset, so it's the
recognized benchmark, not a hand-rolled one. LRA is the standard long-range/structured-reasoning
benchmark for efficient transformers (RoLA's comparison class).

Tokenization is SYMBOL-LEVEL (the canonical LRA scheme, ~17-token vocab): GPT-2 BPE would shatter
`[MIN`/`(`/`)` and inflate the ~2000-symbol sequences to ~3200 BPE tokens. We frame it as causal LM
(decoder reads the expression, predicts the answer digit) with the loss MASKED to the answer token,
which is the natural framing for a recurrent/decoder model and reuses the train_lm harness + the
answer-accuracy metric. Majority-class (digit 0) baseline ~17.8% on test — the floor a real result
must clear.
"""
import os
from functools import lru_cache

DATASET = "fengyang0317/listops-1000"
PAD, ANS = "<pad>", "<ans>"                              # pad id 0; <ans> marks "now emit the answer"


@lru_cache(maxsize=1)
def _raw():
    from datasets import load_dataset
    return load_dataset(DATASET)


@lru_cache(maxsize=1)
def build_vocab():
    """Deterministic symbol vocab from the LRA token set (scan covers it almost immediately)."""
    ds = _raw()
    syms = set()
    for i in range(min(5000, len(ds["train"]))):
        syms.update(ds["train"][i]["Source"].split())
    toks = [PAD, ANS] + sorted(syms)                    # sorted -> stable ids across runs
    return {t: i for i, t in enumerate(toks)}


def vocab_size():
    return len(build_vocab())


def lra_listops_dataset(block_size, max_train=None, max_val=2000):
    """DatasetDict with answer-masked causal-LM examples: input = symbol(Source) + <ans> + digit;
    labels = -100 except the answer digit (predicted from the <ans> position). Right-padded to
    block_size; over-length expressions are dropped (LRA Long-ListOps is bounded ~2000 symbols)."""
    from datasets import Dataset, DatasetDict
    v = build_vocab(); ds = _raw()
    ans_id = v[ANS]

    def enc(split, n):
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        src_split = ds[split]
        lim = len(src_split) if n is None else min(n, len(src_split))
        for i in range(lim):
            ex = src_split[i]
            ids = [v[t] for t in ex["Source"].split() if t in v]
            ans_tok = v[str(ex["Target"])]              # the answer digit is itself a symbol in the vocab
            seq = ids + [ans_id, ans_tok]
            if len(seq) > block_size:                   # drop the rare over-length expression
                continue
            labels = [-100] * len(seq); labels[-1] = ans_tok   # supervise ONLY the answer digit
            attn = [1] * len(seq)
            pad = block_size - len(seq)
            seq += [0] * pad; attn += [0] * pad; labels += [-100] * pad
            rows["input_ids"].append(seq); rows["attention_mask"].append(attn); rows["labels"].append(labels)
        return Dataset.from_dict(rows)

    val_key = "validation" if "validation" in ds else "test"
    return DatasetDict(train=enc("train", max_train), validation=enc(val_key, max_val))
