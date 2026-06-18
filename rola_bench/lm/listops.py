"""LRA ListOps as a NON-RECIPROCAL benchmark for inter-RoLA routing (asym vs sym).

ListOps is nested list arithmetic — e.g. `[MAX 2 9 [MIN 4 7 ] 0 ]` -> 9 — over single digits with
ops MAX / MIN / MED (median) / SM (sum mod 10). It is non-reciprocal in the sense that matters here:
producing the answer requires READING operands according to the *nested structure*, which is decoupled
from the *position* each token was WRITTEN at. So the optimal read-routing differs from the write-
routing — exactly the regime where a separate read router (asymmetric RoLA) should beat a tied one
(symmetric), unlike MQAR where read==write (you read the state of the key you wrote).

We frame it as CAUSAL LM and mask the loss to ONLY the answer token (labels = -100 elsewhere), so the
reported eval loss / ppl is the answer NLL directly — the sharpest asym-vs-sym signal (predicting the
easy structural tokens would otherwise dilute it). Reuses train_lm/HF Trainer; no new training loop.
"""
import random

OPS = ["MAX", "MIN", "MED", "SM"]


def _apply(op, vals):
    if op == "MAX": return max(vals)
    if op == "MIN": return min(vals)
    if op == "SM":  return sum(vals) % 10
    s = sorted(vals); return s[len(s) // 2]              # MED: median (lower-mid), in 0..9


def gen_example(rng, max_depth=5, max_args=5):
    """Generate one (tokens, answer_digit), value computed from the same structure we emit so the
    label is always exactly consistent with the surface form the model sees."""
    def build(depth):
        op = rng.choice(OPS)
        toks = ["[" + op]
        vals = []
        for _ in range(rng.randint(2, max_args)):
            if depth < max_depth and rng.random() < 0.45:
                sub, v = build(depth + 1); toks += sub
            else:
                d = rng.randint(0, 9); toks.append(str(d)); v = d
            vals.append(v)
        toks.append("]")
        return toks, _apply(op, vals)
    toks, val = build(0)
    return toks, val


def listops_dataset(tokenizer, n_train, n_val, max_len=512, max_depth=5, seed=0):
    """Build a DatasetDict of tokenized ListOps with answer-only labels (-100 elsewhere), right-padded
    to max_len. eval_loss == answer NLL. `max_len` comes from train_lm's block_size.

    Difficulty knobs (env-overridable): LM_LISTOPS_DEPTH (max nesting depth; deeper = more structural
    reading, where asymmetric routing should help), LM_LISTOPS_ARGS (max args/node), LM_LISTOPS_MINOPS
    (reject expressions with fewer than this many operators, so trivial shallow trees don't dominate
    after the over-length filter). Defaults push toward the deep, non-reciprocal regime."""
    import os
    max_depth = int(os.environ.get("LM_LISTOPS_DEPTH", max_depth if max_depth != 5 else 10))
    max_args = int(os.environ.get("LM_LISTOPS_ARGS", "5"))
    min_ops = int(os.environ.get("LM_LISTOPS_MINOPS", "4"))
    from datasets import Dataset, DatasetDict
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def make(n, base_seed):
        rng = random.Random(base_seed)
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        made = 0; tries = 0
        while made < n and tries < n * 40:
            tries += 1
            toks, ans = gen_example(rng, max_depth, max_args)
            if toks.count("[") < min_ops:             # drop trivial shallow trees (keep it non-reciprocal-hard)
                continue
            prompt = " ".join(toks) + " ="            # the model reads the expression, then answers
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ans_ids = tokenizer(" " + str(ans), add_special_tokens=False)["input_ids"]
            seq = ids + ans_ids
            if len(seq) > max_len:                    # skip expressions that don't fit (keep it bounded)
                continue
            labels = [-100] * len(seq)
            for j in range(len(ans_ids)):             # supervise ONLY the answer token(s)
                labels[len(ids) + j] = ans_ids[j]
            attn = [1] * len(seq)
            pad = max_len - len(seq)                  # right-pad to a uniform length (causal: pad is after the answer)
            seq += [pad_id] * pad; attn += [0] * pad; labels += [-100] * pad
            rows["input_ids"].append(seq); rows["attention_mask"].append(attn); rows["labels"].append(labels)
            made += 1
        return Dataset.from_dict(rows)

    return DatasetDict(train=make(n_train, seed), validation=make(n_val, seed + 999999))
