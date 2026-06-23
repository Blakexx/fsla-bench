"""AR-bucketed perplexity (Zoology "AR Hits") — the real-text recall metric.

Buckets held-out tokens by whether they complete a bigram (prev, cur) that appeared EARLIER in the
same document, restricted to bigrams occurring <= MAX_FREQ (default 1250) times in the TRAINING
corpus (excludes memorized/common bigrams, isolating genuine in-context recall). Reports perplexity
on the AR-hit slice vs the rest. Per Arora et al., "Zoology" (arXiv:2312.04927): the AR-hit slice is
~6% of tokens but carries the bulk of the attention-vs-linear perplexity gap.

NO public pip implementation exists — this is built from the paper spec. VALIDATION (see the user's
"how do we know it's right" gates):
  * gate 1 (unit): `--selftest` checks the tagger on a synthetic doc with known repeated bigrams.
  * gate 2 (distributional): the run prints `ar_hit_frac`, which should land ~5-7% on natural text.
  * gate 3 (GOLD, manual): run on a PUBLIC attn-vs-Mamba checkpoint and confirm the published
    attn-wins-on-the-AR-slice gap reproduces BEFORE trusting this metric in the paper.

Run:  python -m rola_bench.lm.ar_ppl --selftest
      python -m rola_bench.lm.ar_ppl --pretrained <ckpt> [--dataset Salesforce/wikitext]
            [--dataset_config wikitext-103-raw-v1] [--max_freq 1250] [--freq_tokens 50000000]
            [--max_eval_docs 2000] [--max_len 2048] [--out r.json]
"""
import argparse
import json
import math
from collections import Counter

MAX_FREQ_DEFAULT = 1250


def ar_hit_mask(ids, bigram_freq=None, max_freq=MAX_FREQ_DEFAULT):
    """ids: token ids for ONE document. Returns list `hit` of len(ids): hit[j] (the position that
    PREDICTS ids[j], j>=1) is True iff the bigram (ids[j-1], ids[j]) occurred EARLIER in this doc AND
    its training-corpus frequency is <= max_freq. hit[0]=False. Indexed by the predicted-token pos.
    bigram_freq=None disables the frequency filter (used by --selftest)."""
    ids = list(ids)
    n = len(ids)
    hit = [False] * n
    seen = set()
    for i in range(n - 1):
        bg = (ids[i], ids[i + 1])
        earlier = bg in seen                                   # appeared at some position < i
        rare = True if bigram_freq is None else bigram_freq.get(bg, 0) <= max_freq
        hit[i + 1] = earlier and rare                          # predicting ids[i+1] is an AR hit
        seen.add(bg)
    return hit


def bigram_freqs(docs_ids, cap_tokens):
    """Counter of bigram -> training-corpus count, accumulated until cap_tokens tokens are seen."""
    c, tot = Counter(), 0
    for ids in docs_ids:
        for i in range(len(ids) - 1):
            c[(ids[i], ids[i + 1])] += 1
        tot += len(ids)
        if tot >= cap_tokens:
            break
    return c


def eval_ckpt(model, tok, texts, bigram_freq, max_freq, device, max_len=2048):
    """Per-token NLL bucketed into AR-hit vs other -> {ppl_hit, ppl_other, ar_hit_frac, n_*}."""
    import torch
    import torch.nn.functional as F
    agg = {"hit": [0.0, 0], "other": [0.0, 0]}                 # [sum_nll, n_tokens]
    model.eval()
    with torch.no_grad():
        for txt in texts:
            if not txt or not txt.strip():
                continue
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
            if ids.numel() < 2:
                continue
            logits = model(ids.unsqueeze(0).to(device)).logits[0].float()      # [L, V]
            logp = F.log_softmax(logits[:-1], dim=-1)                          # logits[i] predicts ids[i+1]
            tgt = ids[1:].to(device)
            nll = -logp[torch.arange(tgt.numel(), device=device), tgt]        # [L-1]
            hit = ar_hit_mask(ids.tolist(), bigram_freq, max_freq)            # len L, by predicted pos
            hitmask = torch.tensor(hit[1:], device=device)                    # align with nll (pred pos 1..L-1)
            for key, m in (("hit", hitmask), ("other", ~hitmask)):
                if m.any():
                    agg[key][0] += float(nll[m].sum()); agg[key][1] += int(m.sum())
    out = {}
    for k, (s, n) in agg.items():
        out[f"ppl_{k}"] = (math.exp(s / n) if n else None)
        out[f"n_{k}"] = n
    tot = out["n_hit"] + out["n_other"]
    out["ar_hit_frac"] = (out["n_hit"] / tot if tot else None)
    return out


def selftest():
    A, B, C, D = 10, 11, 12, 13
    ids = [A, B, C, A, B, D]                                   # bigram (A,B) recurs -> predicting 2nd B is the hit
    assert ar_hit_mask(ids, bigram_freq=None) == [False, False, False, False, True, False]
    assert ar_hit_mask(ids, bigram_freq={(A, B): 5000}, max_freq=1250)[4] is False  # common bigram excluded
    assert ar_hit_mask([A, B, A, B], bigram_freq=None) == [False, False, False, True]
    assert ar_hit_mask([A, B, C, D], bigram_freq=None) == [False, False, False, False]  # no recurrence
    print("selftest PASS — tagger flags re-occurring rare bigrams, excludes common + non-recurring")


def _load_docs(dataset, config, split, tok, limit=None):
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split)
    out = []
    for i, row in enumerate(ds):
        t = row.get("text", "")
        if t and t.strip():
            out.append(tok(t, truncation=True, max_length=4096).input_ids)
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description="AR-bucketed perplexity (Zoology AR-hits) recall metric.")
    ap.add_argument("--selftest", action="store_true", help="run the tagger unit-test (gate 1) and exit")
    ap.add_argument("--pretrained", help="checkpoint dir or HF id")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    ap.add_argument("--max_freq", type=int, default=MAX_FREQ_DEFAULT)
    ap.add_argument("--freq_tokens", type=int, default=50_000_000, help="train tokens for the bigram-freq filter")
    ap.add_argument("--max_eval_docs", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.pretrained:
        ap.error("--pretrained required (or use --selftest)")

    import torch
    import fla_rola.models.rola  # noqa: F401 — register the RoLA architecture for AutoModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.pretrained)
    model = AutoModelForCausalLM.from_pretrained(a.pretrained, torch_dtype=torch.bfloat16).to(dev)

    print(f"[ar_ppl] building bigram-freq filter from {a.dataset} train (<= {a.freq_tokens} tokens)...", flush=True)
    train_ids = _load_docs(a.dataset, a.dataset_config, "train", tok)   # generator-ish list
    bg = bigram_freqs(train_ids, a.freq_tokens)
    print(f"[ar_ppl] {len(bg)} distinct bigrams; evaluating on validation...", flush=True)
    val_ids = None
    eval_texts = [tok.decode(ids) for ids in _load_docs(a.dataset, a.dataset_config, "validation", tok, a.max_eval_docs)]
    res = eval_ckpt(model, tok, eval_texts, bg, a.max_freq, dev, a.max_len)
    res["pretrained"] = a.pretrained
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[ar_ppl] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
