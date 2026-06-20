"""RULER-style Single Needle-In-A-Haystack (S-NIAH) long-context recall eval — the claim-aligned
test for RoLA's thesis (recall at length / extrapolation), the same benchmark the GatedDeltaNet /
Mamba / FLA papers use to separate architectures (their advantage is "most pronounced on RULER NIAH").

Self-contained (no RULER repo / data download): we synthesize the haystack + needle directly, so it
runs on ANY AutoModelForCausalLM checkpoint (RoLA is AutoModel-registered via `import fla_rola.models.rola`) at
ARBITRARY sequence lengths — including BEYOND the trained context, which is the length-extrapolation
probe. Greedy decode, exact-match on the retrieved value.

  python niah_eval.py --pretrained <ckpt_dir> --lengths 1024,2048,4096,8192 --out niah.json

Scores retrieval accuracy per (length x depth), averaged over `samples` random placements. A bounded-
state recurrent model that truly carries information should hold accuracy as length grows; attention
is the ceiling within its trained window and the reference under extrapolation.

S-NIAH-1 (passkey, default): a random numeric value in benign repeated-sentence filler.
S-NIAH-2 (--task needle): the value is tied to a key among distractor keys (harder disambiguation).
"""
import argparse, json, random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla_rola.models.rola  # noqa: F401  (registers RoLA AutoClasses so from_pretrained loads RoLA checkpoints)

# RULER S-NIAH-1 filler: benign, low-entropy sentences repeated to fill the haystack (Mohtashami-style).
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
NEEDLE_TMPL = "One of the special magic numbers for {key} is: {value}."
QUERY_TMPL = "\nWhat is the special magic number for {key}?\nThe special magic number for {key} is:"
KEYS = ["the forest", "the ocean", "the mountain", "the desert", "the river",
        "the valley", "the meadow", "the canyon", "the glacier", "the harbor"]


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def build_example(tok, seq_len, depth, rng, task):
    """Construct an input of ~seq_len tokens with the needle at fractional `depth`. Returns
    (input_ids, value_str, key)."""
    value = str(rng.randint(10**6, 10**7 - 1))           # 7-digit passkey
    key = rng.choice(KEYS)
    needle = NEEDLE_TMPL.format(key=key, value=value)
    # 'needle' task: prepend distractor needles with OTHER keys so the model must disambiguate by key.
    distractors = ""
    if task == "needle":
        others = [k for k in KEYS if k != key]
        rng.shuffle(others)
        distractors = " ".join(NEEDLE_TMPL.format(key=k, value=str(rng.randint(10**6, 10**7 - 1)))
                               for k in others[:3]) + " "
    query = QUERY_TMPL.format(key=key)
    needle_ids = _ids(tok, " " + distractors + needle + " ")
    query_ids = _ids(tok, query)
    filler_unit = _ids(tok, FILLER)
    n_filler = max(0, seq_len - len(needle_ids) - len(query_ids))
    n_before = int(n_filler * depth)
    n_after = n_filler - n_before

    def fill(n):
        if n <= 0:
            return []
        reps = n // len(filler_unit) + 1
        return (filler_unit * reps)[:n]

    input_ids = fill(n_before) + needle_ids + fill(n_after) + query_ids
    return input_ids, value, key


@torch.no_grad()
def run(model, tok, device, lengths, depths, samples, task, max_new_tokens, seed):
    rng = random.Random(seed)
    per_len = {}
    for L in lengths:
        hits = total = 0
        for depth in depths:
            for _ in range(samples):
                ids, value, key = build_example(tok, L, depth, rng, task)
                inp = torch.tensor([ids], device=device)
                out = model.generate(inp, max_new_tokens=max_new_tokens, do_sample=False,
                                     num_beams=1, pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True)
                hits += int(value in gen)
                total += 1
        per_len[L] = round(hits / max(1, total), 4)
        print(f"  L={L:>6}: acc={per_len[L]:.3f}  (n={total})", flush=True)
    return per_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", required=True, help="checkpoint dir (save_pretrained) or HF id")
    p.add_argument("--lengths", default="1024,2048,4096,8192", help="comma-sep target token lengths")
    p.add_argument("--depths", default="0,0.25,0.5,0.75,1.0", help="comma-sep needle depth fractions")
    p.add_argument("--samples", type=int, default=8, help="random placements per (length,depth)")
    p.add_argument("--task", default="passkey", choices=["passkey", "needle"])
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--run_id", default=None, help="fleet result id (defaults to --pretrained)")
    p.add_argument("--results", default=None, help="append a fleet-style jsonl row here")
    p.add_argument("--out", default=None, help="write the full per-length JSON here")
    a = p.parse_args()

    lengths = [int(x) for x in a.lengths.split(",") if x]
    depths = [float(x) for x in a.depths.split(",") if x]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.pretrained, torch_dtype=dtype).to(device).eval()
    print(f"[niah] {a.pretrained} task={a.task} lengths={lengths} depths={depths} x{a.samples}", flush=True)

    per_len = run(model, tok, device, lengths, depths, a.samples, a.task, a.max_new_tokens, a.seed)
    overall = round(sum(per_len.values()) / max(1, len(per_len)), 4)
    row = {"run_id": a.run_id or a.pretrained, "ok": True, "eval": "niah", "task": a.task,
           "samples": a.samples, "depths": depths, "niah_acc_by_len": per_len, "niah_acc_mean": overall}
    print("=== NIAH (S-NIAH) results ===")
    print(json.dumps(row, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(row, fh, indent=2)
    if a.results:
        with open(a.results, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
