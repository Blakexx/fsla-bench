"""Seed-confirmation grid → error bars on the headline.

For each (kernel, state-rung), the WINNER = the best-accuracy (shape, LR) cell from the single-seed
canonical_mqar_grid (best-of-LR-and-shape per kernel×rung; SSE excluded — its faithful impl is ~25x
slower). Each winner is re-run at ADDITIONAL seeds so every reported point has ≥3 seeds (the existing
s1337 + these), turning the n=1 grid into mean±range. SSE excluded; MHA included (1 rung-free cell).

Construction is exact + reproducible: we import the full canonical grid (GRID_TIERS=all) and CLONE
each winner's TrainConfig verbatim, swapping only `seed` and the `_s1337` token in run_id (new seed →
new run_id → no collision with the originals). WINNERS is baked (computed from results/mqar on
2026-06-16); regenerate with the picker in this module's __main__ if the grid changes.
"""
import copy
import importlib
import os

os.environ["GRID_TIERS"] = "all"                      # the winners span all rungs; need the full grid
from rola_bench.mqar.experiments import canonical_mqar_grid as _g
importlib.reload(_g)                                   # rebuild configs under GRID_TIERS=all

SEEDS = [int(s) for s in os.environ.get("SEED_CONFIRM_SEEDS", "2024,777").split(",") if s]

WINNERS = [
    "grid-based-wide-nc128_st79872_lr3e-03_s1337", "grid-based-wide-nc16_st9984_lr3e-03_s1337",
    "grid-based-wide-nc256_st159744_lr3e-03_s1337", "grid-based-wide-nc2_st1248_lr3e-03_s1337",
    "grid-based-wide-nc32_st19968_lr3e-03_s1337", "grid-based-wide-nc4_st2496_lr3e-03_s1337",
    "grid-based-wide-nc64_st39936_lr3e-03_s1337", "grid-based-wide-nc8_st4992_lr3e-03_s1337",
    "grid-gdn-square-nc128_st79872_lr1e-03_s1337", "grid-gdn-square-nc256_st159744_lr3e-04_s1337",
    "grid-gdn-square-nc32_st19968_lr3e-04_s1337", "grid-gdn-square-nc64_st39936_lr3e-04_s1337",
    "grid-gdn-square-nc8_st4992_lr3e-02_s1337", "grid-gdn-wide-nc16_st9984_lr1e-02_s1337",
    "grid-gdn-wide-nc2_st1248_lr1e-03_s1337", "grid-gdn-wide-nc4_st2496_lr1e-02_s1337",
    "grid-gla-square-nc128_st79872_lr3e-04_s1337", "grid-gla-square-nc16_st9984_lr1e-03_s1337",
    "grid-gla-square-nc256_st159744_lr3e-04_s1337", "grid-gla-square-nc32_st19968_lr3e-02_s1337",
    "grid-gla-square-nc64_st39936_lr3e-02_s1337", "grid-gla-square-nc8_st4992_lr3e-03_s1337",
    "grid-gla-wide-nc2_st1248_lr3e-04_s1337", "grid-gla-wide-nc4_st2496_lr1e-03_s1337",
    "grid-hedgehog-heads-nc16_st9984_lr1e-03_s1337", "grid-hedgehog-heads-nc4_st2496_lr1e-02_s1337",
    "grid-hedgehog-square-nc128_st79872_lr3e-03_s1337", "grid-hedgehog-square-nc256_st159744_lr3e-03_s1337",
    "grid-hedgehog-square-nc32_st19968_lr3e-03_s1337", "grid-hedgehog-square-nc64_st39936_lr3e-03_s1337",
    "grid-hedgehog-wide-nc2_st1248_lr1e-02_s1337", "grid-hedgehog-wide-nc8_st4992_lr3e-03_s1337",
    "grid-mha_lr1e-03_s1337",
    "grid-rla-square-nc2_st1248_lr3e-03_s1337", "grid-rla-wide-nc128_st79872_lr1e-03_s1337",
    "grid-rla-wide-nc16_st9984_lr3e-03_s1337", "grid-rla-wide-nc256_st159744_lr1e-03_s1337",
    "grid-rla-wide-nc32_st19968_lr3e-03_s1337", "grid-rla-wide-nc4_st2496_lr3e-03_s1337",
    "grid-rla-wide-nc64_st39936_lr3e-03_s1337", "grid-rla-wide-nc8_st4992_lr3e-03_s1337",
    "grid-routed-gla-nc128_st79872_lr1e-02_s1337", "grid-routed-gla-nc16_st9984_lr1e-02_s1337",
    "grid-routed-gla-nc256_st159744_lr1e-02_s1337", "grid-routed-gla-nc2_st1248_lr1e-02_s1337",
    "grid-routed-gla-nc32_st19968_lr1e-02_s1337", "grid-routed-gla-nc4_st2496_lr3e-03_s1337",
    "grid-routed-gla-nc64_st39936_lr1e-02_s1337", "grid-routed-gla-nc8_st4992_lr1e-02_s1337",
    "grid-routed-rla-nc128_st79872_lr3e-03_s1337", "grid-routed-rla-nc16_st9984_lr3e-03_s1337",
    "grid-routed-rla-nc256_st159744_lr3e-03_s1337", "grid-routed-rla-nc2_st1248_lr1e-02_s1337",
    "grid-routed-rla-nc32_st19968_lr1e-02_s1337", "grid-routed-rla-nc4_st2496_lr3e-03_s1337",
    "grid-routed-rla-nc64_st39936_lr3e-03_s1337", "grid-routed-rla-nc8_st4992_lr3e-03_s1337",
]

_base = {c.run_id: c for c in _g.configs}
_base_env = {c.run_id: e for c, e in zip(_g.configs, _g.configs_envs)}   # clone each winner's env verbatim
configs, configs_envs = [], []            # run_rla_sweep.load_configs returns (configs, configs_envs);
_missing = []                             # BOTH must exist + be parallel or the loader AttributeErrors.
for _rid in WINNERS:
    _c0 = _base.get(_rid)
    if _c0 is None:
        _missing.append(_rid); continue
    for _sd in SEEDS:
        _c = copy.deepcopy(_c0)
        _c.seed = _sd                     # model-init + training RNG
        _c.data.seed = _sd                # CRITICAL: also resample the MQAR data. data.seed defaults to
                                          # 123 and the grid never sets it, so without this every seed
                                          # trains on the SAME cached dataset → error bars would capture
                                          # init variance ONLY, not a fresh data draw (the whole point).
        _c.run_id = _rid.replace("_s1337", f"_s{_sd}")   # seed→fresh run_id (no collision)
        configs.append(_c)
        configs_envs.append(dict(_base_env.get(_rid, {"EVAL_EVERY_N": "10"})))

if _missing:
    import sys
    print(f"[seed_confirm] WARNING {len(_missing)} winners not in current grid: {_missing[:3]}...", file=sys.stderr)


def _pick(results_glob="/home/blake/rola-bench/results/mqar/*.jsonl"):
    """Regenerate WINNERS from results (best-of-LR-and-shape per kernel×rung, SSE excluded)."""
    import glob, json, re
    def ker(rid):
        if rid.startswith("grid-routed-"): return "routed-" + rid.split("-")[2]
        if rid.startswith("grid-mha"): return "mha"
        if rid.startswith("grid-sse"): return "sse"
        return rid.split("-")[1]
    best = {}
    for f in glob.glob(results_glob):
        for l in open(f):
            try: j = json.loads(l)
            except Exception: continue
            if not j.get("ok") or not j.get("run_id"): continue
            k = ker(j["run_id"])
            if k == "sse": continue
            nc = re.search(r"nc(\d+)", j["run_id"]); nc = int(nc.group(1)) if nc else -1
            a = j.get("max_acc") or 0.0
            if a > best.get((k, nc), (-1, None))[0]: best[(k, nc)] = (a, j["run_id"])
    return sorted(rid for _, rid in best.values())


if __name__ == "__main__":
    import json
    print(json.dumps(_pick(), indent=0).replace("\n", ""))
