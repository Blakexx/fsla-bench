"""Seed-confirmation: re-run the WINNING cell per (kernel, state-rung) of any MQAR spec at extra
seeds, for error bars. Generic + results-driven (no baked winner list):

  winner((kernel, nc)) = the run_id with the best max_acc across the stored MQAR samples (SSE excluded),
  then each winner's config is cloned with a fresh seed (model + data.seed) + run_id.

Parameterized by env so it works for any experiment:
  SEED_CONFIRM_BASE     base spec name (default: canonical) or a path to a .yaml
  SEED_CONFIRM_SEEDS    comma list of extra seeds (default: 2024,777)
The stage-1 results are every MQAR cell the fleet and the local grid stored (`fsla_bench.results.rows`).

Run as a module via the runner (`--config seed_confirm`); or `python -m ...seed_confirm` to preview
the picked winners.
"""
import copy
import os
import re
import sys

from fsla_bench.mqar.build_configs import build_configs, load_spec
from fsla_bench.mqar.winners import pick_winners

_HERE = os.path.dirname(os.path.abspath(__file__))

BASE = os.environ.get("SEED_CONFIRM_BASE", "canonical")
SEEDS = [int(s) for s in os.environ.get("SEED_CONFIRM_SEEDS", "2024,777").split(",") if s]

# Launch knobs (module-config, so they live here rather than a yaml `fleet:` block).
FLEET = {"batch": 1, "per_batch_timeout": 18000, "env": {"SEED_CONFIRM_SEEDS": "2024,777"}}


def build():
    os.environ["GRID_TIERS"] = "all"              # need every base cell available to match winners
    spec = BASE if BASE.endswith((".yaml", ".yml")) else os.path.join(_HERE, f"{BASE}.yaml")
    base_cfgs, base_envs = build_configs(load_spec(spec))
    by_id = {c.run_id: c for c in base_cfgs}
    env_by = {c.run_id: e for c, e in zip(base_cfgs, base_envs, strict=True)}
    winners = pick_winners(set(by_id))   # only this spec's own cells
    cfgs, envs, missing = [], [], []
    for rid in sorted(winners):
        c0 = by_id.get(rid)
        if c0 is None:
            missing.append(rid)
            continue
        for sd in SEEDS:
            c = copy.deepcopy(c0)
            c.seed = sd
            c.data.seed = sd                              # resample the MQAR data too (fresh draw per seed)
            c.run_id = re.sub(r"_s\d+$", f"_s{sd}", rid)  # fresh run_id -> no collision with stage-1
            cfgs.append(c)
            envs.append(dict(env_by.get(rid, {"EVAL_EVERY_N": "10"})))
    if missing:
        print(f"[seed_confirm] {len(missing)} winners not in base spec {BASE!r}: {missing[:3]}...",
              file=sys.stderr)
    return cfgs, envs


configs, configs_envs = build()


def load_configs_and_envs():
    return configs, configs_envs


if __name__ == "__main__":
    print(f"base={BASE} seeds={SEEDS} winners={len(pick_winners())} "
          f"-> {len(configs)} seed-confirm cells")
