#!/usr/bin/env python3
"""End-to-end per-family CI gate — run ONE epoch of the smallest cell of each family through the
ACTUAL runner path (zoology.launch via run_one). Stronger than ci_cell_check (which is mixer-level):
this builds the full model, trains an epoch, runs the state-logging + eval + result-emit path, and
checks an ok result comes back. It exercises exactly what the fleet runs, so runner/logging
regressions (a stripped mixer module, a missing state_size, a broken result emit) are caught
PRE-DEPLOY instead of on rented GPUs. Local GPU, decoupled from the docker build. Nonzero exit on
any family that doesn't complete ok.
"""
import copy
import sys

from rola_bench.mqar.ci_cell_check import family_of
from rola_bench.mqar.run_rla_sweep import run_one, load_configs


def _set_epochs(cfg, n):
    try:
        cfg.max_epochs = n
        return cfg
    except Exception:
        for meth in ("model_copy", "copy"):
            if hasattr(cfg, meth):
                try:
                    return getattr(cfg, meth)(update={"max_epochs": n})
                except Exception:
                    pass
        raise


def main():
    g_configs, _ = load_configs("canonical")
    fams = {}
    for c in g_configs:
        key, nc = family_of(c)
        size = c.model.d_model * (nc or 1)
        if key not in fams or size < fams[key][0]:
            fams[key] = (size, c)

    print(f"end-to-end families: {len(fams)} (1 epoch each, through zoology.launch)")
    bad = []
    for i, (key, (_, c)) in enumerate(sorted(fams.items(), key=lambda x: str(x[0]))):
        cfg = _set_epochs(copy.deepcopy(c), 1)
        r = run_one(900 + i, cfg.run_id, cfg, {})
        ok = bool(r.get("ok")) and r.get("returncode") == 0
        print(f"  {'PASS' if ok else 'FAIL'}  {key[0]} kernel={key[1]} norm={key[3]} "
              f"-> ok={r.get('ok')} rc={r.get('returncode')} max_acc={r.get('max_acc')}")
        if not ok:
            print("    stderr_tail:", str(r.get("stderr_tail", ""))[-500:])
            bad.append(f"{key[0]}/{key[1]}")
    if bad:
        print(f"CELL-RUN FAILED: {bad}")
        return 1
    print(f"CELL-RUN OK: all {len(fams)} families trained 1 epoch end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
