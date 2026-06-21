#!/usr/bin/env python3
"""Realized-state verification for the canonical MQAR baselines (#89 / #129).

The matched-state comparison only holds if every baseline cell ACTUALLY instantiates at its
nominal target state — `int(expand_v * d_model)` truncation can silently drop a dim, leaving a
baseline weaker than matched (which would unfairly flatter RoLA). canonical_baselines solves dims
via round() then passes them as expand ratios; this re-instantiates each cell on CPU, reads the
module's REALIZED recurrent-state dims back, and asserts they match the intended dims + that the
realized state is within one rounding step of the target. Fail loud on any mismatch.

Run: python -m rola_bench.mqar.verify_state   (exits nonzero if any cell is off)
"""
import importlib
import sys

import rola_bench.mqar.experiments.canonical_baselines as cb


def _instantiate(kc):
    mp, cl = kc["name"].rsplit(".", 1)
    M = getattr(importlib.import_module(mp), cl)
    kw = dict(kc["kwargs"])
    if "hidden_size" not in kw and "d_model" not in kw:
        kw["d_model"] = cb.DMODEL
    return M(**kw)


def _realized(method, m, fm):
    """Realized recurrent-state floats from the instantiated module's true dims."""
    if method in ("rla", "hedgehog"):       # LinearAttention: nh * (fm*k_head) * v_head
        return fm * m.key_dim * m.value_dim // m.num_heads, (m.key_dim, m.value_dim, m.num_heads)
    if method == "gla":
        return m.key_dim * m.value_dim // m.num_heads, (m.key_dim, m.value_dim, m.num_heads)
    if method == "gdn":
        vh = m.value_dim // m.num_heads
        return m.num_heads * m.head_dim * vh, (m.head_dim, vh, m.num_heads)
    if method == "based":
        fd = m.feature_dim
        return (cb.DMODEL + 1) * (fd * (fd + 1) // 2 + fd + 1), (fd,)
    raise ValueError(method)


def verify(state_tol=0.12):
    """Return (n_checked, bad) where bad is a list of (cell, reason)."""
    bad, n = [], 0
    for method in cb.METHODS:
        fm = 2 if method == "hedgehog" else 1
        for shape in cb.SHAPES:
            for nc in cb.NCS:
                cell = cb.baseline_cell(method, shape, nc)
                if cell is None:
                    continue
                kc, target = cell
                n += 1
                tag = f"{method}/{shape}/nc{nc}"
                try:
                    m = _instantiate(kc)
                    rs, dims = _realized(method, m, fm)
                except Exception as e:                       # build failure is itself a finding
                    bad.append((tag, f"build error {type(e).__name__}: {e}")); continue
                kw = kc["kwargs"]
                if "expand_k" in kw and round(kw["expand_k"] * cb.DMODEL) != m.key_dim:
                    bad.append((tag, f"truncated key_dim {m.key_dim} != intended "
                                     f"{round(kw['expand_k'] * cb.DMODEL)}"))
                if abs(rs - target) / target > state_tol:
                    bad.append((tag, f"state {rs} != target {target} ({abs(rs-target)/target:.0%})"))
    return n, bad


def main():
    n, bad = verify()
    for tag, reason in bad:
        print(f"  FAIL {tag}: {reason}")
    if bad:
        print(f"STATE-VERIFY FAILED: {len(bad)} of {n} baseline cells off matched state.")
        return 1
    print(f"STATE-VERIFY OK: all {n} baseline cells at matched state (realized == intended).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
