"""CPU unit tests for the post-hoc RANK evaluator (rola-bench task #85).

Run with the project venv (no GPU needed):

    /mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python \
        -m rola_bench.similarity.test_rank_cpu

or, from /home/blake/rola-bench:

    PYTHONPATH=. /mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python \
        rola_bench/rank/test_rank_cpu.py

Tests:
  1. test_known_rank   — build a rank-r product matrix; assert numerical/stable/eff rank
                          all recover r (the metric core in isolation, no model).
  2. test_rank_stats_shapes — the dict has the same keys as rola.py's RANK_JSON payload.
  3. test_rola_extractor_end_to_end — build a tiny RoLA on CPU, run the full
                          driver (evaluate_rank + build_rola_attention_extractor),
                          assert it produces a finite rank within the nc*d_qk ceiling.
                          (Skipped with a clear message if `rola` isn't importable.)
  4. verify_against_inline_stub — DOCUMENTED stub for checkpoint-vs-inline parity; needs a
                          GPU + a paired checkpoint, so it asserts nothing here.
"""
from __future__ import annotations

import math

import torch

from rola_bench.similarity import (
    build_rola_attention_extractor,
    evaluate_rank,
    make_eval_batch,
    rank_stats,
    rola_attn_from_features,
)


def test_known_rank():
    """A rank-r matrix W = A B^T (A,B: [L,r]) must report rank r on every estimator.

    We make the r retained directions well-separated (orthonormal columns scaled equally)
    so the numerical-rank threshold and the entropy-based estimators all agree on r."""
    torch.manual_seed(0)
    L, r = 128, 7
    # Orthonormal r-dim subspace, equal singular values -> a clean rank-r object with a
    # flat spectrum: numerical rank = r at any tol, stable/eff/pr rank = r exactly.
    A = torch.linalg.qr(torch.randn(L, r))[0]      # [L,r], orthonormal columns
    B = torch.linalg.qr(torch.randn(L, r))[0]
    W = A @ B.T                                    # rank exactly r, all singular values = 1

    out = rank_stats(W)   # [L,L] -> single slice

    # Numerical rank at every tolerance should be exactly r (singular values are {1}^r, 0).
    for key in ("rank_1e-01", "rank_1e-02", "rank_1e-03", "rank_1e-04"):
        assert out[key] == float(r), f"{key}={out[key]} != {r}"
    # Stable rank = sum(s^2)/smax^2 = r (flat spectrum). eff/pr likewise = r.
    assert abs(out["stable_rank"] - r) < 1e-3, out["stable_rank"]
    assert abs(out["eff_rank"] - r) < 1e-3, out["eff_rank"]
    assert abs(out["pr_rank"] - r) < 1e-3, out["pr_rank"]
    print(f"[ok] test_known_rank: rank-{r} matrix -> "
          f"num={out['rank_1e-03']} stable={out['stable_rank']} "
          f"eff={out['eff_rank']} pr={out['pr_rank']}")


def test_known_rank_decaying_spectrum():
    """A matrix with a sharply decaying spectrum: numerical rank at a loose tol < full rank,
    stable rank < numerical rank (the estimators are ordered as they should be)."""
    torch.manual_seed(1)
    L = 64
    U = torch.linalg.qr(torch.randn(L, L))[0]
    V = torch.linalg.qr(torch.randn(L, L))[0]
    # Singular values placed mid-decade (smax * 10^-i / 3) so each sits well INSIDE a tol
    # band, never on a tol*smax boundary where roundoff would flip the count. With smax=1:
    #   s = 1, 1/30, 1/300, 1/3000, ...  i.e. ~3.3e-2, 3.3e-3, 3.3e-4 for i=1,2,3.
    s = torch.tensor([1.0] + [10.0 ** (-i) / 3.0 for i in range(1, L)])
    W = (U * s) @ V.T
    out = rank_stats(W)
    # tol=1e-1: keep s>0.1 -> {1.0} -> rank 1 (3.3e-2 < 0.1).
    assert out["rank_1e-01"] == 1.0, out["rank_1e-01"]
    # tol=1e-2: keep s>1e-2 -> {1.0, 3.3e-2} -> rank 2 (3.3e-3 < 1e-2).
    assert out["rank_1e-02"] == 2.0, out["rank_1e-02"]
    # tol=1e-3: + 3.3e-3 -> rank 3; tol=1e-4: + 3.3e-4 -> rank 4.
    assert out["rank_1e-03"] == 3.0, out["rank_1e-03"]
    assert out["rank_1e-04"] == 4.0, out["rank_1e-04"]
    # stable rank = sum(s^2)/smax^2 ~ 1 + (3.3e-2)^2 + ... ~ 1.0011; rola.py rounds the
    # reported mean to 2 decimals (put() at rola.py:326), so this collapses to 1.0 — the
    # point is stable rank << numerical rank here (sharp spectrum), not its 3rd decimal.
    assert out["stable_rank"] == 1.0, out["stable_rank"]
    assert out["stable_rank"] < out["rank_1e-04"], (out["stable_rank"], out["rank_1e-04"])
    print(f"[ok] test_known_rank_decaying_spectrum: num(1e-1..1e-4)="
          f"{[out[f'rank_{t:.0e}'] for t in (1e-1,1e-2,1e-3,1e-4)]} stable={out['stable_rank']}")


def test_rank_stats_shapes():
    """rank_stats over a [B,H,L,L] batch yields the RANK_JSON key set (means + *_dist +
    spectrum), and n_slices == B*H."""
    torch.manual_seed(2)
    B, H, L = 3, 2, 32
    attn = torch.randn(B, H, L, L)
    out = rank_stats(attn)
    expected_means = {"rank_1e-01", "rank_1e-02", "rank_1e-03", "rank_1e-04",
                      "eff_rank", "pr_rank", "stable_rank"}
    for k in expected_means:
        assert k in out and f"{k}_dist" in out, f"missing {k} / {k}_dist"
        assert len(out[f"{k}_dist"]) == B * H, (k, len(out[f"{k}_dist"]))
    for k in ("spec_idx", "spec_p10", "spec_p50", "spec_p90",
              "sv_ratio_128", "sv_ratio_256", "seq_len", "n_slices"):
        assert k in out, f"missing spectrum key {k}"
    assert out["n_slices"] == B * H
    assert out["seq_len"] == L
    print(f"[ok] test_rank_stats_shapes: {out['n_slices']} slices, "
          f"{len(out['spec_idx'])} spectrum indices")


def test_rola_extractor_end_to_end():
    """Full driver on a tiny CPU RoLA: build -> extract implied W -> rank_stats.

    Asserts the realized rank is finite and within the nc*d_qk per-head ceiling (the rank
    law: rank(W) <= min(nc*d_qk, L)). This exercises build_rola_attention_extractor and the
    feature-map / gate path, which is the only model-specific seam."""
    try:
        import rola
        from rola import RoLA
    except Exception as e:   # pragma: no cover - environment-dependent
        print(f"[skip] test_rola_extractor_end_to_end: `rola` not importable ({e})")
        return

    torch.manual_seed(3)
    d_model, d_qk, d_v, nc, H = 32, 8, 8, 4, 2
    L = 256   # the inline probe only fires at L >= 256 (rola.py:750); match it.
    mixer = RoLA(d_model=d_model, d_qk=d_qk, d_v=d_v, num_chunks=nc, n_heads=H,
                 kernel="rla", phi="elu", state_norm="kappa", tie_routers=False).eval()

    # Bare mixer -> feed hidden states directly (no embedding to discover).
    g = torch.Generator().manual_seed(99)
    x = torch.randn(2, L, d_model, generator=g)

    extractor = build_rola_attention_extractor(layer_index=0)
    out = evaluate_rank(
        checkpoint_path=None,          # evaluate as-built (random init) — we only check sanity
        model_builder=lambda: mixer,   # builder can return a prebuilt module
        attention_extractor=extractor,
        eval_batch=x,
    )
    ceiling = min(nc * d_qk, L)
    num = out["rank_1e-03"]
    assert math.isfinite(num), out
    assert 1.0 <= num <= ceiling + 1e-6, f"rank {num} outside [1, nc*d_qk={ceiling}]"
    assert out["n_slices"] == 2 * H
    print(f"[ok] test_rola_extractor_end_to_end: rank_1e-03={num} "
          f"(ceiling nc*d_qk={ceiling}, stable={out['stable_rank']})")


def test_rola_attn_assembly_matches_inline_formula():
    """rola_attn_from_features must equal rola.py:376's per-slice formula exactly, so the
    batched assembly here and the inline loop measure the SAME object."""
    torch.manual_seed(4)
    B, L, H, dqk, nc = 2, 16, 2, 5, 3
    qf = torch.rand(B, L, H, dqk) + 0.1
    kf = torch.rand(B, L, H, dqk) + 0.1
    r = torch.softmax(torch.randn(B, L, H, nc), -1)
    w = torch.softmax(torch.randn(B, L, H, nc), -1)
    W_batched = rola_attn_from_features(qf, kf, r, w)   # [B,H,L,L]
    # Reference: the exact rola.py:376 per-slice expression.
    for bi in range(B):
        for hi in range(H):
            W_ref = (qf[bi, :, hi] @ kf[bi, :, hi].T) * (r[bi, :, hi] @ w[bi, :, hi].T)
            err = (W_batched[bi, hi] - W_ref).abs().max().item()
            assert err < 1e-5, f"assembly mismatch at (b={bi},h={hi}): {err}"
    print("[ok] test_rola_attn_assembly_matches_inline_formula: == rola.py:376 per-slice")


def verify_against_inline_stub():
    """STUB — checkpoint-vs-inline parity verification (the real correctness gate).

    GOAL: prove this post-hoc evaluator reproduces the inline ROLA_MEASURE_RANK number on
    a shared checkpoint, so we can delete the inline probe in task #84 without losing the
    measurement.

    PROCEDURE (requires a GPU + a paired checkpoint — NOT run here):
      1. On GPU, run a RoLA eval forward with ROLA_MEASURE_RANK=1 on a FIXED eval batch
         (the same one make_eval_batch produces). Capture the emitted `RANK_JSON {...}`
         line for the layer/epoch/seqlen of interest. That batch must be the literal
         tensor flowing through the kernel — see RISK below.
      2. Save the model state_dict to `ckpt.pt`.
      3. Post-hoc:
             out = evaluate_rank(
                 "ckpt.pt",
                 model_builder=lambda: build_same_rola_arch(),
                 attention_extractor=build_rola_attention_extractor(layer_index=LAYER),
                 eval_batch=make_eval_batch(B, L, vocab, seed=SEED, device="cuda"),
             )
      4. Assert the rank fields match the captured RANK_JSON within rounding:
             for k in ("rank_1e-03", "stable_rank", "eff_rank", "pr_rank"):
                 assert abs(out[k] - inline[k]) <= 0.01

    PARITY CONDITIONS (must all hold for an exact match):
      - SAME batch tensor (post-embedding hidden states identical). The inline probe sees
        whatever the trainer fed; this evaluator re-embeds make_eval_batch. To compare,
        feed the inline run the SAME ids and use the SAME embedding weights (they're in the
        checkpoint), OR capture the hidden states the inline forward saw and pass them
        straight in.
      - SAME state_norm path. The inline probe SVDs the read_gates AFTER kappa/per_state
        rescale (rola.py:691/693). _rola_features reproduces that on the torch path; the
        GPU forward may have used the Triton den kernel — numerically equal to within the
        kernel's own 1e-2 gate (rola.py:725), so expect agreement at the 0.01 rounding of
        the reported means, not bit-exactness.
      - max_b / max_l defaults match the inline reference (4 / 4096) — they do.
    """
    print("[stub] verify_against_inline_stub: documented; needs GPU + paired checkpoint "
          "(see docstring). Asserts nothing in the CPU suite.")


def main():
    test_known_rank()
    test_known_rank_decaying_spectrum()
    test_rank_stats_shapes()
    test_rola_attn_assembly_matches_inline_formula()
    test_rola_extractor_end_to_end()
    verify_against_inline_stub()
    print("\nAll CPU rank-evaluator tests passed.")


if __name__ == "__main__":
    main()
