# Rank evaluator — review fix plan (executable)

Source: post-hoc rank evaluator review (2026-06-14). Verdict: the **metric core is a
verbatim, correct port** of `rola.py`'s `_rank_stats` / `_effective_attention_rank` and is
genuinely model-agnostic; the **adapter/usage layer is not yet cross-model-safe**. Work the
phases in order — Phase 1 blocks any cross-model rank table for the paper.

File under work: `rola_bench/rank/evaluator.py` (+ `test_rank_cpu.py`). Line refs are at
review time; re-locate before editing.

---

## DESIGN (supersedes S1): the FINAL-attention-matrix rule
Each `attention_extractor` returns the model's **final** attention matrix `M` (the operator in
`O = M @ V`), with the model's own normalization applied — `to_final_attention(W, normalize, causal)`.
One rule covers all: softmax→`softmax(scores)`; unnormalized linear-attn→`φ(Q)φ(K)ᵀ`; normalized
linear-attn/RoLA→row-normalized. **No rank-invariance argument.** Masking is an orthogonal flag
(default off = the rank-law object). DONE in `evaluator.py` (`to_final_attention`, extractor wiring,
docstring). This dissolves old item S1.

## Phase 1 — CRITICAL (correctness; blocks the cross-model `fig:rank`)

### C1. GLA kappa/per_state uses the WRONG (undecayed) denominator  ✅ DONE
`_rola_features` (evaluator.py:345-353) always calls `_rola_perstate_den(qf, kf, write_gates)`
for `state_norm in {kappa, per_state}`. But the live `ScalarGLAKernel` rescales read-gates by
the **decayed** per-state mass `_rola_gla_perstate_den(qg, kg, write_gates, ld)` (rola.py:191,
used at ScalarGLAKernel.forward ~654-678). So rank on any `rola-gla-kappa*` / `*-per_state`
checkpoint is silently measured on the wrong `W`. RLA path (incl. canonical `rla-kappa-asym`)
is correct.
- [ ] Branch on kernel type in `_rola_features`: if the kernel is GLA (computes a log-decay —
      detect via `isinstance`/`hasattr(kernel, '_log_decay')` or a `state_norm`+kernel-class check),
      compute `ld` exactly as the kernel does and call `_rola_gla_perstate_den(qf, kf, write_gates, ld)`;
      keep `_rola_perstate_den` for the additive (RLA) kernel.
- [ ] Confirm against the inline probe path (rola.py GLA branch) that `ld` is derived identically.

### MASKING decision (LOCKED): run BOTH passes
Run every model/cell at `causal=False` (unmasked content gram — the rank-LAW object, paper
headline, tracks `nc·d_qk`) AND `causal=True` (the literal causal MQAR operator). Masking inflates
rank (Hadamard with full-rank `tril`) and is a shared confound across models, so it can't be the
headline — but we report it as the literal-operator supplement. Extractors already take `causal`;
"run both" = two passes in the runner. Per-model normalization eps is faithful (RoLA 1e-5, fla
1e-10, …) but immaterial to rank (positive feature maps → rowsums ≫ 0).

### C2. Baseline extractors — each returns its model's FINAL matrix (faithful, modular pattern)
Each adapter uses the layer's OWN projections / feature maps / normalization (never reconstructed),
shares the `(layer_index, causal)` contract, routes linear-attn through `to_final_attention`.
**Unit-test each on a tiny built model** (shape `[B,H,L,L]`, finite, rank ballpark):
- [x] `build_fla_linear_attn_extractor` → fla `LinearAttention` (RLA=`elu`, Hedgehog=`hedgehog`,
      t2r/dpfp): own `q/k_proj`+`feature_map_q/k`+`norm_q/k`(1e-4)+`do_feature_map_norm`(1e-10). DONE.
- [x] `build_softmax_attention_extractor` → zoology MHA: `softmax(QKᵀ/√d [+mask])`, own `Wqkv`. DONE.
- [ ] `build_gla_extractor` → zoology `GatedLinearAttention` (read its proj/feature-map/decay).
- [ ] `build_based_extractor` → zoology `Based` (taylor-exp feature map; its own normalization).
- [ ] GDN / delta-net: **NOT a simple gram.** Implied mixing matrix is the delta-rule transition
      `(I − tril(βKKᵀ, −1))⁻¹` composed with `QKᵀ`. Decide: (a) materialize the chunked delta
      transition as `M`, or (b) **exclude GDN with a stated reason**. Document either way.
- [ ] guard `VirtualHeadGLAKernel` (also has `_log_decay`) — the RoLA GLA branch would mishandle it
      (review S-A); reject or document as unsupported.

---

## Phase 2 — SHOULD-FIX (cross-model consistency + cost)

### S1. Enforce ONE normalization convention across all models
RoLA SVDs the RAW `W` (correct; rank-invariant under diagonal rescale). A softmax extractor
returning normalized probabilities `D⁻¹exp(S)` is NOT apples-to-apples for the numerical-rank /
spectrum estimators (only the mathematical rank is invariant; the σ-distribution shifts).
- [ ] Convention: **SVD the pre-normalization scores for every model** (RoLA raw `W`; MHA logits
      `S`). Document in the module docstring + paper methods.

### S2. Default `max_b/max_l` mismatch vs reference
Defaults are `max_b=64, max_l=8192`; the reference `_effective_attention_rank` uses `4/4096`.
At 64/8192 the per-slice SVD loop is minutes-to-OOM and the `*_dist` support is 16× the inline
runs being compared against; the `verify_against_inline_stub` parity comment (test_rank_cpu.py
~209) is consequently false.
- [ ] Align defaults to `4/4096` (or justify 64/8192 explicitly) and fix the stub comment.

### S3. `torch.load(weights_only=False)` note
`_load_state_dict` (evaluator.py:257) is arbitrary-code-execution on untrusted files.
- [ ] Add a comment/flag documenting "trusted local checkpoints only" (loader is otherwise
      correctly fail-loud).

---

## Phase 3 — Verification / deliverable

- [ ] Add a GLA-kappa end-to-end test that asserts **parity vs the inline probe** (not just
      finiteness) — would have caught C1 (current tests are RLA-only).
- [ ] Once Phase 1–2 land AND the MQAR sweep's best-checkpoints are collected: run the evaluator
      across all model families on the saved checkpoints with a pinned eval batch → the
      cross-model rank table / `fig:rank`. (Ties off paper tasks #63, #85.)

---

## Out of scope here
Performance-benchmark review fixes are tracked separately (perf review pending).
