# RoLA kernel-efficiency benchmarks

Canonical, reproducible perf benchmarks for the RoLA kernels.

## PRIMARY — `scaling.py` (the scaling story; everything LAYER-vs-LAYER)

RoLA (rla + gla, swept over routing `{flat,square,tree}`) vs **attention** (the only baseline that
scales differently in L), with RoLA's matched-state **wide-monolith** as the routed-vs-not ablation.
Per-architecture curves over `L × X% (state as % of attn KV cache → nc) × d_qk × routing`, for three
benches: `train` (fwd+bwd), `prefill` (fwd-only), `decode` (RoLA `forward(use_cache)` recurrent vs the
attention KV-cache decode — no op-level call; every arm is the real `nn.Module`). Each cell's WARMUP
reps absorb its first-touch Triton cold-compile, so cells time warm with no separate pre-sweep
cache-warm pass. Output → `results/perf/scaling.csv`.

```
python -m rola_bench.perf.scaling [--benches train,prefill,decode] [--lens …] [--xs …] \
        [--dqks …] [--routings flat,square,tree] [--reps N] [--out DIR]
```

Fleet axes via env: `SCALE_LENS[_TYPE]`, `SCALE_XS`, `SCALE_DQKS`, `SCALE_ROUTINGS`, `PERF_B`.
Each bench-type is its own run_id (`perf-scaling-{train,prefill,decode}`).

## ARCHIVED — `scratch/` (matched-state comparison table; out of the scaling path)

Moved to `rola_bench/perf/scratch/` (kept for provenance, NOT deleted). The scaling path no longer
imports them — it owns its copies of the shared helpers (`_CausalMHA / _instantiate / _e2e_rep`).

| module | regime | measures | run |
|---|---|---|---|
| `scratch/chunked.py`   | training-time chunk-parallel | **fwd + bwd** ms (median+IQR), peak/activation MiB, + params | `python -m rola_bench.perf.scratch.chunked [--reps N] [--out DIR]` |
| `scratch/recurrent.py` | autoregressive decode (op-level) | **fwd only** ms/token, recurrent-state MiB | `python -m rola_bench.perf.scratch.recurrent --bench [--out DIR]` / `--check` |

Default config (mirrors LM training): `B=8, H=8, L=1024, d_qk=d_v=16, bf16`, `nc ∈ {16,64,256}`.
Output → `rola-bench/results/perf/` (`kernel_efficiency.{csv,tex}`, `projection_cost.csv`,
`recurrent_decode.csv`). **Run on an idle GPU** (and NOT on a host running the live cloud
dispatcher).

## Arm registry
`chunked.py` is driven by a single `ARMS` registry (`rola_bench/perf/chunked.py`). Each arm is a
record `{name, variants, make_inputs, run, grad_keys, projection_cfg}`. `variants` is `'rla_gla'`
(run for both the RLA and scalar-GLA inner kernels) or `'own'` (a self-contained architecture run
once, reported under `variant='own'`). `bench_kernel`, `bench_projections`, and `write_tex` all
iterate the registry — adding a kernel is one record, no special-cases. The four original arms'
math is byte-identical to the pre-refactor code.

### RoLA-family arms (per RLA / scalar-GLA inner kernel)
- `rola` — the fla_rola routed `chunk_simple_gla` (shared-Gram Triton): the proposed kernel.
- `vh` — virtual-head over canonical fla (states folded into the head axis): the "no routed
  kernel" baseline.
- `monolith` — canonical fla at one wide head `d_qk=nc·d_qk` (same state, no routing): the
  dense-state kernel-cost floor. (This is the "monolith linear attention" arm.)
- `kappa` — `rola` + per-state-denominator normalization pre-pass (isolates the den-kernel cost).

### Cross-architecture arms (`variant='own'`, at matched total state)
Each holds the recurrent-state float count per head equal to the RoLA cell (`nc·d_qk·d_v`); the
sizing principle is reused from `rola_bench/mqar/experiments/canonical_baselines.py`.
- `gdn` — fla `chunk_gated_delta_rule(q,k,v,g,beta)`. State `d_k·d_v`; widen keys to
  `d_k = nc·d_qk`, hold `d_v=16` (the MQAR "wide" GDN principle — head_dim widens the recall axis).
  Keys are L2-normalized, `beta∈(0,1)`, `g` is a log-space (logsigmoid) per-head scalar gate.
- `based` — fla `fused_chunk_based(q,k,v)`. The kernel applies the 2nd-order Taylor map
  *internally*, so state `=(1+K+K²)·V` with `K` the raw feature dim (kernel-capped at 16). `K` is
  pinned at 16 (richest map) and `V` solved to hit `nc·d_qk·d_v` (V=15/60/240 for nc=16/64/256).
- `hedgehog` — fla `chunk_linear_attn` over the canonical `HedgehogFeatureMap` on q and k. The
  map doubles the feature dim (`cat([2x,-2x]).softmax`), so state `=(2·d_k)·d_v`; `d_k=nc·d_qk/2`,
  `d_v=16`. Normalized readout (`normalize=True`, the published linear-attn form).

## Protocol (why the numbers are trustworthy)
- **Round-robin interleaving**: every timing round cycles through all impls before the next, so
  clock/thermal drift biases each impl equally — between-impl *ratios* (the headline) are drift-free.
- **median + IQR** over `--reps` rounds (default 15), warmup excluded.
- **Activation memory** = bytes retained for backward (saved tensors, deduped by storage
  `data_ptr`) via `saved_tensors_hooks`, read *before* `.backward()`. **Peak** via
  `reset_peak_memory_stats` per round.
- **Fixed seeds** + fixed shapes → re-runs reproduce the table.
- **OOM / cell robustness**: every cell is wrapped in `try/except torch.cuda.OutOfMemoryError`
  (and a catch-all); a failed cell writes an `'OOM'` sentinel row, calls `empty_cache`, and the run
  *continues*. The CSV/tex are written in a `finally`, so a partial or OOM-truncated run is never
  lost. (Matched-state cross-arch arms can have large derived dims — e.g. GDN `d_k=4096` at
  nc=256 — which may exceed a kernel's practical head_dim or the GPU's memory; the sentinel keeps
  the rest of the table intact. See the `# TODO(gpu)` note in `chunked.py`.)
- **Suppressed warning**: the `vh` arm folds states into the head axis, so at large nc the head
  count `H·nc` exceeds `L`, tripping fla's benign `"format mismatch"` heuristic (`seq_len <
  num_heads`). The inputs are deliberately `[B,L,H,*]`; the warning is suppressed around those
  calls (`warnings.catch_warnings`).

### Determinism
GPU timings are **not bitwise-reproducible**; the trustworthy, reproducible quantities are the
between-impl **ratios** and the **memory** columns (peak/activation MiB, byte-exact at fixed seed).
Sources of run-to-run timing variation:
- **Triton autotune** picks a kernel config on first call per shape; warmup rounds (`WARMUP`,
  discarded) absorb this, but the *first* run after a cache wipe pays the autotune cost.
- **TF32 / cuDNN** nondeterministic algorithm selection (and clock/thermal drift) perturb absolute
  ms; round-robin interleaving cancels drift between impls so the ratios stay stable.
- Different GPU / driver / Triton version → different absolute ms (the GPU name is stamped into the
  csv/tex header).

### Timing methodology — note for the paper
`chunked.py` times with **CUDA events** (`elapsed_time`); `recurrent.py` times with
**wall-clock + `cuda.synchronize`** per step (decode is sub-ms, forward-only, and this matches
the original decode protocol). The two are *different methodologies* — do not directly conflate
chunked-ms and decode-ms; compare within a table, not across.

### Decode state is not threaded
`recurrent.py` times a *representative single decode step* (the recurrent state stays at step 0);
per-step cost is step-invariant for these kernels, so this is the correct unit. `state_mib` is the
recurrent-state footprint, not a rollout.

## Provenance
Ported (faithfully — reviewed 2026-06-14) from the validated scratch scripts
`rola-scratch/rola_fla_dev/{bench,paper_bench,decode_bench}.py`, decoupled from `/mnt/c` and the
paper repo. Correctness of the routed kernels themselves is gated separately in the fla_rola fork
tests; `recurrent.py --check` additionally asserts step-decode == the chunked routed kernel.
