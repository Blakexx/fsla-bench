# Perf measurement methodology

The rules every `rola_bench/perf/*.py` script follows. When a number looks surprising, check this
list before the kernel.

## Local 3080Ti vs A100 fleet — which numbers are citable

- **Local (this box, a shared 3080Ti)**: engineering gates only — `--smoke` construct+one-tiny-step
  checks, ad hoc single-cell sanity runs (`engine_bench.py`/`decode_bench.py` invoked directly), and
  the `engine_regression.yaml` internal bake-off when run locally for a quick check. NOT citable
  numbers — the box is shared, thermal/clock behavior is not controlled, and the GPU class doesn't
  match the paper's target hardware.
- **A100 fleet (`rola_bench/fleet/__main__.py perf <config>`, e.g. `paper_v1`)**: the CITABLE systems numbers. Fleet boxes
  are rented SINGLE-TENANT (`fleet.offer_query`/`fleet.batch: 1` in every `experiments/*.yaml`) — no
  `flock` needed there (nothing else touches that GPU for the run's duration); `flock` is a LOCAL-ONLY
  concern (see below).

## GPU exclusivity — `flock` (LOCAL runs only)

This dev box's GPU is shared across concurrent agents/jobs. Every LOCAL perf invocation that touches
CUDA timing **must** run under the box-wide lock, one bench at a time:

```
flock /tmp/rola_gpu.lock python -m rola_bench.perf.engine_bench [...]
flock /tmp/rola_gpu.lock python -m rola_bench.perf.decode_bench [...]
flock /tmp/rola_gpu.lock python -m rola_bench.perf.scaling [...]
```

A concurrent second process on the same GPU corrupts EVERY number in the run (contention, not just
noise) — there is no "close enough" here. `rola_bench.perf.smoke --smoke` runs each script as its own
subprocess sequentially for the same reason (see the module docstring for why: a failed
`torch.cuda.graph` capture can leave the CUDA context in a bad state for the rest of that *process*,
so isolating each bench to its own process also contains that failure mode). On the A100 fleet, this
isolation is structural (single-tenant boxes), not a lock.

## Warm BOTH paths separately (mandatory, not an optimization)

The RoLA chunk kernel's forward-only (inference/prefill) path and its differentiable-with-backward
(training) path are different Triton/CuTe kernel selections/autotune configs. Warming only one and
then timing the other bakes a **cold autotune compile** into the first "timed" rep of the second path,
inflating it by 10-100x on that one measurement. `engine_bench.py`'s `timed()` helper is called twice
— once per path — each with its own warmup loop before its own timing loop. Never share a warmup
between fwd-only and fwd+bwd.

## Sample size and statistic

- **Median-of-N**, N=20 by default (`--reps`), not mean — GPU timing has a long right tail (thermal
  throttling, background OS jitter, occasional slow launch) that a mean lets dominate.
- **Warmup rounds are dropped, not averaged in** — default 4 rounds (`--warmup`) before the timed loop
  starts. This is separate from (and in addition to) the "warm both paths" rule above.
- IQR is reported alongside the median as the noise indicator; a wide IQR on an otherwise-clean box
  means re-run, don't trust the median in isolation.

## Timing primitive

`torch.cuda.Event(enable_timing=True)` bracketing + `torch.cuda.synchronize()`, never host
`time.perf_counter()` around a CUDA call (that measures dispatch latency, not device time) — except in
`decode_bench.py`'s tokens/s loop, where the point is genuinely end-to-end host-observed throughput
(the metric a real decode server cares about), not a single kernel's device time.

## Memory

Peak VRAM via `torch.cuda.reset_peak_memory_stats()` immediately after warmup, then
`torch.cuda.max_memory_allocated()` delta from a post-warmup baseline (`torch.cuda.memory_allocated()`)
— NOT delta-from-zero (that double-counts the model's resident parameters as if they were transient).
Reported in GB (peak_gb). `rola_bench/perf/_common.py::ActivationProbe` additionally tracks bytes
RETAINED for backward (deduped by storage `data_ptr`) where a script needs the saved-tensor breakdown,
not just the transient peak.

## VRAM is a finding, not noise — OOM included

If a cell OOMs, that is a **result** (RoLA's whole value proposition is bounded state vs. a growing KV
cache — a cell that OOMs at some (L, nc) tells you exactly where). Perf scripts wrap each cell so an
OOM/exception writes a sentinel row and the sweep continues (see `scaling.py`'s per-cell `finally`
pattern) — never silently drop a cell, never let one OOM crash the whole sweep.

## Round-robin for cross-impl comparisons

When comparing multiple implementations/arms in the same sweep (e.g. RoLA vs. the wide-monolith
ablation vs. attention in `scaling.py`), interleave ALL impls within every timing round rather than
running each impl's full rep count back-to-back. Clock/thermal drift across a long sweep then biases
every impl equally, so between-impl *ratios* stay drift-free even if absolute numbers drift over the
sweep's wall-clock duration. `rola_bench/perf/_common.py::round_robin`/`rotated` implement this.

## Sequential, not concurrent, suites

Run one suite (mqar / lm / perf) at a time on a given box. Suites are not designed to share a GPU
concurrently — see the `flock` rule above; this is the suite-level restatement of the same constraint.

## Cross-engine comparison (`engine_bench.py --worktree`)

To compare engines (rola / rola-triton / rola-cute / a scratch branch worktree), invoke
`engine_bench.py` once per worktree with identical shape flags (`--L --B --H --d_qk --d_v --levels
--b --kernel --state_norm`) and compare the JSON rows — do NOT set `PYTHONPATH` to switch engines
mid-process; `--worktree` inserts the requested checkout at `sys.path[0]` before the first `fla_rola`
import so each invocation is a clean, single-engine process (avoids import-order engine bleed-through
within one Python process).

## History

See `perf-baseline.md` (rola-scratch) for the prior-generation kernel latency/profile findings this
methodology's rules were extracted from (e.g. the "51% of step is un-fused elementwise glue" profile,
the autotune-per-grad-mode discovery) — those numbers predate the #config-refactor layer surface and
are not re-validated here; only the MEASUREMENT RULES carry forward unchanged.
