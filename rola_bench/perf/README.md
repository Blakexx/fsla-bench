# RoLA scaling perf benchmarks

This package currently owns only the reportable scaling benchmark path.

## Primary path

`scaling.py` compares RoLA layer arms against attention and matched-state monolith arms over:

- sequence length `L`
- state count `nc`
- `d_qk`
- routing `{flat,square,tree}`
- bench type `{train,prefill,decode}`

The launched chunked-forward run uses:

```bash
python launch.py perf scaling_chunked --boxes 1
```

Fleet axes are configured via `SCALE_LENS[_TYPE]`, `SCALE_NCS`, `SCALE_DQKS`,
`SCALE_ROUTINGS`, and `PERF_B`. Each bench type is a separate run id:
`perf-scaling-{train,prefill,decode}`.

## Protocol

- Round-robin interleaving cycles comparable arms within a scaling cell so clock and thermal drift
  affect ratios similarly.
- Each timing sample builds, runs, and frees one arm at a time, so peak memory and OOM sentinels are
  per-arm rather than group-residency artifacts.
- Timings are reported as median plus IQR over kept reps, with warmup excluded.
- Prefill peak memory uses CUDA's `max_memory_allocated() - base` around the forward call.
- Training also reports retained activation memory from `saved_tensors_hooks`, before backward.

## Archive

Older perf experiments, matched-state scratch benches, debug launch configs, and stale perf runtime
artifacts were moved out of the main benchmark package. They live under:

```text
/home/blake/rola-scratch/archived_rola_bench_perf/
```
