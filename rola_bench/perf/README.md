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
  affect ratios similarly. The arm order rotates each round so fixed position effects are shared too.
- Each timing sample builds, runs, and frees one arm at a time, so peak memory and OOM sentinels are
  per-arm rather than group-residency artifacts.
- Timings are reported as median plus IQR over kept reps, with warmup excluded. CSV rows also include
  raw sample arrays and IQR percentages so noisy cells can be filtered before plotting.
- Prefill peak memory uses CUDA's `max_memory_allocated() - base` around the forward call.
- Training also reports retained activation memory from `saved_tensors_hooks`, before backward.

## Diagnostic profiling

`profile_graph.py` profiles one concrete scaling cell and writes an actionable report instead of a
paper table. Use it when a scaling row looks wrong and you need the computation graph broken into
phase high-water memory, whole-forward peak, analytic tensor sizes, and optional PyTorch profiler
tables/traces.

Example:

```bash
PERF_B=2 python -m rola_bench.perf.profile_graph \
  --arm rola --L 65536 --d-qk 16 --nc 64 --routing square \
  --torch-profiler --out /home/blake/rola-scratch/results/profile_graph
```

Fleet diagnostic launch:

```bash
python launch.py perf profile_graph --boxes 1 --batch 1
```

The fleet config's `profiles:` list owns the cells to profile. Each entry becomes a run id:
`perf-profile-graph-<name>`. A profile entry specifies `arm`, `L`, `batch`, `d_qk`, `nc`,
`routing`, `warmup`, `reps`, and whether to collect `torch_profiler` artifacts. Use `reps>=7` for
diagnostic timing cells; deep Nsight profiles can stay at `reps=1` because the profiler serializes
kernel replay. `PROFILE_*` env vars are kept only as an ad hoc fallback for manual/local one-off runs.

Outputs:

- `summary.json`: machine-readable cell metadata, phase timings, memory deltas, robust timing stats,
  raw samples, peak-memory samples, and analytic sizes.
- `report.md`: short human-readable diagnosis scaffold.
- `profiler_cuda.txt` / `profiler_memory.txt` / `trace.json.gz` when `--torch-profiler` is set.
- Synced profiler artifacts: `summary.json` embeds `torch_profiler.cuda_table`,
  `torch_profiler.memory_table`, and `torch_profiler.trace_json_gz_base64`
  (`trace_encoding=chrome_trace_json_gzip_base64`), so the fleet result row is useful even after the
  Vast worker and sidecar files are gone. It also includes compact `top_cuda` and `top_memory` rows for
  quick inspection.
- Buffer ledger: `summary.json` includes `buffer_ledger` and `diagnostics.unattributed_routed_peak_mib`
  to separate known model-sized buffers from hidden allocator/kernel workspace.

For Triton kernel internals, use the separate deep config:

```bash
python launch.py perf profile_graph_deep --boxes 1 --batch 1
```

`deep_profile: ncu` runs Nsight Compute against a child profile run and imports the `.ncu-rep` details
back into `summary.json` under `deep_profile.tools.ncu.details_text`. `deep_ncu_set: basic` is the
default cloud-safe pass; switch one profile to `full` only for a targeted source/stall investigation.
The fleet base image already includes Nsight Compute, but host permissions may still decide whether
hardware counters are available on a rented box.

This diagnostic entrypoint is deliberately separate from the scaling launch configs. It is for root
causing constants and memory walls, not for producing headline curves.

## Archive

Older perf experiments, matched-state scratch benches, debug launch configs, and stale perf runtime
artifacts were moved out of the main benchmark package. They live under:

```text
/home/blake/rola-scratch/archived_rola_bench_perf/
```
