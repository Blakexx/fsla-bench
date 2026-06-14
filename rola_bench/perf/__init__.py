"""Canonical kernel-efficiency benchmarks for RoLA (rola-bench).

Two regimes, both at MATCHED total state (nc states x d_qk x d_v per head):
  - chunked   : training-time chunk-parallel attention, forward + backward
                (`python -m rola_bench.perf.chunked`)
  - recurrent : autoregressive decode, forward only
                (`python -m rola_bench.perf.recurrent --bench` / `--check`)

Both compare the routed fla_rola kernel against virtual-head and matched-state monolith
baselines on the canonical fla kernels. See README.md for protocol + provenance.
"""
