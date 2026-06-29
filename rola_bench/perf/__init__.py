"""Scaling perf benchmarks for RoLA (rola-bench).

Primary path:
  - scaling: L x X% x d_qk x routing{flat,square,tree} over train (fwd+bwd), prefill (fwd),
             and decode (RoLA.forward(use_cache) recurrent vs attention KV-cache decode).

Older matched-state/debug perf benches are archived in rola-scratch/archived_rola_bench_perf.
See README.md for protocol and provenance.
"""
