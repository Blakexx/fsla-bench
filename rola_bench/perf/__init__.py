"""Canonical kernel-efficiency benchmarks for RoLA (rola-bench).

PRIMARY (the scaling story — RoLA vs attention, per-architecture curves, everything LAYER-vs-LAYER):
  - scaling   : L × X% × d_qk × routing{flat,square,tree} over train (fwd+bwd) | prefill (fwd) |
                decode (RoLA.forward(use_cache) recurrent vs the attention KV-cache decode).
                (`python -m rola_bench.perf.scaling`)

ARCHIVED (moved to perf/scratch/ by #40 — superseded for scaling, kept for provenance + the
matched-state comparison table; out of the scaling sweep's dependency graph):
  - scratch/chunked   : the original matched-state e2e fwd+bwd table (RoLA vs GDN/GLA/Based/Hedgehog).
  - scratch/recurrent : the original op-level decode bench (vh_expand / decode_routed_step).

See README.md for protocol + provenance.
"""
