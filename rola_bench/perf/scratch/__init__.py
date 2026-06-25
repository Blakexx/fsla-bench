"""ARCHIVED perf benches (moved out of the active scaling path; #40).

These are the OLD, pre-scaling-refactor benchmarks, kept for provenance — NOT deleted:

  * chunked.py    — the original matched-state e2e fwd+bwd table (RoLA vs GDN/GLA/Based/Hedgehog
                    monoliths). Superseded for the scaling story by perf/scaling.py (which now owns
                    its own copies of the shared helpers _CausalMHA / _e2e_rep / _instantiate, so the
                    scaling path no longer imports chunked.py).
  * recurrent.py  — the original op-level decode bench (incl. the dead vh_expand stub and
                    decode_routed_step the old scaling decode arm called directly). The scaling
                    decode arm is now LAYER-LEVEL (RoLA.forward(use_cache) vs an attention KV-cache
                    decode), so this op-level path is no longer on the scaling route.

They still import cleanly (`from .. import _common as C`) and can be run for the matched-state
comparison table, but they are out of the scaling sweep's dependency graph.
"""
