"""Post-hoc realized-attention RANK evaluator (rola-bench task #85).

Standalone successor to rola.py's inline ROLA_MEASURE_RANK / RANK_JSON probe (which task
#84 removes). Import-light (torch only); decoupled from the rola.py runtime.

Public API:
  rank_stats(attn)                       — numerical / stable / effective rank of [B,H,L,L]
  evaluate_rank(ckpt, builder, extractor, batch) — generic checkpoint-driven driver
  build_rola_attention_extractor()       — per-model adapter that returns RoLA's implied W
  rola_attn_from_features(qf,kf,r,w)     — assemble W = (phi(Q)phi(K)^T) ∘ (read write^T)
  make_mqar_eval(...)                    — real MQAR eval batch (per kv-slice) for the rank probe

Metric definitions are mirrored verbatim from rola.py:311-401 (see evaluator.py docstring
for the exact line map).
"""
from .evaluator import (
    DEFAULT_TOLS,
    build_rola_attention_extractor,
    evaluate_rank,
    make_mqar_eval,
    rank_stats,
    rola_attn_from_features,
)

__all__ = [
    "DEFAULT_TOLS",
    "build_rola_attention_extractor",
    "evaluate_rank",
    "make_mqar_eval",
    "rank_stats",
    "rola_attn_from_features",
]
