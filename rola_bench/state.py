"""Canonical matched-state definition, shared across all benchmarks (#126).

The matched-state x-axis is the CONTENT recurrent state — the key/value memory each method carries:

    content_state(H, nc, d_qk, d_v) = H * nc * d_qk * d_v

RoLA additionally spends a per-state denominator (normalizer) column. That is reported SEPARATELY as
overhead — it is NOT folded into the matched x-axis:

    rola_den_overhead(H, nc, d_qk) = H * nc * d_qk     # the (d_v+1) column = d_qk extra entries

Why this exists: the three benchmarks historically counted state two different ways, so the same
`nc` landed at different x-coordinates:
  * perf (`_common.shaped_dims`) and LM (`arms.MATCHED_STATE`) already use content-only.
  * the MQAR canonical grid bakes `st = REF(nc) = 624*nc = 4*12*13*nc` into the run_id, which
    INCLUDES the den column (content 4*12*12 = 576*nc + den 4*12 = 48*nc). Those run_ids are
    immutable (changing them orphans completed cells), so MQAR rows still carry `st = 624*nc`.

Convention (use everywhere a number is PLOTTED or COMPARED across benchmarks):
  - x-axis  := content_state(...)  — so MQAR / perf / LM share one axis.
  - overhead:= rola_den_overhead(...) shown as a separate series, never added to the x-axis.
  - to place an existing MQAR row on that axis, convert its baked `st` with `mqar_ref_to_content`.
"""


def content_state(H, nc, d_qk, d_v):
    """The matched recurrent KV state — the canonical cross-benchmark x-axis."""
    return H * nc * d_qk * d_v


def rola_den_overhead(H, nc, d_qk):
    """RoLA per-state denominator — the (d_v+1) normalizer column = d_qk extra entries per head per
    state: H*nc*d_qk. Reported as overhead, NOT part of the matched axis."""
    return H * nc * d_qk


def mqar_ref_to_content(st, d_v=12):
    """Convert a baked MQAR run_id state (REF(nc)=H*nc*d_v*(d_v+1), content+den) to content-only,
    so a stored MQAR row sits on the same x-axis as perf/LM. content = st * d_v/(d_v+1)."""
    return st * d_v / (d_v + 1)
