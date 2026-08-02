"""Canonical matched-state definition, shared across all benchmarks (#126).

REWRITTEN 2026-08-02 onto the V3 state shape (finding F1). The V2 formulas this module carried —
`H * nc * d_qk * d_v` and `H * nc * d_qk` — described a per-state `[d_qk, d_v]` MATRIX. The V3
architecture deleted `d_qk` outright: one state is a `[d_v]` value row, so a head's state is
`[N, d_v]`. Anything still computing with `d_qk` overstates RoLA's state by exactly that factor.

The matched-state x-axis is the CONTENT recurrent state — the value memory each method carries:

    rola_content_state(H, N, d_v) = H * N * d_v

RoLA additionally spends ONE denominator (normalizer) column per state under `state_norm='global'`
(and none at all under `'raw'`). That is reported SEPARATELY as overhead, never folded into the axis:

    rola_den_overhead(H, N) = H * N        # the (d_v + 1) column = 1 extra entry per state

**These are documentation of the convention, not the authority.** For any live number, ask the
engine: `fla_rola.layers.RoLA.get_stats()` reports `{d_v, n_heads, num_chunks, state_floats}` for
the layer that was actually built, and `rola_bench.lm.arms.recurrent_state_floats` reads it from
there. A formula in a shared module is exactly what let a 16x error survive across every published
LM row.

Why this module exists: the three benchmarks historically counted state two different ways, so the
same `N` landed at different x-coordinates:
  * perf (`_common.shaped_dims`) and LM (`arms.recurrent_state_floats`) use content-only.
  * the MQAR canonical grid bakes `st = REF(nc) = 624*nc = 4*12*13*nc` into the run_id, which
    INCLUDES the den column (content 4*12*12 = 576*nc + den 4*12 = 48*nc). Those run_ids are
    immutable (changing them orphans completed cells), so MQAR rows still carry `st = 624*nc` — a
    V2-era number kept for identity only. It is NOT a V3 state size and must not be plotted as one.

Convention (use everywhere a number is PLOTTED or COMPARED across benchmarks):
  - x-axis  := content state — so MQAR / perf / LM share one axis.
  - overhead:= den overhead shown as a separate series, never added to the x-axis.
  - to place an existing (V2-era) MQAR row on that axis, convert its baked `st` with
    `mqar_ref_to_content`.
"""


def rola_content_state(H, N, d_v):
    """The matched recurrent value state of a V3 RoLA layer — the cross-benchmark x-axis.

    `N` = total addressable states per head (`prod_l b_l`), `d_v` = value width per state,
    `H` = heads. No `d_qk`: the architecture has none (F1).
    """
    return H * N * d_v


def rola_den_overhead(H, N):
    """RoLA denominator sidecar under `state_norm='global'` — the (d_v + 1) normalizer column, i.e.
    ONE extra entry per head per state: H*N. Zero under `state_norm='raw'`. Reported as overhead,
    NOT part of the matched axis."""
    return H * N


def mqar_ref_to_content(st, d_v=12):
    """Convert a baked MQAR run_id state (REF(nc)=H*nc*d_v*(d_v+1), content+den) to content-only,
    so a stored MQAR row sits on the same x-axis as perf/LM. content = st * d_v/(d_v+1)."""
    return st * d_v / (d_v + 1)
