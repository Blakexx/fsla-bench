"""Clean-room canonical baselines for the MQAR matched-state comparison.

**THE CONTENDER LADDER (ruling, 2026-08-02).** The baseline set is the TAXONOMY LADDER — one
canonical occupant per TRANSITION-STRUCTURE rung, and nothing else:

  | rung | transition structure `S_t = f(S_{t-1}, k_t, v_t)` | occupant |
  |---|---|---|
  | vanilla linear attention | `S + k v^T` (no forgetting)             | `rla`  (FLA `LinearAttention`, elu feature map, normalized readout) |
  | gated linear attention   | `diag(g) S + k v^T` (elementwise decay) | `gla`  (zoology `GatedLinearAttention`) |
  | gated delta rule         | `g(I - b k k^T) S + b k v^T` (state-dependent erasure) | `gdn` (FLA `GatedDeltaNet`) |
  | attention                | unbounded KV cache                      | `attn` (the MHA oracle, in `build_configs`) |

RETIRED from the ladder, with reasons (`RETIRED_METHODS`): `hedgehog` and `based` occupy the
vanilla-LA rung ALREADY held by `rla` — they differ only in the feature map phi, which is a
map-family axis, not a transition-structure rung, and RoLA's own map-family control is the D1-dense
preset (`mqar.instances.rola-d1-dense`). `sse_ladder` is a ROUTING-family reimplementation and the
ruling excludes those outright: a second routed method is not a baseline for the routed method, it
is an unratified competitor implementation whose cells nobody here can defend.

Every surviving baseline is the method's OWN published implementation, sharing ZERO code with RoLA.
No routing, no virtual heads, no shared feature-map or normalization scaffold.

Conv policy (option A, the literature-standard MQAR form): mixer-internal short conv
is OFF. The harness wraps every model as Hybrid([BaseConv, mixer]) over 2 layers, so
layer 0 is a shared BaseConv and layer 1 is the conv-free mixer, identically for RoLA
and every baseline. This is how these baselines were evaluated on MQAR in the source
papers (zoology's own GLA defaults use_short_conv=False for exactly this reason).
Method-defining non-conv defaults (GLA output gate, GDN gate, Based taylor map, norms)
are kept.

**Matched state, RE-DERIVED 2026-08-02 (Phase 1.2).** `REF(nc)` is the RoLA CONTENT state at the
pinned MQAR geometry, read from `rola_bench.mqar.instances.content_state_floats` — `H*N*d_v` with
`H=4, d_v=64`, i.e. `256*nc`. The previous `REF(nc) = 624*nc` was written in the DELETED dimension
(`624 = H*d_qk*(d_v+1)` at `d_qk=d_v=12`): a V3 RoLA state is `[N, d_v]`, so that reference
over-sized every baseline by a factor of `d_qk = 12` at the same nc. This is finding F1's error on
the MQAR tier, and it invalidates the matched-state x-coordinate of every pre-V3 row in
`results/mqar/` (see `mqar-retrospective.md`). The denominator sidecar (`H*N` under
`state_norm='global'`) is reported as separate overhead, never on the matched axis (#126).

Baselines spend the same total recurrent floats on content (d_k*d_v), three ways, in their own
parameterization:
  wide   = widen key/feature dim, d_v held ~12, H=4
  square = key dim = value dim, H=4
  heads  = base d_qk=d_v=12, scale head count
Each method's state formula is its own; the companion verifier reads realized state
back from the built projections, so cells are confirmed, never assumed. The realized
state (not the nominal target) is the matched-state x-coordinate in the table.
"""
import math

from rola_bench.mqar.instances import MQAR_GEOM, content_state_floats

#: The HELD value width for the `wide` and `heads` shapes. Pinned to RoLA's own `d_v` at the MQAR
#: geometry so "wide" means "same value width as RoLA, wider keys" rather than "a different value
#: width AND wider keys" — one axis moves per shape, which is what makes the shape family readable.
DV = MQAR_GEOM["d_v"]
DMODEL = 128
#: Realized head-count cap for the `heads` shape. This is an ACTIVATION-memory bound, not a taste
#: threshold: baseline activations scale as H*L*B and the grid runs at train_batch 128, where H
#: beyond ~64 does not fit on a 24 GB box. Expressed on the realized head count (the thing that
#: actually costs), never on `nc` (which no longer determines it after the REF re-derivation).
MAX_BASELINE_HEADS = 64
#: FLA's GatedDeltaNet kernel caps `head_dim`; a cell above it is a kernel refusal, not a run failure.
MAX_GDN_HEAD_DIM = 256
BASE_H = MQAR_GEOM["n_heads"]
#: The matched-state reference: RoLA's CONTENT recurrent floats at the pinned MQAR geometry.
#: Derived from `instances.MQAR_GEOM`, never a literal — a geometry re-pin moves the baselines with
#: it, which is the property the pre-V3 `624*nc` constant did not have. `st` in every run_id.
REF = lambda nc: int(content_state_floats(nc))


def _gla(d_k_total, d_v_total, num_heads):
    return dict(name="zoology.mixers.gla.GatedLinearAttention",
                kwargs=dict(expand_k=d_k_total / DMODEL, expand_v=d_v_total / DMODEL,
                            num_heads=num_heads, use_short_conv=False))


def _lin(feature_map, d_k_total, d_v_total, num_heads):
    # do_feature_map_norm=True restores the canonical NORMALIZED linear-attention readout
    # (Katharopoulos 2020 / Hedgehog): o / (phi(q) . sum_j phi(k_j)). FLA's default is False
    # (unnormalized + output RMSNorm) — a modern variant, not the published form. The denominator
    # is recomputed from k (FLA's final_state stays [B,H,K,V]), so the matched-state formula is
    # unchanged. output_norm='rmsnorm' (FLA default) still applies on top. Uses the fla-rola FORK's
    # layer = upstream fla's LinearAttention verbatim + a one-line fused_recurrent scale-default fix
    # (fla's fused_recurrent omits the `scale=k.shape[-1]**-0.5` default chunk has, crashing the
    # normalized short-seq path). The fix is committed/visible in the fork, so results reproduce for
    # anyone using fla-rola — no hidden runtime monkeypatch of the canonical library.
    return dict(name="fla_rola.layers.LinearAttention",
                kwargs=dict(hidden_size=DMODEL, expand_k=d_k_total / DMODEL,
                            expand_v=d_v_total / DMODEL, num_heads=num_heads,
                            feature_map=feature_map, do_feature_map_norm=True))


def _gdn(head_dim, num_heads, dv_head):
    # FLA's canonical GatedDeltaNet, explicit head_dim widens KEYS (recall axis); expand_v
    # holds d_v at the task level. v_out = expand_v*num_heads*head_dim, so d_v_head=dv_head
    # needs expand_v = dv_head/head_dim. head_dim kernel-capped at 256.
    return dict(name="fla.layers.GatedDeltaNet",
                kwargs=dict(hidden_size=DMODEL, num_heads=num_heads, head_dim=head_dim,
                            expand_v=dv_head / head_dim, use_short_conv=False))


def baseline_cell(method, shape, nc):
    """(kernel_dict, target_state) for a canonical baseline cell, or None if (method, shape)
    is not well-defined or is structurally out of reach. Dims are solved to the nominal target
    `REF(nc)`; the realized state is read back and verified by the companion verifier.

    A `None` return is NEVER a run failure. It is one of two things, and both are reportable
    columns rather than missing data: an UNDEFINED cell (the method has no such shape) or a
    STRUCTURAL EXIT (the method's own kernel/memory bound is reached at this rung — GDN's
    `head_dim` cap, the `heads` shape's activation bound). `build_configs` logs every omission.
    """
    if method in RETIRED_METHODS:
        raise ValueError(f"MQAR baseline method {method!r} is RETIRED: {RETIRED_METHODS[method]}")
    S = REF(nc)
    H = BASE_H
    if method == "rla":                                   # vanilla LA rung, elu feature map
        if shape == "wide":                               # H*dk_head*DV = S
            dk = max(1, round(S / (H * DV)));             return _lin("elu", dk * H, DV * H, H), S
        if shape == "square":                             # H*d*d = S
            d = max(1, round(math.sqrt(S / H)));          return _lin("elu", d * H, d * H, H), S
        if shape == "heads":                              # h*DV*DV = S
            h = max(1, round(S / (DV * DV)))
            return (None if h > MAX_BASELINE_HEADS else (_lin("elu", DV * h, DV * h, h), S))
    if method == "gla":                                   # gated LA rung
        if shape == "wide":
            dk = max(1, round(S / (H * DV)));             return _gla(dk * H, DV * H, H), S
        if shape == "square":
            d = max(1, round(math.sqrt(S / H)));          return _gla(d * H, d * H, H), S
        if shape == "heads":
            h = max(1, round(S / (DV * DV)))
            return (None if h > MAX_BASELINE_HEADS else (_gla(DV * h, DV * h, h), S))
    if method == "gdn":                                   # gated delta-rule rung
        # FLA GatedDeltaNet, head_dim widens KEYS (the recall axis), d_v held via expand_v.
        if shape == "wide":                               # H*head_dim*DV = S, d_v=DV held
            hd = max(1, round(S / (H * DV)))
            return (None if hd > MAX_GDN_HEAD_DIM else (_gdn(hd, H, DV), S))
        if shape == "square":                             # H*d*d = S, head_dim=d_v=d
            d = max(1, round(math.sqrt(S / H)))
            return (None if d > MAX_GDN_HEAD_DIM else (_gdn(d, H, d), S))
        if shape == "heads":                              # h*DV*DV = S, head_dim=DV
            h = max(1, round(S / (DV * DV)))
            return (None if h > MAX_BASELINE_HEADS else (_gdn(DV, h, DV), S))
    return None


#: THE CONTENDER LADDER, in taxonomy order (weakest transition structure first). One occupant per
#: rung; `attn` is the fourth rung and is built by `build_configs.attn` (it has no state to match).
METHODS = ["rla", "gla", "gdn"]
SHAPES = ["wide", "square", "heads"]
NCS = [2, 4, 8, 16, 32, 64, 128, 256]

#: Retirement ledger for the contender set (ruling 2026-08-02: one occupant per transition-structure
#: rung; no routing-family reimplementations). `baseline_cell` refuses these by name so a stale spec
#: fails loudly instead of quietly running a different comparison.
RETIRED_METHODS: dict[str, str] = {
    "hedgehog":
        "occupies the VANILLA-LA rung, which `rla` already holds. It differs from `rla` only in the "
        "feature map phi (softmax([x,-x]) vs elu) — a map-family axis, not a transition-structure "
        "rung. RoLA's own map-family control is the D1-dense preset "
        "(mqar.instances.'rola-d1-dense'), so the comparison the hedgehog arm was making is now made "
        "inside the RoLA family at matched geometry.",
    "based":
        "also the vanilla-LA rung (taylor_exp feature map), and additionally it has no independent "
        "value width — its single head sees the full d_model, so `square`/`heads` were undefined and "
        "its state ladder was never shape-matched to the others. Retired for the same rung-collision "
        "reason as hedgehog.",
    "sse":
        "a ROUTING-family method. The ruling excludes routing-family reimplementations from the "
        "baseline set outright: a second routed method is not a control for the routed method under "
        "test, and its cells would be an unratified third-party implementation this repo cannot "
        "defend at publication. The `sse_ladder` arm builder retires with it.",
}
