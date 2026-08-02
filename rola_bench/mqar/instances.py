"""THE MQAR RoLA PRESET REGISTRY — preset name -> `zoology.mixers.rola.RoLAMixer` kwargs.

REBUILT 2026-08-02 (Phase 1.2). The pre-V3 registry was 13 presets keyed on axes the V3
architecture deleted (`d_qk`, `kernel='gla_scalar'`, `state_norm in {'kappa','per_state'}`,
`phi`). None of them were translated: a preset defined by a deleted knob cannot be re-pinned to a
live knob without silently publishing a different cell under a published name. They retire by name
with a reason, in `RETIRED_INSTANCES`, exactly as `named_configs.RETIRED_CONFIGS` does.

**Symbols.** `N` = addressable states per head (`nc`); `D` = `levels_per_head`; `b_l` =
`states_per_level`, `N = prod_l b_l`; `S = sum_l b_l` = addressing entropy; `d_v` = value width per
state per head; `H` = heads; `alpha` = entmax order of a sparse duty (1.5 entmax / 2.0 sparsemax),
a per-LEVEL field of `LevelRoutingConfig` applying to whichever side of that level is `'sparse'`.
"duty" = one (side, level) pair.

**THE SLATE — the three canonical wirings plus the map-family control.** These are the SAME cells
`rola_bench.named_configs` publishes on the layer path; this module is their MQAR spelling (conv
OFF, and topology given as an explicit (D, b) so the grid can sweep N over non-square rungs, which
`named_configs._square_topology` cannot). The two registries are gated against each other in
`rola_bench/test_mqar_grid.py` — a divergence would mean the MQAR cell and the layer/perf cell
of the same published name are different experiments.

  * `rola-arm1-densread-sparsewrite` — ARM 1, the `(N, 1)` touch-product point. Coarse level untied
    dense-read/sparse-write, fine level untied dense/dense. O(N) read touch, O(1) write.
  * `rola-arm2-union`  — ARM 2, the UNION cell. Coarse level untied sparse/sparse, which the engine
    resolves to `UnionRouting` (one shared entmax support, split by sigma); fine untied dense/dense.
    Off the touch hyperbola, O(1) both sides.
  * `rola-arm3-levelsplit` — ARM 3, the equal-width LEVEL SPLIT, the `(sqrt N, sqrt N)` point.
    Coarse untied dense-read/sparse-write, fine untied sparse-read/dense-write: each level carries
    exactly one sparse duty and the two levels carry OPPOSITE duties.
  * `rola-d1-dense` — **THE D1 DENSE CONTROL**, the map-family control. `D=1`, `b=[N]`, one dense
    (softmax) routing level, untied. This is monolithic normalized-exponential linear attention
    written in RoLA's own vocabulary: no factorization, no sparsity, every state touched by every
    token. It is the degenerate corner of the whole design space and it is what the theory's
    structured wirings must be measured AGAINST — if a wiring cannot beat D1-dense at matched N and
    d_v, the structure bought nothing. It is also the arm that carries the E2 "does a routed feature
    map train at all" burden in its simplest form.

**VARIANTS — only where a grid axis demands one.**
  * `rola-arm3-levelsplit-a2w` — ARM 3 with `alpha=2.0` (sparsemax) on the level whose sparse duty
    is the WRITE and `alpha=1.5` on the level whose sparse duty is the READ ("commit hard, address
    softly"). The alpha axis; mirrors `named_configs.RoLA-base-levelsplit-a2w-global`.
  * `rola-arm1-tied` / `rola-d1-dense-tied` — the tied spellings, for the coupling-independence
    cells: `tied=True` shares ONE projection between read and write on that level, which is the
    exact structural statement "the read cue and the write cue are the same cue". The
    cue-consistency boundary (the theory's Eq. 7) is a statement about read/write routing agreement,
    so an untied-vs-tied pair at matched N is the only in-registry probe of it that exists today.
    (The degrading-cue DATA generator, the other half, is not built — audit §4.1 item 2.)

**GEOMETRY (`MQAR_GEOM`) — a stated derivation, ratifiable, not a magic number.** The LM flagship
was re-pinned 2026-08-02 (finding F1) to `H=8`, `d_v=128` at `d_model=512`, i.e. a readout width
`H*d_v = 2*d_model` and a per-state value width equal to the standard head width. MQAR's backbone is
`d_model=128`, so the same two invariants give `H*d_v = 256` at MQAR scale; holding the MQAR
baseline suite's own head convention (`canonical_baselines.BASE_H = 4`) fixes `d_v = 64`. Content
recurrent state is then `H*N*d_v = 256*N` floats/layer with a `H*N = 4*N` denominator sidecar under
`state_norm='global'`. Both numbers are REALIZED (`RoLA.get_stats()`), never asserted — see
`realized_state_floats`.

  The pre-V3 grid's `d_qk=d_v=12, H=4` and its `REF(nc) = 624*nc` matched-state reference were
  written in the DELETED dimension (`624 = 4*12*13 = H*d_qk*(d_v+1)`). A V3 state is `[N, d_v]`, so
  that reference over-sizes every baseline by `d_qk = 12x` at the same nc — the same class of error
  as finding F1 on the LM tier. `canonical_baselines.REF` is re-derived from this geometry.

Every preset returns kwargs for `zoology.mixers.rola.RoLAMixer`, which validates them against the
V3 layer's OWN signature — so a stale key here fails at construction, by name.
"""
from __future__ import annotations

from rola_bench.rola_compat import topology_from_legacy

# --- the pinned MQAR head geometry (see the module docstring for the derivation) -----------------
MQAR_GEOM = dict(n_heads=4, d_v=64)

#: THE READOUT TILE, pinned rather than inherited (found by the Phase 1 smoke, 2026-08-02).
#:
#: The V3 layer defaults `state_block` to `topology.widths[-1]` — the FINEST branch width. That
#: default couples a cell's EVAL-mode buildability to its factorization, because the CUDA consumer
#: only instantiates `BC in _SUPPORTED_BC = (16, 64)`: `b=[16,16]` evaluates and `b=[4,4]`,
#: `b=[32,32]`, `b=[256]` all raise `NotImplementedError` out of `resolve_occupancy`. The grid
#: SWEEPS the factorization, so inheriting that default would make half the N ladder unrunnable at
#: eval time for a reason that has nothing to do with the experiment — and it would have been
#: discovered per-cell on a rented box.
#:
#: Pinning `state_block = 16` decouples them: every factorization on the ladder evaluates, and the
#: readout tile is a constant across the sweep instead of a hidden covariate that moves with N.
#: Verified on GPU across `b in {[4,4], [8,16], [16,16], [32,32], [256]}` at `d_v=64`.
MQAR_STATE_BLOCK = 16

#: Content recurrent floats per layer per state-rung, as a function of N. Derived from MQAR_GEOM,
#: NOT a literal — `canonical_baselines.REF` reads it so the baseline matched-state solve and the
#: RoLA geometry can never drift apart.
def content_state_floats(nc: int) -> int:
    """H * N * d_v — the V3 RoLA content state at the pinned MQAR geometry."""
    return MQAR_GEOM["n_heads"] * int(nc) * MQAR_GEOM["d_v"]


def denominator_state_floats(nc: int, state_norm: str = "global") -> int:
    """H * N under `state_norm='global'` (the denominator sidecar), exactly 0 under `'raw'`."""
    return MQAR_GEOM["n_heads"] * int(nc) if state_norm == "global" else 0


# --- the routing rows, one per canonical wiring ---------------------------------------------------
# Rows are plain dicts (pydantic-serializable through zoology's ModuleConfig.kwargs); the mixer
# materializes them into the engine's frozen LevelRoutingConfig with strict key validation.
_DENSE = dict(tied=False, read="dense", write="dense")

_WIRINGS: dict[str, dict] = {
    # ARM 1 — dense read / sparse write on the coarse level; dense/dense fine. D=2.
    "rola-arm1-densread-sparsewrite": dict(
        D=2, rows=[dict(tied=False, read="dense", write="sparse"), dict(_DENSE)]),
    # ARM 2 — union: coarse untied sparse/sparse resolves to UnionRouting; dense/dense fine. D=2.
    "rola-arm2-union": dict(
        D=2, rows=[dict(tied=False, read="sparse", write="sparse"), dict(_DENSE)]),
    # ARM 3 — level split: one sparse duty per level, opposite across levels. D=2.
    "rola-arm3-levelsplit": dict(
        D=2, rows=[dict(tied=False, read="dense", write="sparse"),
                   dict(tied=False, read="sparse", write="dense")]),
    # ARM 3's ALPHA-CHOICE variant: sparsemax on the sparse WRITE duty, entmax-1.5 on the sparse READ.
    "rola-arm3-levelsplit-a2w": dict(
        D=2, rows=[dict(tied=False, read="dense", write="sparse", alpha=2.0),
                   dict(tied=False, read="sparse", write="dense", alpha=1.5)]),
    # THE D1 DENSE CONTROL — one dense softmax level, no factorization. D=1.
    "rola-d1-dense": dict(D=1, rows=[dict(_DENSE)]),
    # Coupling-independence pair: the tied spellings (one shared projection per level => read cue
    # IS the write cue).
    "rola-arm1-tied": dict(
        D=2, rows=[dict(tied=True, read="dense", write="sparse"),
                   dict(tied=True, read="dense", write="dense")]),
    "rola-d1-dense-tied": dict(D=1, rows=[dict(tied=True, read="dense", write="dense")]),
}

#: Declaration order == the order the grid reports them in.
ROLA_INSTANCES = tuple(_WIRINGS)

#: The three canonical wirings plus the control — the slate every law-grid cell runs.
CANONICAL_WIRINGS = ("rola-arm1-densread-sparsewrite", "rola-arm2-union", "rola-arm3-levelsplit",
                     "rola-d1-dense")


class PresetRungUnavailable(ValueError):
    """A preset cannot be built at this state rung for a STRUCTURAL reason.

    Not an error in the spec and not a run failure: a D=2 wiring needs `N` to factor into two
    branches of width >= 2, so `N = 2` has no D=2 spelling at all. The grid reports these the same
    way it reports GDN's `head_dim` ceiling — an omitted cell with a printed reason, i.e. a cost
    column for that wiring, never missing data. Callers that do not want the omission (a gate, a
    single-cell build) simply let it propagate.
    """


def _balanced_pair(nc: int) -> list[int]:
    """The most balanced (b0, b1) with `b0 * b1 == nc` and `b0 <= b1`.

    The pre-V3 grid used `rola_compat.topology_from_legacy(nc, 'square')`, which demands a PERFECT
    SQUARE and therefore excluded half of every power-of-two ladder (2, 8, 32, 128, 32768...). The
    D=2 wirings are defined by their DUTY assignment, not by equal branch widths, so the right
    factorization is the most balanced one that exists — `[8, 16]` at N=128 is the same wiring as
    `[16, 16]` at N=256, which is exactly how the LM flagship is already pinned (mixed radix).
    """
    nc = int(nc)
    best = None
    for b0 in range(2, int(nc ** 0.5) + 1):
        if nc % b0 == 0:
            best = [b0, nc // b0]
    if best is None:
        raise PresetRungUnavailable(
            f"N={nc} admits no two-level factorization with both branch widths >= 2 (its only "
            "factor pair is 1 x N, and a width-1 level is a no-op that would silently make a D=2 "
            "wiring a D=1 one). This rung has no spelling for this preset; drop it from the ladder "
            "or use the D=1 control at this rung.")
    return best


def _topology(name: str, nc: int, states_per_level=None) -> tuple[int, list[int] | int]:
    """(D, b) for a preset at state-rung `nc`.

    `states_per_level` (an explicit mixed-radix list, e.g. `[8, 16]`) OVERRIDES the default
    factorization — this is the addressing-entropy ladder `S = sum_l b_l` the audit (§4.1 item 3)
    named as missing: at fixed `N`, `[2,2,2,2,2,2,2,2]` (S=16) and `[256]` (S=256) are different
    addressing budgets. When given, it must multiply to `nc` and have the preset's own depth.
    """
    D = _WIRINGS[name]["D"]
    if states_per_level is not None:
        b = [int(x) for x in states_per_level]
        prod = 1
        for x in b:
            prod *= x
        if prod != int(nc):
            raise ValueError(
                f"states_per_level={b} multiplies to {prod}, not nc={nc}. The state rung and the "
                "addressing factorization must agree — N = prod_l b_l is the definition, not a hint.")
        if len(b) != D:
            raise ValueError(
                f"preset {name!r} has depth D={D} but states_per_level={b} has depth {len(b)}. "
                "Depth is part of the WIRING (which duties exist on which level); a different depth "
                "is a different preset, not a different rung.")
        return D, b
    b = [int(nc)] if D == 1 else _balanced_pair(nc)   # D=1: the unfactorized control
    _check_engine_caps(name, nc, b)
    return D, b


def _check_engine_caps(name: str, nc: int, b: list[int]) -> None:
    """Refuse a rung the ENGINE cannot build, with the engine's own numbers.

    The V3 routing tree caps branch width at `MAX_BRANCH_WIDTH` and depth at `MAX_ROUTING_DEPTH`.
    The consequence is structural and worth stating plainly, because it is an argument the paper
    makes rather than a limitation it works around: **the D1 dense control cannot exist above
    N = MAX_BRANCH_WIDTH.** An unfactorized router is one softmax over N logits, and the kernel does
    not instantiate one that wide — so at the separation scales the monolithic control has no
    spelling at all, while the factorized wirings reach N = 65536 as `[256, 256]`. That is the same
    kind of reported ceiling as GDN's `head_dim` cap, and it is reported the same way.
    """
    from rola_bench.engine import MAX_BRANCH_WIDTH, MAX_ROUTING_DEPTH
    if int(nc) % MQAR_STATE_BLOCK:
        raise PresetRungUnavailable(
            f"N={nc} is not a multiple of the pinned readout tile "
            f"state_block={MQAR_STATE_BLOCK}, and the engine requires the tile to divide N (owners "
            "partition the leaf space). Leaving state_block to follow the finest branch width "
            "instead would put BC outside the consumer's instantiated set, so eval-mode cells below "
            f"N={MQAR_STATE_BLOCK} have no launchable geometry at all. This is a reported floor on "
            "the state ladder, not a spec error: the matched-state curve for a ROUTED arm starts at "
            f"N={MQAR_STATE_BLOCK}, while the monolithic baselines have no such floor and are "
            "plotted from their smallest rung.")
    if len(b) > MAX_ROUTING_DEPTH:
        raise PresetRungUnavailable(
            f"preset {name!r} at N={nc} needs depth {len(b)}, above the engine's "
            f"MAX_ROUTING_DEPTH={MAX_ROUTING_DEPTH}.")
    widest = max(b)
    if widest > MAX_BRANCH_WIDTH:
        raise PresetRungUnavailable(
            f"preset {name!r} at N={nc} factorizes as {b}, whose widest level ({widest}) exceeds the "
            f"engine's MAX_BRANCH_WIDTH={MAX_BRANCH_WIDTH}. For the D=1 control this is the ceiling "
            "on an unfactorized router — one softmax over N logits — and it is a REPORTED result, "
            f"not a gap: above N={MAX_BRANCH_WIDTH} the monolithic control has no spelling while the "
            "factorized wirings still do.")


def rola_instance(name: str, nc: int, *, n_heads: int | None = None, d_v: int | None = None,
                  states_per_level=None, state_norm: str = "global", decay: str = "none",
                  router_bias: bool = True, use_short_conv: bool = False,
                  decay_target_leaf_rate: float | None = None) -> dict:
    """`zoology.mixers.rola.RoLAMixer` kwargs for one named preset at state rung `nc`.

    There is no `d_qk` parameter and no `kernel` parameter: V3 owns no Q/K surface, and the decay
    vocabulary is the layer's own `decay in {'none','leaf_mass'}`. `router_bias` defaults to True
    (the flagship pin — every `named_configs` builder passes it), NOT to the layer default.
    """
    if name in RETIRED_INSTANCES:
        raise ValueError(f"RoLA MQAR preset {name!r} is RETIRED: {RETIRED_INSTANCES[name]}")
    if name not in _WIRINGS:
        raise ValueError(f"unknown RoLA instance {name!r}; expected one of {sorted(_WIRINGS)}")
    D, b = _topology(name, nc, states_per_level)
    kw = dict(
        n_heads=MQAR_GEOM["n_heads"] if n_heads is None else n_heads,
        d_v=MQAR_GEOM["d_v"] if d_v is None else d_v,
        levels_per_head=D, states_per_level=b,
        level_routing_configs=[dict(r) for r in _WIRINGS[name]["rows"]],
        decay=decay, state_norm=state_norm, router_bias=router_bias,
        use_short_conv=use_short_conv, state_block=MQAR_STATE_BLOCK,
        # F5's gain pin, MQAR spelling: `gain_bias_init` is the ONE gain kwarg the layer takes and
        # its default is a flagged model decision in the engine docstring. Pinned, not inherited.
        gain_bias_init=None,
    )
    if decay_target_leaf_rate is not None:
        kw["decay_target_leaf_rate"] = decay_target_leaf_rate
    return kw


def realized_state_floats(name: str, nc: int, **kw) -> dict:
    """Construct the preset on CPU and read `RoLA.get_stats()` back off it.

    The MQAR grid's x-axis is recurrent state, so that number must be MEASURED (finding F1: the LM
    tier published a formula-derived state that was 16x wrong for a year). Returns
    `{d_v, n_heads, num_chunks, state_floats, content, denominator}`.
    """
    from zoology.mixers.rola import RoLAMixer
    kwargs = rola_instance(name, nc, **kw)
    d_model = kwargs["n_heads"] * kwargs["d_v"]
    layer = RoLAMixer(d_model=d_model, layer_idx=0, **kwargs)
    stats = dict(layer.get_stats())
    stats["content"] = stats["n_heads"] * stats["num_chunks"] * stats["d_v"]
    stats["denominator"] = stats["state_floats"] - stats["content"]
    return stats


# --- RETIREMENT LEDGER ---------------------------------------------------------------------------
# Same contract as `named_configs.RETIRED_CONFIGS`: these are not "needs an edit". The knob that
# DEFINED each preset does not exist in the V3 architecture, so re-pinning the name to a live knob
# would publish a different cell under a name that already has results attached to it. `rola_instance`
# refuses them by name. Dispositions are recorded so the 683 completed pre-V3 cells in `results/mqar/`
# stay interpretable (see mqar-retrospective.md).
RETIRED_INSTANCES: dict[str, str] = {
    "rola-rla-asym":
        "pre-V3 flat (D=1) dense/dense router at d_qk=12. DISPOSITION: succeeded by "
        "'rola-d1-dense' (the D1 dense control), which is the same WIRING at the V3 geometry — but "
        "not the same cell: d_qk is gone, so the state is [N,d_v] not [d_qk,d_v].",
    "rola-rla-sym":
        "the tied spelling of 'rola-rla-asym'. DISPOSITION: succeeded by 'rola-d1-dense-tied'.",
    "rola-rla-asym-ps":
        "defined by state_norm='per_state'. RETIRED with no successor: the V3 normalization set is "
        "{'raw','global'} (campaign-prereqs E2); per-state normalization has no V3 spelling.",
    "rola-rla-sym-ps":
        "state_norm='per_state', tied. RETIRED with no successor (see 'rola-rla-asym-ps').",
    "rola-rla-kappa-asym":
        "defined by state_norm='kappa'. RETIRED with no successor: kappa was retired outright "
        "(E2, [[rola-self-normalizing-dead-leaves-inert]] — the readout ratio makes dead-leaf reads "
        "inert, so the kappa machinery was solving a non-problem). This was the canonical arm of the "
        "old grid; its rows in results/mqar/ are pre-V3 evidence and do not carry forward.",
    "rola-rla-kappa-sym":
        "state_norm='kappa' with tied routers. RETIRED with no successor, for the same reason as "
        "'rola-rla-kappa-asym': kappa is not a V3 normalization. The TYING half of its question "
        "survives as 'rola-d1-dense-tied'.",
    "rola-rla-asym-tieinit":
        "defined by tie_router_init=True, which is not a V3 layer kwarg. RETIRED: tying is "
        "structural and per-level (LevelRoutingConfig.tied), not an init trick. The question it "
        "asked — does a tied INIT of untied routers help — is not on the paper's axis list.",
    "rola-gla-scalar-sym":
        "kernel='gla_scalar' (per-token scalar decay). RETIRED (E3): the spelling exists in no "
        "engine checkout. The live decay vocabulary is decay='leaf_mass' (detached leaf mass), which "
        "is a DIFFERENT clock, so this is not a rename.",
    "rola-gla-scalar-norm-sym":
        "kernel='gla_scalar' with the normalized readout flag. RETIRED (E3): both the kernel name "
        "and its nested `kernel_kwargs={'normalized': ...}` channel are gone; V3 spells "
        "normalization as `state_norm in {'raw','global'}` on the layer itself.",
    "rola-gla-scalar-asym":
        "kernel='gla_scalar', untied. RETIRED (E3): the spelling exists in no engine checkout, and "
        "the live decay='leaf_mass' clock is a different operation, not a rename.",
    "rola-gla-scalar-norm-asym":
        "kernel='gla_scalar' with the normalized readout flag, untied. RETIRED (E3), for both "
        "reasons above: the kernel name and the nested normalization channel are each gone.",
    "rola-gla-kappa-sym":
        "kernel='gla_scalar' AND state_norm='kappa', tied — doubly retired (E2 + E3). Neither half "
        "has a V3 spelling, so there is nothing to re-pin it to.",
    "rola-gla-kappa-asym":
        "kernel='gla_scalar' AND state_norm='kappa' — doubly retired (E2 + E3). This was the "
        "'normalized (paper)' arm of the old canonical grid and of ood_recall.yaml.",
}
