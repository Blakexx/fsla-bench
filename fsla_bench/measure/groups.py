"""THE GROUPS: which central cells launch together, the claim they hold, and which arms are timed in one session.

A GROUP is a selection of central cells (`fsla_devtools.cells`) and its relation: `holds`, the claim in words; `equal`,
the parameters every cell of the group carries with one value; and `together`, the arm sets each timed in one session
over the group's cells, in every checkout a run measures. `check` holds a group to its claim against the central
registry. Bazel reads this file: the bench's `groups` extension (`bazel/groups.bzl`) runs `described()` and writes the
groups as Starlark (`@bench_groups//:groups.bzl`), which the sessions' macro (`measure/sessions.bzl`) declares targets
over -- a group that breaks its claim fails the fetch.

    attention(tokens, states, dv, cells, heads) FSLA's carry cells of one length, state count N and value width beside
                                                the attention cell of that length and width; each arm of
                                                `ATTENTION_ARMS` timed beside `flash` in its own session; `heads` the
                                                heads every cell draws a call
    layer(cell)                                 a layer cell (its input, and the routing and gain its layer is built
                                                with); its solve and its decode step each timed alone
    layer_prefill(cell, ...)                    THE LAYER-LEVEL PREFILL: FSLA's stateless prefill beside
                                                FlashAttention with its QKV projection, one session; each of the op's
                                                stages (`LAYER_STAGES`) in its own
    incremental(chunk, dv, contexts, ...)       THE INCREMENTAL CELL (ruled 2026-10-18): c = `chunk` new tokens after
                                                each context T, FlashAttention over the T + c KV cache (with `hidden`,
                                                its QKV projection for `heads` heads too), every context in one session;
                                                FSLA's side joins once its carry starts from a stored state

`GATE` names the groups the kernel's gate reads first. The capacity-fair comparison is at N = L (fsla's
docs/measurement.md); a group's claim says whether its FSLA cells' state count equals their tokens.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    name: str
    holds: str
    equal: tuple[str, ...]
    cells: tuple[str, ...]
    together: tuple[tuple[str, ...], ...]
    states: int | None = None
    #: the attention reference's arms this group registers itself: (arm, its parameters)
    flash_arms: tuple[tuple[str, dict], ...] = ()
    #: the heads a call draws, every cell of the group alike (FSLA's write heads, attention's heads)
    heads: int = 1


#: FSLA's arms an attention group times, EACH IN ITS OWN SESSION beside `flash`: a session holds every member set up at
#: once, so one FSLA arm a session bounds the device memory a group needs
ATTENTION_ARMS = ("prefill",)


def attention(tokens: int, states: int, dv: int, fsla_cells: tuple[str, ...], heads: int = 1) -> Group:
    fair = "N = L, capacity-fair" if states == tokens else f"N {'>' if states > tokens else '<'} L, not capacity-fair"
    suffix = "" if heads == 1 else f"-h{heads}"
    return Group(f"L{tokens}-N{states}-dv{dv}{suffix}",
                 f"{tokens} tokens and value width {dv}, bf16, attention {'one head' if heads == 1 else f'{heads} heads'}"
                 f" of width {dv}; FSLA's state N = {states} a head ({fair})",
                 ("tokens", "dv"), (*fsla_cells, f"qkv-L{tokens}-dv{dv}{suffix}"),
                 tuple((arm, "flash") for arm in ATTENTION_ARMS), states, heads=heads)


def incremental(chunk: int, dv: int, contexts: tuple[int, ...], hidden: int = 0, heads: int = 1) -> Group:
    level = f"-h{hidden}x{heads}" if hidden else ""
    arms = tuple((f"flash@T={t}" + (f",h={hidden}x{heads}" if hidden else ""),
                  {"context": t, **({"hidden": hidden, "heads": heads} if hidden else {})}) for t in contexts)
    what = (f"with its QKV projection from hidden {hidden} for {heads} heads in one GEMM" if hidden
            else "attention alone, one head")
    return Group(f"inc-c{chunk}-dv{dv}{level}",
                 f"{chunk} new tokens after T = {', '.join(map(str, contexts))} absorbed, value width {dv}, bf16: "
                 f"FlashAttention over the T + c cache, bottom-right causal, {what}",
                 ("tokens", "dv"), (f"qkv-L{chunk}-dv{dv}",), (tuple(a for a, _ in arms),), None, arms)


def layer(cell: str) -> Group:
    return Group(cell, f"the layer cell {cell}: its producer's solve, and its decode step where it takes steps", (),
                 (cell,), (("entmax_solve",), ("decode_step",)))


#: the op's stages a layer-prefill group times on the layer's own bundle, EACH IN ITS OWN SESSION (one FSLA arm a session
#: bounds the device memory, as `ATTENTION_ARMS`): they split the layer's time by launch
LAYER_STAGES = ("layer_op",)


def layer_prefill(cell: str, tokens: int, dv: int, hidden: int, heads: int) -> Group:
    flash = f"flash@T=0,h={hidden}x{heads}"
    suffix = "" if heads == 1 else f"-h{heads}"
    return Group(f"layer-prefill-{cell.removeprefix('layer-')}",
                 f"the layer cell {cell}: its stateless prefill beside FlashAttention with its QKV projection from "
                 f"hidden {hidden} for {heads} heads, the output projection outside both; each of the op's stages "
                 f"alone on the layer's own bundle",
                 ("tokens", "dv"), (cell, f"qkv-L{tokens}-dv{dv}{suffix}"),
                 (("layer_prefill", flash), *((stage,) for stage in LAYER_STAGES)), None,
                 ((flash, {"hidden": hidden, "heads": heads}),))


GROUPS = {g.name: g for g in (
    attention(16, 256, 64, ("carry-dead-16x16-T16-carried", "carry-deposit-16x16-T16")),
    attention(32, 256, 64, ("carry-dense-16x16-T32",)),
    attention(64, 256, 64, ("carry-alt-k4-16x16-T64",)),
    attention(128, 4096, 64, ("carry-dense-64x64-T128-carried",)),
    attention(256, 4096, 64, ("carry-dense-64x64-T256-sup0.25-carried", "carry-dense-64x64-T256-sup0.25",
                              "carry-dense-16x16x16-T256", "carry-alt-k4-16x16x16-T256")),
    attention(256, 65536, 64, ("carry-dense-16x16x16x16-T256", "carry-alt-k2-16x16x16x16-T256")),
    attention(1024, 65536, 32, ("carry-alt-k4-256x256-T1024-dv32",)),
    attention(1024, 256, 64, ("carry-dense-16x16-T1024", "carry-alt-k4-16x16-T1024")),
    attention(1024, 65536, 64, ("carry-dense-256x256-T1024", "carry-alt-k4-256x256-T1024", "carry-alt-k16-256x256-T1024",
                                "carry-cohort-k4-c64-256x256-T1024", "carry-cohort-k16-c64-256x256-T1024",
                                "carry-dense-256x256-T1024-sup0.5")),
    attention(1024, 65536, 128, ("carry-alt-k4-256x256-T1024-dv128",)),
    attention(1792, 65536, 64, ("carry-alt-k4-256x256-T1792",)),
    attention(4096, 65536, 64, ("carry-alt-k4-256x256-T4096",)),
    attention(16384, 16384, 64, ("carry-dense-128x128-T16384", "carry-alt-k4-128x128-T16384")),
    attention(16384, 65536, 64, ("carry-dense-256x256-T16384", "carry-alt-k16-256x256-T16384",
                                 "carry-alt-k4-256x256-T16384", "carry-cohort-k16-c64-256x256-T16384")),
    attention(65536, 65536, 64, ("carry-dense-256x256-T65536", "carry-alt-k4-256x256-T65536")),
    attention(65536, 65536, 64, ("carry-dense-256x256-T65536-h4", "carry-alt-k4-256x256-T65536-h4"), heads=4),
    layer("layer-alt-entmax1.5-64x64-T2048"),
    layer("layer-dense-softmax-64x64-T512"),
    layer("layer-dense-softmax-64x64-T4096"),
    layer("layer-alt-entmax1.5-8x8x8x8-T512-g8"),
    layer("layer-alt-entmax1.5-16x16-T128-h2-dec8-fp32"),
    layer("layer-dense-softmax-256-T128-h2-dec8-fp32"),
    *(layer_prefill(c, 65536, 64, 256, 4) for c in ("layer-alt-entmax1.5-256x256-T65536-h4",
                                                   "layer-alt-entmax1.5-256x256-T65536-h4-g4",
                                                   "layer-alt-entmax1.5-256x256-T65536-h4-g16")),
    *(incremental(c, 64, (0, 4096, 16384, 65536, 262144, 1048576)) for c in (1024, 4096, 16384)),
    *(incremental(c, 64, (0, 4096, 16384, 65536, 262144), hidden=2048, heads=16) for c in (1024, 4096)),
)}
GATE = ("L65536-N65536-dv64", "L1024-N65536-dv64")


def check(group: Group, registry) -> None:
    """Refuses a group whose cells the registry lacks or whose cells break its claim, read off the cells' own fields
    (`fsla_devtools.cells.form.Cell`): an `equal` field with two values, an FSLA cell whose state count is not the
    group's N, a cell drawing other than the group's heads, an attention cell that is not bf16."""
    from fsla_devtools.cells import build

    missing = sorted(set(group.cells) - set(registry.cells))
    if missing:
        raise KeyError(f"group {group.name} names cells the central registry does not hold: {missing}")
    cells = {c: build(registry.cell(c)) for c in group.cells}
    for field in group.equal:
        values = {c: getattr(cell, field) for c, cell in cells.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"group {group.name} holds {field} equal, but its cells carry {values}")
    if group.states is None:
        return
    for c, cell in cells.items():
        if cell.kind == "carry" and group.states != cell.routing.N:
            raise ValueError(f"group {group.name} claims N = {group.states}, but {c} has {cell.routing.N} states")
        if group.heads != cell.H:
            raise ValueError(f"group {group.name} claims {group.heads} head(s) a call, but {c} draws {cell.H}")
        if cell.kind == "qkv" and cell.dtype != "bf16":
            raise ValueError(f"group {group.name} claims bf16 attention, but {c} is {cell.dtype}")


def described() -> dict:
    """Every group, checked against the central registry, as plain data: what the sessions' macro reads (`GROUPS`, by
    name, each its fields; `GATE`)."""
    import dataclasses

    from fsla_devtools.cells import central

    for group in GROUPS.values():
        check(group, central())
    return {"GROUPS": {name: dataclasses.asdict(group) for name, group in GROUPS.items()}, "GATE": list(GATE)}
