"""THE ENGINE SEAM — the one import boundary between rola-bench and the RoLA engine package.

**Purpose.** Every rola-bench module reaches the engine (layers, ops, routing references, the HF
model registration) through THIS file and nowhere else. The engine package is about to be renamed
and re-homed twice over: the thunder branches (`rola-thunder-track{D,E,K,P}`) merge into mainline,
and mainline is then extracted as the standalone `rola` package (`fla_rola` -> `rola`). Before this
file existed that rename was ~60 edits across 18 files, each an independent chance to miss one and
discover it at runtime on a rented box. Now it is a package constant, here, once.

**TWO CONSTANTS, TWO ROLES (2026-08-17).** They are no longer two engines. `fla_rola` is the
FLA-INTEGRATION VEHICLE — the HF model (`models.rola` / `RoLAConfig` / the AutoClass registration),
the mixer class the LM tier constructs, and FLA's own baseline layers and ops — and its RoLA layer
is now a thin shim over the standalone. `rola` is THE ENGINE: the operator, the producer, the
routing vocabulary and the oracle. So `ENGINE_PACKAGE` names the vehicle (LM tier, perf tier, the
baseline ops) and `GRID_ENGINE_PACKAGE` names the engine (the grid, and now every routing/oracle
symbol any tier reads), with a symbol table each. A name lives in exactly one table.

It also gives the migration a single place to answer "which engine did this result come from?" and
a single place to shim a symbol that moves between engine submodules — instead of 18 files each
knowing that `LevelRoutingConfig` happens to live in `layers.rola` today.

**Zero behaviour change.** This module resolves nothing at import time. Symbols are looked up
lazily through PEP-562 `__getattr__`, so `from rola_bench.engine import LevelRoutingConfig` inside a
function body triggers exactly the same import, at exactly the same moment, as the
`from fla_rola.layers.rola import LevelRoutingConfig` it replaced. That laziness is load-bearing:
several call sites import inside functions specifically to keep Triton (and therefore a CUDA driver)
off the module-import path, which is what lets the config gates run CPU-only.

**A symbol that does not exist on the pinned engine raises at ACCESS time, as before.** The
V2-surface entries (`chunk_rola`, `_rola_chunk_core`, `rola_perstate_den_*`) are GONE from this
table with the engine that never had them: `fla_rola/ops/rola/` was deleted by the 2026-08-17
repoint, so they now name a package that does not exist rather than a symbol a package lacks.
Their callers (`mqar/verify.py`, `similarity/`, `perf/profile_graph.py`) are the same known
untouched V2 debt (rola-bench-audit §5.1) and fail as they already did — through the seam's own
"add it to a table" message instead of an ImportError.

**One deliberate exception.** `perf/engine_bench.py` still imports `fla_rola` directly. Its
uncommitted V2-era WIP is parked at
`rola-scratch/workflows/20260802-thunder/engine_bench_v2era.patch`, and that patch carries the
import line as diff CONTEXT — repointing it here would stop the patch applying. Repoint that file
when the patch lands or is dropped. It is explicitly not paper material and retires with the V2 line.

Usage:
    from rola_bench.engine import RoLA, LevelRoutingConfig      # lazy symbol access
    from rola_bench import engine; engine.register_hf_models()  # HF AutoClass registration
    engine.import_grid_engine("routing.reference")              # a whole engine submodule
"""
from __future__ import annotations

import importlib
from types import ModuleType

# ---------------------------------------------------------------------------------------------
# THE knobs — ONE PER TIER, because the repoint is not simultaneous.
#
# `ENGINE_PACKAGE` is the LEGACY engine: the LM tier's whole vehicle (`models.rola` / `RoLAConfig`
# / the AutoClass registration, which the standalone does not have), the perf tier and the V2-era
# gates. `GRID_ENGINE_PACKAGE` is the STANDALONE engine the MQAR/presence grid trains on
# (campaign-dispatch B3). They are two constants rather than one because a single knob cannot be
# flipped without breaking the tier that has no standalone surface yet — the seam's one-knob design
# assumed the migration would be atomic, and it is not.
# ---------------------------------------------------------------------------------------------
ENGINE_PACKAGE = "fla_rola"
GRID_ENGINE_PACKAGE = "rola"


def import_engine(suffix: str = "") -> ModuleType:
    """Import an engine submodule by its path RELATIVE to the engine package.

    `import_engine()` returns the package itself (the `import fla_rola  # load the kernel stack`
    side-effect form); `import_engine("ops.rola.routing.reference")` returns that submodule.
    """
    return importlib.import_module(f"{ENGINE_PACKAGE}.{suffix}" if suffix else ENGINE_PACKAGE)


def import_grid_engine(suffix: str = "") -> ModuleType:
    """Import a STANDALONE-engine submodule by its path relative to `GRID_ENGINE_PACKAGE`."""
    return importlib.import_module(
        f"{GRID_ENGINE_PACKAGE}.{suffix}" if suffix else GRID_ENGINE_PACKAGE)


def chunk_arm_for(D: int, B: int) -> tuple[int, int] | None:
    """The `(C, BC)` the operator would run a uniform `(D, B)` topology on, or None if unbuilt.

    Asked of the ENGINE's own selector rather than reimplemented, so the grid's refusal and the
    launch cannot disagree about which arm a rung takes ([[matching-the-naive-antipattern]]).
    """
    eng_rules = import_grid_engine("engine.rules")
    eng_manifest = import_grid_engine("engine.facts.manifest")
    return eng_rules.arm((int(B),) * int(D), eng_manifest.chunk_arms())


def chunk_scan_arms() -> set[tuple[int, int, int, int]]:
    """The `(C, BC, D, B)` arms the two backward SCANS are built for — the TRAINABLE set, read off
    the compiled binary's census."""
    return import_grid_engine("engine.facts.manifest").scan_arms()


def register_hf_models() -> ModuleType:
    """Import the engine's HF model package, registering RoLA with the transformers AutoClasses.

    This is the `import fla_rola.models.rola  # noqa: F401` side effect that `arm_config`,
    `train_lm`, `eval_lm`, `ar_ppl` and `niah_eval` all depend on: without it
    `AutoModelForCausalLM.from_config(RoLAConfig(...))` does not know the architecture.

    The HF model is the VEHICLE's, and stays there: the standalone engine has no `models/rola/`
    and wants none. Since the 2026-08-17 repoint the mixer that model builds is a shim over the
    engine, so this call registers an architecture whose arithmetic is the standalone's.
    """
    return import_engine("models.rola")


def engine_revision() -> str | None:
    """Best-effort provenance: the engine package's `__file__` root, for result-row stamping."""
    try:
        return getattr(import_engine(), "__file__", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# Symbol table: exported name -> engine submodule that owns it. Lazily resolved by __getattr__.
# Grouped by engine area; a symbol moving between submodules is a one-line edit HERE.
# ---------------------------------------------------------------------------------------------
_SYMBOLS: dict[str, str] = {
    # --- layers (the construction surface rola-bench actually builds) ---
    "RoLA": "layers",
    "LinearAttention": "layers",
    "GatedDeltaNet": "layers",
    "GatedLinearAttention": "layers.gla",
    "DeltaNet": "layers.delta_net",
    "GatedSlotAttention": "layers.gsa",
    "LevelRoutingConfig": "layers.rola",
    "RoLADecodeGraph": "layers.rola",
    # `_resolve_level_routing` is private to the vehicle but is the ONLY faithful public-config ->
    # engine-level mapping; the gates use it rather than reimplementing it
    # ([[matching-the-naive-antipattern]]). Since the 2026-08-17 repoint it takes `(config, width)`
    # and returns a STANDALONE level object, because that is what the engine's level carries.
    "_resolve_level_routing": "layers.rola",

    # --- HF model surface (the LM tier's entire vehicle) ---
    "RoLAConfig": "models.rola",
    "Cache": "models.utils",

    # --- baseline ops used by the similarity tier ---
    "chunk_linear_attn": "ops.linear_attn",
    "chunk_gated_delta_rule": "ops.gated_delta_rule",
}

# ---------------------------------------------------------------------------------------------
# STANDALONE-engine symbols (`GRID_ENGINE_PACKAGE`), same lazy contract. The MQAR/presence grid's
# tier reads its structural facts here; a name may appear in ONE table only, and the two engines'
# `MAX_ROUTING_DEPTH` / `MAX_BRANCH_WIDTH` agree (4 / 256), so moving them is a repoint and not a
# change of the numbers any caller sees.
# ---------------------------------------------------------------------------------------------
_GRID_SYMBOLS: dict[str, str] = {
    "MAX_ROUTING_DEPTH": "routing.topology",
    "MAX_BRANCH_WIDTH": "routing.topology",
    #: The operator's token quantum: `T` must be a multiple of it, which is a constraint on the
    #: grid's sequence lengths and therefore read rather than restated.
    "CHUNK_TOKENS": "engine.rules",

    # --- routing vocabulary + the canonical fp64 oracle the gates check against. These moved
    # here from the legacy table on 2026-08-17, when `fla_rola/ops/rola/` was deleted and the
    # vehicle's layer became a shim: there is now ONE routing vocabulary and ONE oracle, and they
    # are the engine's.
    "resolve_b_spec": "routing.topology",
    "ResolvedRouting": "routing.types",
    "IndependentRouting": "routing.types",
    "UnionRouting": "routing.types",
    "SoftmaxActivation": "routing.types",
    "EntmaxActivation": "routing.types",
    "Topology": "routing.types",
    "LeafMassDecay": "routing.types",
    "routing_factor_levels": "routing.reference",
    #: THE ORACLE, under the engine's own name. The legacy `naive_rola_v3` spelling retired with
    #: the engine that had two generations to tell apart; the contract is unchanged (same
    #: positional arguments, same `eps=1e-5`, same `(y, final_state)`), except that this one
    #: returns fp64 unconditionally instead of casting back to the caller's dtype.
    "naive_rola": "ops.naive",
}
#: DELETED FROM THE TABLES rather than repointed, each with its reason:
#:   `LevelConfig`      — the engine's level object carries its own width, so a separate
#:                        `LevelConfig(width, routing)` wrapper has no counterpart to name.
#:   `_SUPPORTED_BC` / `_SUPPORTED_DV`
#:                      — buildability is answered by `chunk_arm_for` / `chunk_scan_arms` off the
#:                        compiled binary's census, which is a fact rather than a pair of lists.
#:   `chunk_rola`, `chunk_rola_routed`, `rola_perstate_den_*`, `_rola_chunk_core`
#:                      — V2 surface of a package that no longer exists (see the module docstring).
#: The engine's CLASSES are in the table now: the shadowing hazard that kept them out was two
#: engines exporting one set of names, and there is one engine.


def __getattr__(name: str):
    """PEP 562 lazy attribute access — the import happens here, at first use, never at seam import."""
    if name in _GRID_SYMBOLS:
        return getattr(import_grid_engine(_GRID_SYMBOLS[name]), name)
    try:
        module = _SYMBOLS[name]
    except KeyError:
        raise AttributeError(
            f"{__name__!r} exposes no engine symbol {name!r}. Add it to `_SYMBOLS` (name -> the "
            f"engine submodule that owns it) rather than importing {ENGINE_PACKAGE} directly — "
            "this module is the single seam the coming package rename edits.") from None
    return getattr(import_engine(module), name)


def __dir__():
    return sorted([*globals(), *_SYMBOLS, *_GRID_SYMBOLS])
