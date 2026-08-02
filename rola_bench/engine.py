"""THE ENGINE SEAM — the one import boundary between rola-bench and the RoLA engine package.

**Purpose.** Every rola-bench module reaches the engine (layers, ops, routing references, the HF
model registration) through THIS file and nowhere else. The engine package is about to be renamed
and re-homed twice over: the thunder branches (`rola-thunder-track{D,E,K,P}`) merge into mainline,
and mainline is then extracted as the standalone `rola` package (`fla_rola` -> `rola`). Before this
file existed that rename was ~60 edits across 18 files, each an independent chance to miss one and
discover it at runtime on a rented box. Now it is `ENGINE_PACKAGE = "rola"`, here, once.

It also gives the migration a single place to answer "which engine did this result come from?" and
a single place to shim a symbol that moves between engine submodules — instead of 18 files each
knowing that `LevelRoutingConfig` happens to live in `layers.rola` today.

**Zero behaviour change.** This module resolves nothing at import time. Symbols are looked up
lazily through PEP-562 `__getattr__`, so `from rola_bench.engine import LevelRoutingConfig` inside a
function body triggers exactly the same import, at exactly the same moment, as the
`from fla_rola.layers.rola import LevelRoutingConfig` it replaced. That laziness is load-bearing:
several call sites import inside functions specifically to keep Triton (and therefore a CUDA driver)
off the module-import path, which is what lets the config gates run CPU-only.

**A symbol that does not exist on the pinned engine raises at ACCESS time, as before.** Some entries
below are V2-surface (`chunk_rola`, `_rola_chunk_core`, `rola_perstate_den_*`) and are absent from
the V3 mainline engine; they are listed so the seam is complete over the tree as it stands, not
because they resolve. Their callers (`mqar/verify.py`, `similarity/`, `perf/profile_graph.py`) are
known untouched V2 debt (rola-bench-audit §5.1) and will fail the same way through the seam as
around it.

**One deliberate exception.** `perf/engine_bench.py` still imports `fla_rola` directly. Its
uncommitted V2-era WIP is parked at
`rola-scratch/workflows/20260802-thunder/engine_bench_v2era.patch`, and that patch carries the
import line as diff CONTEXT — repointing it here would stop the patch applying. Repoint that file
when the patch lands or is dropped. It is explicitly not paper material and retires with the V2 line.

Usage:
    from rola_bench.engine import RoLA, LevelRoutingConfig      # lazy symbol access
    from rola_bench import engine; engine.register_hf_models()  # HF AutoClass registration
    engine.import_engine("ops.rola.routing.reference")          # a whole submodule
"""
from __future__ import annotations

import importlib
from types import ModuleType

# ---------------------------------------------------------------------------------------------
# THE knob. When the engine package is renamed (fla_rola -> rola, on extraction to the standalone
# repo), this string is the entire change on the rola-bench side.
# ---------------------------------------------------------------------------------------------
ENGINE_PACKAGE = "fla_rola"


def import_engine(suffix: str = "") -> ModuleType:
    """Import an engine submodule by its path RELATIVE to the engine package.

    `import_engine()` returns the package itself (the `import fla_rola  # load the kernel stack`
    side-effect form); `import_engine("ops.rola.routing.reference")` returns that submodule.
    """
    return importlib.import_module(f"{ENGINE_PACKAGE}.{suffix}" if suffix else ENGINE_PACKAGE)


def register_hf_models() -> ModuleType:
    """Import the engine's HF model package, registering RoLA with the transformers AutoClasses.

    This is the `import fla_rola.models.rola  # noqa: F401` side effect that `arm_config`,
    `train_lm`, `eval_lm`, `ar_ppl` and `niah_eval` all depend on: without it
    `AutoModelForCausalLM.from_config(RoLAConfig(...))` does not know the architecture.

    NOTE (staleness, rola-bench-audit §5.2): the standalone `rola` package has `layers/`, `ops/`,
    `planner/`, `routing/` and `integrations/hf.py` but NO `models/rola/` — no `RoLAConfig`, no
    AutoModel registration. The whole LM tier rests on this call, so when `ENGINE_PACKAGE` flips,
    this function is the one that must be answered for first.
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
    # `_resolve_level_routing` is private to the engine but is the ONLY faithful config ->
    # ResolvedRouting mapping; the gates use it rather than reimplementing the mapping
    # ([[matching-the-naive-antipattern]]).
    "_resolve_level_routing": "layers.rola",

    # --- HF model surface (the LM tier's entire vehicle) ---
    "RoLAConfig": "models.rola",
    "Cache": "models.utils",

    # --- routing vocabulary + the canonical CPU/fp64 references the gates check against ---
    "resolve_b_spec": "ops.rola.routing.topology",
    # The engine's own structural CAPS on the routing tree. rola-bench reads them rather than
    # re-declaring them, so a preset that cannot be built at a rung says so with the engine's number
    # (mqar.instances.PresetRungUnavailable) instead of failing deep inside `resolve_b_spec`.
    "MAX_ROUTING_DEPTH": "ops.rola.routing.topology",
    "MAX_BRANCH_WIDTH": "ops.rola.routing.topology",
    # The (BC, d_v) the CUDA consumer INSTANTIATES. Private to the engine by name, but it is a
    # hard buildability constraint on any EVAL-mode cell, so the benches read it rather than
    # rediscovering it as a NotImplementedError on a rented box.
    "_SUPPORTED_BC": "ops.rola.routing.schedule_v3",
    "_SUPPORTED_DV": "ops.rola.routing.schedule_v3",
    "ResolvedRouting": "ops.rola.routing.types",
    "IndependentRouting": "ops.rola.routing.types",
    "UnionRouting": "ops.rola.routing.types",
    "SoftmaxActivation": "ops.rola.routing.types",
    "EntmaxActivation": "ops.rola.routing.types",
    "LevelConfig": "ops.rola.routing.types",
    "Topology": "ops.rola.routing.types",
    "LeafMassDecay": "ops.rola.routing.types",
    "routing_factor_levels": "ops.rola.routing.reference",
    "naive_rola_v3": "ops.rola.naive",

    # --- baseline ops used by the similarity tier ---
    "chunk_linear_attn": "ops.linear_attn",
    "chunk_gated_delta_rule": "ops.gated_delta_rule",

    # --- V2-SURFACE, absent from the V3 mainline engine. Listed so the seam is complete over the
    # tree as it stands; they raise at access exactly as their direct imports did.
    "chunk_rola": "ops.rola",
    "chunk_rola_routed": "ops.rola",
    "rola_perstate_den_triton": "ops.rola",
    "rola_perstate_den_gla_triton": "ops.rola",
    "_rola_chunk_core": "ops.rola.interface",
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access — the import happens here, at first use, never at seam import."""
    try:
        module = _SYMBOLS[name]
    except KeyError:
        raise AttributeError(
            f"{__name__!r} exposes no engine symbol {name!r}. Add it to `_SYMBOLS` (name -> the "
            f"engine submodule that owns it) rather than importing {ENGINE_PACKAGE} directly — "
            "this module is the single seam the coming package rename edits.") from None
    return getattr(import_engine(module), name)


def __dir__():
    return sorted([*globals(), *_SYMBOLS])
