"""RoLA SCALING benchmark — per-architecture characterization, NOT cross-architecture comparison.

Arms: **RoLA** (rla + gla inner kernels, swept over routing structure {flat,square,tree}), its
matched-state **wide-monolith** equivalent (nc=1, all the state in one wide head — the "what if you
didn't route" ablation, RoLA's own control, NOT a zoo baseline), and **softmax attention**. Rationale
for excluding the rest: GDN / GLA / Based / ... are the SAME complexity class as RoLA (O(L) compute,
bounded state), so on an L-axis they are parallel lines with different constants — "the constants" is
a matched-state COMPARISON question (see perf/comparison), not a scaling-shape one. Attention is the
ONLY baseline that scales DIFFERENTLY in L (O(L^2) compute, O(L) growing KV cache), the entire reason
RoLA exists. The wide monolith earns its place as RoLA's structural ablation (matched state,
routed-vs-not), not as a competitor.

IMPORTANT: the numbers here are NOT a fairness verdict. RoLA sitting at X<100% of attention's KV
cache is the bounded-state VALUE PROPOSITION (the whole point), not an unfair tilt — bounded-vs-
unbounded is the legitimate contrast. The output is the SHAPE of each curve and where RoLA crosses
attention; head-to-head verdicts at matched state live in the (separate) comparison benchmark.

EVERYTHING IS LAYER-vs-LAYER: every arm instantiates its CANONICAL nn.Module at d_model=DM and times
`module(x)` — no hand-assembled op-on-synthetic-tensor scaffold (the user was burnt by op-level
benches). The decode arm too: RoLA's layer `forward(use_cache=True)` recurrent path vs the attention
KV-cache decode, both timed as one fused layer region.

Axes:
  * L        sequence length (the primary scaling axis)
  * nc       RoLA states per head; X% is recorded as the realized fraction of attention's KV cache
  * d_qk     feature dim {16,64,128} -- the occupancy / operating-point knob (16 = the starved corner)
  * routing  the per-head factor topology {flat,square,tree} the RoLA layer routes through in-kernel
             (flat = dense router; square/tree = factored). RoLA-only axis (monolith/attn don't route).
  * bench    training (chunked fwd+bwd) | prefill (chunked fwd-only) | decode (recurrent, per-token)

attention depends only on L (no X / d_qk / routing axis) -> timed once per (bench, L) and reused as the
reference curve. Each cell is wrapped so an OOM/error writes a sentinel and the sweep continues;
the CSV is written in a `finally`. Use small PERF_B (e.g. 2) so long L fits.

This module STANDS ALONE: it owns its copies of the shared helpers (_CausalMHA / _instantiate /
_e2e_rep) — no import from the archived perf/scratch/{chunked,recurrent}.py.

Run:  python -m rola_bench.perf.scaling [--benches train,prefill,decode]
            [--lens 1024,4096,16384,65536] [--ncs 16,64,256,1024] [--dqks 16,64,128]
            [--routings flat,square,tree] [--reps N] [--out DIR]

WARMUP reps absorb each cell's first-touch Triton cold-compile, so cells time warm — no separate
pre-sweep cache-warm pass (the old warm_gate was net-negative: its capture phase serialized the
autotune in-process so the parallel phase only re-hit cache, and it warmed the raw-forward kernels
while this bench fires the kappa-norm kernels — 0/4 overlap).

**STATUS 2026-08-02 — the RoLA arms are BLOCKED on the zoology mixer migration.** The V3 mainline
engine owns no Q/K surface, and `zoology.mixers.rola.RoLAMixer` still forwards `head_k_dim=d_qk`,
`kernel=`, `phi=`, `qk_norm=` into `fla_rola.layers.RoLA` — every one of those kwargs was deleted by
V3. So `build_rola` / `build_rola_named` cannot construct a RoLA layer against the campaign engine,
whatever this file writes; they raise `_zoology_blocker()` by name instead of TypeError-ing deep in
the layer, exactly as `rola_bench.suites.mqar._construct_named` does. No V2 back-translation is
written here on purpose: the V2 spellings (`kernel`, `kernel_kwargs={'state_norm': 'kappa'}`,
`d_qk`) are retired vocabulary and would be dead code the moment the mixer is migrated
(rola-bench-audit §3, Phase 1.2/1.4). The `attn`, `monolith` and `gla/gdn/deltanet/gsa` arms are
unaffected and still run.

The NAMED-CONFIG arms are derived from `rola_bench.named_configs.PERF_SUITE_CONFIGS`, not hardcoded.
Hardcoding is precisely how this file silently rotted: it pinned `'RoLA-base'`, which the E1-E7
migration retired, so `build_rola_named` KeyError'd and no test noticed.
`test_perf_scaling_arms.py` is that missing gate.
"""
import argparse
import contextlib
import gc
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/rola-bench-hf")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])

import torch

# NOTE: `zoology.mixers.rola.RoLAMixer` and `rola_bench.mqar.instances.rola_instance` are no longer
# imported here — the RoLA arms are blocked on the mixer migration (module docstring) and an unused
# import of a V2-surface module would keep this file coupled to it for nothing. Restore both when
# Phase 1.2 lands.

from . import _common as C
import rola_bench.mqar.experiments.canonical_baselines as cb

DM = 512  # d_model backbone for the scaling layers
ROUTINGS = ("flat", "square", "tree")  # the per-head factor topologies the RoLA layer supports
DECODE_STEPS = 100
DECODE_WARMUP = 20
ROLA_STATE_DTYPE = torch.float32  # recurrent/final-state cache is physically fp32 today


def _nearest_even_pow2(n):
    """Nearest nc that is BOTH a perfect square AND a power of two — i.e. an EVEN power of two
    (1,4,16,64,256,1024,...) = 4^m. Such an nc is the one value valid for every routing topology at
    once: flat (any), square (sqrt = 2^m is an integer), tree (2^(2m) is a power of two). So all three
    routings in a cell run at the SAME matched state. Rounded in log space (geometric nearest)."""
    if n <= 1:
        return 1
    m = round(math.log2(n) / 2.0)        # 4^m == 2^(2m)
    return 1 << (2 * m)


def nc_for(L, x_pct, d):
    """RoLA states to hit ~X% of attention's KV cache at d_qk=d_v=d (X = nc*d/(2L)), SNAPPED to the
    nearest even-power-of-two so the single nc is valid for flat|square|tree alike (perfect-square AND
    power-of-two — see _nearest_even_pow2). The snap rule is consistent across all cells; realized X =
    nc*d/(2L) follows from the snapped nc (recorded via the row's nc field — no silent corruption)."""
    raw = max(1, round(2 * L * (x_pct / 100.0) / d))
    return _nearest_even_pow2(raw)


def _infeasible_excs():
    """(CUDA OOM, Triton OutOfResources) — the "this config can't run on THIS GPU" signals: out of
    memory, or a kernel that won't fit the GPU's shared-mem/registers at this nc. A cell that raises
    one is an infeasible-frontier SKIP (sentinel row), not a bug — matching the same rule warm uses.
    Any OTHER exception is a real bug (still recorded per-cell, but as ERR, and printed loudly)."""
    oor = ()
    for m in ("triton.runtime.errors", "triton.compiler.errors"):
        try:
            oor = (__import__(m, fromlist=["OutOfResources"]).OutOfResources,)
            break
        except Exception:  # noqa: BLE001
            pass
    return (torch.cuda.OutOfMemoryError,) + oor


_INFEASIBLE = _infeasible_excs()


def _dtype_bytes(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _mib(*dims, dtype=C.DT):
    n = 1
    for x in dims:
        n *= x
    return round(n * _dtype_bytes(dtype) / 2 ** 20, 1)


def _mib_elems(n, dtype=C.DT):
    return round(n * _dtype_bytes(dtype) / 2 ** 20, 1)


def _attn_cache_mib(L):
    return _mib(C.B, C.H, L, DM // C.H, 2, dtype=C.DT)


def _rola_state_elems_per_batch(nc, d):
    # scaling.py builds kappa-normalized RoLA, whose recurrent state carries value + denominator columns.
    return C.H * nc * d * (d + 1)


def _rola_state_mib(nc, d):
    return _mib_elems(C.B * _rola_state_elems_per_batch(nc, d), dtype=ROLA_STATE_DTYPE)


def _module_state_mib(module):
    return _mib_elems(C.B * module.get_stats()["state_floats"], dtype=ROLA_STATE_DTYPE)


def _state_x_pct(nc, d, L):
    rola_bytes = _rola_state_elems_per_batch(nc, d) * _dtype_bytes(ROLA_STATE_DTYPE)
    attn_bytes = (2 * L * DM) * _dtype_bytes(C.DT)
    return round(100.0 * rola_bytes / attn_bytes, 1)


def _round_ms(x, digits=3):
    return round(float(x), digits)


def _sdpa_flags():
    """Record the SDPA policy knobs PyTorch will use for the attention baseline.

    PyTorch does not expose a stable public "which backend was selected for this exact call" return
    value, so reportable runs should either stamp these enabled flags or force one backend with
    SCALE_ATTN_BACKEND={flash,efficient,math,cudnn}.
    """
    return {
        "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math": bool(torch.backends.cuda.math_sdp_enabled()),
        "cudnn": bool(getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)()),
    }


def _sdpa_backend_label():
    forced = os.environ.get("SCALE_ATTN_BACKEND", "").strip().lower()
    if forced:
        return f"forced:{forced}"
    flags = _sdpa_flags()
    enabled = [name for name, ok in flags.items() if ok]
    return "auto:" + "+".join(enabled)


def _sdpa_context():
    forced = os.environ.get("SCALE_ATTN_BACKEND", "").strip().lower()
    if not forced:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    aliases = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    if forced not in aliases:
        raise ValueError(f"unknown SCALE_ATTN_BACKEND={forced!r}; expected one of {sorted(aliases)}")
    return sdpa_kernel(aliases[forced])


# ------------------------------------------------------------------ shared helpers (own copies)
def _instantiate(kernel_cfg, DM):
    """Build a canonical baseline module from a baseline_cell kernel dict. `name` is a dotted import
    path; kwargs already carry hidden_size for the fla classes, so d_model is injected only for the
    zoology mixers (which take d_model= and have neither key in their kwargs). Relocated here so the
    scaling path stands alone (was imported from the now-archived chunked.py)."""
    mod_path, cls_name = kernel_cfg["name"].rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_path), cls_name)
    kwargs = dict(kernel_cfg["kwargs"])
    if "hidden_size" not in kwargs and "d_model" not in kwargs:
        kwargs["d_model"] = DM
    return cls(**kwargs).to(C.DEV, C.DT)


class _CausalMHA(torch.nn.Module):
    """Canonical causal multi-head attention via SDPA (flash backend on A100/bf16) — the O(L^2)
    reference for the scaling sweep (RoLA's O(L) crossover + the within-window ceiling). Relocated
    here so scaling.py stands alone. Supports a forward `kv` cache for the layer-level decode arm:
    pass `cache=(k_cache,v_cache)` and `use_cache=True` to append the new token and attend to the
    full grown KV (the O(L)-growing-cache decode RoLA's bounded state is measured against)."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.nh, self.hd = n_heads, d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, self.nh, self.hd).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            kc, vc = cache
            k = torch.cat([kc, k], dim=2)
            v = torch.cat([vc, v], dim=2)
        # prefill (L>1) is causal; single-token decode attends to the whole cached past (no mask).
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(L > 1))
        o = self.proj(o.transpose(1, 2).reshape(B, L, D))
        if use_cache:
            return o, (k, v)
        return o


def _e2e_rep(forward, leaves, module):
    """One fused fwd+bwd rep -> (fwd_ms, fwd_bwd_ms, peak_mib, act_mib). act_mib = bytes RETAINED for
    backward (saved tensors, deduped by storage) — the memory that accumulates across depth; peak_mib
    is the transient max. Clears x AND param grads per rep. Relocated here from chunked.py."""
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    probe = C.ActivationProbe()
    with probe:
        e0.record(); o = forward(leaves); e1.record()
    torch.cuda.synchronize()
    act = probe.mib                                       # retained-for-backward (across-depth) memory
    o.float().sum().backward(); e2.record(); torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    leaves["x"].grad = None
    module.zero_grad(set_to_none=True)
    return e0.elapsed_time(e1), e0.elapsed_time(e2), peak, act


# ------------------------------------------------------------------ builders
_ZOOLOGY_BLOCKER = (
    "the perf scaling harness's RoLA arms are blocked on the zoology mixer migration: "
    "zoology.mixers.rola.RoLAMixer forwards head_k_dim=d_qk, kernel=, phi= and qk_norm= into "
    "fla_rola.layers.RoLA, and the V3 mainline engine deleted all four (it owns no Q/K surface). "
    "d_qk has no V3 meaning to translate to, so there is no construction that preserves the cell. "
    "Migrate zoology/mixers/rola.py onto the V3 layer surface first (rola-bench-audit §3, "
    "Phase 1.2/1.4); reuse named_configs._decay_from_kernel for the kernel->decay mapping. "
    "The attn / monolith / gla / gdn / deltanet / gsa arms are unaffected.")


def _zoology_blocker(detail: str) -> NotImplementedError:
    return NotImplementedError(f"{detail}: {_ZOOLOGY_BLOCKER}")


def build_rola(gla, nc, d, routing="flat"):
    """Routed RoLA LAYER at d_v=d, nc states, `routing` factor topology, perf backbone.

    BLOCKED — see the module docstring. The presets this used to name
    (`rola-rla-kappa-asym` / `rola-gla-kappa-asym`) are doubly dead: `state_norm='kappa'` is not a
    normalization in any engine checkout (campaign-prereqs E2) and the whole `gla_*` family is
    `kernel='gla_scalar'` (E3). Re-pinning them to live presets would still not construct, because
    the mixer's own signature is the blocker."""
    raise _zoology_blocker(
        f"build_rola(gla={gla}, nc={nc}, d={d}, routing={routing!r})")


# --- REAL external linear-attention baselines (paper_v1: RoLA vs the FLA library's own canonical
# layers, layer-vs-layer at matched d_model=DM, NOT matched-state -- these are the SAME complexity
# class as RoLA per this module's own docstring rationale, included here specifically for paper_v1's
# "final RoLA vs real implementations" ask; the scaling-SHAPE argument for excluding them from the
# original RoLA-vs-monolith-vs-attn sweep still holds for that sweep's own purpose). Canonical geometry
# per arm mirrors rola_bench.lm.arms's GDN_GEOM/GLA_GEOM (the paper's LM-suite defaults), NOT the (d,nc)
# sweep axis -- these layers don't have a states_per_head knob to sweep. use_short_conv=False for all
# (arms.py's uniform "isolate the mixer" policy). Layers come from `fla_rola.layers` (the engine's own
# fork of the canonical fla layers) rather than the separately pip-installed `fla` package, so they
# track whatever fla commit this engine worktree forked from.
_FLA_BASELINE_GEOM = {
    "gla": dict(num_heads=4, expand_k=2.0, expand_v=8.0, gate_logit_normalizer=16),
    "gdn": dict(num_heads=8, head_dim=256, expand_v=2.0),
    "deltanet": dict(num_heads=8, expand_k=1.0, expand_v=1.0),
    "gsa": dict(num_heads=4, expand_k=1.0, expand_v=1.0, num_slots=64),
}


def _build_fla_baseline(kind):
    """One canonical `fla_rola.layers` baseline layer at hidden_size=DM. `kind` in `_FLA_BASELINE_GEOM`."""
    if kind == "gla":
        from fla_rola.layers.gla import GatedLinearAttention as Cls
    elif kind == "gdn":
        from fla_rola.layers.gated_deltanet import GatedDeltaNet as Cls
    elif kind == "deltanet":
        from fla_rola.layers.delta_net import DeltaNet as Cls
    elif kind == "gsa":
        from fla_rola.layers.gsa import GatedSlotAttention as Cls
    else:
        raise ValueError(f"unknown fla baseline kind {kind!r}")
    return Cls(hidden_size=DM, use_short_conv=False, layer_idx=0,
              **_FLA_BASELINE_GEOM[kind]).to(C.DEV, C.DT)


def _named_arm_kind(name: str) -> str:
    """The CLI/`SCALE_ARMS` spelling of a named config (it is also the `arm` column in the CSV)."""
    return name.lower()


def named_rola_arm_names() -> tuple[str, ...]:
    """The named configs this harness exposes as scaling arms — DERIVED from the registry's own
    perf slate (`named_configs.PERF_SUITE_CONFIGS`), never hardcoded here.

    This is the anti-rot property `test_perf_scaling_arms.py` gates: the previous hardcoded pair
    `("rola-base", "rola-base-rla")` outlived `RoLA-base`'s retirement by the E1-E7 migration and
    turned into a silent `KeyError` in the paper_v1 kernel harness, because nothing tested it.
    Deriving means a retirement can only ever remove an arm, never leave a dangling name."""
    from rola_bench.named_configs import PERF_SUITE_CONFIGS
    return tuple(PERF_SUITE_CONFIGS)


def named_rola_arm_kinds() -> tuple[str, ...]:
    return tuple(_named_arm_kind(n) for n in named_rola_arm_names())


_NAMED_ROLA_ARM_KINDS = named_rola_arm_kinds()   # paper_v1's flagship arms


def resolve_named_arm(kind, nc=256):
    """Resolve a named-config scaling arm to `(name, layer_kwargs)` — PURE REGISTRY WORK, no engine
    import, no GPU, so it can be gated on CPU.

    `layer_kwargs` are the V3 `fla_rola.layers.RoLA` kwargs the registry emits
    (`levels_per_head` / `states_per_level` / `level_routing_configs` / `decay` / `state_norm` /
    `router_bias` / `use_short_conv`). Note `decay`, not `kernel`: the E1-E7 migration moved the
    builders onto the V3 layer's own vocabulary, and reading the removed `kernel` key is the second
    of the two bugs this file carried."""
    from rola_bench.named_configs import NAMED_CONFIGS, RETIRED_CONFIGS
    by_kind = {_named_arm_kind(n): n for n in named_rola_arm_names()}
    if kind not in by_kind:
        raise ValueError(
            f"unknown named RoLA arm {kind!r}; expected one of {sorted(by_kind)} "
            "(derived from named_configs.PERF_SUITE_CONFIGS)")
    name = by_kind[kind]
    if name in RETIRED_CONFIGS:
        raise ValueError(
            f"named config {name!r} is on the perf slate but RETIRED: {RETIRED_CONFIGS[name]}")
    if name not in NAMED_CONFIGS:
        raise KeyError(
            f"named config {name!r} is on the perf slate but absent from NAMED_CONFIGS")
    kwargs = dict(NAMED_CONFIGS[name].build_kwargs(nc, True))   # conv=True (paper/LM geometry)
    return name, kwargs


def build_rola_named(kind, nc=256):
    """Build a flagship named-config RoLA layer (`rola_bench.named_configs`) as a scaling arm, at ITS
    OWN fixed geometry (H=8, d_v=64, square topology) rather than the (d, nc) sweep axis -- same
    rationale as `_build_fla_baseline`: the flagship isn't parameterized by that axis.

    The registry resolution runs FIRST, so a dangling/retired arm name fails with its own message;
    the construction itself is BLOCKED on the zoology mixer (see the module docstring)."""
    name, _kwargs = resolve_named_arm(kind, nc)
    raise _zoology_blocker(f"build_rola_named({kind!r}, nc={nc}) -> {name!r}")


def build_monolith(gla, nc, d):
    """The matched-state WIDE monolith (nc=1, one big state of size H·nc·d·d, d_k widened to nc·d) —
    RoLA's routed-vs-not ablation. Reuses canonical_baselines.baseline_cell at the perf shape so the
    state exactly matches build_rola(gla,nc,d). Returns the module, or None if infeasible."""
    cb.DMODEL = DM; cb.BASE_H = C.H; cb.DV = d
    cb.REF = lambda _nc: C.H * _nc * d * d           # match RoLA's content state at this (nc,d) cell
    cell = cb.baseline_cell("gla" if gla else "rla", "wide", nc)
    if cell is None:
        return None
    kernel_cfg, _state = cell
    return _instantiate(kernel_cfg, DM)


def _fwd(module, x):
    o = module(x)
    return o[0] if isinstance(o, tuple) else o


# ------------------------------------------------------------------ timing reps
def _train_cell(module, L):
    """fwd+bwd median over reps -> (fwd_ms, fb_ms, peak_mib, act_mib)."""
    x = torch.randn(C.B, L, DM, device=C.DEV, dtype=C.DT)
    leaves = {"x": x.detach().requires_grad_()}
    fwd = lambda lv: _fwd(module, lv["x"])
    recs = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*format mismatch.*")
        for r in range(C.WARMUP + C.REPS):
            rec = _e2e_rep(fwd, leaves, module)
            if r >= C.WARMUP:
                recs.append(rec)
    f, fb, pk, ac = zip(*recs)
    return (_round_ms(C.median(f)), _round_ms(C.iqr(f)),
            _round_ms(C.median(fb)), _round_ms(C.iqr(fb)),
            round(max(pk)), round(max(ac)))


def _prefill_cell(module, L):
    """forward-only (inference_mode, no backward) median -> (fwd_ms, peak_mib)."""
    x = torch.randn(C.B, L, DM, device=C.DEV, dtype=C.DT)
    times, peak = [], 0
    with torch.inference_mode():
        for _ in range(C.WARMUP):
            _fwd(module, x)
        torch.cuda.synchronize()
        for _ in range(C.REPS):
            t, p = _prefill_rep(module, x)
            times.append(t)
            peak = max(peak, p)
    return _round_ms(C.median(times)), _round_ms(C.iqr(times)), round(peak)


def _prefill_rep(module, x):
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); _fwd(module, x); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1), (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def _new_cache():
    """A fresh fla_rola Cache for the layer-level decode arm (carries the per-layer recurrent state)."""
    from fla_rola.models.utils import Cache
    return Cache()


_SEED_PREFILL = 256   # tokens used to SEED the recurrent state (fixed-size; seeding doesn't need full L)


def _decode_layer_sample(module, L):
    """One isolated LAYER-LEVEL decode sample: seed recurrent state, then time one single-token step.

    Samples are intentionally one-step so decode arms can be scheduled round-robin across comparable
    RoLA/monolith implementations while preserving the train/prefill memory semantics: each sample
    builds, seeds, times, and frees only one arm.
    """
    if getattr(module, "layer_idx", None) is None:
        module.layer_idx = 0
    seed_len = min(max(1, L), _SEED_PREFILL)
    x_pre = torch.randn(C.B, seed_len, DM, device=C.DEV, dtype=C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated()   # pre-cache baseline (params only)
    with torch.inference_mode():
        cache = _new_cache()
        module(x_pre, past_key_values=cache, use_cache=True)        # prefill -> seeds recurrent_state
        if len(cache) == 0:
            raise RuntimeError(
                "decode arm: cache NOT seeded after prefill (len==0) — the layer dropped the "
                "past_key_values/use_cache kwargs, so this would measure a COLD un-seeded step. "
                "Check the module's forward threads the cache (zoology RoLAMixer wrapper).")
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()  # forget the chunked-seed spike
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); module(x1, past_key_values=cache, use_cache=True); t1.record(); torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        peak = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    return ms, peak


def _decode_layer_step(module, L):
    """LAYER-LEVEL decode: SEED the layer's recurrent state from a prefill, then time the single-token
    decode step `module.forward(x1, past_key_values=cache, use_cache=True)` — the shipped recurrent
    decode path (RoLA auto-selects fused_recurrent for L<=64). The recurrent state is FIXED-SIZE (it
    does NOT grow with steps nor with the prefill length — that's the whole point vs attention's
    growing KV), so seeding from a bounded prefill is faithful and each timed step runs into the SAME
    persisted cache: a drift-free steady-state seeded decode (no per-step clone, no op-level call).

    ASSERTS the cache is actually seeded after the prefill (len>0) — a fail-loud guard against a
    silently-dropped cache kwarg (the zoology wrapper used to swallow it, which secretly measured a
    COLD un-seeded step; #40 gate-B). Returns (ms/token, peak_mib) — peak_mib is the MEASURED peak
    allocated above the pre-cache baseline during the decode steps (resident state + per-step
    transients); the chunked seed's spike is reset out so it reflects decode, not the prefill. Carried
    alongside the analytic state size as an overhead/sanity cross-check (literature reports ms/token)."""
    times, peak = [], 0
    for r in range(DECODE_WARMUP + DECODE_STEPS):
        ms, pk = _decode_layer_sample(module, L)
        peak = max(peak, pk)
        if r >= DECODE_WARMUP:
            times.append(ms)
    return _round_ms(C.median(times), 4), _round_ms(C.iqr(times), 4), peak


def _decode_layer_sample_graphed(module, L):
    """SAME seeding contract as `_decode_layer_sample` (fixed-size recurrent state, seeded from a
    bounded prefill, `len(cache)==0` fail-loud guard), but the timed single-token step is a CUDA-graph
    REPLAY rather than an eager call — the C9 pattern
    (`rola-scratch/workflows/dag/dag_tasks/C9-decode-swap-graph.md`): prefer the engine's own
    `RoLADecodeGraph` (bit-identical-to-eager, correctness-gated) when the checked-out engine has it;
    else fall back to a generic `torch.cuda.graph` capture of the raw decode call (best-effort, NOT
    correctness-gated — a graph-breaker there, e.g. a host sync on the decode path, surfaces as a real
    exception, which the caller's round-robin loop already turns into an ERR sentinel row like any
    other cell failure — no special-case handling needed here)."""
    if getattr(module, "layer_idx", None) is None:
        module.layer_idx = 0
    seed_len = min(max(1, L), _SEED_PREFILL)
    x_pre = torch.randn(C.B, seed_len, DM, device=C.DEV, dtype=C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        cache = _new_cache()
        module(x_pre, past_key_values=cache, use_cache=True)
        if len(cache) == 0:
            raise RuntimeError(
                "decode arm (graphed): cache NOT seeded after prefill (len==0) -- see "
                "_decode_layer_sample's identical guard.")
        try:
            from fla_rola.layers.rola import RoLADecodeGraph
            g = RoLADecodeGraph(module, cache, x1)
            step = lambda: g.step(x1)
        except ImportError:
            static_in = x1.clone()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    module(static_in, past_key_values=cache, use_cache=True)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                module(static_in, past_key_values=cache, use_cache=True)

            def step():
                static_in.copy_(x1)
                graph.replay()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); step(); t1.record(); torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        peak = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    return ms, peak


def _decode_rola_cell(module, L):
    """RoLA layer decode -> (ms/token, state_mib_analytic, peak_mib_measured). state_mib is the FIXED
    recurrent footprint (the comparison axis); peak_mib is the measured decode-step peak."""
    ms, iqr_ms, peak = _decode_layer_step(module, L)
    return ms, iqr_ms, _module_state_mib(module), peak


def _decode_monolith_cell(module, L):
    """Matched-state monolith layer decode -> (ms/token, None, peak_mib_measured) via its own use_cache
    path. No clean analytic state axis (the wide nc=1 state), so the analytic slot is None."""
    ms, iqr_ms, peak = _decode_layer_step(module, L)
    return ms, iqr_ms, None, peak


def _decode_attn_cell(L):
    """LAYER-LEVEL attention decode: one query token through _CausalMHA attending to an L-long cached
    KV (the O(L)-GROWING cache RoLA's bounded state is measured against). Reports
    (ms/token, cache_mib_analytic, peak_mib_measured) — cache_mib is the analytic K+V footprint (the
    comparison axis), peak_mib the measured decode-step peak (cache + SDPA transient)."""
    times, peak = [], 0
    for r in range(DECODE_WARMUP + DECODE_STEPS):
        ms, pk = _decode_attn_sample(L)
        peak = max(peak, pk)
        if r >= DECODE_WARMUP:
            times.append(ms)
    hd = DM // C.H
    ms, iqr_ms = C.median(times), C.iqr(times)
    return _round_ms(ms, 4), _round_ms(iqr_ms, 4), _attn_cache_mib(L), peak


def _decode_attn_sample(L):
    hd = DM // C.H
    m = _CausalMHA(DM, C.H).to(C.DEV, C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated()   # pre-cache baseline (params only)
    kc = torch.randn(C.B, C.H, L, hd, device=C.DEV, dtype=C.DT)
    vc = torch.randn(C.B, C.H, L, hd, device=C.DEV, dtype=C.DT)
    with torch.inference_mode(), _sdpa_context():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); m(x1, cache=(kc, vc), use_cache=False); t1.record(); torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        peak = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    return ms, peak


# ------------------------------------------------------------------ sweep
def _row(**kw):
    base = dict(bench=None, L=None, state_axis=None, X_pct=None, d_qk=None, nc=None, routing=None, arm=None,
                fwd_ms=None, fwd_iqr_ms=None, fb_ms=None, fb_iqr_ms=None, peak_mib=None, act_mib=None,
                fwd_iqr_pct=None, fb_iqr_pct=None, decode_iqr_pct=None,
                fwd_samples_ms=None, fb_samples_ms=None, decode_samples_ms=None,
                peak_samples_mib=None, act_samples_mib=None,
                state_mib=None, attn_backend=None, sdpa_flags=None)
    base.update(kw)
    return base


_FLA_BASELINE_KINDS = tuple(_FLA_BASELINE_GEOM)   # ("gla", "gdn", "deltanet", "gsa")


def _arm_kinds():
    raw = os.environ.get("SCALE_ARMS", "attn,rola,monolith")
    kinds = tuple(x.strip() for x in raw.split(",") if x.strip())
    allowed = {"attn", "rola", "monolith", *_FLA_BASELINE_KINDS, *_NAMED_ROLA_ARM_KINDS}
    unknown = sorted(set(kinds) - allowed)
    if unknown:
        raise ValueError(f"unknown SCALE_ARMS entries {unknown}; expected subset of {sorted(allowed)}")
    return kinds


def _decode_graphed_enabled():
    """SCALE_DECODE_GRAPHED=1 adds a `graphed: true` cell variant to every RoLA decode arm (eager and
    graphed as cells of the SAME decode bench, per the paper_v1 methodology) -- OFF by default (byte-
    identical to before when unset), and only meaningful for `bench=='decode'`."""
    return _env_flag("SCALE_DECODE_GRAPHED", False)


def _arm_specs(routings, include_attention=False, include_graphed=False):
    kinds = _arm_kinds()
    idx = 0
    if include_attention and "attn" in kinds:
        yield dict(idx=idx, kind="attn", routing=None, gla=None, arm="attn", graphed=False)
        idx += 1
    for kind in ("rola", "monolith"):
        if kind not in kinds:
            continue
        kind_routings = routings if kind == "rola" else (None,)
        for routing in kind_routings:
            for gla, var in ((False, "rla"), (True, "gla")):
                yield dict(idx=idx, kind=kind, routing=routing, gla=gla, arm=f"{kind}-{var}", graphed=False)
                idx += 1
                # graphed decode variant: RoLA only (monolith/attn stay eager -- the graph capture is
                # RoLA's own decode-step machinery, not a generic layer capability here).
                if include_graphed and kind == "rola":
                    yield dict(idx=idx, kind=kind, routing=routing, gla=gla,
                               arm=f"{kind}-{var}-graphed", graphed=True)
                    idx += 1
    # Real external FLA baselines (paper_v1): one fixed-geometry variant each, no routing/gla axis,
    # never graphed (that machinery is RoLA-decode-specific).
    for kind in _FLA_BASELINE_KINDS:
        if kind not in kinds:
            continue
        yield dict(idx=idx, kind=kind, routing=None, gla=None, arm=kind, graphed=False)
        idx += 1
    # The flagship named-config arms (paper_v1): fixed geometry, eligible for the graphed decode
    # variant (it IS RoLA's own decode-step machinery).
    for kind in _NAMED_ROLA_ARM_KINDS:
        if kind not in kinds:
            continue
        yield dict(idx=idx, kind=kind, routing=None, gla=None, arm=kind, graphed=False)
        idx += 1
        if include_graphed:
            yield dict(idx=idx, kind=kind, routing=None, gla=None, arm=f"{kind}-graphed", graphed=True)
            idx += 1


def _safe_empty_cache():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _env_flag(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "", "false", "False", "no", "No")


def _preflight_enabled():
    return _env_flag("SCALE_PREFLIGHT_ARMS", False)


def _preflight_timeout():
    return int(os.environ.get("SCALE_PREFLIGHT_TIMEOUT", "900"))


def _append_error(rows, bench, L, x_pct, d, nc, spec, label, exc=None):
    row = _row(bench=bench, L=L, state_axis="nc", X_pct=x_pct, d_qk=d, nc=nc,
               routing=spec["routing"], arm=spec["arm"])
    if bench == "decode":
        row["decode_ms"] = label
    else:
        row["fwd_ms"] = label
        if bench == "train":
            row["fb_ms"] = label
    if exc is not None:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["trace"] = traceback.format_exc()
    rows.append(row)
    return row


def _preflight_child(args):
    routing = None if args.preflight_routing == "__none__" else args.preflight_routing
    spec = {
        "idx": args.preflight_idx,
        "kind": args.preflight_kind,
        "routing": routing,
        "gla": bool(args.preflight_gla),
        "arm": args.preflight_arm,
        "graphed": bool(args.preflight_graphed),
    }
    try:
        if args.preflight_bench in ("train", "prefill"):
            rec = _prepare_timed_arm(
                args.preflight_bench,
                args.preflight_L,
                args.preflight_d,
                args.preflight_nc,
                spec,
            )
            if rec is None:
                data = {"ok": False, "label": "OOM", "error": "arm build returned None"}
            else:
                sample = _prepared_sample(rec)
                peak = sample[2] if args.preflight_bench == "train" else sample[1]
                data = {"ok": True, "peak_mib": round(float(peak), 1)}
        else:
            sample = _timed_decode_sample(args.preflight_L, args.preflight_d, args.preflight_nc, spec)
            if sample is None:
                data = {"ok": False, "label": "OOM", "error": "arm build returned None"}
            else:
                data = {"ok": True, "peak_mib": round(float(sample[-1]), 1)}
    except _INFEASIBLE as e:
        data = {"ok": False, "label": "OOM", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()}
    except Exception as e:  # noqa: BLE001
        data = {"ok": False, "label": "ERR", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()}

    Path(args.preflight_out).write_text(json.dumps(data))


def _preflight_arm(bench, L, d, nc, spec):
    if not _preflight_enabled():
        return {"ok": True}

    fd, out_path = tempfile.mkstemp(prefix="rola-preflight-", suffix=".json")
    os.close(fd)
    routing = spec["routing"] if spec["routing"] is not None else "__none__"
    cmd = [
        sys.executable, "-m", "rola_bench.perf.scaling",
        "--preflight-out", out_path,
        "--preflight-bench", bench,
        "--preflight-L", str(L),
        "--preflight-d", str(d),
        "--preflight-nc", str(nc),
        "--preflight-kind", spec["kind"],
        "--preflight-routing", routing,
        "--preflight-gla", "1" if spec["gla"] else "0",
        "--preflight-arm", spec["arm"],
        "--preflight-idx", str(spec["idx"]),
        "--preflight-graphed", "1" if spec.get("graphed") else "0",
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=True,
            timeout=_preflight_timeout(),
            check=False,
        )
        try:
            data = json.loads(Path(out_path).read_text())
        except Exception as e:  # noqa: BLE001
            data = {"ok": False, "label": "ERR",
                    "error": f"preflight produced no JSON: {type(e).__name__}: {e}"}
        if proc.returncode != 0 and data.get("ok", False):
            data = {"ok": False, "label": "ERR", "error": f"preflight exited {proc.returncode}"}
        if not data.get("ok", False):
            stderr = proc.stderr.strip()
            stdout = proc.stdout.strip()
            if stderr:
                data["stderr"] = stderr[-4000:]
            if stdout:
                data["stdout"] = stdout[-4000:]
        return data
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "label": "ERR", "error": f"preflight timed out after {e.timeout}s"}
    finally:
        try:
            Path(out_path).unlink()
        except OSError:
            pass


def _build_scaling_arm(spec, L, d, nc):
    C.set_shape(L=L, DQK=d, DV=d)
    if spec["kind"] == "attn":
        return _CausalMHA(DM, C.H).to(C.DEV, C.DT)
    if spec["kind"] == "rola":
        return build_rola(spec["gla"], nc, d, routing=spec["routing"])
    if spec["kind"] in _FLA_BASELINE_KINDS:
        return _build_fla_baseline(spec["kind"])
    if spec["kind"] in _NAMED_ROLA_ARM_KINDS:
        module, _used = build_rola_named(spec["kind"])
        return module
    return build_monolith(spec["gla"], nc, d)


def _summarize_train(rows, L, x_pct, d, nc, spec, recs):
    f, fb, pk, ac = zip(*recs)
    f_stats = C.sample_stats(f)
    fb_stats = C.sample_stats(fb)
    row = _row(bench="train", L=L, state_axis="nc", X_pct=x_pct, d_qk=d, nc=nc,
               routing=spec["routing"], arm=spec["arm"],
               fwd_ms=f_stats["median"], fwd_iqr_ms=f_stats["iqr"], fwd_iqr_pct=f_stats["iqr_pct"],
               fb_ms=fb_stats["median"], fb_iqr_ms=fb_stats["iqr"], fb_iqr_pct=fb_stats["iqr_pct"],
               peak_mib=round(max(pk)), act_mib=round(max(ac)),
               fwd_samples_ms=C.samples_json(f), fb_samples_ms=C.samples_json(fb),
               peak_samples_mib=C.samples_json(pk, digits=1), act_samples_mib=C.samples_json(ac, digits=1),
               attn_backend=_sdpa_backend_label() if spec["kind"] == "attn" else None,
               sdpa_flags=str(_sdpa_flags()) if spec["kind"] == "attn" else None)
    rows.append(row); print(row, flush=True)


def _summarize_prefill(rows, L, x_pct, d, nc, spec, recs):
    f, pk = zip(*recs)
    f_stats = C.sample_stats(f)
    row = _row(bench="prefill", L=L, state_axis="nc", X_pct=x_pct, d_qk=d, nc=nc,
               routing=spec["routing"], arm=spec["arm"],
               fwd_ms=f_stats["median"], fwd_iqr_ms=f_stats["iqr"], fwd_iqr_pct=f_stats["iqr_pct"],
               fwd_samples_ms=C.samples_json(f), peak_samples_mib=C.samples_json(pk, digits=1),
               peak_mib=round(max(pk)),
               attn_backend=_sdpa_backend_label() if spec["kind"] == "attn" else None,
               sdpa_flags=str(_sdpa_flags()) if spec["kind"] == "attn" else None)
    rows.append(row); print(row, flush=True)


def _prepared_sample(rec):
    """Run one timing sample against a warm, reused module/input pair."""
    bench, spec, module, x = rec["bench"], rec["spec"], rec["module"], rec["x"]
    if bench == "train":
        leaves = {"x": x.detach().requires_grad_()}
        forward = lambda lv: _fwd(module, lv["x"])
        if spec["kind"] == "attn":
            with _sdpa_context():
                return _e2e_rep(forward, leaves, module)
        return _e2e_rep(forward, leaves, module)
    if spec["kind"] == "attn":
        with _sdpa_context(), torch.inference_mode():
            return _prefill_rep(module, x)
    with torch.inference_mode():
        return _prefill_rep(module, x)


def _prepare_timed_arm(bench, L, d, nc, spec):
    """Build one module/input pair for the whole cell timing pass.

    Keeping the same module and input across warmup + measured reps avoids first-use allocator/cuBLAS/
    module construction jitter in every retained sample. Peak memory is still measured by a separate
    isolated pass below, so the extra resident arms in the round-robin timing pass do not define the
    reported memory frontier.
    """
    seed = 10_000 + L + d * 31 + nc * 17 + spec["idx"]
    torch.manual_seed(seed)
    module = _build_scaling_arm(spec, L, d, nc)
    if module is None:
        return None
    torch.manual_seed(seed + 1)
    x = torch.randn(C.B, L, DM, device=C.DEV, dtype=C.DT)
    return {"bench": bench, "spec": spec, "module": module, "x": x}


def _isolated_metric_sample(bench, L, d, nc, spec):
    """One isolated warm sample for peak/activation accounting.

    This intentionally builds only one arm, warms it once, then records the memory metrics from a
    second sample. Timing from this pass is discarded; the round-robin timing pass owns latency.
    """
    rec = _prepare_timed_arm(bench, L, d, nc, spec)
    if rec is None:
        return None
    _prepared_sample(rec)
    sample = _prepared_sample(rec)
    del rec
    gc.collect()
    _safe_empty_cache()
    return sample


def _with_isolated_metrics(bench, data, metric_sample):
    if not data or metric_sample is None:
        return data
    if bench == "train":
        _, _, peak, act = metric_sample
        return [(f, fb, peak, act) for f, fb, _, _ in data]
    _, peak = metric_sample
    return [(f, peak) for f, _ in data]


def _timed_decode_sample(L, d, nc, spec):
    """Run one isolated seeded decode sample for a RoLA/monolith scaling arm."""
    seed = 20_000 + L + d * 31 + nc * 17 + spec["idx"]
    torch.manual_seed(seed)
    if spec["kind"] == "attn":
        torch.manual_seed(seed + 1)
        ms, peak = _decode_attn_sample(L)
        return ms, _attn_cache_mib(L), peak
    module = _build_scaling_arm(spec, L, d, nc)
    if module is None:
        return None
    torch.manual_seed(seed + 1)
    sampler = _decode_layer_sample_graphed if spec.get("graphed") else _decode_layer_sample
    ms, peak = sampler(module, L)
    state_mib = None
    if spec["kind"] == "rola":
        state_mib = _module_state_mib(module)
    return ms, state_mib, peak


def _timed_group(bench, L, x_pct, d, nc, routings, rows):
    """Time comparable RoLA/monolith arms round-robin for one scaling cell.

    Keeping the same `(bench,L,d,nc)` arms in round-robin order makes thermal/clock drift affect their
    ratios similarly. Each arm now reuses one warm module/input pair for latency; per-arm peak memory is
    measured afterward in an isolated one-arm pass.
    """
    specs = list(_arm_specs(routings, include_attention=True))
    survivors = []
    for spec in specs:
        try:
            preflight = _preflight_arm(bench, L, d, nc, spec)
            if not preflight.get("ok", False):
                row = _append_error(rows, bench, L, x_pct, d, nc, spec,
                                    preflight.get("label", "ERR"))
                row["error"] = preflight.get("error")
                if preflight.get("trace"):
                    row["trace"] = preflight["trace"]
                print(f"{row['fwd_ms']} {spec['arm']} {bench} L{L} X{x_pct} d{d} "
                      f"r{spec['routing']} nc{nc}: {row.get('error')}", flush=True)
                print(row, flush=True)
                continue
            survivors.append(spec)
        except _INFEASIBLE:
            row = _append_error(rows, bench, L, x_pct, d, nc, spec, "OOM")
            _safe_empty_cache()
            print(f"OOM {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}", flush=True)
            print(row, flush=True)
        except Exception as e:  # noqa: BLE001
            row = _append_error(rows, bench, L, x_pct, d, nc, spec, "ERR", e)
            _safe_empty_cache()
            print(f"ERR {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}: "
                  f"{type(e).__name__}: {e}", flush=True)
            print(row, flush=True)

    active = []
    for spec in survivors:
        try:
            prepared = _prepare_timed_arm(bench, L, d, nc, spec)
            if prepared is not None:
                active.append(dict(key=f"{spec['arm']}:{spec['routing']}", spec=spec,
                                   prepared=prepared, data=[]))
        except _INFEASIBLE:
            row = _append_error(rows, bench, L, x_pct, d, nc, spec, "OOM")
            _safe_empty_cache()
            print(f"OOM {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}", flush=True)
            print(row, flush=True)
        except Exception as e:  # noqa: BLE001
            row = _append_error(rows, bench, L, x_pct, d, nc, spec, "ERR", e)
            _safe_empty_cache()
            print(f"ERR {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}: "
                  f"{type(e).__name__}: {e}", flush=True)
            print(row, flush=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*format mismatch.*")
        for r in range(C.WARMUP + C.REPS):
            still_active = []
            for rec in C.rotated(active, r):
                spec = rec["spec"]
                try:
                    sample = _prepared_sample(rec["prepared"])
                    if r >= C.WARMUP:
                        rec["data"].append(sample)
                    still_active.append(rec)
                except _INFEASIBLE:
                    row = _append_error(rows, bench, L, x_pct, d, nc, spec, "OOM")
                    _safe_empty_cache()
                    print(f"OOM {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}", flush=True)
                    print(row, flush=True)
                except Exception as e:  # noqa: BLE001
                    row = _append_error(rows, bench, L, x_pct, d, nc, spec, "ERR", e)
                    _safe_empty_cache()
                    print(f"ERR {spec['arm']} {bench} L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    print(row, flush=True)
                finally:
                    _safe_empty_cache()
            active = still_active
            if not active:
                break

    for rec in active:
        if not rec["data"]:
            continue
        try:
            rec["data"] = _with_isolated_metrics(
                bench, rec["data"], _isolated_metric_sample(bench, L, d, nc, rec["spec"]))
        except _INFEASIBLE:
            row = _append_error(rows, bench, L, x_pct, d, nc, rec["spec"], "OOM")
            print(f"OOM {rec['spec']['arm']} {bench} L{L} X{x_pct} d{d} "
                  f"r{rec['spec']['routing']} nc{nc} during isolated memory pass", flush=True)
            print(row, flush=True)
            continue
        if bench == "train":
            _summarize_train(rows, L, x_pct, d, nc, rec["spec"], rec["data"])
        else:
            _summarize_prefill(rows, L, x_pct, d, nc, rec["spec"], rec["data"])

    del active
    gc.collect()
    _safe_empty_cache()


def _summarize_decode(rows, L, x_pct, d, nc, spec, recs):
    ms, state_mib, pk = zip(*recs)
    ms_stats = C.sample_stats(ms, digits=4)
    row = _row(bench="decode", L=L, state_axis="nc", X_pct=x_pct, d_qk=d, nc=nc,
               routing=spec["routing"], arm=spec["arm"],
               decode_ms=ms_stats["median"], decode_iqr_ms=ms_stats["iqr"],
               decode_iqr_pct=ms_stats["iqr_pct"], decode_samples_ms=C.samples_json(ms, digits=4),
               state_mib=next((m for m in state_mib if m is not None), None),
               peak_mib=round(max(pk)), peak_samples_mib=C.samples_json(pk, digits=1),
               attn_backend=_sdpa_backend_label() if spec["kind"] == "attn" else None,
               sdpa_flags=str(_sdpa_flags()) if spec["kind"] == "attn" else None)
    rows.append(row); print(row, flush=True)


def _timed_decode_group(L, x_pct, d, nc, routings, rows):
    """Time comparable decode arms round-robin for one scaling cell."""
    active = []
    for spec in _arm_specs(routings, include_attention=True, include_graphed=_decode_graphed_enabled()):
        preflight = _preflight_arm("decode", L, d, nc, spec)
        if not preflight.get("ok", False):
            row = _append_error(rows, "decode", L, x_pct, d, nc, spec,
                                preflight.get("label", "ERR"))
            row["error"] = preflight.get("error")
            if preflight.get("trace"):
                row["trace"] = preflight["trace"]
            print(f"{row.get('decode_ms')} {spec['arm']} decode L{L} X{x_pct} d{d} "
                  f"r{spec['routing']} nc{nc}: {row.get('error')}", flush=True)
            print(row, flush=True)
            continue
        active.append(dict(key=f"{spec['arm']}:{spec['routing']}", spec=spec, data=[]))

    for r in range(DECODE_WARMUP + DECODE_STEPS):
        still_active = []
        for rec in C.rotated(active, r):
            spec = rec["spec"]
            try:
                sample = _timed_decode_sample(L, d, nc, spec)
                if sample is None:
                    continue
                if r >= DECODE_WARMUP:
                    rec["data"].append(sample)
                still_active.append(rec)
            except _INFEASIBLE:
                row = _append_error(rows, "decode", L, x_pct, d, nc, spec, "OOM")
                _safe_empty_cache()
                print(f"OOM {spec['arm']} decode L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}", flush=True)
                print(row, flush=True)
            except Exception as e:  # noqa: BLE001
                row = _append_error(rows, "decode", L, x_pct, d, nc, spec, "ERR", e)
                _safe_empty_cache()
                print(f"ERR {spec['arm']} decode L{L} X{x_pct} d{d} r{spec['routing']} nc{nc}: "
                      f"{type(e).__name__}: {e}", flush=True)
                print(row, flush=True)
            finally:
                _safe_empty_cache()
        active = still_active
        if not active:
            break

    for rec in active:
        if rec["data"]:
            _summarize_decode(rows, L, x_pct, d, nc, rec["spec"], rec["data"])


def sweep(benches, lens, ncs, dqks, rows, routings=ROUTINGS):
    for bench in benches:
        for L in lens:
            C.set_shape(L=L)
            # ---- Attention + RoLA (routing sweep) + matched-state wide-monolith: (X%, d_qk) grid ----
            for d in dqks:
                for nc in ncs:
                    # BOUNDED-STATE REGIME ONLY. Compare PHYSICAL bytes: RoLA's fp32 recurrent state
                    # H·nc·d·(d+1) vs attention's bf16 K+V cache 2·L·d_model. At/above 100% RoLA holds
                    # more physical state than the full KV cache — the regime it is not for — so skip it.
                    x_pct = _state_x_pct(nc, d, L)
                    if x_pct >= 100.0:
                        continue
                    if bench in ("train", "prefill"):
                        _timed_group(bench, L, x_pct, d, nc, routings, rows)
                        continue
                    _timed_decode_group(L, x_pct, d, nc, routings, rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description="RoLA scaling benchmark (RoLA vs attention; per-arch curves).")
    ap.add_argument("--preflight-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-bench", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-L", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-d", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-nc", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-kind", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-routing", default="__none__", help=argparse.SUPPRESS)
    ap.add_argument("--preflight-gla", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-idx", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--preflight-graphed", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--benches", default="train,prefill,decode", help="comma: train,prefill,decode")
    ap.add_argument("--lens", default=os.environ.get("SCALE_LENS", "1024,4096,16384,65536"))
    ap.add_argument("--ncs", default=os.environ.get("SCALE_NCS", "16,64,256,1024"),
                    help="fixed states_per_head to sweep (cells kept only while state<KV: nc*d<2L)")
    ap.add_argument("--dqks", default=os.environ.get("SCALE_DQKS", "16,64,128"))
    ap.add_argument("--routings", default=os.environ.get("SCALE_ROUTINGS", "flat,square,tree"),
                    help="comma: RoLA per-head factor topologies to sweep")
    ap.add_argument("--reps", type=int, default=C.REPS)
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    if a.preflight_out:
        C.require_cuda()
        _preflight_child(a)
        return
    C.require_cuda()
    C.REPS = a.reps
    benches = [b for b in a.benches.split(",") if b.strip()]
    lens = [int(v) for v in a.lens.split(",") if v.strip()]
    ncs = [int(v) for v in a.ncs.split(",") if v.strip()]
    dqks = [int(v) for v in a.dqks.split(",") if v.strip()]
    routings = [r for r in a.routings.split(",") if r.strip()]
    print(f"# scaling | {C.gpu_name()} | B={C.B} H={C.H} d_model={DM} bf16 | benches={benches} "
          f"| L={lens} nc={ncs} d_qk={dqks} routing={routings} (cells: state<KV only) | RoLA(rla,gla) + attn", flush=True)
    rows = []
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    try:
        sweep(benches, lens, ncs, dqks, rows, routings=routings)
    finally:
        if rows:
            C.write_csv(out / "scaling.csv", rows)
            print(f"\nWROTE {out}/scaling.csv  [{C.gpu_name()}]  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
