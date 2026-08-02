"""Generic MQAR config generator: a declarative spec -> zoology (configs, configs_envs).

One generator owns all the grid LOGIC (cell builders, the baseline shape/state-match solve, run_id
naming); an experiment is just a YAML spec declaring the axes (backbone, data, lrs/seeds, and an
`arms` list grouped into tiers).

**REBUILT 2026-08-02 (Phase 1.3), against the V3 preset registry and the two-factor grid design.**
What changed, and why each change is structural rather than cosmetic:

  * the `routed` arm builder speaks `rola_bench.mqar.instances.rola_instance(name, nc, ...)` — no
    `d_qk`, no `kernel`, no `kernel_kwargs`. Those axes were deleted by the V3 architecture and the
    presets keyed on them retired (`instances.RETIRED_INSTANCES`).
  * **`seq_lens` is a first-class axis, crossed with `ncs`.** `L` used to be baked into the data
    preset, so `alpha_cap = N/L` was derivable post-hoc and sweepable never. Crossing them is what
    lets a spec walk `alpha_cap` through 1 from BOTH directions at fixed everything-else, which is
    the form the N = L boundary claim is stated in (audit §4.1 item 1).
  * **L-extrapolation is an axis, not a by-product**: `data.protocol.test_seq_lens` may contain
    lengths greater than the cell's train length, and those slices are reported separately.
  * **the addressing-entropy ladder `S = sum_l b_l`** is expressible: a routed arm may carry an
    explicit `states_per_level` list (mixed radix), which `rola_instance` validates against `nc`
    (audit §4.1 item 3).
  * **the attention `d_qk` ladder** is expressible: `{build: attn, head_dims: [...]}` sweeps the
    per-head key width, which for zoology's MHA is `d_model / num_heads` — the one place `d_qk`
    legitimately survives the V3 migration, on the attention baseline only (audit §4.1 item 4).
  * `sse_ladder` is GONE (`canonical_baselines.RETIRED_METHODS['sse']`: routing-family
    reimplementations are excluded from the baseline set by ruling).

Spec schema (see experiments/*.yaml):
  project_name: str
  backbone: {d_model, n_layers, vocab, l_max}        # any omitted -> common.DEFAULTS
  data:
    # either a fixed named preset ...
    {preset: ext, train_batch, test_batch, cache_dir}
    # ... or the law-grid protocol, which makes L an axis:
    {train_batch, test_batch, cache_dir,
     protocol: {train_kv: [..], test_seq_lens: [..], test_kv_by_len: {L: [..]},
                train_examples: int, test_examples: int}}
  train:    {max_epochs, weight_decay, seed | seeds:[..], early_stopping_threshold,
             early_stopping_metric, slice_keys, eval_every}
  lrs:  [..]                  # default LR sweep (an arm may override with its own `lrs`)
  ncs:  [..]                  # default state-rung ladder (an arm may override with `ncs`)
  seq_lens: [..]              # OPTIONAL train-length ladder; requires data.protocol. Crossed with ncs.
  tiers: {"<id>": [arm, ...]} # selected by the GRID_TIERS env (default all)
  arm kinds:
    {build: routed, instance, tag, n_heads?, d_v?, states_per_level?, state_norm?, decay?,
     router_bias?, kwargs?, lrs?, ncs?, seq_lens?, run_id?}
        run_id template vars: {tag} {nc} {st} {lr:.0e} {seed} {d_model} {L} {alpha_cap}
        default: "grid-routed-{tag}-nc{nc}_L{L}_st{st}_lr{lr:.0e}_s{seed}"
    {build: baselines, methods: [..], shapes: [..]}
    {build: attn, head_dims: [..]?}
"""
import os

from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig

from rola_bench.mqar import common as C
from rola_bench.mqar.experiments import canonical_baselines as _cb
from rola_bench.mqar import realized_state
from rola_bench.mqar.experiments.canonical_baselines import baseline_cell
from rola_bench.mqar.instances import PresetRungUnavailable, rola_instance

_ROUTED_RUNID = "grid-routed-{tag}-nc{nc}_L{L}_st{st}_lr{lr:.0e}_s{seed}"
_BASELINE_RUNID = "grid-{method}-{shape}-nc{nc}_L{L}_st{st}_lr{lr:.0e}_s{seed}"
_ATTN_RUNID = "grid-mha-dqk{d_qk}_L{L}_lr{lr:.0e}_s{seed}"

#: Minimum seeds for a published cell — see the check in `build_configs`.
MIN_SEEDS = 3

#: Retirement ledger for arm BUILDERS (not presets — those live in `instances.RETIRED_INSTANCES`).
RETIRED_BUILDERS = {
    "sse_ladder": "SSE is a routing-family method; the contender ladder admits one occupant per "
                  "TRANSITION-STRUCTURE rung and no routing-family reimplementations "
                  "(canonical_baselines.RETIRED_METHODS['sse']).",
}


def _data_for(spec, bb, L):
    """The DataConfig for train length `L` (or the spec's fixed preset when L is None)."""
    dat = spec["data"]
    if L is None:
        if "preset" not in dat:
            raise ValueError(
                "spec declares no `seq_lens` axis and no `data.preset`: one of the two must say "
                "what sequence lengths the cell trains on.")
        return C.make_data(dat["preset"], bb["vocab"], dat["train_batch"], dat["test_batch"],
                           dat["cache_dir"])
    proto = dat.get("protocol")
    if proto is None:
        raise ValueError(
            "`seq_lens` makes L an axis, which requires `data.protocol` (train_kv / test_seq_lens / "
            "test_kv_by_len / train_examples / test_examples). A named `data.preset` bakes its own "
            "fixed L ladder in and cannot be crossed with `ncs` — that is the axis the pre-V3 grid "
            "was missing, so silently falling back to the preset here would reproduce the gap.")
    # `by_train_len` lets one spec carry protocols that differ PER TRAIN LENGTH. It exists for the
    # separation cells, where a rung's evaluation ladder must scale with the rung (holding a fixed
    # item/length ratio across three orders of magnitude) and where evaluating a 16K cell at 64K
    # would be pure waste. Top-level protocol keys are the defaults; the per-length block overrides.
    per_len = (proto.get("by_train_len") or {})
    override = per_len.get(L, per_len.get(str(L)))
    if override is not None:
        proto = {**proto, **override}
    elif per_len:
        raise ValueError(
            f"data.protocol.by_train_len is declared but has no entry for train length {L}. Every "
            "length in `seq_lens` must be answered explicitly — an unstated rung would silently "
            "inherit another rung's evaluation ladder.")
    test_lens = [int(x) for x in proto["test_seq_lens"]]
    kv_by_len = {int(k): list(v) for k, v in proto["test_kv_by_len"].items()}
    missing = [x for x in test_lens if x not in kv_by_len]
    if missing:
        raise ValueError(f"data.protocol.test_kv_by_len has no entry for test length(s) {missing}")
    return C.make_law_data(
        bb["vocab"], dat["train_batch"], dat["test_batch"], dat["cache_dir"],
        train_seq_len=L, train_kv=proto["train_kv"], test_seq_lens=test_lens,
        test_kv_by_len=kv_by_len, train_examples=proto["train_examples"],
        test_examples=proto["test_examples"])


def build_configs(spec):
    """Expand a spec dict into (configs, configs_envs)."""
    bb = {**C.DEFAULTS, **spec.get("backbone", {})}   # spec overrides the library defaults
    _cb.DMODEL = bb["d_model"]                         # baseline projection solve reads this
    tr = spec["train"]
    def_lrs, def_ncs = spec["lrs"], spec["ncs"]
    def_seq_lens = spec.get("seq_lens") or [None]      # [None] => the fixed data preset
    seeds = tr.get("seeds") or [tr["seed"]]
    # SEED FLOOR (ruling 2026-08-02, from the MQAR retrospective): the pre-V3 grid was single-seed
    # EVERYWHERE, so no cell in it carries a variance estimate — and the law claim is a claim about
    # where a boundary sits, which is not readable off one draw. Three seeds is the floor for any
    # published cell. The escape is deliberately a WRITTEN REASON, not a flag: a spec that runs
    # single-seed must say in the spec why that is defensible, and the reason travels with the file.
    if len(seeds) < MIN_SEEDS and not spec.get("single_seed_rationale"):
        raise ValueError(
            f"spec {spec.get('project_name')!r} declares {len(seeds)} seed(s); published MQAR cells "
            f"require >= {MIN_SEEDS} (variance is part of the law claim, and the pre-V3 grid's "
            "single-seed cells are exactly why). Either widen `train.seeds`, or state a "
            "`single_seed_rationale:` in the spec explaining why this experiment does not need one.")
    l_max = bb["l_max"]
    env = {"EVAL_EVERY_N": str(tr["eval_every"])} if tr.get("eval_every") else {}

    configs, envs = [], []
    _data_cache = {}

    def data_for(L):
        if L not in _data_cache:
            _data_cache[L] = _data_for(spec, bb, L)
        return _data_cache[L]

    def add(kernel, run_id, lr, seed, L):
        configs.append(TrainConfig(
            data=data_for(L),
            model=ModelConfig(block_type="TransformerBlock",
                              sequence_mixer=C.wrap_hybrid(kernel, l_max),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=bb["d_model"], n_layers=bb["n_layers"],
                              max_position_embeddings=0, vocab_size=bb["vocab"]),
            logger=LoggerConfig(project_name=spec["project_name"], entity=""),
            max_epochs=tr["max_epochs"], learning_rate=lr, weight_decay=tr["weight_decay"],
            seed=seed, run_id=run_id,
            early_stopping_threshold=tr["early_stopping_threshold"],
            early_stopping_metric=tr["early_stopping_metric"],
            slice_keys=list(tr["slice_keys"])))
        envs.append(dict(env))

    def _axes(arm):
        """(lrs, ncs, seq_lens) for one arm — arm-level overrides beat the spec defaults."""
        return (arm.get("lrs", def_lrs), arm.get("ncs", def_ncs),
                arm.get("seq_lens", def_seq_lens))

    def _fmt(rid, **kw):
        # `alpha_cap = N/L` is the capacity ratio the boundary claim is stated in; it is available to
        # every run_id template so a cell names its own position relative to the N = L diagonal.
        L, nc = kw.get("L"), kw.get("nc")
        kw.setdefault("alpha_cap", (f"{nc / L:g}" if (L and nc) else "na"))
        kw.setdefault("L", "preset")
        return rid.format(**kw)

    def realized_st(kernel):
        """The cell's CAPACITY COORDINATE — realized content floats, read back off the constructed
        module (`realized_state`). Never the nominal target, never a formula: the retrospective's
        ruling, and the same fix as finding F1. `unbounded` for attention, which has no rung."""
        content, _overhead = realized_state.state_floats(kernel, bb["d_model"])
        return "unbounded" if content is None else content

    # --- arm builders (run_id formats are load-bearing: they key the result rows) ---
    def routed(arm):
        rid = arm.get("run_id", _ROUTED_RUNID)
        extra = dict(arm.get("kwargs", {}))       # forwarded verbatim to the mixer (V3 kwargs only)
        preset_kw = {k: arm[k] for k in ("n_heads", "d_v", "states_per_level", "state_norm",
                                         "decay", "router_bias", "decay_target_leaf_rate")
                     if k in arm}
        a_lrs, a_ncs, a_lens = _axes(arm)
        dropped = []
        for seed in seeds:
            for L in a_lens:
                for nc in a_ncs:
                    try:
                        kw = rola_instance(arm["instance"], nc, **preset_kw)
                    except PresetRungUnavailable as e:
                        # Structural, not a failure: a D=2 wiring has no spelling at N=2. Reported
                        # exactly like a baseline's kernel ceiling — an omitted cell with a reason.
                        if seed == seeds[0] and L == a_lens[0]:
                            dropped.append(f"{arm['tag']}/nc{nc}: {e}")
                        continue
                    kw.update(extra)
                    kernel = dict(name="zoology.mixers.rola.RoLAMixer", kwargs=kw)
                    st = realized_st(kernel)
                    for lr in a_lrs:
                        add(kernel, _fmt(rid, tag=arm["tag"], nc=nc, st=st, lr=lr, seed=seed,
                                         d_model=bb["d_model"], L=L), lr, seed, L)
        if dropped:
            print(f"[build_configs] {len(dropped)} routed cell(s) omitted -- the preset has no "
                  f"spelling at that rung (NOT a run failure): {dropped}", flush=True)

    def baselines(arm):
        rid = arm.get("run_id", _BASELINE_RUNID)
        a_lrs, a_ncs, a_lens = _axes(arm)
        dropped = []
        for seed in seeds:
            for L in a_lens:
                for method in arm["methods"]:
                    for shape in arm["shapes"]:
                        for nc in a_ncs:
                            cell = baseline_cell(method, shape, nc)
                            if cell is None:
                                # NOT a run failure: an undefined shape or a STRUCTURAL EXIT (the
                                # method's own kernel/memory bound). Logged once, and it is a
                                # reportable cost column for that method, never missing data.
                                if seed == seeds[0] and L == a_lens[0]:
                                    dropped.append(f"{method}/{shape}/nc{nc}")
                                continue
                            kernel, _nominal = cell
                            st = realized_st(kernel)
                            for lr in a_lrs:
                                add(kernel, _fmt(rid, method=method, shape=shape, nc=nc, st=st,
                                                 lr=lr, seed=seed, d_model=bb["d_model"], L=L),
                                    lr, seed, L)
        if dropped:
            print(f"[build_configs] {len(dropped)} baseline cell(s) omitted — matched-state "
                  f"undefined / structural exit (NOT a run failure): {dropped}", flush=True)

    def attn(arm):
        """The MHA oracle. `head_dims` sweeps the per-head KEY width — the attention `d_qk` ladder.

        zoology's MHA derives `head_dim = d_model // num_heads`, so a head width `d_qk` is requested
        by asking for `d_model // d_qk` heads. This is the ONE place `d_qk` survives the V3
        migration: it is attention's own capacity dimension, and the fitted exchange rate between it
        and RoLA's `N` is a paper row (audit §4.1 item 4). Omit `head_dims` for the single canonical
        oracle cell.
        """
        rid = arm.get("run_id", _ATTN_RUNID)
        a_lrs, _, a_lens = _axes(arm)
        head_dims = arm.get("head_dims")
        if head_dims is None:
            cells = [(C.mha, C.mha["kwargs"]["num_heads"] and bb["d_model"] // C.mha["kwargs"]["num_heads"])]
        else:
            cells = []
            for d_qk in head_dims:
                if bb["d_model"] % d_qk:
                    raise ValueError(
                        f"attention head width d_qk={d_qk} does not divide d_model="
                        f"{bb['d_model']}; zoology's MHA derives head_dim = d_model // num_heads, "
                        "so the ladder must be built from divisors.")
                cells.append((dict(name="zoology.mixers.attention.MHA",
                                   kwargs=dict(num_heads=bb["d_model"] // d_qk, dropout=0.1)), d_qk))
        for seed in seeds:
            for L in a_lens:
                for kernel, d_qk in cells:
                    for lr in a_lrs:
                        add(kernel, _fmt(rid, d_qk=d_qk, lr=lr, seed=seed,
                                         d_model=bb["d_model"], L=L), lr, seed, L)

    builders = {"routed": routed, "baselines": baselines, "attn": attn}

    tiers = spec["tiers"]
    sel = os.environ.get("GRID_TIERS", "all")
    chosen = list(tiers) if sel == "all" else [t for t in tiers if t in sel]
    for t in chosen:                       # preserve declaration order (== run order)
        for arm in tiers[t]:
            kind = arm["build"]
            if kind in RETIRED_BUILDERS:
                raise ValueError(f"arm builder {kind!r} is RETIRED: {RETIRED_BUILDERS[kind]}")
            if kind not in builders:
                raise ValueError(f"unknown arm builder {kind!r}; expected one of {sorted(builders)}")
            builders[kind](arm)
    if not configs:
        raise ValueError(
            f"spec {spec.get('project_name')!r} expanded to ZERO cells (GRID_TIERS={sel!r}). An "
            "empty grid is always a spec or tier-selection bug — refusing rather than launching a "
            "no-op fleet job.")
    return configs, envs


def load_spec(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_configs_and_envs(spec_path):
    return build_configs(load_spec(spec_path))
