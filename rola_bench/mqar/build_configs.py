"""Generic MQAR config generator: a declarative spec -> zoology (configs, configs_envs).

One generator owns all the grid LOGIC (cell builders, the baseline shape/state-match solve, the SSE
ladder, run_id naming); an experiment is just a YAML spec declaring the axes (backbone, data,
lrs/seeds, and an `arms` list grouped into tiers). Replaces the one-off `*.py` experiment modules.

Spec schema (see experiments/*.yaml):
  project_name: str
  backbone: {d_model, n_layers, vocab, l_max}        # any omitted -> common.DEFAULTS
  data:     {preset, train_batch, test_batch, cache_dir}
  train:    {max_epochs, weight_decay, seed | seeds:[..], early_stopping_threshold,
             early_stopping_metric, slice_keys, eval_every}
  lrs:  [..]                  # default LR sweep (an arm may override with its own `lrs`)
  ncs:  [..]                  # default state-rung ladder (an arm may override with `ncs`)
  tiers: {"<id>": [arm, ...]} # selected by the GRID_TIERS env (default all)
  arm kinds:
    {build: routed, instance, tag, d_qk, d_v, n_heads, kwargs?, lrs?, ncs?, run_id?}
        run_id template vars: {tag} {nc} {st} {lr:.0e} {seed} {d_model}
        default: "grid-routed-{tag}-nc{nc}_st{st}_lr{lr:.0e}_s{seed}"
    {build: baselines, methods: [..], shapes: [..]}
    {build: sse_ladder}
    {build: attn}
"""
import os

from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig

from rola_bench.mqar import common as C
from rola_bench.mqar.experiments import canonical_baselines as _cb
from rola_bench.mqar.experiments.canonical_baselines import baseline_cell, REF
from rola_bench.mqar.instances import rola_instance

_ROUTED_RUNID = "grid-routed-{tag}-nc{nc}_st{st}_lr{lr:.0e}_s{seed}"


def build_configs(spec):
    """Expand a spec dict into (configs, configs_envs)."""
    bb = {**C.DEFAULTS, **spec.get("backbone", {})}   # spec overrides the library defaults
    _cb.DMODEL = bb["d_model"]                         # baseline projection solve reads this
    dat, tr = spec["data"], spec["train"]
    def_lrs, ncs = spec["lrs"], spec["ncs"]
    seeds = tr.get("seeds") or [tr["seed"]]
    l_max = bb["l_max"]
    data_cfg = C.make_data(dat["preset"], bb["vocab"], dat["train_batch"], dat["test_batch"],
                           dat["cache_dir"])
    env = {"EVAL_EVERY_N": str(tr["eval_every"])} if tr.get("eval_every") else {}

    configs, envs = [], []

    def add(kernel, run_id, lr, seed):
        configs.append(TrainConfig(
            data=data_cfg,
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

    # --- arm builders (run_id formats are load-bearing: they key the result rows) ---
    def routed(arm):
        rid = arm.get("run_id", _ROUTED_RUNID)
        extra = arm.get("kwargs", {})            # e.g. {router_bias: true}
        a_lrs, a_ncs = arm.get("lrs", def_lrs), arm.get("ncs", ncs)
        for seed in seeds:
            for nc in a_ncs:
                kw = rola_instance(arm["instance"], d_qk=arm["d_qk"], d_v=arm["d_v"],
                                   num_chunks=nc, n_heads=arm["n_heads"])
                kw.update(extra)
                kernel = dict(name="zoology.mixers.rola.RoLAMixer", kwargs=kw)
                for lr in a_lrs:
                    add(kernel, rid.format(tag=arm["tag"], nc=nc, st=REF(nc), lr=lr, seed=seed,
                                           d_model=bb["d_model"]), lr, seed)

    def baselines(arm):
        rid = arm.get("run_id", "grid-{method}-{shape}-nc{nc}_st{st}_lr{lr:.0e}_s{seed}")
        a_lrs, a_ncs = arm.get("lrs", def_lrs), arm.get("ncs", ncs)
        dropped = []
        for seed in seeds:
            for method in arm["methods"]:
                for shape in arm["shapes"]:
                    for nc in a_ncs:
                        cell = baseline_cell(method, shape, nc)
                        if cell is None:
                            if seed == seeds[0]:        # log infeasible cells once — never drop silently (#129)
                                dropped.append(f"{method}/{shape}/nc{nc}")
                            continue
                        kernel, st = cell
                        for lr in a_lrs:
                            add(kernel, rid.format(method=method, shape=shape, nc=nc, st=st, lr=lr,
                                                   seed=seed, d_model=bb["d_model"]), lr, seed)
        if dropped:
            print(f"[build_configs] {len(dropped)} baseline cell(s) omitted — matched-state "
                  f"undefined / capped (NOT a run failure): {dropped}", flush=True)

    def sse_ladder(arm):
        # SSE scaled the way it is strongest: NATIVE (few big partitions) + PARALLEL (N~nc small).
        # Per-head state (N+1)*c*dh matches RoLA's nc*dqk*(dv+1)=156*nc at dh=12 -> (N+1)*c = 13*nc.
        a_lrs = arm.get("lrs", def_lrs)
        for seed in seeds:
            for nc in (8, 16, 32, 64):
                budget = 13 * nc
                native = (4, max(2, round(budget / 5)))
                par = (max(1, nc), max(2, round(budget / (nc + 1))))
                for (N, c), tag, topks in ((native, "native", (1, 2)), (par, "parallel", (2,))):
                    for tk in topks:
                        kernel = dict(name="zoology.mixers.sse.SSE",
                                      kwargs=dict(n_heads=4, num_partitions=N, num_rows=c, topk=tk))
                        st = 4 * (N + 1) * c * 12
                        for lr in a_lrs:
                            add(kernel, f"grid-sse-{tag}t{tk}-nc{nc}_N{N}c{c}_st{st}_lr{lr:.0e}_s{seed}", lr, seed)

    def attn(arm):
        a_lrs = arm.get("lrs", def_lrs)
        for seed in seeds:
            for lr in a_lrs:
                add(C.mha, f"grid-mha_lr{lr:.0e}_s{seed}", lr, seed)

    builders = {"routed": routed, "baselines": baselines, "sse_ladder": sse_ladder, "attn": attn}

    tiers = spec["tiers"]
    sel = os.environ.get("GRID_TIERS", "all")
    chosen = list(tiers) if sel == "all" else [t for t in tiers if t in sel]
    for t in chosen:                       # preserve declaration order (== run order)
        for arm in tiers[t]:
            builders[arm["build"]](arm)
    return configs, envs


def load_spec(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_configs_and_envs(spec_path):
    return build_configs(load_spec(spec_path))
