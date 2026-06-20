"""Generic MQAR config generator: a declarative spec -> zoology (configs, configs_envs).

One generator owns all the grid LOGIC (cell builders, the baseline shape/state-match solve, the SSE
ladder, run_id naming); an experiment is just a YAML spec declaring the axes (backbone, data,
lrs/seed, and an `arms` list grouped into tiers). Replaces the one-off `*.py` experiment modules.

Spec schema (see experiments/canonical.yaml):
  project_name: str
  backbone: {d_model, n_layers, vocab, l_max}
  data:     {preset, train_batch, test_batch, cache_dir}
  train:    {max_epochs, weight_decay, seed, early_stopping_threshold, early_stopping_metric,
             slice_keys, eval_every}
  lrs:  [..]            # swept for every arm
  ncs:  [..]            # state-rung ladder
  tiers: {"<id>": [arm, ...], ...}    # selected by the GRID_TIERS env (default all)
  arm kinds:
    {build: routed,    instance, tag, d_qk, d_v, n_heads}
    {build: baselines, methods: [..], shapes: [..]}
    {build: sse_ladder}
    {build: attn}
"""
import os

from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig

from rola_bench.mqar import common as C
from rola_bench.mqar.experiments.canonical_baselines import baseline_cell, REF
from rola import rola_instance


def build_configs(spec):
    """Expand a spec dict into (configs, configs_envs)."""
    bb = {**C.DEFAULTS, **spec.get("backbone", {})}   # spec overrides the library defaults
    dat, tr = spec["data"], spec["train"]
    lrs, ncs = spec["lrs"], spec["ncs"]
    seed = tr["seed"]
    l_max = bb["l_max"]
    data_cfg = C.make_data(dat["preset"], bb["vocab"], dat["train_batch"], dat["test_batch"],
                           dat["cache_dir"])
    env = {"EVAL_EVERY_N": str(tr["eval_every"])} if tr.get("eval_every") else {}

    configs, envs = [], []

    def add(kernel, run_id, lr):
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
        for nc in ncs:
            kw = rola_instance(arm["instance"], d_qk=arm["d_qk"], d_v=arm["d_v"],
                               num_chunks=nc, n_heads=arm["n_heads"])
            kernel = dict(name="zoology.mixers.rola.RoLAMixer", kwargs=kw)
            for lr in lrs:
                add(kernel, f"grid-routed-{arm['tag']}-nc{nc}_st{REF(nc)}_lr{lr:.0e}_s{seed}", lr)

    def baselines(arm):
        for method in arm["methods"]:
            for shape in arm["shapes"]:
                for nc in ncs:
                    cell = baseline_cell(method, shape, nc)
                    if cell is None:
                        continue
                    kernel, st = cell
                    for lr in lrs:
                        add(kernel, f"grid-{method}-{shape}-nc{nc}_st{st}_lr{lr:.0e}_s{seed}", lr)

    def sse_ladder(arm):
        # SSE scaled the way it is strongest: NATIVE (few big partitions) + PARALLEL (N~nc small).
        # Per-head state (N+1)*c*dh matches RoLA's nc*dqk*(dv+1)=156*nc at dh=12 -> (N+1)*c = 13*nc.
        for nc in (8, 16, 32, 64):
            budget = 13 * nc
            native = (4, max(2, round(budget / 5)))
            par = (max(1, nc), max(2, round(budget / (nc + 1))))
            for (N, c), tag, topks in ((native, "native", (1, 2)), (par, "parallel", (2,))):
                for tk in topks:
                    kernel = dict(name="zoology.mixers.sse.SSE",
                                  kwargs=dict(n_heads=4, num_partitions=N, num_rows=c, topk=tk))
                    st = 4 * (N + 1) * c * 12
                    for lr in lrs:
                        add(kernel, f"grid-sse-{tag}t{tk}-nc{nc}_N{N}c{c}_st{st}_lr{lr:.0e}_s{seed}", lr)

    def attn(arm):
        for lr in lrs:
            add(C.mha, f"grid-mha_lr{lr:.0e}_s{seed}", lr)

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
