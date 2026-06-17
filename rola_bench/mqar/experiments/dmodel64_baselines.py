"""d_model=64 matched-state MONOLITH baselines for the nc256 comparison (pairs with
dmodel64_ablation, the routed RoLA d64 nc256 cell).

Tests whether ROUTING beats the monolith forms at tiny width: RLA-wide and Hedgehog-square,
each matched to the routed RoLA nc256 recurrent state (S = REF(256) = 159744), built at
d_model=64. Same canonical baseline builder as the main grid, but DMODEL overridden to 64
(it's read at call time; the state target S is d_model-independent, so the match holds — the
key/feature projection just widens relative to the 64-wide backbone). 5-LR sweep, seed 1337.

Distinct run_id tag (grid-{method}-{shape}-d64-*) so no collision with the d128 canonical rows.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from rola_bench.mqar.experiments.rla_sweep import wrap_hybrid, VOCAB
from rola_bench.mqar.experiments.rola_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
import rola_bench.mqar.experiments.canonical_baselines as cb

D_MODEL, NC, SEED, TEST_BS = 64, 256, 1337, 8
cb.DMODEL = D_MODEL                              # build baseline projections at d_model=64 (default is 128)
_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
CELLS = [("rla", "wide"), ("hedgehog", "square")]

configs, configs_envs = [], []


def add(kernel, run_id, lr):
    configs.append(TrainConfig(
        data=DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                        batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext"),
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="canonical-mqar-grid", entity=""),
        max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED, run_id=run_id,
        early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"]))
    configs_envs.append({"EVAL_EVERY_N": "10"})


for method, shape in CELLS:
    built = cb.baseline_cell(method, shape, NC)
    assert built is not None, f"{method}-{shape} undefined at nc{NC}"
    kernel, S = built
    for lr in _GRID:
        rid = f"grid-{method}-{shape}-d{D_MODEL}-nc{NC}_st{S}_lr{lr:.0e}_s{SEED}"
        add(kernel, rid, lr)


def load_configs_and_envs():
    return configs, configs_envs


if __name__ == "__main__":
    print(f"{len(configs)} d64 baseline cells (cb.DMODEL={cb.DMODEL}):")
    for c in configs:
        print("  ", c.run_id)
