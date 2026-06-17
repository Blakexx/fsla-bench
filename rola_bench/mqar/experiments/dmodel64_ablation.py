"""Quick MQAR ablation: RoLA-RLA-kappa-asym at a TINY d_model=64 with nc=256 states.

Tests RoLA's "best form" (task #111): can a small backbone (d_model=64) still solve MQAR when given
many routed states (nc=256)? The recurrent state (n_h*nc*d_qk*d_v) is large relative to d_model, so
this probes whether routing capacity decouples from model width. Single cell, best LR (3e-3), seed
1337, otherwise byte-identical to the canonical routed RLA-kappa cell except d_model 128 -> 64.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from rola_bench.mqar.experiments.rla_sweep import wrap_hybrid, VOCAB
from rola_bench.mqar.experiments.rola_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

D_MODEL, NC, SEED, TEST_BS = 64, 256, 1337, 8
_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]      # full LR sweep (same grid as the canonical routed cells)
INSTANCE = "rola-rla-kappa-asym"

configs, configs_envs = [], []

for LR in _GRID:
    kw = rola_instance(INSTANCE, d_qk=12, d_v=12, num_chunks=NC, n_heads=4)
    kernel = dict(name="zoology.mixers.rola.RoLAMixer", kwargs=kw)
    run_id = f"grid-routed-rla-d{D_MODEL}-nc{NC}_lr{LR:.0e}_s{SEED}"
    configs.append(TrainConfig(
        data=DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                        batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext"),
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="canonical-mqar-grid", entity=""),
        max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=SEED, run_id=run_id,
        early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"]))
    configs_envs.append({"EVAL_EVERY_N": "10"})


def load_configs_and_envs():
    return configs, configs_envs


if __name__ == "__main__":
    print(f"{len(configs)} cell: {[c.run_id for c in configs]} d_model={D_MODEL} nc={NC}")
