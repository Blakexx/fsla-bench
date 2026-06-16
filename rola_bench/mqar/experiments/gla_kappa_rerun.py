"""MQAR re-run: routed-GLA ladder with the KAPPA variant (rola-gla-kappa-asym).

The canonical_mqar_grid routed-GLA ladder used `rola-gla-scalar-sym` — state_norm='raw'
(UNNORMALIZED), sym — while routed-RLA used `rola-rla-kappa-asym` (state_norm='kappa', asym).
This re-runs the GLA ladder with `rola-gla-kappa-asym` (kernel='gla_scalar',
kernel_kwargs={'state_norm':'kappa'} — read gates rescaled by the DECAYED per-state mass, the
SAME kappa construction as RLA), so the routed RLA/GLA pairing is finally consistent.

Collision-safety: the MQAR run_id encodes the TAG + state, NOT the instance. The original ladder
is tagged `gla` (grid-routed-gla-*); this uses a DISTINCT tag `glakappa` (grid-routed-glakappa-*),
so the new run_ids never collide with — and never overwrite/skip — the preserved scalar-sym rows
(the non-norm data is kept for comparison).

Cells are byte-identical to canonical_mqar_grid's routed() (same TRAIN/TEST configs, d_model,
40 epochs, 5-LR grid, seed 1337, d_qk=d_v=12, n_heads=4, state token REF(nc)); only the routed
instance and the run_id tag differ.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from rola_bench.mqar.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from rola_bench.mqar.experiments.rola_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola_bench.mqar.experiments.canonical_baselines import NCS, REF
from rola import rola_instance

SEED, TEST_BS = 1337, 8
_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]      # identical to the canonical grid's routed LR sweep
INSTANCE, TAG = "rola-gla-kappa-asym", "glakappa"

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


for nc in NCS:
    kw = rola_instance(INSTANCE, d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
    kernel = dict(name="zoology.mixers.rola.RoLAMixer", kwargs=kw)
    for lr in _GRID:
        add(kernel, f"grid-routed-{TAG}-nc{nc}_st{REF(nc)}_lr{lr:.0e}_s{SEED}", lr)


def load_configs_and_envs():
    return configs, configs_envs


if __name__ == "__main__":
    print(f"{len(configs)} glakappa cells; sample: {[c.run_id for c in configs[:2]]}")
