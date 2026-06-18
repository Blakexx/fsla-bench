"""Limited norm x symmetry ablation on MQAR (the paper's normalization/routing equivalence section).

Sweeps the 3 normalized RoLA-RLA modes x {sym, asym} routing across 3 state rungs, each with a small
LR sweep so the comparison is AT each cell's optimum (the legacy grid used a single fixed LR — the
"equivalence at optimum" claim needs per-cell optima). Inter-RoLA only; canonical MQAR geometry
(d_qk=d_v=12, n_heads=4, d_model=128), so the only varying factors are norm mode, routing, nc, LR.

  3 norm (global / per_state / kappa) x 2 routing (sym=tied / asym=untied) x nc {16,64,256} x 4 LR
  = 72 cells, seed 1337. (`raw`/GLA-unnormalized is a separate control, omitted.)
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from rola_bench.mqar.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from rola_bench.mqar.experiments.rola_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

import os
SEED, TEST_BS = 1337, 8
# nc rungs — env-overridable (NORMSYM_NCS=64 runs just the mid rung: 6 variants x 1 nc x 4 LR = 24 cells).
NCS = [int(x) for x in os.environ.get("NORMSYM_NCS", "16,64,256").split(",") if x]
LRS = [1e-3, 3e-3, 1e-2, 3e-2]                  # bracket the known RLA optimum (~1e-2) to find per-cell best
# (tag, rola_instance): the 6 normalized norm x routing variants.
VARIANTS = [("g_asym",  "rola-rla-asym"),       ("g_sym",  "rola-rla-sym"),
            ("ps_asym", "rola-rla-asym-ps"),    ("ps_sym", "rola-rla-sym-ps"),
            ("k_asym",  "rola-rla-kappa-asym"), ("k_sym",  "rola-rla-kappa-sym")]

configs, configs_envs = [], []
for tag, inst in VARIANTS:
    for nc in NCS:
        for LR in LRS:
            kw = rola_instance(inst, d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
            kernel = dict(name="zoology.mixers.rola.RoLAMixer", kwargs=kw)
            run_id = f"normsym-{tag}-nc{nc}_lr{LR:.0e}_s{SEED}"
            configs.append(TrainConfig(
                data=DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                                batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext"),
                model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                                  state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                                  d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
                logger=LoggerConfig(project_name="rola-normsym-resweep", entity=""),
                max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=SEED, run_id=run_id,
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            configs_envs.append({"EVAL_EVERY_N": "10"})

assert len(configs) == len(VARIANTS) * len(NCS) * len(LRS), len(configs)


def load_configs_and_envs():
    return configs, configs_envs


if __name__ == "__main__":
    print(f"{len(configs)} cells: {len(VARIANTS)} variants x {len(NCS)} nc x {len(LRS)} LR")
    for c in configs[:6]:
        print("  ", c.run_id)
