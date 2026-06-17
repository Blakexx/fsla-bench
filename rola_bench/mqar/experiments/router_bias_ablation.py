"""MQAR ablation: does a router BIAS help RoLA-RLA-kappa at nc128?

Motivation: a bias-free router (z = W·x) routes x=0 to a UNIFORM distribution — it cannot
express a content-independent default/"garbage-bin" state. A bias (z = W·x + b) makes one state
the zero-content default (content perturbs off it), à la attention sinks / softmax-off-by-one.
Question: does that help the hardest recall slice, where filler-token pollution bites?

Target cell: routed RLA-kappa, nc128, the BEST LR (3e-3: overall 0.997, but kv1024 only 0.923 —
~8% headroom on the hardest OOD slice). Paired bias={False,True} x seeds={1337,2024}, everything
else byte-identical to canonical_mqar_grid's routed() so the ONLY difference is router_bias.

Collision-safety: tag `rlabias0`/`rlabias1` (NOT the canonical `rla` tag) so run_ids never collide
with the preserved no-bias nc128 rows. router_bias is injected into the mixer kwargs; it reaches
RoLA.__init__ (added in rola.py) — REQUIRES the v13+ image (v12 predates the flag and would
silently swallow it via **kwargs, making bias=True a no-op).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from rola_bench.mqar.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from rola_bench.mqar.experiments.rola_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola_bench.mqar.experiments.canonical_baselines import REF
from rola import rola_instance

NC, LR, TEST_BS = 128, 3e-3, 8
SEEDS = [1337, 2024]
INSTANCE = "rola-rla-kappa-asym"

configs, configs_envs = [], []


def add(kernel, run_id, seed):
    configs.append(TrainConfig(
        data=DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                        batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext"),
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="canonical-mqar-grid", entity=""),
        max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=seed, run_id=run_id,
        early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"]))
    configs_envs.append({"EVAL_EVERY_N": "10"})


for seed in SEEDS:
    for bias in (False, True):
        kw = rola_instance(INSTANCE, d_qk=12, d_v=12, num_chunks=NC, n_heads=4)
        kw["router_bias"] = bias                       # the one knob under test
        kernel = dict(name="zoology.mixers.rola.RoLAMixer", kwargs=kw)
        tag = f"rlabias{int(bias)}"
        add(kernel, f"grid-routed-{tag}-nc{NC}_st{REF(NC)}_lr{LR:.0e}_s{seed}", seed)


def load_configs_and_envs():
    return configs, configs_envs


if __name__ == "__main__":
    print(f"{len(configs)} bias-ablation cells:")
    for c in configs:
        print(" ", c.run_id)
