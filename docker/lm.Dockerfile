# rola-bench LM image — FROM the project-agnostic fleet base.
#
# Same pattern as mqar.Dockerfile but for the language-modeling benchmark: rola + fla-rola + the HF
# stack + the vendored flame harness (in rola_bench/lm). zoology (MQAR baselines) is NOT installed.
#
# SCAFFOLD: the LM box-side result-row contract is not yet wired (see rola_bench/lm/job.py). This
# image builds the LM runtime; FLEET_RUN_CMD is a placeholder until train_lm emits {run_id, ok} rows.
FROM blakeresearch/fleet-base:v1

ENV DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline WANDB_CONSOLE=off

# LM stack: HF + canonical FLA (the rola_hf model imports fla.layers) + datasets/eval.
RUN pip install --no-cache-dir fla-core==0.4.2 flash-linear-attention==0.4.2 \
        transformers accelerate datasets evaluate einops pydantic numpy tqdm wandb rich pyyaml scipy

COPY rola       /opt/rola
COPY fla-rola   /opt/fla-rola
COPY rola-bench /opt/rola-bench
RUN pip install --no-cache-dir --no-deps -e /opt/rola \
 && pip install --no-cache-dir --no-deps -e /opt/fla-rola \
 && pip install --no-cache-dir --no-deps -e /opt/rola-bench

RUN python -c "import torch, fla.layers, fla_rola, rola, fleet; import rola_bench.lm.job; \
print('rola-bench lm image OK; torch', torch.__version__)"

ENV FLEET_WORKDIR=/opt/rola-bench/rola_bench/lm \
    FLEET_VERIFY_CMD="" \
    FLEET_RUN_CMD="python train_lm.py --arm \$RUN_IDS --results \$FLEET_RESULTS"

WORKDIR /opt/rola-bench/rola_bench/lm
# CMD inherited from fleet-base: python -u -m fleet.box_server
