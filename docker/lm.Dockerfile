# rola-bench LM image — FROM the project-agnostic fleet base.
#
# Same pattern as mqar.Dockerfile but for the language-modeling benchmark: rola + fla-rola + the HF
# stack + the vendored flame harness (in rola_bench/lm). zoology (MQAR baselines) is NOT installed.
#
# The box-side contract is wired: run_lm.py maps RUN_IDS ("<arm>_s<seed>") -> train_lm.py, which
# appends {run_id, ok, params_m, state_floats, ppl, ...} to $FLEET_RESULTS + saves the ckpt dir.
FROM blakeresearch/fleet-base:v1

ENV DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline WANDB_CONSOLE=off

# LM stack: HF + canonical FLA (the rola_hf model imports fla.layers) + datasets/eval.
RUN pip install --no-cache-dir fla-core==0.4.2 flash-linear-attention==0.4.2 \
        transformers accelerate datasets evaluate einops pydantic numpy tqdm wandb rich pyyaml scipy lm-eval
# flash-attn: ONLY the softmax-attention CEILING arm (fla.layers.Attention) needs it; the 4
# bounded arms (RoLA x3, GDN) do not. Try a prebuilt wheel first; fall back to a source build
# (needs the CUDA toolkit in the base). If it can't install, the `attn` arm fails loudly while the
# bounded arms still run — so the image build does NOT hard-fail on flash-attn.
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "WARN: flash-attn not installed — the attn ceiling arm will be unavailable on this image"

COPY fla-rola   /opt/fla-rola
COPY rola-bench /opt/rola-bench
# --no-build-isolation: build against the image's torch (some setup.py's import torch at build).
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/fla-rola \
 && pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/rola-bench

# Verify the BOX-side path only (run_lm -> train_lm -> arms + deps incl. flash-attn for the attn arm).
# NOT rola_bench.lm.job — that's DISPATCHER-side (constructs fleet.Job with fields the box's baked
# fleet-base fleet may predate); the box runs run_lm and never imports the Job.
RUN python -c "import torch, fla.layers, fla_rola, fleet, flash_attn; \
import sys; sys.path.insert(0, '/opt/rola-bench/rola_bench/lm'); import arms, run_lm; \
print('rola-bench lm image OK; torch', torch.__version__, 'flash_attn', flash_attn.__version__, 'arms', len(arms.ARMS))"

ENV FLEET_WORKDIR=/opt/rola-bench/rola_bench/lm \
    FLEET_VERIFY_CMD="python -c 'import torch; assert torch.cuda.is_available(); import run_lm, arms; assert len(arms.ARMS)>=5; print(\"VERIFY OK\")'" \
    FLEET_RUN_CMD="python run_lm.py"

WORKDIR /opt/rola-bench/rola_bench/lm
# CMD inherited from fleet-base: python -u -m fleet.box_server
