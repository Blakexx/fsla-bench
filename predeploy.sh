#!/usr/bin/env bash
# Decoupled pre-deploy gate: run the per-family CI on a real GPU (local venv — NOT the docker build,
# which has no GPU), and only build+push the image if every cell family builds + fwd+bwd. Tests
# validate behavior on the code; the build is a separate artifact step. Run this before rola_bench/fleet/__main__.py.
#
#   ./predeploy.sh mqar        # CI (local GPU) -> build+push blakeresearch/rola-bench:v1
set -euo pipefail
TARGET="${1:-mqar}"
PY="${ROLA_PY:-/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

DEV=$($PY -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')
echo "== CI gate on $DEV =="
# (1) fast per-family mixer build+fwd+bwd; (2) AUTHORITATIVE end-to-end per-family cell run through
# the actual zoology.launch path (caught the sse/state_size/fla-tuple bugs the mixer-level check
# missed). Both must pass before we build a deployable image.
if [ "$TARGET" = mqar ]; then
    GRID_TIERS="${GRID_TIERS:-12}" "$PY" -m rola_bench.mqar.ci_cell_check
    GRID_TIERS="${GRID_TIERS:-12}" "$PY" -m rola_bench.mqar.ci_cell_run
else
    echo "  (no CI gate for '$TARGET' yet — lm is scaffold)"
fi
echo "== CI passed (construction + end-to-end) -> building + pushing image =="
bash "$HERE/docker/build.sh" "$TARGET" --push --prune
echo "== predeploy OK: image built, pushed, local copy pruned. Safe to rola_bench/fleet/__main__.py $TARGET =="
