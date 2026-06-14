#!/usr/bin/env bash
# Decoupled pre-deploy gate: run the per-family CI on a real GPU (local venv — NOT the docker build,
# which has no GPU), and only build+push the image if every cell family builds + fwd+bwd. Tests
# validate behavior on the code; the build is a separate artifact step. Run this before launch.py.
#
#   ./predeploy.sh mqar        # CI (local GPU) -> build+push blakeresearch/rola-bench:v1
set -euo pipefail
TARGET="${1:-mqar}"
PY="${ROLA_PY:-/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== CI: per-family build+fwd+bwd on $($PY -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")') =="
GRID_TIERS="${GRID_TIERS:-12}" "$PY" -m rola_bench.${TARGET}.ci_cell_check
echo "== CI passed -> building + pushing image =="
bash "$HERE/docker/build.sh" "$TARGET" --push --prune
echo "== predeploy OK: image built, pushed, local copy pruned. Safe to launch.py $TARGET =="
