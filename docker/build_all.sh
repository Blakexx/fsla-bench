#!/usr/bin/env bash
# One command to build + push everything the perf/rank fleets need (no GPU required — pure build).
#
#   bash rola-bench/docker/build_all.sh
#
# Builds, in order:
#   1. blakeresearch/fleet-base:v2   — fleet base + the box_server base64-PAYLOAD patch (rank ships
#                                      checkpoints this way). Context = the fleet repo.
#   2. blakeresearch/rola-bench:v2   — FROM fleet-base:v2; the whole rola_bench suite (mqar+perf+rank
#                                      modules + evaluator). build.sh stages clean sources.
# then pushes both to Docker Hub (creds via Docker Desktop). v1 images are left untouched, so the
# live MQAR sweep keeps running on the proven rola-bench:v1.
#
# Requires the Docker daemon (start Docker Desktop on Windows first). After this finishes:
#   cd /home/blake/rola-bench
#   PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
#   $PY rola_bench/fleet/__main__.py perf --boxes 2      # kernel-efficiency on a clean cloud GPU
#   $PY rola_bench/fleet/__main__.py rank --boxes 8      # post-hoc rank over every checkpoint (ships .pt via payload)
set -euo pipefail

V="${IMG_VERSION:-v2}"
FLEET="/home/blake/fleet"
RB="/home/blake/rola-bench"

# The sub-scripts (build_base.sh/build.sh) invoke bare `docker`, which in WSL is Docker Desktop's
# integration shim. Require it to reach a running daemon; otherwise tell the user exactly what to fix.
docker version >/dev/null 2>&1 || {
  echo "ERROR: 'docker' can't reach a daemon in WSL." >&2
  echo "  1) Start Docker Desktop on Windows, and" >&2
  echo "  2) ensure WSL integration is ON for this distro (Docker Desktop > Settings > Resources > WSL integration)." >&2
  echo "Then re-run: bash rola-bench/docker/build_all.sh" >&2
  exit 1; }

echo "==> [1/2] fleet base  blakeresearch/fleet-base:$V"
bash "$FLEET/docker/build_base.sh" "blakeresearch/fleet-base:$V"
docker push "blakeresearch/fleet-base:$V"

echo "==> [2/2] suite image blakeresearch/rola-bench:$V  (mqar+perf+rank)"
IMG_VERSION="$V" bash "$RB/docker/build.sh" mqar --push

echo "==> done. pushed fleet-base:$V + rola-bench:$V"
echo "    launch:  python -m rola_bench.fleet perf --boxes 2   |   python -m rola_bench.fleet rank --boxes 8"
