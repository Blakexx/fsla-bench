#!/usr/bin/env bash
# Build a rola-bench plug-in image (FROM blakeresearch/fleet-base). Stages clean package sources into
# one context — only runtime code (no .git/__pycache__/data/results/docs), so images are lean and
# publication-safe. Requires the fleet base first: fleet/docker/build_base.sh.
#
#   build.sh mqar                 # build blakeresearch/rola-bench:v1   (MQAR benchmark)
#   build.sh lm                   # build blakeresearch/rola-lm:v1      (LM benchmark, scaffold)
#   build.sh mqar --push          # build, then push to the registry
#   build.sh mqar --push --prune  # build, push, then remove the local image (keep base layers)
#
# Process: build local -> upload -> nuke unneeded. --prune drops the freshly-built plug-in image
# locally after a successful push (the fleet base + CUDA base stay cached for the next build).
set -euo pipefail

# Fail loud if docker isn't reachable — a background/non-login shell can lack /usr/bin on PATH, and a
# bare `docker build` then dies with "command not found" mid-script. (Bit us once: the failure got
# masked by a `| tail` pipe, so the build "succeeded" with exit 0 and the image never got pushed.)
command -v docker >/dev/null 2>&1 || { echo "FATAL: docker not on PATH (have you got /usr/bin?)" >&2; exit 3; }
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not reachable" >&2; exit 3; }

TARGET="${1:?usage: build.sh <mqar|lm> [--push] [--prune]}"; shift || true
PUSH=0; PRUNE=0
for a in "$@"; do case "$a" in --push) PUSH=1;; --prune) PRUNE=1;; esac; done

# IMG_VERSION pins the tag (default v2: v1 = the proven MQAR image, left untouched; v2 adds the
# perf/rank code + the box_server base64-payload base). The mqar image is the WHOLE rola_bench
# suite (perf+rank+mqar modules); perf/rank select their runner via FLEET_RUN_CMD at launch.
V="${IMG_VERSION:-v2}"
case "$TARGET" in
  mqar) TAG="blakeresearch/rola-bench:$V"; DF="mqar.Dockerfile"; WITH_ZOO=1;;
  lm)   TAG="blakeresearch/rola-lm:$V";    DF="lm.Dockerfile";   WITH_ZOO=0;;
  *) echo "unknown target '$TARGET' (mqar|lm)"; exit 2;;
esac

CLA="${ROLA_CLA_SRC:-/mnt/c/Users/Blake/Documents/VSCode/CLA}"
FLA="${ROLA_FLA_SRC:-/home/blake/flash-linear-attention-rola}"
ZOO="${ROLA_ZOO_SRC:-/home/blake/zoology}"
RB="${ROLA_BENCH_SRC:-/home/blake/rola-bench}"
HERE="$(cd "$(dirname "$0")" && pwd)"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
RS=(rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' --exclude='build')

# rola.py is deleted — the model is fla-rola, its helpers moved into rola-bench. Nothing imports
# `rola`, so it's no longer staged/installed in the image.
echo "staging fla-rola (fork)…"
"${RS[@]}" "$FLA/" "$STAGE/fla-rola/"

if [ "$WITH_ZOO" = 1 ]; then
  echo "staging zoology (MQAR baselines package + packaging)…"
  mkdir -p "$STAGE/zoology"
  "${RS[@]}" "$ZOO/zoology" "$STAGE/zoology/"
  cp "$ZOO/pyproject.toml" "$STAGE/zoology/" 2>/dev/null || true
  cp "$ZOO/setup.py" "$STAGE/zoology/" 2>/dev/null || true
fi

# rola-bench: the whole suite package, minus the other benchmark's module (lean per-image) + legacy.
echo "staging rola-bench ($TARGET)…"
mkdir -p "$STAGE/rola-bench"
OTHER=$([ "$TARGET" = mqar ] && echo lm || echo mqar)
"${RS[@]}" --exclude="mqar/experiments/legacy" --exclude="$OTHER" "$RB/rola_bench" "$STAGE/rola-bench/"
cp "$RB/pyproject.toml" "$STAGE/rola-bench/"

cp "$HERE/$DF" "$STAGE/Dockerfile"
echo "build context staged at $STAGE ($(du -sh "$STAGE" | cut -f1)); building $TAG…"
docker build -t "$TAG" "$STAGE"
echo "built $TAG"

if [ "$PUSH" = 1 ]; then
  echo "pushing $TAG…"; docker push "$TAG"; echo "pushed $TAG"
fi
if [ "$PRUNE" = 1 ]; then
  if [ "$PUSH" = 1 ]; then
    echo "pruning local image $TAG (base layers kept)…"; docker image rm "$TAG" >/dev/null 2>&1 || true
  else
    echo "skip --prune: not pushed (refusing to drop an un-uploaded image)"
  fi
  docker builder prune -f >/dev/null 2>&1 || true
fi
