#!/usr/bin/env bash
# Build a fsla-bench plug-in image (FROM blakeresearch/fleet-base). Stages clean package sources into
# one context — only runtime code (no .git/__pycache__/data/results/docs), so images are lean and
# publication-safe. Requires the fleet base first: fleet/docker/build_base.sh.
#
#   build.sh mqar                 # build blakeresearch/fsla-bench:$IMG_VERSION   (MQAR)
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

TARGET="${1:?usage: build.sh mqar [--push] [--prune]}"; shift || true
PUSH=0; PRUNE=0
for a in "$@"; do case "$a" in --push) PUSH=1;; --prune) PRUNE=1;; esac; done

# IMG_VERSION pins the tag. The mqar image is the fsla_bench package.
V="${IMG_VERSION:-v2}"
case "$TARGET" in
  mqar) TAG="blakeresearch/fsla-bench:$V"; DF="mqar.Dockerfile"; WITH_ZOO=1; BASE_DEFAULT=v2;;
  *) echo "unknown target '$TARGET' (mqar)"; exit 2;;
esac
BASE_TAG="${FLEET_BASE_TAG:-$BASE_DEFAULT}"

# The pins the graph's environments name (MODULE.bazel): the image is built from a checkout at exactly these commits.
ZOO_PIN=8befca418e90ae0bf6fe45fb892d8eae885169f2   # Blakexx/fsla-zoology, branch fsla-pins
CR_PIN=ffdd03b3582b6c82af27fa00f19950c1378e652e    # Blakexx/chromatic-recall, main

# The package sources are this machine's checkouts, named in the environment (no defaults: a checkout lives wherever
# the machine keeps it).
FLA="${FSLA_FLA_SRC:?set FSLA_FLA_SRC to the fla fork checkout (branch fsla)}"
ZOO="${FSLA_ZOO_SRC:?set FSLA_ZOO_SRC to the zoology checkout (Blakexx/fsla-zoology at $ZOO_PIN)}"
CR="${FSLA_RECALL_SRC:?set FSLA_RECALL_SRC to the chromatic-recall checkout (Blakexx/chromatic-recall at $CR_PIN)}"
RB="$(cd "$(dirname "$0")/.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
RS=(rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' --exclude='build')

echo "staging fla (the fork's fsla branch)…"
"${RS[@]}" "$FLA/" "$STAGE/fla/"

if [ "$WITH_ZOO" = 1 ]; then
  echo "staging zoology (the fork's fsla-pins) and chromatic-recall (the recall tasks, the launcher)…"
  [ "$(git -C "$ZOO" rev-parse HEAD)" = "$ZOO_PIN" ] || { echo "FSLA_ZOO_SRC is not at $ZOO_PIN" >&2; exit 2; }
  [ "$(git -C "$CR" rev-parse HEAD)" = "$CR_PIN" ] || { echo "FSLA_RECALL_SRC is not at $CR_PIN" >&2; exit 2; }
  mkdir -p "$STAGE/zoology" "$STAGE/chromatic-recall"
  "${RS[@]}" --exclude='notebooks' --exclude='assets' "$ZOO/" "$STAGE/zoology/"
  "${RS[@]}" "$CR/" "$STAGE/chromatic-recall/"
fi

# fsla-bench: the suite package, less the legacy experiments.
echo "staging fsla-bench ($TARGET)…"
mkdir -p "$STAGE/fsla-bench"
"${RS[@]}" --exclude="mqar/experiments/legacy" "$RB/fsla_bench" "$STAGE/fsla-bench/"
cp "$RB/pyproject.toml" "$STAGE/fsla-bench/"

cp "$HERE/$DF" "$STAGE/Dockerfile"
echo "build context staged at $STAGE ($(du -sh "$STAGE" | cut -f1)); building $TAG from fleet-base:$BASE_TAG…"
docker build --build-arg "FLEET_BASE_TAG=$BASE_TAG" -t "$TAG" "$STAGE"
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
