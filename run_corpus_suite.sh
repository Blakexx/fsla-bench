#!/bin/bash
# FULLY-SEQUENTIAL corpus-suite driver: TinyStories -> WikiText-103 -> text8 -> PG-19, ONE fleet at a
# time on the shared rola-lm label (sequential => no label collision, no race). Fixed RoLA geometry
# (d=256, 8 heads, d_v=32, d_qk=16 -> output n_h*d_v=256=d_model; d_v=32 avoids the d_v=64 backward
# smem OOM, task #115); only nc + L + tokenizer move so RoLA state stays ½ attn KV at each context
# length. Each corpus gets its OWN results+ckpt dir (run_id is <arm>_s<seed> -> would collide otherwise).
set -u
VENV=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
cd /home/blake/rola-bench
IMG=blakeresearch/rola-lm:v13
ARMS="rola-rla-kappa-asym,rola-rla-kappa-sym,rola-gla-kappa-asym,rola-gla-kappa-sym,attn"
# Count live boxes through the ACTIVE provider seam ($FLEET_PROVIDER, default vast) rather than
# importing fleet.vast directly -- the label check must work whichever provider the fleet runs on.
nboxes() { $VENV -c "import sys;sys.path.insert(0,'/home/blake/fleet');import fleet;print(len(fleet.provider().instances_with_label('rola-lm')))" 2>/dev/null; }

echo "[driver $(date +%H:%M)] start; ensuring rola-lm label is free..."
while [ "$(nboxes)" != "0" ]; do sleep 30; done

run_corpus() {  # name dataset config L nc batch seeds
  local name=$1 dataset=$2 cfg=$3 L=$4 nc=$5 bs=$6 seeds=$7
  echo "[driver $(date +%H:%M)] === $name : L=$L nc=$nc batch=$bs seeds=$seeds (tok/step=$((L*bs))) ==="
  mkdir -p "/mnt/e/rola/rola_data/lm_ckpts_$name"
  env ROLA_LM_IMAGE=$IMG \
    ROLA_LM_RESULTS="/home/blake/rola-bench/results/$name" \
    ROLA_LM_RUNTIME="/home/blake/rola-bench/runtime/$name" \
    ROLA_LM_CKPTS="/mnt/e/rola/rola_data/lm_ckpts_$name" \
    LM_ARMS="$ARMS" LM_SEEDS="$seeds" LM_DATASET="$dataset" LM_DATASET_CONFIG="$cfg" \
    LM_HIDDEN=256 LM_LAYERS=6 LM_ROLA_NC=$nc LM_ROLA_DQK=16 LM_ROLA_DV=32 LM_ROLA_NH=8 LM_ATTN_HEADS=4 \
    LM_MAX_POS=2048 LM_BLOCK_SIZE=$L LM_BATCH_SIZE=$bs LM_GRAD_ACCUM=1 LM_LR=3e-4 \
    LM_MAX_STEPS=15000 LM_SAVE_STEPS=1500 LM_GRAD_CKPT=0 LM_LIVE_PULL=900 \
    OFFER_QUERY="gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500" \
    $VENV rola_bench/fleet/__main__.py lm --boxes 8 > "/tmp/lm_$name.log" 2>&1
  echo "[driver $(date +%H:%M)] $name finished (exit $?)"
  while [ "$(nboxes)" != "0" ]; do sleep 30; done      # fully drained before next corpus
}

# TinyStories first (reuses results/tinystories_final: 5 attn cells already done -> only 20 RoLA re-run)
run_corpus tinystories_final "roneneldan/TinyStories" "" 512  32  32 "1337,2024,11,42,7"
run_corpus wikitext103       Salesforce/wikitext    wikitext-103-raw-v1 1024 64 16 "1337,2024,11"
run_corpus text8             text8                  ""                  2048 128 8 "1337,2024,11"
run_corpus pg19              pg19                   ""                  2048 128 8 "1337,2024,11"
echo "[driver $(date +%H:%M)] ALL CORPORA DONE"
