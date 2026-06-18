#!/bin/bash
# Sequential corpus-suite driver: after TinyStories (rola-lm) drains, run WikiText-103 -> text8 -> PG-19
# one fleet at a time (same label, so sequential = no collision). Fixed RoLA geometry (d=256, 8 heads,
# d_v=32, d_qk=16 -> output n_h*d_v=256=d_model; d_v=32 avoids the d_v=64 backward smem OOM, task #115);
# only nc + L + tokenizer move so RoLA state stays ½ attn KV at each context length.
# Each corpus gets its OWN results+ckpt dir (run_id is <arm>_s<seed> -> would cross-contaminate otherwise).
set -u
VENV=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
cd /home/blake/rola-bench
IMG=blakeresearch/rola-lm:v13
ARMS="rola-rla-kappa-asym,rola-rla-kappa-sym,rola-gla-kappa-asym,rola-gla-kappa-sym,attn"
SEEDS="1337,2024,11"          # 3 seeds for the heavier corpora (TinyStories already has 5)
nboxes() { $VENV -c "import sys;sys.path.insert(0,'/home/blake/fleet');from fleet import vast;print(len(vast.instances_with_label('rola-lm')))" 2>/dev/null; }

echo "[driver $(date +%H:%M)] waiting for TinyStories (rola-lm) to drain before reusing the label..."
while [ "$(nboxes)" != "0" ]; do sleep 60; done
echo "[driver $(date +%H:%M)] rola-lm free; starting corpus suite."

run_corpus() {  # name dataset config L nc batch
  local name=$1 dataset=$2 cfg=$3 L=$4 nc=$5 bs=$6
  echo "[driver $(date +%H:%M)] === $name : L=$L nc=$nc batch=$bs (tok/step=$((L*bs))) ==="
  rm -rf "/mnt/d/rola_data/lm_ckpts_$name"; mkdir -p "/mnt/d/rola_data/lm_ckpts_$name"
  env ROLA_LM_IMAGE=$IMG \
    ROLA_LM_RESULTS="/home/blake/rola-bench/results/$name" \
    ROLA_LM_RUNTIME="/home/blake/rola-bench/runtime/$name" \
    ROLA_LM_CKPTS="/mnt/d/rola_data/lm_ckpts_$name" \
    LM_ARMS="$ARMS" LM_SEEDS="$SEEDS" LM_DATASET="$dataset" LM_DATASET_CONFIG="$cfg" \
    LM_HIDDEN=256 LM_LAYERS=6 LM_ROLA_NC=$nc LM_ROLA_DQK=16 LM_ROLA_DV=32 LM_ROLA_NH=8 LM_ATTN_HEADS=4 \
    LM_MAX_POS=2048 LM_BLOCK_SIZE=$L LM_BATCH_SIZE=$bs LM_GRAD_ACCUM=1 LM_LR=3e-4 \
    LM_MAX_STEPS=15000 LM_SAVE_STEPS=1500 LM_GRAD_CKPT=0 LM_LIVE_PULL=900 \
    OFFER_QUERY="gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500" \
    $VENV launch.py lm --boxes 8 > "/tmp/lm_$name.log" 2>&1
  echo "[driver $(date +%H:%M)] $name finished (exit $?)"
  while [ "$(nboxes)" != "0" ]; do sleep 30; done      # ensure fully drained before next corpus
}

run_corpus wikitext103 wikitext wikitext-103-raw-v1 1024 64 16
run_corpus text8        text8   ""                   2048 128 8
run_corpus pg19         pg19    ""                   2048 128 8
echo "[driver $(date +%H:%M)] ALL CORPORA DONE"
