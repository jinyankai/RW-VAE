#!/usr/bin/env bash
# v3 pipeline for ONE arm whose tokenizer is already trained (scripts/v3_train_tokenizer.py), on the 8-GPU node:
#   1. in parallel on single GPUs: reconstruction eval of the arm; once per experiment the VA-VAE (init.ckpt)
#      baseline through the same protocol and the FID references of val and train_dev (07 ref-stats)
#   2. posterior cache + DiT normalisation stats (torchrun, NGPU processes)
#   3. LightningDiT training on the posterior (torchrun); extra arguments go to v3_train_dit.py (e.g. --repa)
# Usage (inside tmux):  scripts/run_v3_arm.sh T-c [--repa]
# Env: PY (default .venv/bin/python), NGPU (default 8). Logs: runs/imagenet256_v3/<stage>/... and stdout.
set -euo pipefail
ARM=${1:?usage: scripts/run_v3_arm.sh ARM [v3_train_dit args]}
shift
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
NGPU=${NGPU:-8}
TORCHRUN="$(dirname "$PY")/torchrun --nproc_per_node $NGPU"
C="--config configs/dataset_imagenet256.yaml --config configs/baseline_common.yaml --config configs/v3.yaml \
--config configs/server_a800.yaml --config configs/v3_server.yaml"
EXP=runs/imagenet256_v3

echo "[$(date '+%F %T')] stage 1: recon eval + references"
pids=()
CUDA_VISIBLE_DEVICES=0 $PY scripts/v3_evaluate.py recon $C --arm "$ARM" --overwrite & pids+=($!)
if [ ! -f $EXP/recon/init_baseline/recon.json ]; then
  CUDA_VISIBLE_DEVICES=1 $PY scripts/v3_evaluate.py recon $C --arm "$ARM" --baseline --overwrite & pids+=($!)
fi
if [ ! -f $EXP/fid_ref/val.npz ]; then
  CUDA_VISIBLE_DEVICES=2 $PY scripts/07_evaluate.py ref-stats $C --split val --overwrite & pids+=($!)
fi
if [ ! -f $EXP/fid_ref/train_dev.npz ]; then
  CUDA_VISIBLE_DEVICES=3 $PY scripts/07_evaluate.py ref-stats $C --split train_dev --overwrite & pids+=($!)
fi
for p in "${pids[@]}"; do wait "$p"; done

echo "[$(date '+%F %T')] stage 2: posterior cache"
$TORCHRUN scripts/v3_cache_posterior.py $C --arm "$ARM" --resume

echo "[$(date '+%F %T')] stage 3: DiT"
$TORCHRUN scripts/v3_train_dit.py $C --arm "$ARM" "$@"
echo "[$(date '+%F %T')] done"
