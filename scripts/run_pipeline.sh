#!/usr/bin/env bash
# RW-VAE v2.1 pipeline on the Linux GPU node (8x A800-80GB). One command per stage; stops on the first error.
#
#   bash scripts/run_pipeline.sh svp  SVP (sanity-validation pilot): random 10-class ImageNet subset through
#                                     00 -> 07; the DiT is trained to overfit it (configs/svp.yaml)
#   bash scripts/run_pipeline.sh imagenet  ImageNet-1k 00 -> 05, then STOPS: R*, w* must be chosen from the 05
#                                     results and frozen in configs/arms.yaml before any 06 run (plan Sec. 8)
#
# Options (flags or environment variables):
#   --train-root DIR  / IMAGENET_TRAIN  source ImageFolder for the SVP subset (default /home/jinyankai/data/imagenet/train)
#   --val-root DIR    / IMAGENET_VAL    (default /home/jinyankai/data/imagenet/val)
#   --svp-out DIR     / SVP_OUT         where make_svp_subset writes the subset (default: configs/svp.yaml's
#                                       dataset roots, /home/jinyankai/data/svp/imagenet_c10_n100)
#   --name NAME       / EXP_NAME        SVP only: experiment name (default svp). Any other name also moves the
#                                       manifest, latent cache and dataset.name to <NAME>-specific paths, so a
#                                       second SVP (e.g. svp_imagenette on another source) never collides with runs/svp
#   --set K=V                           extra config override for every stage (repeatable), e.g.
#                                       --set train.max_steps=8000 to fit 06 into a time budget
#   --to STAGE        / TO              stop after STAGE (default 07), e.g. to time 06 before committing to it
#   --from STAGE      / FROM            skip the stages before STAGE (00 01 02 03 04 05 05b 06 07), e.g. after a
#                                       failure; a finished stage refuses to re-run into its run dir (Run rules)
#   PY (default .venv/bin/python), NGPU (default 8), LOG_DIR (default runs/pipeline_logs/<mode>)
# Per-process logs of the sharded stages go to $LOG_DIR; every stage also writes log.txt in its run dir and one
# record (with git_commit / git_dirty) to manifests/run_registry.jsonl. Prepare ImageNet-1k first with
# scripts/tools/imagenet_parquet_to_folder.py (README).
set -euo pipefail

MODE="${1:-}"
shift || true
IMAGENET_TRAIN="${IMAGENET_TRAIN:-/home/jinyankai/data/imagenet/train}"
IMAGENET_VAL="${IMAGENET_VAL:-/home/jinyankai/data/imagenet/val}"
SVP_OUT="${SVP_OUT:-/home/jinyankai/data/svp/imagenet_c10_n100}"
FROM="${FROM:-00}"
TO="${TO:-07}"
EXP_NAME="${EXP_NAME:-svp}"
EXTRA_SET=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-root) IMAGENET_TRAIN="$2"; shift 2 ;;
        --val-root)   IMAGENET_VAL="$2"; shift 2 ;;
        --svp-out)    SVP_OUT="$2"; shift 2 ;;
        --from)       FROM="$2"; shift 2 ;;
        --to)         TO="$2"; shift 2 ;;
        --name)       EXP_NAME="$2"; shift 2 ;;
        --set)        EXTRA_SET+=(--set "$2"); shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
NGPU="${NGPU:-8}"

C=(--config configs/dataset_imagenet256.yaml --config configs/baseline_common.yaml --config configs/arms.yaml
   --config configs/server_a800.yaml)
case "$MODE" in
    svp)      C+=(--config configs/svp.yaml
                  --set "dataset.train_root=$SVP_OUT/train" --set "dataset.val_root=$SVP_OUT/val")
              if [[ "$EXP_NAME" != svp ]]; then
                  C+=(--set "experiment.name=$EXP_NAME" --set "dataset.name=$EXP_NAME"
                      --set "dataset.manifest=manifests/${EXP_NAME}_seed2026.json"
                      --set "cache.dir=/home/jinyankai/cache/${EXP_NAME}_ldm_f16d32")
              fi
              EXP="runs/$EXP_NAME" ;;
    imagenet) EXP=runs/imagenet256_ldm_f16d32 ;;
    *) echo "usage: $0 svp|imagenet [--train-root DIR] [--val-root DIR] [--svp-out DIR] [--from STAGE]" >&2; exit 2 ;;
esac

C+=("${EXTRA_SET[@]}")
LOG_DIR="${LOG_DIR:-runs/pipeline_logs/${MODE}}"
mkdir -p "$LOG_DIR"

want() { [[ ! "$1" < "$FROM" && ! "$1" > "$TO" ]]; }    # stage ids sort lexically: 00 < 01 < ... < 05 < 05b < 06 < 07
step() { echo; echo "=== [$MODE] $*"; }

# N independent processes (one per GPU, --num-procs/--proc-id), waited for; any failure stops the pipeline.
# Each process gets nproc/N BLAS/OpenMP threads: with the default (all cores each), 8 processes running the
# 03 alignment scan (float64 numpy) thrash the CPU -- measured >200 s instead of 15 s on the 128-core node.
SHARD_THREADS="${SHARD_THREADS:-$(( $(nproc) / NGPU > 0 ? $(nproc) / NGPU : 1 ))}"
sharded() {
    local name="$1"; shift
    local pids=() k fail=0
    for ((k = 0; k < NGPU; k++)); do
        OMP_NUM_THREADS="$SHARD_THREADS" OPENBLAS_NUM_THREADS="$SHARD_THREADS" MKL_NUM_THREADS="$SHARD_THREADS"             "$PY" "$@" --num-procs "$NGPU" --proc-id "$k" --device "cuda:$k" > "$LOG_DIR/${name}_proc$k.log" 2>&1 &
        pids+=($!)
    done
    for k in "${!pids[@]}"; do
        wait "${pids[$k]}" || { echo "$name process $k failed: see $LOG_DIR/${name}_proc$k.log" >&2; fail=1; }
    done
    [[ $fail -eq 0 ]]
}

if want 00; then
    step "00 verify the allocation math (CPU)"
    "$PY" scripts/00_verify_math.py "${C[@]}" --figures
fi

if want 01; then
    if [[ "$MODE" == svp ]]; then
        step "SVP subset: 10 random classes x (100 train + 50 val) -> $SVP_OUT"
        "$PY" scripts/tools/make_svp_subset.py --train-root "$IMAGENET_TRAIN" --val-root "$IMAGENET_VAL" \
            --out "$SVP_OUT" --num-classes 10 --train-per-class 100 --val-per-class 50 --seed 2026
    fi
    step "01 manifest + DATA_CARD"
    "$PY" scripts/01_prepare_manifest.py "${C[@]}"
fi

if want 02; then
    step "02 latent cache: $NGPU shard processes, then finalize + re-encode check"
    sharded 02_cache scripts/02_cache_latents.py "${C[@]}"
    "$PY" scripts/02_cache_latents.py "${C[@]}" --finalize
fi

if want 03; then
    step "03 joint [mu; DINO] statistics: $NGPU shard processes, then merge (alignment gate: exit 2 on FAIL)"
    sharded 03_stats scripts/03_estimate_stats.py "${C[@]}"
    "$PY" scripts/03_estimate_stats.py "${C[@]}" --merge --figures
fi

if want 04; then
    step "04 allocations for every (rule, R, w) + go/no-go"
    "$PY" scripts/04_solve_channel.py "${C[@]}" --figures
fi

if want 05; then
    step "05 generator-free M3 metrics on train_dev (rule selection split)"
    "$PY" scripts/05_eval_recon_probe.py "${C[@]}" --split train_dev --figures
    if [[ "$MODE" == svp ]]; then
        step "05 held-out check on val at R = 22 (all rules; never used for rule selection)"
        "$PY" scripts/05_eval_recon_probe.py "${C[@]}" --split val --R 22 --figures
    fi
fi

if [[ "$MODE" == imagenet ]]; then
    cat <<EOF

=== STOP before 06 (plan Sec. 8, pre-registration).
Read $EXP/recon_probe/train_dev/results.md and $EXP/allocations/allocations_summary.md, choose R* and w*
from these generator-free results ONLY (never from gFID), set preregistration.frozen: true, R_star, w_star in
configs/arms.yaml and commit. Then train and evaluate, e.g.:
  torchrun --standalone --nproc_per_node=$NGPU scripts/06_train_dit.py ${C[*]} --arm B0
  torchrun --standalone --nproc_per_node=$NGPU scripts/06_train_dit.py ${C[*]} --arm B3 --R <R*> --w <w*>
  $PY scripts/07_evaluate.py ref-stats ${C[*]} --split val
  $PY scripts/07_evaluate.py generate ${C[*]} --arm B3 --R <R*> --w <w*>                           # FID-10K screening
  $PY scripts/07_evaluate.py generate ${C[*]} --arm B3 --R <R*> --w <w*> --num-samples 50000 --save-npz  # FID-50K + ADM
EOF
    exit 0
fi

# ------------------------------------------------------------------ SVP only from here on
# Lossy arms at R = 22 nats/token (grid.R_nats) and w = 2 (grid.w); exploratory = off the (unfrozen) preregistration.
ARMS=("--arm B0" "--arm B2 --R 22" "--arm B3 --R 22 --w 2")

if want 05b; then
    step "05b decoder adaptation, B0-adapt control (M4 recipe exercised once)"
    "$PY" scripts/05b_adapt_decoder.py "${C[@]}" --arm B0 --val-images 100
fi

if want 06; then
    for arm in "${ARMS[@]}"; do
        step "06 DiT overfit run ($arm), torchrun x $NGPU"
        # shellcheck disable=SC2086  # $arm holds several flags
        "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" scripts/06_train_dit.py "${C[@]}" \
            $arm --exploratory
    done
fi

if want 07; then
    F="$EXP/fid_ref"
    step "07 FID references: train_fit (500, equal to val), train_fit features (+ h-flips), val (+ features)"
    "$PY" scripts/07_evaluate.py ref-stats "${C[@]}" --split train_fit --max-images 500
    "$PY" scripts/07_evaluate.py ref-stats "${C[@]}" --split train_fit --save-features --hflip
    "$PY" scripts/07_evaluate.py ref-stats "${C[@]}" --split val --save-features
    for arm in "${ARMS[@]}"; do
        step "07 generate ($arm): gFID vs train_fit and val, nearest neighbours vs train_fit"
        # shellcheck disable=SC2086
        "$PY" scripts/07_evaluate.py generate "${C[@]}" $arm \
            --ref-stats "train=$F/train_fit_n500.npz" --ref-stats "val=$F/val.npz" \
            --nn-ref "train=$F/train_fit_features.npz" --nn-baseline "$F/val_features.npz" --save-grid 64
    done
    step "07 report (SVP verdict)"
    "$PY" scripts/07_evaluate.py report "${C[@]}"
    echo; echo "SVP report: $EXP/report/REPORT.md"
fi
