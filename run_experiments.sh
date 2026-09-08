#!/usr/bin/env bash
# Both experiments, start to finish.
#
#   bash run_experiments.sh                     # GPU if available, 400 products
#   bash run_experiments.sh --device cpu        # overnight on a laptop
#   bash run_experiments.sh --sizes "debug tiny"  # skip the slow one
#   bash run_experiments.sh --exclude-holdout   # keep held-out products out of pretraining
#
# Experiment 1 varies model size with everything else fixed. Experiment 2 fine-tunes
# the same architecture from pretrained weights and from random init.
#
# Each stage is skipped if its checkpoint already exists, so an interrupted run
# continues where it stopped. Delete runs/<name>/ to force a stage to redo.
#
# Output: runs/*/ per stage, then RESULTS.md with every table.

set -euo pipefail

PY=${PY:-python}
DEVICE=""
PRODUCTS=400
STEPS=700
SIZES="debug tiny small"
EXCLUDE=""
LOG=experiments.log

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)           DEVICE="--device $2"; shift 2 ;;
    --products)         PRODUCTS="$2"; shift 2 ;;
    --steps)            STEPS="$2"; shift 2 ;;
    --sizes)            SIZES="$2"; shift 2 ;;
    --exclude-holdout)  EXCLUDE="--exclude-products data/out/holdout_products.json"; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m  (%s)\n' "$1" "$(date +%H:%M:%S)" | tee -a "$LOG"; }
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

START=$(date +%s)
: > "$LOG"

say "Data: $PRODUCTS products"
run $PY -m data.generate --products "$PRODUCTS"
run $PY -m data.instructions

if [[ -n "$EXCLUDE" ]]; then
  echo "Held-out products WILL be excluded from pretraining." | tee -a "$LOG"
else
  echo "Held-out products will NOT be excluded from pretraining." | tee -a "$LOG"
  echo "The report can then claim generalisation of the instruction format to" | tee -a "$LOG"
  echo "products never instruction-tuned on, which is narrower than generalisation" | tee -a "$LOG"
  echo "to unseen products. Pass --exclude-holdout for the stronger claim." | tee -a "$LOG"
fi

# ── Experiment 1: model scale ────────────────────────────────────────────────
for size in $SIZES; do
  say "Experiment 1 / $size / pretrain"
  if [[ -f "runs/pre-$size/ckpt_best.pt" ]]; then
    echo "  runs/pre-$size exists, skipping" | tee -a "$LOG"
  else
    run $PY -m train.pretrain --preset "$size" --context-length 256 \
        --steps "$STEPS" --name "pre-$size" --sample-every 0 $DEVICE $EXCLUDE
  fi

  say "Experiment 1 / $size / finetune"
  if [[ -f "runs/ft-$size/ckpt_best.pt" ]]; then
    echo "  runs/ft-$size exists, skipping" | tee -a "$LOG"
  else
    run $PY -m train.finetune --init "runs/pre-$size/ckpt_best.pt" \
        --name "ft-$size" --samples 0 $DEVICE
  fi

  say "Experiment 1 / $size / evaluate"
  run $PY -m evals.run --ckpt "runs/ft-$size/ckpt_best.pt" --baselines $DEVICE
done

# ── Experiment 2: does pretraining help ──────────────────────────────────────
# The control has to be the same architecture as the treatment. --like copies it
# out of the checkpoint rather than trusting me to pass matching flags.
CONTROL_BASE=tiny
if [[ ! -f "runs/pre-$CONTROL_BASE/ckpt_best.pt" ]]; then
  CONTROL_BASE=$(echo "$SIZES" | awk '{print $NF}')
fi

say "Experiment 2 / control / finetune from random init"
if [[ -f "runs/ft-scratch/ckpt_best.pt" ]]; then
  echo "  runs/ft-scratch exists, skipping" | tee -a "$LOG"
else
  run $PY -m train.finetune --scratch --like "runs/pre-$CONTROL_BASE/ckpt_best.pt" \
      --name ft-scratch --samples 0 $DEVICE
fi

say "Experiment 2 / control / evaluate"
run $PY -m evals.run --ckpt runs/ft-scratch/ckpt_best.pt --baselines $DEVICE

# ── Tables ───────────────────────────────────────────────────────────────────
say "Collecting results"
run $PY -m evals.collect

ELAPSED=$(( $(date +%s) - START ))
printf '\nDone in %dh %dm. Tables in RESULTS.md, full log in %s.\n' \
  $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) "$LOG" | tee -a "$LOG"
echo "Experiment 2 is the ft-$CONTROL_BASE row against the ft-scratch row." | tee -a "$LOG"
