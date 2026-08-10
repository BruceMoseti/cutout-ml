#!/usr/bin/env bash
# Train every CPU-feasible architecture on an identical data budget.
#
# The point of this script is comparability. cutoutnet-{small,base,tiny} all see the
# same 2048 procedurally generated samples per epoch for the same 14 epochs from the
# same dataset seed, so the differences between their rows in the benchmark table are
# attributable to capacity and nothing else.
#
# u2net-lite is the exception and is deliberately called out here rather than hidden:
# it costs roughly 4x more per sample than cutoutnet-small on CPU (measured, not
# assumed), so an identical budget would take over two hours. It is trained on a
# smaller budget and every document that quotes its accuracy says so. Do not compare
# its IoU with the CutoutNet rows as though the budgets matched.
#
# Runs are sequential on purpose: each one is given the whole box so that the
# samples_per_second recorded in its run JSON means something.
#
# Usage:  scripts/train_suite.sh [arch ...]     (default: all of the below)

set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

EPOCHS="${EPOCHS:-14}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2048}"
VAL_SAMPLES="${VAL_SAMPLES:-192}"
BATCH_SIZE="${BATCH_SIZE:-16}"

run() {
  local arch="$1"; shift
  echo "=============================================================="
  echo "training ${arch} at $(date -u +%FT%TZ)"
  echo "=============================================================="
  "$PYTHON" -m cutoutml.training.train --arch "$arch" "$@"
}

archs=("$@")
if [ ${#archs[@]} -eq 0 ]; then
  # Ordered so that the identical-budget CutoutNet capacity sweep completes first;
  # u2net-lite is the odd budget out and therefore goes last.
  archs=(cutoutnet-base cutoutnet-tiny u2net-lite)
fi

for arch in "${archs[@]}"; do
  case "$arch" in
    u2net-lite)
      # Reduced budget: see the note at the top of this file.
      run "$arch" --epochs 6 --train-samples 1024 --val-samples "$VAL_SAMPLES" \
        --batch-size 8 --resolution 256
      ;;
    *)
      run "$arch" --epochs "$EPOCHS" --train-samples "$TRAIN_SAMPLES" \
        --val-samples "$VAL_SAMPLES" --batch-size "$BATCH_SIZE"
      ;;
  esac
done

echo "all runs complete at $(date -u +%FT%TZ)"
