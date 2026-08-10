#!/usr/bin/env bash
# Train every CPU-feasible architecture on an identical budget.
#
# The point of this script is comparability. Every architecture below sees the same
# 2048 procedurally generated samples per epoch, for the same 14 epochs, from the same
# dataset seed, at the same 256px resolution and batch size 16. The differences between
# their rows in the benchmark table are therefore attributable to the architecture and
# nothing else -- which is the only way a capacity curve or a cross-architecture
# comparison means anything.
#
# Holding the budget fixed is not free: a measured training step at batch 8 costs
# 261 ms for cutoutnet-tiny, 667 ms for cutoutnet-small, 422 ms for cutoutnet-base and
# 1194 ms for u2net-lite on this 8-core CPU, so u2net-lite's run takes roughly four
# times as long as tiny's for the same number of samples. That is the honest cost of a
# comparison, and it is why this script exists instead of a set of ad-hoc invocations
# with whatever budget happened to fit.
#
# Runs are sequential on purpose: each one is given the whole box so that the
# samples_per_second recorded in its run JSON is a number about the architecture rather
# than a number about how many other runs were competing for cores.
#
# Usage:  scripts/train_suite.sh [arch ...]     (default: the three below)
#
# The GPU-only architectures (u2net-full, birefnet-*) are excluded by default. They are
# registered and trainable, but a useful run needs a GPU, so this repository ships no
# checkpoint for them and claims no accuracy figure. See docs/benchmarks.md.

set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

EPOCHS="${EPOCHS:-14}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2048}"
VAL_SAMPLES="${VAL_SAMPLES:-192}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RESOLUTION="${RESOLUTION:-256}"

archs=("$@")
if [ ${#archs[@]} -eq 0 ]; then
  # Cheapest first, so that an interrupted suite still leaves a complete comparison
  # behind rather than one finished run and one truncated one.
  archs=(cutoutnet-tiny cutoutnet-base u2net-lite)
fi

for arch in "${archs[@]}"; do
  echo "=============================================================="
  echo "training ${arch} at $(date -u +%FT%TZ)"
  echo "=============================================================="
  "$PYTHON" -m cutoutml.training.train \
    --arch "$arch" \
    --epochs "$EPOCHS" \
    --train-samples "$TRAIN_SAMPLES" \
    --val-samples "$VAL_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --resolution "$RESOLUTION"
done

echo "all runs complete at $(date -u +%FT%TZ)"
