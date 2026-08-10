"""Reproducibility of the benchmark's accuracy pass.

GrabCut seeds its colour model from OpenCV's process-global RNG, so its output depends on
how many draws happened earlier in the process. Because the harness times every case
before it scores it, that made a published accuracy number a function of `--repetitions`:
a flag that should only decide how much latency data is collected was changing the score.

The harness now resets that RNG immediately before each accuracy pass
(:data:`cutoutml.benchmarks.harness.EVAL_RNG_SEED`), so the pass starts from a fixed state
whatever ran before it. These tests hold that invariant from both ends - the score must not
move when the timing work around it changes, and the reset itself must still be there.
"""

from __future__ import annotations

import numpy as np
import pytest

from cutoutml.benchmarks.harness import (
    EVAL_RNG_SEED,
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkHarness,
)
from cutoutml.models.classical.baseline import grabcut_mask

#: The published `cutoutnet-fp32` IoU. Pinned so that a change to the eval set, the
#: preprocessing or the metric implementation cannot pass as a change to the RNG fix.
PUBLISHED_CUTOUTNET_IOU = 0.8544288202661507


def _harness(*, repetitions: int, samples: int = 2) -> BenchmarkHarness:
    """A harness on the committed defaults, shrunk to what a test needs.

    `load_sample_seconds=0` skips the contention probe, which measures the machine rather
    than the model and costs a second per case.
    """
    return BenchmarkHarness(
        BenchmarkConfig(
            warmup=0,
            repetitions=repetitions,
            accuracy_samples=samples,
            load_sample_seconds=0.0,
        )
    )


def _grabcut_iou(harness: BenchmarkHarness) -> float:
    result = harness.run_case(BenchmarkCase(model="classical", label="classical-grabcut"))
    assert result.status == "ok", result.error
    assert result.accuracy is not None
    return result.accuracy["iou"]


def _burn_opencv_rng() -> None:
    """Advance OpenCV's global RNG the way a preceding case would."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 255
    for _ in range(3):
        grabcut_mask(image, iterations=2)


def test_the_same_evaluation_twice_gives_the_same_score():
    """The weakest form of the invariant, and the one that failed before: two identical
    evaluations in one process differ if the second inherits the first's RNG state."""
    first = _grabcut_iou(_harness(repetitions=1))
    second = _grabcut_iou(_harness(repetitions=1))

    assert first == second


def test_the_repetition_count_does_not_move_the_accuracy():
    """`--repetitions` decides how many times the model is timed. Each of those calls used
    to advance the RNG that the scoring pass then drew from, so the committed default and
    a single repetition produced two different, individually stable, published scores."""
    committed_default = BenchmarkConfig().repetitions
    assert committed_default > 1, "this test needs two distinct repetition counts"

    at_one = _grabcut_iou(_harness(repetitions=1))
    at_default = _grabcut_iou(_harness(repetitions=committed_default))

    assert at_one == at_default


def test_unrelated_opencv_work_beforehand_does_not_move_the_accuracy():
    """Execution order is the general case of the repetition count: any earlier case that
    draws from the RNG would otherwise shift every classical score after it."""
    baseline = _grabcut_iou(_harness(repetitions=1))

    _burn_opencv_rng()
    after_burn = _grabcut_iou(_harness(repetitions=1))

    assert after_burn == baseline


def test_the_accuracy_pass_reseeds_rather_than_happening_to_agree():
    """Behavioural tests above would still pass if the reset were deleted on a day the
    surrounding call counts happened to line up. This one fails the moment the call goes,
    which is the point: the guarantee is the reset, not today's arithmetic."""
    seeds: list[int] = []
    harness = _harness(repetitions=1)

    with pytest.MonkeyPatch.context() as patch:
        import cutoutml.benchmarks.harness as harness_module

        real = harness_module.cv2.setRNGSeed

        def record(seed: int) -> None:
            seeds.append(seed)
            real(seed)

        patch.setattr(harness_module.cv2, "setRNGSeed", record)
        _grabcut_iou(harness)

    assert seeds == [EVAL_RNG_SEED]


def test_a_learned_model_scores_exactly_what_it_published():
    """The fix must not have touched anything that does not draw from OpenCV's RNG."""
    harness = _harness(repetitions=1, samples=64)

    result = harness.run_case(BenchmarkCase(model="cutoutnet", label="cutoutnet-fp32"))
    if result.status == "skipped":
        # The harness turns a missing checkpoint into a skipped case rather than raising,
        # so this has to read the status: CutoutNet is trained here, not committed.
        pytest.skip(f"cutoutnet weights unavailable: {result.error}")

    assert result.status == "ok", result.error
    assert result.accuracy is not None
    assert result.accuracy["iou"] == PUBLISHED_CUTOUTNET_IOU
