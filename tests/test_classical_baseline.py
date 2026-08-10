"""The non-learned baselines.

These exist to give the learned models something to beat, so the only thing that matters
about them is that their scores mean what the benchmark says they mean. One of them does
not: GrabCut's score depends on how many times it has already been called in the process,
which the benchmark documentation has to qualify because the harness times every case
before it scores it. The test below pins that behaviour so the qualification cannot
quietly become wrong in either direction - if OpenCV ever makes GrabCut reproducible, this
fails and `docs/benchmarks.md` should stop warning about it.
"""

from __future__ import annotations

import numpy as np

from cutoutml.datasets.synthetic import SyntheticSegmentationDataset
from cutoutml.models.classical.baseline import grabcut_mask, spectral_residual_saliency


def _eval_image() -> np.ndarray:
    """The first test-split sample, which is what the benchmark scores first."""
    image, _ = SyntheticSegmentationDataset(count=4, seed=20240817, split="test")[0]
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


def test_grabcut_is_not_reproducible_call_to_call():
    """OpenCV seeds GrabCut's colour model from a process-global RNG, so the call index is
    a hidden input. This is why `classical-grabcut` reproduces its published IoU only at
    the committed repetition count."""
    image = _eval_image()

    masks = [grabcut_mask(image, iterations=5) for _ in range(6)]
    means = {round(float(m.mean()), 10) for m in masks}

    assert len(means) > 1, "GrabCut became reproducible; the benchmark caveat is now stale"


def test_saliency_is_reproducible_call_to_call():
    """The contrast with GrabCut is the point: nothing about running a baseline twice is
    inherently unstable, so the caveat belongs to GrabCut rather than to the classical
    family."""
    image = _eval_image()

    masks = [spectral_residual_saliency(image) for _ in range(4)]

    for later in masks[1:]:
        assert np.array_equal(masks[0], later)
