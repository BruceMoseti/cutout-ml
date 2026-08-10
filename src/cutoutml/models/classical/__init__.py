"""Zero-training classical baselines."""

from cutoutml.models.classical.baseline import (
    ClassicalBaseline,
    TrivialBaseline,
    grabcut_mask,
    otsu_threshold,
    spectral_residual_saliency,
)

__all__ = [
    "ClassicalBaseline",
    "TrivialBaseline",
    "grabcut_mask",
    "otsu_threshold",
    "spectral_residual_saliency",
]
