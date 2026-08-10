"""CutoutNet: original small segmentation/matting network trained in this repo."""

from cutoutml.models.cutoutnet.adapter import DEFAULT_CHECKPOINT, CutoutNetAdapter
from cutoutml.models.cutoutnet.arch import (
    ARCHITECTURES,
    CutoutNet,
    InvertedResidual,
    RefinementHead,
    cutoutnet_base,
    cutoutnet_small,
    cutoutnet_tiny,
)

__all__ = [
    "ARCHITECTURES",
    "DEFAULT_CHECKPOINT",
    "CutoutNet",
    "CutoutNetAdapter",
    "InvertedResidual",
    "RefinementHead",
    "cutoutnet_base",
    "cutoutnet_small",
    "cutoutnet_tiny",
]
