"""Compact bilateral-reference segmentation network (BiRefNet-inspired)."""

from cutoutml.models.birefnet.adapter import BiRefNetAdapter
from cutoutml.models.birefnet.arch import (
    BilateralReferenceBlock,
    BiRefNetCompact,
    birefnet_compact,
    birefnet_tiny,
    sobel_gradient,
)

__all__ = [
    "BiRefNetAdapter",
    "BiRefNetCompact",
    "BilateralReferenceBlock",
    "birefnet_compact",
    "birefnet_tiny",
    "sobel_gradient",
]
