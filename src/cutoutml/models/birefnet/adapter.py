"""Adapter for the compact bilateral-reference network."""

from __future__ import annotations

from typing import Any

from torch import nn

from cutoutml.models.base import ModelMetadata, TorchSegmentationModel
from cutoutml.models.birefnet.arch import birefnet_compact, birefnet_tiny

_LICENSE_WARNING = (
    "Architecture-inspired reimplementation. Official BiRefNet code is MIT and its "
    "weights are hosted on HuggingFace; SOME third-party fine-tuned BiRefNet "
    "checkpoints are released under non-commercial terms. Verify the licence of any "
    "weights you load. See docs/models.md."
)


class BiRefNetAdapter(TorchSegmentationModel):
    """Bilateral-reference high-resolution segmentation.

    Runs at 512x512 by default rather than U^2-Net's 320x320, because the whole
    point of the bilateral reference is high-resolution edge fidelity and there is
    little to gain from it at low input sizes.
    """

    VARIANTS = ("compact", "tiny")

    def __init__(self, *, variant: str = "compact", **kwargs: Any) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(
                f"unknown BiRefNet variant {variant!r}; expected one of {self.VARIANTS}"
            )
        kwargs.setdefault("name", f"birefnet-{variant}")
        kwargs.setdefault("input_size", (512, 512))
        super().__init__(**kwargs)
        self.variant = variant

    def build_module(self) -> nn.Module:
        return birefnet_compact() if self.variant == "compact" else birefnet_tiny()

    def default_weights_hint(self) -> str:
        return f"models/birefnet/{self.name}.pt"

    def weights_hint(self) -> str:
        return (
            "This repository ships no BiRefNet weights: the official checkpoints are "
            "on HuggingFace (blocked in some environments) and target the full Swin "
            "backbone, whose tensor shapes do not match this compact "
            "reimplementation. Either train this architecture yourself "
            "(`python -m cutoutml.training.train --arch birefnet`) or pass "
            "random_init=True for latency-only benchmarking. " + _LICENSE_WARNING
        )

    def metadata(self) -> ModelMetadata:
        meta = super().metadata()
        return ModelMetadata(
            **{
                **meta.as_dict(),
                "input_size": self.input_size,
                "architecture": f"BiRefNetCompact-{self.variant}",
                "notes": (meta.notes + " " + _LICENSE_WARNING).strip(),
            }
        )
