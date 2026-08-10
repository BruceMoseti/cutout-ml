"""Adapter for CutoutNet, the model trained in-repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from cutoutml.core.config import get_settings
from cutoutml.models.base import ModelMetadata, TorchSegmentationModel
from cutoutml.models.cutoutnet.arch import ARCHITECTURES

DEFAULT_CHECKPOINT = "cutoutnet-small.pt"


class CutoutNetAdapter(TorchSegmentationModel):
    """CutoutNet: original, small, trained on the synthetic dataset in this repo.

    Unlike the other PyTorch adapters this one has weights committed to the
    repository (``models/cutoutnet/``), because they are small enough and were
    produced here. ``load()`` therefore succeeds out of the box, and its accuracy
    numbers in ``benchmarks/results`` are real measurements rather than
    placeholders.
    """

    VARIANTS = tuple(ARCHITECTURES)

    def __init__(self, *, variant: str = "small", **kwargs: Any) -> None:
        if variant not in ARCHITECTURES:
            raise ValueError(
                f"unknown CutoutNet variant {variant!r}; expected one of {self.VARIANTS}"
            )
        kwargs.setdefault("name", f"cutoutnet-{variant}")
        kwargs.setdefault("input_size", (256, 256))
        super().__init__(**kwargs)
        self.variant = variant

    def build_module(self) -> nn.Module:
        return ARCHITECTURES[self.variant]()

    def resolve_weights_path(self) -> Path:
        """Fall back to the committed checkpoint under ``models/cutoutnet/``."""
        if self.weights_path is not None and self.weights_path.is_file():
            return self.weights_path
        candidate = get_settings().model_weights_dir / "cutoutnet" / f"{self.name}.pt"
        if candidate.is_file():
            return candidate
        self.weights_path = candidate
        return super().resolve_weights_path()

    def default_weights_hint(self) -> str:
        return f"models/cutoutnet/{self.name}.pt"

    def weights_hint(self) -> str:
        return (
            "CutoutNet weights are produced in-repo. Train them with "
            "`make train` (a few minutes on 8 CPU cores) or "
            "`python -m cutoutml.training.train --variant "
            f"{self.variant}`."
        )

    @property
    def normalization(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Plain ``[-1, 1]`` scaling.

        ImageNet statistics exist to match a pretrained backbone's expectations;
        CutoutNet has no pretrained backbone, so symmetric scaling is the simpler
        and equally effective choice. It also means the refinement head's raw-RGB
        input is already centred.
        """
        return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def metadata(self) -> ModelMetadata:
        meta = super().metadata()
        return ModelMetadata(
            **{
                **meta.as_dict(),
                "input_size": self.input_size,
                "architecture": f"CutoutNet-{self.variant}",
                "notes": (
                    meta.notes
                    + " Original architecture; trained in-repo on the synthetic dataset."
                ).strip(),
            }
        )
