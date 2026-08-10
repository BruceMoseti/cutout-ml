"""U^2-Net adapter, including official-checkpoint key translation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cutoutml.core.logging import get_logger
from cutoutml.models.base import ModelMetadata, TorchSegmentationModel
from cutoutml.models.u2net.arch import U2Net, u2net_full, u2net_lite

log = get_logger(__name__)

# --------------------------------------------------------------------- remapping
# Official U^2-Net names the six encoder stages `stage1..stage6`, the decoders
# `stage5d..stage1d`, and every RSU sub-convolution `rebnconv{n}` /
# `rebnconv{n}d`. This implementation uses ModuleLists, so the keys differ while
# the tensors are identical. Depth per stage is needed to place the *bottom*
# convolution, whose official index equals the RSU depth.
_FULL_DEPTHS = {1: 7, 2: 6, 3: 5, 4: 4, 5: 4, 6: 4}

_STAGE_RE = re.compile(r"^stage(?P<idx>[1-6])(?P<dec>d)?\.(?P<rest>.+)$")
_SIDE_RE = re.compile(r"^side(?P<idx>[1-6])\.(?P<rest>.+)$")
_RSU_RE = re.compile(r"^rebnconv(?P<n>\d+)(?P<dec>d)?\.(?P<rest>.+)$")


def _remap_rsu_key(inner: str, depth: int) -> str | None:
    """Translate one key *inside* an RSU block, or ``None`` if unrecognised."""
    if inner.startswith("rebnconvin."):
        return inner
    m = _RSU_RE.match(inner)
    if not m:
        return None
    n = int(m.group("n"))
    rest = m.group("rest")
    if m.group("dec"):
        # decoder: official rebnconv{depth-1-i}d  ->  decoder.{i}
        i = depth - 1 - n
        if not 0 <= i <= depth - 2:
            return None
        return f"decoder.{i}.{rest}"
    if n == depth:
        return f"rebnconv_bottom.{rest}"
    if 1 <= n <= depth - 1:
        return f"encoder.{n - 1}.{rest}"
    return None


def remap_official_state_dict(
    state: dict[str, torch.Tensor], depths: dict[int, int] | None = None
) -> dict[str, torch.Tensor]:
    """Rename an official U^2-Net checkpoint onto this implementation's keys.

    Keys that are already in this implementation's naming (``encoders.0.``,
    ``decoders.0.``, ``side.0.``) are passed through untouched, so the function is
    idempotent and safe to run on checkpoints produced by this repo.
    """
    depths = depths or _FULL_DEPTHS
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, value in state.items():
        if key.startswith(("encoders.", "decoders.", "side.")):
            out[key] = value
            continue
        if key.startswith("outconv."):
            out[key] = value
            continue

        m = _SIDE_RE.match(key)
        if m:
            out[f"side.{int(m.group('idx')) - 1}.{m.group('rest')}"] = value
            continue

        m = _STAGE_RE.match(key)
        if m:
            idx = int(m.group("idx"))
            depth = depths.get(idx, 4)
            inner = _remap_rsu_key(m.group("rest"), depth)
            if inner is None:
                unmapped.append(key)
                continue
            prefix = f"decoders.{5 - idx}" if m.group("dec") else f"encoders.{idx - 1}"
            out[f"{prefix}.{inner}"] = value
            continue

        unmapped.append(key)

    if unmapped:
        log.warning("u2net_unmapped_keys", count=len(unmapped), sample=unmapped[:5])
    return out


class U2NetAdapter(TorchSegmentationModel):
    """Adapter for U^2-Net (``full``, 44M params) and U^2-Net-P (``lite``, 1.1M).

    Weights
    -------
    Pretrained U^2-Net weights are distributed by the authors under Apache-2.0 but
    are hosted on Google Drive / HuggingFace mirrors, neither of which is
    guaranteed to be reachable from a build machine. Therefore:

    * without weights, :meth:`load` raises
      :class:`~cutoutml.models.base.WeightsUnavailableError` naming the path and
      the download command;
    * with ``random_init=True`` the network is built with Kaiming-initialised
      weights so **latency** can still be benchmarked. Accuracy from that mode is
      meaningless and every metadata/benchmark record it produces is flagged
      ``accuracy_valid=False``.
    """

    VARIANTS = ("full", "lite")

    def __init__(self, *, variant: str = "full", **kwargs: Any) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown U2Net variant {variant!r}; expected one of {self.VARIANTS}")
        kwargs.setdefault("name", f"u2net-{variant}")
        kwargs.setdefault("input_size", (320, 320))
        super().__init__(**kwargs)
        self.variant = variant

    def build_module(self) -> nn.Module:
        return u2net_full() if self.variant == "full" else u2net_lite()

    def transform_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Translate official parameter names onto this implementation's."""
        return remap_official_state_dict(state)

    def default_weights_hint(self) -> str:
        fname = "u2net.pth" if self.variant == "full" else "u2netp.pth"
        return f"models/u2net/{fname}"

    def weights_hint(self) -> str:
        """What to actually do about the missing checkpoint.

        The already-solved route is named first. This architecture is registered twice:
        once expecting a checkpoint trained here, and once (``u2netp`` / ``u2net``)
        pointing at the authors' published weights, which ``make weights-pretrained``
        fetches and converts. Anyone hitting this error most likely wants that pair and
        should not be sent to train for an hour to find out.

        The caveat belongs in the error rather than only in the docs: the published
        weights are Apache-2.0 and their accuracy is *not* comparable with an in-repo run,
        because the two see different datasets.
        """
        pretrained = "u2netp" if self.variant == "lite" else "u2net"
        trainable = self.variant == "lite"
        train_hint = (
            f"train this one with `scripts/train_suite.sh u2net-{self.variant}` "
            "(~1 hour on 8 CPU cores)"
            if trainable
            else f"train it yourself (u2net-{self.variant} is 44M parameters and needs a GPU)"
        )
        return (
            f"The same architecture with the authors' pretrained weights is already "
            f"registered as `{pretrained}` (Apache-2.0): run `make weights-pretrained` to "
            f"fetch and convert them, then benchmark `{pretrained}` instead. Note that its "
            "accuracy is not comparable with an in-repo run - different training data. "
            f"Otherwise, {train_hint}, or drop a checkpoint into "
            f"{self.default_weights_hint()} yourself. For latency-only benchmarking pass "
            "random_init=True."
        )

    @property
    def normalization(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """U^2-Net's own mean/std, which differ from the usual ImageNet triple."""
        return ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def intensity_divisor(self, image: np.ndarray) -> float | None:
        """Reproduce the reference pipeline's per-image max division.

        Upstream rescales to ``[0, 1]`` and then divides by the image's own maximum
        before the mean/std step, so an image whose brightest pixel is 128 is stretched
        to full range. This is only a no-op for images containing a saturated pixel;
        on the synthetic eval set fewer than half qualify, and skipping it feeds the
        pretrained weights inputs they were never trained on. Measured cost of getting
        this wrong on the eval set: several IoU points, which is more than the
        difference between two of the models in the benchmark table.

        A fully black image would divide by zero, so it falls back to no scaling.
        """
        peak = float(np.asarray(image).max())
        return peak / 255.0 if peak > 0 else None

    def metadata(self) -> ModelMetadata:
        meta = super().metadata()
        return ModelMetadata(
            **{
                **meta.as_dict(),
                "input_size": self.input_size,
                "architecture": f"U2Net-{self.variant}",
                "notes": (meta.notes + " Nested U-structure, 6 side outputs + fusion.").strip(),
            }
        )


def load_u2net_from_official(path: Path | str, variant: str = "full") -> U2Net:
    """Build a :class:`U2Net` and populate it from an official checkpoint.

    Provided as a standalone helper so the remapping can be exercised (and unit
    tested) without going through the adapter/registry machinery.
    """
    from cutoutml.models.base import load_state_dict

    module = u2net_full() if variant == "full" else u2net_lite()
    state = remap_official_state_dict(load_state_dict(path))
    module.load_state_dict(state, strict=True)
    module.eval()
    return module
