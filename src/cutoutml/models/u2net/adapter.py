"""U^2-Net adapter, including official-checkpoint key translation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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

    def _load(self) -> None:
        """Load with official-checkpoint key remapping applied."""
        module = self.build_module()
        if not self.random_init:
            path = self.resolve_weights_path()
            from cutoutml.models.base import load_state_dict

            raw = load_state_dict(path)
            state = remap_official_state_dict(raw)
            missing, unexpected = module.load_state_dict(state, strict=False)
            if missing or unexpected:
                log.warning(
                    "u2net_state_dict_mismatch",
                    missing=len(missing),
                    unexpected=len(unexpected),
                    missing_sample=list(missing)[:5],
                )
            self.weights_path = path
        else:
            log.warning(
                "random_init_model",
                model=self.name,
                warning="weights are random; latency is valid, accuracy is NOT",
            )
        module.eval()
        module.to(self.device)
        self.module = module

    def default_weights_hint(self) -> str:
        fname = "u2net.pth" if self.variant == "full" else "u2netp.pth"
        return f"models/u2net/{fname}"

    def weights_hint(self) -> str:
        return (
            "U^2-Net weights are published by the authors (Apache-2.0) but hosted "
            "off-PyPI; run `python -m cutoutml.models.download_weights --model "
            f"{self.name}` while the mirror is reachable, or drop the .pth into "
            f"{self.default_weights_hint()}. For latency-only benchmarking pass "
            "random_init=True."
        )

    @property
    def normalization(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """U^2-Net's own scheme: divide by max then ImageNet-normalise.

        The reference preprocessing rescales to ``[0, 1]``, divides by the image
        maximum and then applies ``mean=(0.485, 0.456, 0.406)`` with
        ``std=(0.229, 0.224, 0.225)``. The max-division is a no-op for images that
        contain a saturated pixel (almost all photographs), so plain ImageNet
        normalisation is used and the difference is documented rather than
        silently absorbed.
        """
        return ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

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
