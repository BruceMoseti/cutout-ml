"""Losses for segmentation / matting training.

The combination used by default is ``BCE + IoU + edge``, which is a deliberate
choice rather than a grab-bag:

* **BCE** is per-pixel and gives a well-conditioned gradient everywhere, including
  in the soft-alpha band where a region loss is nearly flat. Alone, it is
  dominated by the (usually much larger) background class and converges to a
  timid, low-confidence mask.
* **Soft IoU** is a *region* loss. It is scale-invariant with respect to object
  size, so a small object contributes as much as a large one, which is exactly
  what BCE fails at. Alone, it has vanishing gradients when the prediction and
  target barely overlap - the classic "IoU loss won't start training" failure.
* **Edge loss** (L1 on Sobel magnitude) concentrates capacity on the boundary,
  which is the only place the two losses above are ambivalent about and the only
  place a human looks.

Together: BCE gets training moving and keeps the soft band calibrated, IoU fixes
the class imbalance, edge loss buys boundary sharpness. SSIM is available and adds
structural sensitivity, but it costs ~30% more time per step on CPU for a small
gain, so it is off by default.

Every loss takes **logits**, never probabilities: ``binary_cross_entropy_with_logits``
fuses the sigmoid for numerical stability, and doing it by hand overflows for
|logit| > ~40, which happens routinely once a model is confident.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy from logits."""
    return F.binary_cross_entropy_with_logits(logits, target)


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``1 - soft IoU``, computed per sample and averaged.

    Per-sample (rather than per-batch) reduction matters: pooling the intersection
    over a whole batch lets one large object dominate, defeating the purpose of
    using a region loss in the first place.
    """
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims) - inter
    return (1.0 - (inter + eps) / (union + eps)).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``1 - Dice``. Slightly gentler gradients than IoU near convergence."""
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    total = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (total + eps)).mean()


def _sobel(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, kx.transpose(2, 3), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def edge_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between the Sobel magnitude of the prediction and of the target.

    Operating on gradients rather than values makes this loss blind to any
    constant offset and maximally sensitive to boundary placement and sharpness.
    """
    return F.l1_loss(_sobel(torch.sigmoid(logits)), _sobel(target))


def ssim_loss(
    logits: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """``1 - SSIM`` with a Gaussian window, computed on probabilities.

    Structural similarity compares local means, variances and covariance, so it
    penalises a mask that has the right average alpha but the wrong local
    structure - the failure mode that region losses cannot see.
    """
    probs = torch.sigmoid(logits)
    coords = torch.arange(window, dtype=probs.dtype, device=probs.device) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, -1)
    kernel = (g.transpose(2, 1) @ g).view(1, 1, window, window)

    pad = window // 2

    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad)

    mu_x, mu_y = filt(probs), filt(target)
    sigma_x = filt(probs * probs) - mu_x * mu_x
    sigma_y = filt(target * target) - mu_y * mu_y
    sigma_xy = filt(probs * target) - mu_x * mu_y

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim_map.mean()


@dataclasses.dataclass(slots=True)
class LossWeights:
    """Relative weights of the loss terms and of deep supervision."""

    bce: float = 1.0
    iou: float = 1.0
    edge: float = 0.5
    ssim: float = 0.0
    gradient: float = 0.3
    side: float = 0.4

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


class SegmentationLoss(nn.Module):
    """Composite loss with deep supervision over a network's multiple outputs.

    ``outputs`` is whatever the architecture returns. Element 0 is the primary
    prediction and is weighted 1.0; every subsequent logit map is a side output
    weighted by ``weights.side``. Deep supervision is not cosmetic here: with a
    randomly initialised (no pretrained backbone) network, supervising only the
    final output means the encoder receives gradient only through the whole
    decoder, and it trains visibly slower.

    ``gradient_output_index`` names an output that predicts the *edge map* rather
    than the mask (BiRefNet's gradient head); it is supervised against the Sobel
    magnitude of the target instead of the target itself.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        *,
        gradient_output_index: int | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.gradient_output_index = gradient_output_index

    def forward(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(total_loss, per_term_scalars)`` for JSON metric logging."""
        outs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        total = torch.zeros((), dtype=outs[0].dtype, device=outs[0].device)
        parts: dict[str, float] = {}

        for index, logits in enumerate(outs):
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(
                    logits, size=target.shape[-2:], mode="bilinear", align_corners=False
                )

            if index == self.gradient_output_index:
                grad_target = _sobel(target).clamp(0.0, 1.0)
                term = F.binary_cross_entropy_with_logits(logits, grad_target)
                total = total + self.weights.gradient * term
                parts["gradient"] = float(term.detach())
                continue

            scale = 1.0 if index == 0 else self.weights.side
            sub = torch.zeros((), dtype=logits.dtype, device=logits.device)

            if self.weights.bce > 0:
                v = bce_loss(logits, target)
                sub = sub + self.weights.bce * v
                if index == 0:
                    parts["bce"] = float(v.detach())
            if self.weights.iou > 0:
                v = soft_iou_loss(logits, target)
                sub = sub + self.weights.iou * v
                if index == 0:
                    parts["iou"] = float(v.detach())
            if self.weights.edge > 0:
                v = edge_loss(logits, target)
                sub = sub + self.weights.edge * v
                if index == 0:
                    parts["edge"] = float(v.detach())
            if self.weights.ssim > 0:
                v = ssim_loss(logits, target)
                sub = sub + self.weights.ssim * v
                if index == 0:
                    parts["ssim"] = float(v.detach())

            total = total + scale * sub

        parts["total"] = float(total.detach())
        return total, parts
