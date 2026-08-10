"""Classical, zero-training segmentation baselines.

A benchmark table without a non-learned baseline is not interpretable: if a
neural network scores 0.82 IoU, the reader needs to know whether Otsu on a
saliency map scores 0.35 or 0.80. Two methods are provided:

**Spectral-residual saliency** (Hou & Zhang, CVPR 2007). The log-amplitude
spectrum of natural images is close to ``1/f``; whatever *departs* from that
smooth trend tends to be the salient object. So: FFT, take the log amplitude,
subtract its local average, invert with the original phase, square, blur. This is
implemented directly with NumPy because ``cv2.saliency`` lives in
opencv-contrib, which this project does not depend on. Roughly 15 lines and no
training data.

**GrabCut** (Rother et al., 2004). Iterative graph-cut with Gaussian-mixture
colour models, initialised from a rectangle or from a saliency-derived trimap.
Genuinely strong on high-contrast subjects and hopeless on cluttered ones, which
makes it an informative baseline.

Both are exposed through the same :class:`~cutoutml.models.base.SegmentationModel`
interface, so the benchmark harness cannot tell them apart from a network. They
carry no parameters, no device and no weights, so ``predict`` runs on NumPy and
the tensor plumbing exists only to satisfy the contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import cv2
import numpy as np
import torch

from cutoutml.core.imaging import LetterboxInfo, letterbox, unletterbox_mask
from cutoutml.models.base import ModelMetadata, SegmentationModel

Method = Literal["saliency", "grabcut", "saliency+grabcut"]


def spectral_residual_saliency(
    image: np.ndarray, *, work_size: int = 64, blur_sigma: float = 2.5
) -> np.ndarray:
    """Spectral-residual saliency map, normalised to ``[0, 1]``.

    ``work_size`` follows the paper's recommendation of downsampling to ~64 px:
    the spectral residual is a *global* statistic and computing it at full
    resolution mostly amplifies texture noise. The result is resized back up.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (work_size, work_size), interpolation=cv2.INTER_AREA)

    fft = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)

    # The "spectral residual" is the log spectrum minus its local average.
    averaged = cv2.blur(log_amplitude.astype(np.float32), (3, 3))
    residual = log_amplitude - averaged

    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency = np.abs(reconstructed) ** 2

    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), blur_sigma)
    saliency = cv2.resize(
        saliency, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    lo, hi = float(saliency.min()), float(saliency.max())
    if hi - lo < 1e-12:
        return np.zeros_like(saliency, dtype=np.float32)
    return ((saliency - lo) / (hi - lo)).astype(np.float32)


def otsu_threshold(saliency: np.ndarray) -> tuple[np.ndarray, float]:
    """Binarise a ``[0, 1]`` map with Otsu's method.

    Returns the binary mask and the chosen threshold (in ``[0, 1]``), because the
    threshold is useful for building a GrabCut trimap.
    """
    quantised = np.clip(np.rint(saliency * 255.0), 0, 255).astype(np.uint8)
    thresh_value, binary = cv2.threshold(quantised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.float32), float(thresh_value) / 255.0


def grabcut_mask(
    image: np.ndarray,
    *,
    iterations: int = 5,
    border_fraction: float = 0.1,
    init_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run GrabCut and return a ``{0, 1}`` float mask.

    Without ``init_mask`` the algorithm is seeded with a rectangle inset by
    ``border_fraction`` on each side - the standard "subject is roughly centred"
    assumption. With ``init_mask`` (typically from saliency) a proper trimap is
    built: eroded interior = definite foreground, dilated exterior = definite
    background, the band between = probable. Seeding from saliency is what lifts
    GrabCut above its rectangle-only performance on off-centre subjects.
    """
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        return np.ones((h, w), dtype=np.float32)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    if init_mask is None:
        mask = np.zeros((h, w), np.uint8)
        inset_x = max(1, int(w * border_fraction))
        inset_y = max(1, int(h * border_fraction))
        rect = (inset_x, inset_y, max(1, w - 2 * inset_x), max(1, h - 2 * inset_y))
        try:
            cv2.grabCut(bgr, mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            # Degenerate colour distributions (e.g. a flat image) make the GMM
            # fit fail; fall back to the seed rectangle rather than crashing.
            out = np.zeros((h, w), np.float32)
            out[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]] = 1.0
            return out
    else:
        binary = (np.asarray(init_mask) > 0.5).astype(np.uint8)
        if binary.sum() == 0 or binary.sum() == binary.size:
            return binary.astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        sure_fg = cv2.erode(binary, k, iterations=2)
        sure_bg = 1 - cv2.dilate(binary, k, iterations=3)

        mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        mask[binary > 0] = cv2.GC_PR_FGD
        mask[sure_fg > 0] = cv2.GC_FGD
        mask[sure_bg > 0] = cv2.GC_BGD
        # GrabCut needs at least one definite pixel of each class.
        if not (mask == cv2.GC_FGD).any():
            mask[binary > 0] = cv2.GC_PR_FGD
        try:
            cv2.grabCut(bgr, mask, None, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return binary.astype(np.float32)

    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    return fg.astype(np.float32)


class ClassicalBaseline(SegmentationModel):
    """Non-learned baseline exposed through the standard model interface.

    ``method``:

    * ``saliency`` - spectral residual + Otsu. Very fast, no assumptions.
    * ``grabcut`` - GrabCut from a centred rectangle.
    * ``saliency+grabcut`` (default) - saliency builds the trimap, GrabCut refines
      the boundary using colour. Slowest but clearly the best of the three.

    ``device`` and ``precision`` are accepted and reported for interface parity
    but have no effect: this runs on the CPU in NumPy/OpenCV regardless.
    """

    def __init__(
        self,
        *,
        method: Method = "saliency+grabcut",
        grabcut_iterations: int = 5,
        soften: float = 1.5,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", f"classical-{method}")
        kwargs.setdefault("input_size", (320, 320))
        super().__init__(**kwargs)
        self.method: Method = method
        self.grabcut_iterations = grabcut_iterations
        self.soften = soften
        self._images: list[np.ndarray] = []

    def _load(self) -> None:
        """Nothing to load; present so cold-start time is measured as ~0 honestly."""

    def preprocess(
        self, images: Sequence[np.ndarray] | np.ndarray
    ) -> tuple[torch.Tensor, list[LetterboxInfo]]:
        """Letterbox as usual, but stash the uint8 pixels for OpenCV.

        The tensor is still produced (and still letterboxed at ``input_size``) so
        the harness measures the same resize cost it measures for the networks;
        the algorithms themselves need the uint8 image, which is carried
        alongside.
        """
        batch = [images] if isinstance(images, np.ndarray) and images.ndim == 3 else list(images)
        padded_images: list[np.ndarray] = []
        infos: list[LetterboxInfo] = []
        for img in batch:
            padded, info = letterbox(np.asarray(img), self.input_size)
            padded_images.append(padded)
            infos.append(info)
        self._images = padded_images
        stacked = np.stack(padded_images, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(stacked.transpose(0, 3, 1, 2))), infos

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return pseudo-logits so :meth:`postprocess` can stay shared.

        The classical methods produce probabilities directly, so they are mapped
        through a logit with a finite clamp: a hard 0/1 mask becomes +-6, which
        sigmoids back to 0.0025 / 0.9975 and survives the same soft-clip stage the
        networks go through.
        """
        images = self._images or [
            (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for t in tensor
        ]
        masks = [self._segment(img) for img in images]
        arr = np.stack(masks, axis=0)[:, None, :, :]
        clamped = np.clip(arr, 1e-3, 1 - 1e-3)
        logits = np.log(clamped / (1.0 - clamped)).astype(np.float32)
        return torch.from_numpy(logits)

    def _segment(self, image: np.ndarray) -> np.ndarray:
        if self.method == "grabcut":
            mask = grabcut_mask(image, iterations=self.grabcut_iterations)
        elif self.method == "saliency":
            saliency = spectral_residual_saliency(image)
            mask, _ = otsu_threshold(saliency)
        else:
            saliency = spectral_residual_saliency(image)
            seed, _ = otsu_threshold(saliency)
            mask = grabcut_mask(image, iterations=self.grabcut_iterations, init_mask=seed)

        if self.soften > 0:
            # A hard graph-cut boundary composites with visible aliasing; a small
            # blur is the fairest representation of what this baseline can do.
            mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), self.soften)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def postprocess(
        self, logits: torch.Tensor, infos: Sequence[LetterboxInfo]
    ) -> list[np.ndarray]:
        probs = torch.sigmoid(logits.float())
        if probs.ndim == 4:
            probs = probs[:, 0]
        arr = probs.cpu().numpy()
        return [unletterbox_mask(arr[i], infos[i]) for i in range(len(infos))]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            architecture=f"Classical/{self.method}",
            param_count=0,
            trainable_param_count=0,
            runtime="opencv+numpy",
            license="baseline implementation, MIT (algorithms: Hou & Zhang 2007; Rother et al. 2004)",
            source="https://doi.org/10.1109/CVPR.2007.383267",
            weights_sha256=None,
            notes=(
                "Zero-training baseline. Runs on CPU regardless of the requested "
                "device; reported device is the requested one for interface parity."
            ),
            **{**self._base_metadata_kwargs(), "randomly_initialized": False, "accuracy_valid": True},
        )


TrivialMethod = Literal["ones", "zeros", "center_ellipse"]


class TrivialBaseline(SegmentationModel):
    """Content-blind predictions, included to calibrate the accuracy column.

    IoU and MAE are only interpretable relative to what a model that *looks at
    nothing* achieves. On a dataset where the foreground covers 35% of the frame,
    predicting all-foreground already scores IoU 0.35, and a fixed centred ellipse
    scores more still if objects tend to be centred. Publishing those floors next to
    the real numbers is the difference between "0.60 IoU" meaning something and
    meaning nothing.

    ``center_ellipse`` doubles as a diagnostic for the dataset itself: if it scores
    highly, the generator has a centre-prior bias that inflates every other row.
    """

    def __init__(self, *, method: TrivialMethod = "center_ellipse", radius_frac: float = 0.32, **kwargs: Any) -> None:
        kwargs.setdefault("name", f"trivial-{method}")
        kwargs.setdefault("input_size", (320, 320))
        super().__init__(**kwargs)
        self.method: TrivialMethod = method
        self.radius_frac = radius_frac

    def _load(self) -> None:
        """Nothing to load."""

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        n, _, h, w = tensor.shape
        if self.method == "ones":
            mask = np.ones((h, w), dtype=np.float32)
        elif self.method == "zeros":
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            radial = ((xx - w / 2) / (self.radius_frac * w)) ** 2 + (
                (yy - h / 2) / (self.radius_frac * h)
            ) ** 2
            mask = (radial <= 1.0).astype(np.float32)
        clamped = np.clip(mask, 1e-3, 1 - 1e-3)
        logits = np.log(clamped / (1.0 - clamped)).astype(np.float32)
        return torch.from_numpy(np.broadcast_to(logits, (n, 1, h, w)).copy())

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            architecture=f"Trivial/{self.method}",
            param_count=0,
            trainable_param_count=0,
            runtime="numpy",
            license="MIT (this implementation)",
            source="metric calibration reference, not a real method",
            weights_sha256=None,
            notes=(
                "Content-blind reference. Any model that does not clearly beat this "
                "row has learned nothing about the images."
            ),
            **{**self._base_metadata_kwargs(), "randomly_initialized": False, "accuracy_valid": True},
        )
