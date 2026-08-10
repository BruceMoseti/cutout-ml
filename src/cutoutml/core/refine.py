"""Alpha refinement.

A segmentation network run at 256-1024 px produces a *low-resolution, slightly
blobby* mask. Upsampling it naively to a 4000 px photo gives stair-stepped,
haloed edges. The refinement stack here is what turns a mask into a usable
cutout, and every stage is individually switchable so the benchmark harness can
measure its cost and its effect:

1. **Guided filter** - edge-aware joint upsampling using the colour image as the
   guide. Implemented directly (``cv2.ximgproc`` lives in opencv-contrib, which
   we do not depend on) so it works in the headless wheel.
2. **Threshold shaping** - optional gamma / soft-clip to pull near-0 and near-1
   values to the rails without hardening genuine soft edges (hair, motion blur).
3. **Morphological cleanup** - close pinholes, open speckles.
4. **Small-component removal** - drop connected foreground islands below a
   fraction of the largest component.
5. **Feathering** - a narrow blur restricted to the boundary band, which softens
   aliasing without eroding the interior.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np


@dataclasses.dataclass(slots=True)
class RefineConfig:
    """Knobs for :func:`refine_alpha`.

    Defaults are tuned for photographic cutouts at >=512 px; the video pipeline
    deliberately disables the guided filter by default because it is the most
    expensive stage per frame.
    """

    guided_filter: bool = True
    guided_radius: int = 8
    guided_eps: float = 1e-4
    soft_clip: bool = True
    clip_low: float = 0.02
    clip_high: float = 0.98
    gamma: float = 1.0
    morph_close: int = 0
    morph_open: int = 0
    min_component_ratio: float = 0.0
    feather_radius: int = 0
    boundary_band: int = 6

    def as_dict(self) -> dict[str, float | int | bool]:
        return dataclasses.asdict(self)

    @classmethod
    def fast(cls) -> RefineConfig:
        """Cheap preset: no guided filter, just rail-snapping. Used for video."""
        return cls(guided_filter=False, soft_clip=True, feather_radius=0)

    @classmethod
    def quality(cls) -> RefineConfig:
        """Full stack, for single-image requests where latency matters less."""
        return cls(
            guided_filter=True,
            guided_radius=12,
            soft_clip=True,
            morph_close=3,
            morph_open=3,
            min_component_ratio=0.02,
            feather_radius=2,
        )

    @classmethod
    def off(cls) -> RefineConfig:
        """No refinement at all - the baseline the benchmark compares against."""
        return cls(guided_filter=False, soft_clip=False)


def guided_filter(
    guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-4
) -> np.ndarray:
    """Edge-preserving joint filter (He et al., 2010), colour-guide variant.

    For each pixel we solve a local ridge regression ``a * I + b ~= p`` over a
    ``(2r+1)`` window, then average the per-window coefficients. This is the
    O(N) box-filter formulation, so cost is independent of ``radius``.

    ``guide`` may be ``(H, W)`` grayscale or ``(H, W, 3)``; the grayscale path is
    used because the full 3x3 colour covariance inverse costs ~4x more for a
    marginal quality gain on alpha maps.
    """
    if radius < 1:
        return np.asarray(src, dtype=np.float32)

    guide_f = np.asarray(guide, dtype=np.float32)
    if guide_f.ndim == 3:
        guide_f = cv2.cvtColor(guide_f, cv2.COLOR_RGB2GRAY)
    if guide_f.max() > 1.5:
        guide_f = guide_f / 255.0
    src_f = np.asarray(src, dtype=np.float32)

    if guide_f.shape[:2] != src_f.shape[:2]:
        src_f = cv2.resize(
            src_f, (guide_f.shape[1], guide_f.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    mean_i = box(guide_f)
    mean_p = box(src_f)
    corr_i = box(guide_f * guide_f)
    corr_ip = box(guide_f * src_f)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    return np.asarray(box(a) * guide_f + box(b), dtype=np.float32)


def soft_clip(alpha: np.ndarray, low: float = 0.02, high: float = 0.98) -> np.ndarray:
    """Rescale ``[low, high]`` onto ``[0, 1]`` and clamp outside.

    Networks trained with BCE rarely saturate fully, leaving a faint grey wash
    over the background (alpha ~0.03) that shows up as a halo when composited.
    This removes that wash while keeping the interior of the soft-edge band
    monotonic, unlike a hard threshold.
    """
    if high <= low:
        return np.clip(alpha, 0.0, 1.0)
    out = (np.asarray(alpha, dtype=np.float32) - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


def apply_gamma(alpha: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-shape the alpha ramp (``gamma > 1`` thins soft edges)."""
    if gamma == 1.0:
        return np.asarray(alpha, dtype=np.float32)
    return np.power(np.clip(alpha, 0.0, 1.0), gamma, dtype=np.float32)


def morphological_cleanup(alpha: np.ndarray, close: int = 0, open_: int = 0) -> np.ndarray:
    """Close small holes then remove small speckles.

    Order matters: closing first fills pinholes that opening would otherwise
    widen into visible gaps.
    """
    out = np.asarray(alpha, dtype=np.float32)
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_ > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_ + 1, 2 * open_ + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    return np.asarray(out, dtype=np.float32)


def remove_small_components(
    alpha: np.ndarray, min_ratio: float = 0.02, threshold: float = 0.5
) -> np.ndarray:
    """Zero foreground islands smaller than ``min_ratio`` of the largest one.

    Saliency-style models frequently fire on a bright background detail; the
    result is a correct main subject plus a few stray blobs. Keeping only
    components within a size ratio of the biggest is a cheap, effective fix that
    never touches single-object images.
    """
    if min_ratio <= 0:
        return np.asarray(alpha, dtype=np.float32)

    binary = (np.asarray(alpha, dtype=np.float32) >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:  # background + at most one component
        return np.asarray(alpha, dtype=np.float32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    keep = {i + 1 for i, area in enumerate(areas) if area >= min_ratio * largest}

    mask = np.isin(labels, list(keep))
    return np.where(mask, alpha, 0.0).astype(np.float32)


def feather_edges(alpha: np.ndarray, radius: int = 2, band: int = 6) -> np.ndarray:
    """Blur only the boundary band of the alpha map.

    A global blur would soften the whole silhouette and eat into the interior;
    restricting the blend to a dilated-minus-eroded band keeps solid regions
    exactly 1.0 and empty regions exactly 0.0.
    """
    if radius < 1:
        return np.asarray(alpha, dtype=np.float32)

    a = np.asarray(alpha, dtype=np.float32)
    ksize = 2 * radius + 1
    blurred = cv2.GaussianBlur(a, (ksize, ksize), 0, borderType=cv2.BORDER_REFLECT)

    binary = (a >= 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    edge = cv2.dilate(binary, k) - cv2.erode(binary, k)
    edge_f = edge.astype(np.float32)

    return np.asarray(a * (1.0 - edge_f) + blurred * edge_f, dtype=np.float32)


def refine_alpha(
    alpha: np.ndarray,
    image: np.ndarray | None = None,
    config: RefineConfig | None = None,
) -> np.ndarray:
    """Run the configured refinement stack over a full-resolution alpha map.

    ``image`` is the original RGB frame, required only for the guided filter.
    Returns a fresh ``float32`` array in ``[0, 1]``; the input is not mutated.
    """
    cfg = config or RefineConfig()
    out = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)

    if cfg.guided_filter and image is not None:
        out = guided_filter(image, out, radius=cfg.guided_radius, eps=cfg.guided_eps)
        out = np.clip(out, 0.0, 1.0)

    if cfg.soft_clip:
        out = soft_clip(out, cfg.clip_low, cfg.clip_high)

    if cfg.gamma != 1.0:
        out = apply_gamma(out, cfg.gamma)

    if cfg.morph_close or cfg.morph_open:
        out = morphological_cleanup(out, cfg.morph_close, cfg.morph_open)

    if cfg.min_component_ratio > 0:
        out = remove_small_components(out, cfg.min_component_ratio)

    if cfg.feather_radius > 0:
        out = feather_edges(out, cfg.feather_radius, cfg.boundary_band)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------- temporal (video)


def ema_smooth(previous: np.ndarray | None, current: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Exponential moving average between consecutive alpha maps.

    ``alpha`` weights the *current* frame: 1.0 disables smoothing, lower values
    trade responsiveness for stability. Cheap (one lerp per frame) and needs only
    one frame of state, which matters for a streaming pipeline.
    """
    cur = np.asarray(current, dtype=np.float32)
    if previous is None:
        return cur
    w = float(np.clip(alpha, 0.0, 1.0))
    return np.asarray(w * cur + (1.0 - w) * np.asarray(previous, dtype=np.float32), dtype=np.float32)


def temporal_median(window: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median over a short window of alpha maps.

    Removes single-frame dropouts that an EMA would merely attenuate, at the cost
    of ``len(window)`` frames of latency and memory.
    """
    if not window:
        raise ValueError("temporal_median requires a non-empty window")
    if len(window) == 1:
        return np.asarray(window[0], dtype=np.float32)
    return np.median(np.stack(window, axis=0), axis=0).astype(np.float32)


def temporal_flicker(alphas: list[np.ndarray]) -> float:
    """Mean absolute difference between consecutive alpha maps.

    This is the metric used to *measure* whether temporal smoothing helps: lower
    means less frame-to-frame jitter. It says nothing about accuracy - a frozen
    mask scores 0 - so it is always reported next to IoU.
    """
    if len(alphas) < 2:
        return 0.0
    diffs = [
        float(np.abs(alphas[i].astype(np.float32) - alphas[i - 1].astype(np.float32)).mean())
        for i in range(1, len(alphas))
    ]
    return float(np.mean(diffs))
