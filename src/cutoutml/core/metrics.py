"""Segmentation / matting quality metrics.

The definitions follow the salient-object-detection literature so numbers are
comparable with published DUTS / DIS5K results:

* **MAE** - mean absolute error over the continuous alpha, no threshold.
* **IoU** - intersection over union of the binarised masks at 0.5.
* **F-beta** with beta^2 = 0.3 - the SOD standard, which weights precision more
  heavily than recall because false positives are visually worse in a cutout.
* **max/mean F-beta** - swept over 255 thresholds, removing the arbitrariness of
  a single operating point.
* **S-measure** - structure similarity (Fan et al., 2017), object- and
  region-aware, catches masks with the right area but the wrong shape.
* **Boundary F1** - F1 restricted to a tolerance band around the true contour;
  this is what actually correlates with "does the cutout look good".
* **BER** - balanced error rate, robust to the heavy background bias of
  segmentation masks.

Every function takes ``pred``/``gt`` as float arrays in ``[0, 1]`` with the same
shape and returns plain Python floats so results are JSON-serialisable.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

EPS = 1e-8
FBETA_SQ = 0.3
"""beta^2 for F-beta, matching the salient object detection convention."""


@dataclasses.dataclass(frozen=True, slots=True)
class MaskMetrics:
    """Full metric bundle for one predicted alpha map."""

    mae: float
    iou: float
    dice: float
    f_beta: float
    f_beta_max: float
    f_beta_mean: float
    s_measure: float
    boundary_f1: float
    ber: float
    precision: float
    recall: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def _validate(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)
    if p.ndim == 3:
        p = p[..., 0]
    if g.ndim == 3:
        g = g[..., 0]
    if p.shape != g.shape:
        raise ValueError(f"prediction shape {p.shape} != ground truth shape {g.shape}")
    if p.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    return np.clip(p, 0.0, 1.0), np.clip(g, 0.0, 1.0)


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute error on the continuous alpha map."""
    p, g = _validate(pred, gt)
    return float(np.abs(p - g).mean())


def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Intersection over union after binarising both masks.

    An all-background prediction against an all-background truth is defined as
    1.0 (perfect) rather than 0/0.
    """
    p, g = _validate(pred, gt)
    pb, gb = p >= threshold, g >= threshold
    inter = float(np.logical_and(pb, gb).sum())
    union = float(np.logical_or(pb, gb).sum())
    if union == 0.0:
        return 1.0
    return inter / union


def dice(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Dice / F1 coefficient of the binarised masks."""
    p, g = _validate(pred, gt)
    pb, gb = p >= threshold, g >= threshold
    inter = float(np.logical_and(pb, gb).sum())
    total = float(pb.sum() + gb.sum())
    if total == 0.0:
        return 1.0
    return 2.0 * inter / total


def soft_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Threshold-free IoU on continuous values (the training-time IoU loss)."""
    p, g = _validate(pred, gt)
    inter = float((p * g).sum())
    union = float(p.sum() + g.sum() - inter)
    if union <= EPS:
        return 1.0
    return inter / union


def precision_recall(
    pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    """Pixel precision and recall of the foreground class."""
    p, g = _validate(pred, gt)
    pb, gb = p >= threshold, g >= threshold
    tp = float(np.logical_and(pb, gb).sum())
    fp = float(np.logical_and(pb, ~gb).sum())
    fn = float(np.logical_and(~pb, gb).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return prec, rec


def f_beta(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, beta_sq: float = FBETA_SQ) -> float:
    """F-beta at a fixed threshold with beta^2 = 0.3 by default."""
    prec, rec = precision_recall(pred, gt, threshold)
    denom = beta_sq * prec + rec
    if denom <= EPS:
        return 0.0
    return float((1.0 + beta_sq) * prec * rec / denom)


def f_beta_curve(
    pred: np.ndarray, gt: np.ndarray, *, steps: int = 255, beta_sq: float = FBETA_SQ
) -> np.ndarray:
    """F-beta swept across ``steps`` thresholds, vectorised via histograms.

    Naively looping 255 thresholds over a multi-megapixel mask is slow, so we
    build cumulative histograms of the prediction values inside and outside the
    ground-truth foreground and read precision/recall off them.
    """
    p, g = _validate(pred, gt)
    gb = g >= 0.5

    bins = steps + 1
    quant = np.clip((p * steps).astype(np.int32), 0, steps)

    hist_fg = np.bincount(quant[gb], minlength=bins).astype(np.float64)
    hist_bg = np.bincount(quant[~gb], minlength=bins).astype(np.float64)

    # tp[t] = #fg pixels with value >= t, fp[t] = #bg pixels with value >= t
    tp = np.cumsum(hist_fg[::-1])[::-1]
    fp = np.cumsum(hist_bg[::-1])[::-1]
    n_fg = float(hist_fg.sum())

    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, EPS), 1.0)
    rec = tp / n_fg if n_fg > 0 else np.ones_like(tp)

    denom = beta_sq * prec + rec
    fb = np.where(denom > EPS, (1.0 + beta_sq) * prec * rec / np.maximum(denom, EPS), 0.0)
    return np.asarray(fb[1:], dtype=np.float64)


def s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    """Structure measure S_alpha = alpha * S_object + (1 - alpha) * S_region."""
    p, g = _validate(pred, gt)
    gb = (g >= 0.5).astype(np.float32)
    y = float(gb.mean())
    if y == 0.0:
        return float(1.0 - p.mean())
    if y == 1.0:
        return float(p.mean())
    return float(alpha * _s_object(p, gb) + (1.0 - alpha) * _s_region(p, gb))


def _s_object(p: np.ndarray, gb: np.ndarray) -> float:
    def obj(vals: np.ndarray) -> float:
        if vals.size == 0:
            return 0.0
        mu = float(vals.mean())
        sigma = float(vals.std())
        return 2.0 * mu / (mu * mu + 1.0 + sigma + EPS)

    fg_score = obj(p[gb > 0.5])
    bg_score = obj(1.0 - p[gb <= 0.5])
    mu = float(gb.mean())
    return mu * fg_score + (1.0 - mu) * bg_score


def _s_region(p: np.ndarray, gb: np.ndarray) -> float:
    """Divide at the GT centroid into 4 quadrants and average weighted SSIM."""
    h, w = gb.shape
    total = float(gb.sum())
    if total == 0:
        return 0.0
    ys, xs = np.nonzero(gb > 0.5)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    cy = int(np.clip(cy, 1, h - 1))
    cx = int(np.clip(cx, 1, w - 1))

    quads = [
        (slice(0, cy), slice(0, cx)),
        (slice(0, cy), slice(cx, w)),
        (slice(cy, h), slice(0, cx)),
        (slice(cy, h), slice(cx, w)),
    ]
    score = 0.0
    for sy, sx in quads:
        weight = float(gb[sy, sx].sum()) / total
        score += weight * _ssim(p[sy, sx], gb[sy, sx])
    return score


def _ssim(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    n = float(x.size)
    mx, my = float(x.mean()), float(y.mean())
    vx = float(((x - mx) ** 2).sum() / (n - 1 + EPS))
    vy = float(((y - my) ** 2).sum() / (n - 1 + EPS))
    cov = float(((x - mx) * (y - my)).sum() / (n - 1 + EPS))
    num = 4.0 * mx * my * cov
    den = (mx**2 + my**2) * (vx + vy)
    if den > EPS:
        return num / den
    if num < EPS and den < EPS:
        return 1.0
    return 0.0


def boundary_f1(
    pred: np.ndarray, gt: np.ndarray, *, tolerance: int = 3, threshold: float = 0.5
) -> float:
    """F1 between predicted and true contours within ``tolerance`` pixels.

    Contours are extracted with a morphological gradient, then each side is
    matched against the other's dilated contour. This is the "trimap-free" BF
    measure and is far more sensitive to hair/edge quality than region IoU.
    """
    p, g = _validate(pred, gt)
    pb = (p >= threshold).astype(np.uint8)
    gb = (g >= threshold).astype(np.uint8)

    p_edge = _contour(pb)
    g_edge = _contour(gb)

    if p_edge.sum() == 0 and g_edge.sum() == 0:
        return 1.0
    if p_edge.sum() == 0 or g_edge.sum() == 0:
        return 0.0

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    p_dil = cv2.dilate(p_edge, k)
    g_dil = cv2.dilate(g_edge, k)

    precision = float((p_edge & g_dil).sum()) / float(p_edge.sum())
    recall = float((g_edge & p_dil).sum()) / float(g_edge.sum())
    if precision + recall <= EPS:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _contour(binary: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    return np.asarray(cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, k) > 0, dtype=np.uint8)


def ber(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Balanced error rate: ``1 - (sensitivity + specificity) / 2``.

    Reported as a percentage-free fraction in ``[0, 1]``; lower is better. Unlike
    plain accuracy it does not reward predicting everything as background on a
    mask that is 90% background.
    """
    p, g = _validate(pred, gt)
    pb, gb = p >= threshold, g >= threshold
    tp = float(np.logical_and(pb, gb).sum())
    tn = float(np.logical_and(~pb, ~gb).sum())
    fp = float(np.logical_and(pb, ~gb).sum())
    fn = float(np.logical_and(~pb, gb).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    return float(1.0 - 0.5 * (sens + spec))


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of correctly classified pixels (reported for completeness only)."""
    p, g = _validate(pred, gt)
    return float(((p >= threshold) == (g >= threshold)).mean())


def compute_all(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    threshold: float = 0.5,
    boundary_tolerance: int = 3,
) -> MaskMetrics:
    """Compute the full metric bundle in one pass."""
    curve = f_beta_curve(pred, gt)
    prec, rec = precision_recall(pred, gt, threshold)
    return MaskMetrics(
        mae=mae(pred, gt),
        iou=iou(pred, gt, threshold),
        dice=dice(pred, gt, threshold),
        f_beta=f_beta(pred, gt, threshold),
        f_beta_max=float(curve.max()),
        f_beta_mean=float(curve.mean()),
        s_measure=s_measure(pred, gt),
        boundary_f1=boundary_f1(pred, gt, tolerance=boundary_tolerance, threshold=threshold),
        ber=ber(pred, gt, threshold),
        precision=prec,
        recall=rec,
        accuracy=pixel_accuracy(pred, gt, threshold),
    )


def aggregate(metrics: list[MaskMetrics]) -> dict[str, float]:
    """Mean of each field across a dataset, plus the sample count."""
    if not metrics:
        return {}
    fields = [f.name for f in dataclasses.fields(MaskMetrics)]
    out = {name: float(np.mean([getattr(m, name) for m in metrics])) for name in fields}
    out["n_samples"] = float(len(metrics))
    return out
