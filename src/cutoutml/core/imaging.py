"""Image primitives: EXIF orientation, letterboxing, normalisation, compositing.

These functions are deliberately pure and NumPy-only so they can be unit tested
without a model, and so the exact same code runs in the image pipeline, the
video pipeline and the benchmark harness. Conventions used throughout CutoutML:

* **Images** are ``uint8`` arrays shaped ``(H, W, 3)`` in **RGB** order.
  OpenCV's native BGR is converted at the boundary, never carried around.
* **Alpha / masks** are ``float32`` arrays shaped ``(H, W)`` in ``[0, 1]``.
* **Tensors** are ``float32`` ``(N, 3, H, W)``, normalised.
"""

from __future__ import annotations

import dataclasses
import io
import math
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Resampling = Literal["bilinear", "bicubic", "area", "nearest"]

_CV_INTERP: dict[str, int] = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "nearest": cv2.INTER_NEAREST,
}


@dataclasses.dataclass(frozen=True, slots=True)
class LetterboxInfo:
    """Everything needed to invert a letterbox operation exactly.

    ``scale`` is the single factor applied to both axes (aspect ratio is
    preserved); ``pad_left``/``pad_top`` locate the resized content inside the
    padded canvas.
    """

    orig_width: int
    orig_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    scale: float

    @property
    def canvas_size(self) -> tuple[int, int]:
        """``(width, height)`` of the padded canvas."""
        return (
            self.resized_width + self.pad_left + self.pad_right,
            self.resized_height + self.pad_top + self.pad_bottom,
        )

    def as_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------- decoding


def decode_image(
    data: bytes, *, apply_exif: bool = True, max_pixels: int | None = None
) -> np.ndarray:
    """Decode encoded image bytes into an RGB ``uint8`` array.

    PIL is used rather than ``cv2.imdecode`` because it exposes EXIF and handles
    palette/16-bit/CMYK inputs more predictably. ``max_pixels`` guards against
    decompression-bomb uploads.
    """
    with Image.open(io.BytesIO(data)) as img:
        if apply_exif:
            img = ImageOps.exif_transpose(img) or img
        if max_pixels is not None and img.width * img.height > max_pixels:
            raise ValueError(
                f"image has {img.width * img.height} pixels which exceeds the limit of {max_pixels}"
            )
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def encode_image(
    image: np.ndarray,
    fmt: Literal["png", "webp", "jpeg"] = "png",
    *,
    alpha: np.ndarray | None = None,
    quality: int = 90,
) -> bytes:
    """Encode an RGB array (optionally with alpha) to bytes.

    JPEG has no alpha channel, so an alpha argument is ignored there; callers
    that need transparency must pick ``png`` or ``webp``.
    """
    if alpha is not None and fmt in {"png", "webp"}:
        rgba = np.dstack([image, to_uint8_alpha(alpha)])
        pil = Image.fromarray(rgba, mode="RGBA")
    else:
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")

    buf = io.BytesIO()
    if fmt == "png":
        pil.save(buf, format="PNG", optimize=True)
    elif fmt == "webp":
        pil.save(buf, format="WEBP", quality=quality, method=4)
    else:
        pil.save(buf, format="JPEG", quality=quality, subsampling=1)
    return buf.getvalue()


def encode_mask(alpha: np.ndarray) -> bytes:
    """Encode a float alpha map as an 8-bit grayscale PNG."""
    pil = Image.fromarray(to_uint8_alpha(alpha), mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_uint8_alpha(alpha: np.ndarray) -> np.ndarray:
    """Quantise a ``[0, 1]`` float alpha to ``uint8`` with rounding."""
    return np.clip(np.rint(np.asarray(alpha, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ letterbox


def letterbox(
    image: np.ndarray,
    target: tuple[int, int],
    *,
    pad_value: int = 0,
    interpolation: Resampling = "bilinear",
    allow_upscale: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize ``image`` into a ``target`` canvas preserving aspect ratio.

    Padding is split evenly so the content stays centred, which matters because
    several segmentation backbones are sensitive to where the object sits
    relative to the receptive-field centre. Returns the padded image plus the
    :class:`LetterboxInfo` needed by :func:`unletterbox_mask`.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB image, got shape {image.shape}")
    tw, th = target
    if tw <= 0 or th <= 0:
        raise ValueError(f"target must be positive, got {target}")

    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    if not allow_upscale:
        scale = min(scale, 1.0)

    # Round to nearest so that e.g. a 1:1 image into a square canvas needs no pad.
    nw = max(1, min(tw, round(w * scale)))
    nh = max(1, min(th, round(h * scale)))

    resized = (
        image
        if (nw, nh) == (w, h)
        else cv2.resize(image, (nw, nh), interpolation=_CV_INTERP[interpolation])
    )

    pad_w, pad_h = tw - nw, th - nh
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top

    if pad_w or pad_h:
        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
        )
    else:
        canvas = resized

    info = LetterboxInfo(
        orig_width=w,
        orig_height=h,
        resized_width=nw,
        resized_height=nh,
        pad_left=left,
        pad_top=top,
        pad_right=right,
        pad_bottom=bottom,
        scale=scale,
    )
    return canvas, info


def unletterbox_mask(
    mask: np.ndarray,
    info: LetterboxInfo,
    *,
    interpolation: Resampling = "bilinear",
) -> np.ndarray:
    """Crop the padding off ``mask`` and resize it back to the original size.

    This is the exact inverse of :func:`letterbox` up to resampling error, which
    the round-trip unit test asserts.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected (H, W) mask, got shape {mask.shape}")

    canvas_w, canvas_h = info.canvas_size
    if mask.shape[:2] != (canvas_h, canvas_w):
        mask = cv2.resize(mask, (canvas_w, canvas_h), interpolation=_CV_INTERP[interpolation])

    cropped = mask[
        info.pad_top : info.pad_top + info.resized_height,
        info.pad_left : info.pad_left + info.resized_width,
    ]
    if cropped.shape[:2] == (info.orig_height, info.orig_width):
        return np.ascontiguousarray(cropped, dtype=np.float32)
    out = cv2.resize(
        np.ascontiguousarray(cropped, dtype=np.float32),
        (info.orig_width, info.orig_height),
        interpolation=_CV_INTERP[interpolation],
    )
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------- normalisation


def normalize(
    image: np.ndarray,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Scale ``uint8`` RGB to ``float32`` CHW and apply per-channel normalisation."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def denormalize(
    chw: np.ndarray,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Inverse of :func:`normalize`, returning ``uint8`` HWC RGB."""
    arr = np.asarray(chw, dtype=np.float32).transpose(1, 2, 0)
    arr = arr * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- compositing


def composite_over_color(
    image: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    """Alpha-composite ``image`` over a solid colour.

    Straight (non-premultiplied) alpha: ``out = fg * a + bg * (1 - a)``.
    """
    a = _alpha3(alpha, image.shape[:2])
    bg = np.empty_like(image, dtype=np.float32)
    bg[:] = np.asarray(color, dtype=np.float32)
    out = image.astype(np.float32) * a + bg * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def composite_over_image(
    image: np.ndarray, alpha: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Composite over an arbitrary background, cover-cropping it to fit.

    Backgrounds are scaled to *cover* the foreground and centre-cropped rather
    than stretched, so user-supplied backdrops keep their aspect ratio.
    """
    h, w = image.shape[:2]
    bg = cover_resize(background, (w, h))
    a = _alpha3(alpha, (h, w))
    out = image.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def composite_blurred_background(
    image: np.ndarray, alpha: np.ndarray, *, blur_sigma: float = 12.0
) -> np.ndarray:
    """Portrait-mode style output: keep the subject, blur its own background.

    The blur is applied to the whole frame and then composited under the subject;
    blurring only the masked region would drag subject colours outward and halo.
    """
    if blur_sigma <= 0:
        return np.ascontiguousarray(image)
    ksize = int(max(3, 2 * round(3 * blur_sigma) + 1))
    blurred = cv2.GaussianBlur(image, (ksize, ksize), blur_sigma, borderType=cv2.BORDER_REFLECT)
    a = _alpha3(alpha, image.shape[:2])
    out = image.astype(np.float32) * a + blurred.astype(np.float32) * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def cover_resize(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Scale to cover ``(width, height)`` then centre-crop to exactly that size."""
    tw, th = target
    h, w = image.shape[:2]
    scale = max(tw / w, th / h)
    nw, nh = max(tw, math.ceil(w * scale)), max(th, math.ceil(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2
    return np.ascontiguousarray(resized[y0 : y0 + th, x0 : x0 + tw])


def _alpha3(alpha: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Broadcast a 2-D alpha map to ``(H, W, 1)`` float32, resizing if needed."""
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape[:2] != hw:
        a = cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(a, 0.0, 1.0)[..., None]


def unmultiply_alpha(image: np.ndarray, alpha: np.ndarray, *, eps: float = 1e-3) -> np.ndarray:
    """Recover straight-alpha colour from a premultiplied composite.

    Useful when exporting transparent PNGs from a source that has already been
    composited over black: without this, semi-transparent edges look too dark.
    """
    a = _alpha3(alpha, image.shape[:2])
    out = np.where(a > eps, image.astype(np.float32) / np.maximum(a, eps), image.astype(np.float32))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
