"""Procedural primitives for the synthetic dataset.

No real segmentation dataset can be downloaded in this environment, so the
benchmark set is generated. That is a genuine limitation (documented in
``docs/decisions/ADR-004-synthetic-dataset.md``) but a generated set can be made
*hard in the ways that matter* for matting, which is what these primitives are
for:

* **Fractional alpha at edges.** Every shape mask is rasterised at 4x resolution
  and box-downsampled, so boundary pixels get true fractional coverage exactly
  like an anti-aliased photograph. A model that only ever predicts 0 or 1 loses
  measurable MAE, which is the point.
* **Shapes with real curvature statistics.** Smooth random blobs (radial Fourier
  series), superellipses, star polygons and text glyphs cover convex, concave,
  spiky and thin-stroke topologies. Thin strokes are where cheap models fail.
* **Non-trivial backgrounds.** Value-noise fBm, gradients, stripes and blurred
  colour fields, so the foreground cannot be found by "not flat".
* **Colour ambiguity.** Foreground and background palettes are drawn so they
  sometimes collide, which is the failure mode that separates a network from a
  colour-clustering baseline like GrabCut.

Every function takes an explicit ``numpy.random.Generator``; nothing touches
global random state, which is what makes the whole dataset reproducible from one
integer seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MakerFn = Callable[[np.random.Generator, "tuple[int, int]"], np.ndarray]
"""Every maker below takes ``(rng, (h, w))`` and returns a float32 array. Naming the
shape means the registry dicts stay callable to a type checker instead of collapsing to
``object`` across their heterogeneous keyword-only parameters."""

SUPERSAMPLE = 4
"""Rasterisation factor for anti-aliased masks. 4x gives 17 distinct alpha levels
per boundary pixel, which is plenty and costs 16x the raster area."""


# --------------------------------------------------------------------- noise


def value_noise(rng: np.random.Generator, size: tuple[int, int], cells: int = 4) -> np.ndarray:
    """Smooth ``[0, 1]`` noise from a bicubically upsampled random lattice.

    A true Perlin implementation buys nothing here: what matters is a smooth,
    band-limited random field, and bicubic interpolation of a small lattice gives
    that in three lines. ``cells`` sets the spatial frequency.
    """
    h, w = size
    cells = max(2, cells)
    lattice = rng.random((cells, cells), dtype=np.float32)
    out = cv2.resize(lattice, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(out.min()), float(out.max())
    if hi - lo < 1e-8:
        return np.full((h, w), 0.5, dtype=np.float32)
    return ((out - lo) / (hi - lo)).astype(np.float32)


def fbm(
    rng: np.random.Generator,
    size: tuple[int, int],
    *,
    octaves: int = 4,
    base_cells: int = 3,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Fractal Brownian motion: octaves of :func:`value_noise` at halving weight.

    Produces the ``1/f``-ish spectrum of natural textures, so a saliency baseline
    that keys on spectral irregularity is genuinely challenged by the background
    rather than trivially defeating it.
    """
    h, w = size
    total = np.zeros((h, w), dtype=np.float32)
    amplitude, weight_sum = 1.0, 0.0
    cells = float(base_cells)
    for _ in range(max(1, octaves)):
        total += amplitude * value_noise(rng, size, cells=round(cells))
        weight_sum += amplitude
        amplitude *= gain
        cells *= lacunarity
    out = total / max(weight_sum, 1e-8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# -------------------------------------------------------------------- shapes


def _downsample_mask(mask: np.ndarray, factor: int = SUPERSAMPLE) -> np.ndarray:
    """Box-average a supersampled binary mask into a fractional-alpha mask."""
    h, w = mask.shape[0] // factor, mask.shape[1] // factor
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def blob_mask(
    rng: np.random.Generator,
    size: tuple[int, int],
    *,
    harmonics: int = 6,
    roughness: float = 0.35,
    radius_frac: float = 0.38,
) -> np.ndarray:
    """A smooth closed random blob from a radial Fourier series.

    ``r(theta) = R * (1 + sum_k a_k cos(k*theta + phi_k))`` with amplitudes decaying
    as ``1/k``. Guaranteed star-convex and self-intersection-free, unlike random
    control-point splines, so the resulting mask is always a single clean region.
    """
    h, w = size
    hh, ww = h * SUPERSAMPLE, w * SUPERSAMPLE

    theta = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False, dtype=np.float32)
    radius = np.ones_like(theta)
    for k in range(1, harmonics + 1):
        amp = rng.normal(0.0, roughness / k)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        radius += amp * np.cos(k * theta + phase)
    radius = np.clip(radius, 0.25, 1.9)

    base_r = radius_frac * min(hh, ww)
    cx = ww / 2.0 + rng.uniform(-0.06, 0.06) * ww
    cy = hh / 2.0 + rng.uniform(-0.06, 0.06) * hh
    pts = np.stack(
        [cx + base_r * radius * np.cos(theta), cy + base_r * radius * np.sin(theta)],
        axis=1,
    )

    canvas = np.zeros((hh, ww), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], color=1)
    return _downsample_mask(canvas)


def superellipse_mask(
    rng: np.random.Generator,
    size: tuple[int, int],
    *,
    exponent: float | None = None,
    radius_frac: float = 0.38,
) -> np.ndarray:
    """A rotated superellipse ``|x/a|^n + |y/b|^n <= 1``.

    ``n`` sweeps from a diamond (n<1) through a circle (n=2) to a squarish shape
    (n>4), all with analytically exact boundaries. Evaluated directly on the
    supersampled grid rather than polygonised, which is both faster and gives a
    sharper reference.
    """
    h, w = size
    hh, ww = h * SUPERSAMPLE, w * SUPERSAMPLE
    n = float(exponent if exponent is not None else rng.uniform(0.6, 6.0))

    a = radius_frac * ww * rng.uniform(0.8, 1.2)
    b = radius_frac * hh * rng.uniform(0.8, 1.2)
    angle = rng.uniform(0.0, math.pi)

    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    xx = xx - ww / 2.0
    yy = yy - hh / 2.0
    ca, sa = math.cos(angle), math.sin(angle)
    xr = xx * ca + yy * sa
    yr = -xx * sa + yy * ca

    value = np.abs(xr / a) ** n + np.abs(yr / b) ** n
    return _downsample_mask((value <= 1.0).astype(np.uint8))


def star_polygon_mask(
    rng: np.random.Generator,
    size: tuple[int, int],
    *,
    points: int | None = None,
    inner_ratio: float | None = None,
    radius_frac: float = 0.42,
) -> np.ndarray:
    """A rotated star polygon: sharp concave corners that test boundary metrics.

    Low ``inner_ratio`` produces thin spikes, which is exactly the regime where
    IoU stays high while boundary F1 collapses - useful for showing why both
    metrics are reported.
    """
    h, w = size
    hh, ww = h * SUPERSAMPLE, w * SUPERSAMPLE
    k = int(points if points is not None else rng.integers(4, 10))
    ratio = float(inner_ratio if inner_ratio is not None else rng.uniform(0.32, 0.62))

    r_out = radius_frac * min(hh, ww)
    r_in = r_out * ratio
    offset = rng.uniform(0.0, 2.0 * math.pi)

    pts = []
    for i in range(2 * k):
        angle = offset + i * math.pi / k
        r = r_out if i % 2 == 0 else r_in
        pts.append((ww / 2.0 + r * math.cos(angle), hh / 2.0 + r * math.sin(angle)))

    canvas = np.zeros((hh, ww), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(np.array(pts)).astype(np.int32)], color=1)
    return _downsample_mask(canvas)


def rounded_rect_mask(
    rng: np.random.Generator, size: tuple[int, int], *, radius_frac: float = 0.4
) -> np.ndarray:
    """An axis-rotated rounded rectangle. The easy case, kept for calibration."""
    h, w = size
    hh, ww = h * SUPERSAMPLE, w * SUPERSAMPLE
    rw = int(radius_frac * ww * rng.uniform(1.0, 1.7))
    rh = int(radius_frac * hh * rng.uniform(1.0, 1.7))
    corner = int(min(rw, rh) * rng.uniform(0.05, 0.45))

    canvas = np.zeros((hh, ww), dtype=np.uint8)
    x0, y0 = (ww - rw) // 2, (hh - rh) // 2
    cv2.rectangle(canvas, (x0 + corner, y0), (x0 + rw - corner, y0 + rh), 1, -1)
    cv2.rectangle(canvas, (x0, y0 + corner), (x0 + rw, y0 + rh - corner), 1, -1)
    for cx, cy in (
        (x0 + corner, y0 + corner),
        (x0 + rw - corner, y0 + corner),
        (x0 + corner, y0 + rh - corner),
        (x0 + rw - corner, y0 + rh - corner),
    ):
        cv2.circle(canvas, (cx, cy), corner, 1, -1)

    mask = _downsample_mask(canvas)
    angle = rng.uniform(-45.0, 45.0)
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)


_GLYPHS = "ABCDEFGHKMNPRSVWXYZ2345689&@#%?"
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """First available system font at ``px``, falling back to PIL's bitmap font.

    The fallback keeps the generator working on a minimal container where no
    TrueType font is installed; glyph masks are then blockier, which the manifest
    records via ``font``.
    """
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def glyph_mask(
    rng: np.random.Generator, size: tuple[int, int], *, text: str | None = None
) -> np.ndarray:
    """A rendered letter/digit: thin strokes, holes and sharp corners.

    Glyphs are the hardest class in this dataset. They have high perimeter per unit
    area, interior holes (``A``, ``8``, ``@``) that punish morphological
    hole-filling, and stroke widths near the model's effective resolution.
    """
    h, w = size
    hh, ww = h * SUPERSAMPLE, w * SUPERSAMPLE
    char = text if text is not None else str(rng.choice(list(_GLYPHS)))

    img = Image.new("L", (ww, hh), 0)
    draw = ImageDraw.Draw(img)
    font = _load_font(int(min(hh, ww) * rng.uniform(0.62, 0.92)))
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((ww - tw) / 2 - bbox[0], (hh - th) / 2 - bbox[1]), char, fill=255, font=font)

    mask = _downsample_mask((np.asarray(img) > 127).astype(np.uint8))
    angle = rng.uniform(-25.0, 25.0)
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)


def layered_mask(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Union of 2-3 primitives at random offsets: multi-part, sometimes disjoint.

    Disjoint components are what make ``remove_small_components`` a real
    trade-off rather than a free win, so the dataset has to contain them.
    """
    h, w = size
    makers = (blob_mask, superellipse_mask, star_polygon_mask, rounded_rect_mask)
    count = int(rng.integers(2, 4))
    out = np.zeros((h, w), dtype=np.float32)
    for _ in range(count):
        maker = makers[int(rng.integers(0, len(makers)))]
        part = maker(rng, size, radius_frac=float(rng.uniform(0.16, 0.3)))
        dx = int(rng.uniform(-0.28, 0.28) * w)
        dy = int(rng.uniform(-0.28, 0.28) * h)
        shifted = cv2.warpAffine(
            part,
            np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
        out = np.maximum(out, shifted)
    return out


SHAPE_MAKERS: dict[str, MakerFn] = {
    "blob": blob_mask,
    "superellipse": superellipse_mask,
    "star": star_polygon_mask,
    "rounded_rect": rounded_rect_mask,
    "glyph": glyph_mask,
    "layered": layered_mask,
}


# ---------------------------------------------------------------- backgrounds


def _random_color(rng: np.random.Generator, *, lo: int = 10, hi: int = 245) -> np.ndarray:
    return rng.integers(lo, hi, size=3).astype(np.float32)


def bg_linear_gradient(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Two-colour linear gradient at a random angle."""
    h, w = size
    c0, c1 = _random_color(rng), _random_color(rng)
    angle = rng.uniform(0.0, math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = xx * math.cos(angle) + yy * math.sin(angle)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
    return (c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]).astype(
        np.float32
    )


def bg_radial_gradient(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Radial gradient with an off-centre focus - a vignette-like backdrop."""
    h, w = size
    c0, c1 = _random_color(rng), _random_color(rng)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    t = np.clip(r / max(float(r.max()), 1e-6), 0.0, 1.0)
    return (c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]).astype(
        np.float32
    )


def bg_fbm(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Coloured fBm noise: a cluttered, natural-ish texture."""
    c0, c1 = _random_color(rng), _random_color(rng)
    n = fbm(rng, size, octaves=int(rng.integers(3, 6)), base_cells=int(rng.integers(2, 6)))
    t = n[..., None]
    return (c0[None, None, :] * (1 - t) + c1[None, None, :] * t).astype(np.float32)


def bg_stripes(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Rotated periodic stripes; high-frequency structure that fools saliency."""
    h, w = size
    c0, c1 = _random_color(rng), _random_color(rng)
    period = float(rng.uniform(6.0, 42.0))
    angle = rng.uniform(0.0, math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = (xx * math.cos(angle) + yy * math.sin(angle)) / period
    wave = 0.5 + 0.5 * np.sin(2 * math.pi * phase)
    if rng.random() < 0.5:  # hard-edged bars rather than a sine
        wave = (wave > 0.5).astype(np.float32)
    t = wave[..., None]
    return (c0[None, None, :] * (1 - t) + c1[None, None, :] * t).astype(np.float32)


def bg_blurred_field(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Heavily blurred random colour patches - a bokeh stand-in."""
    h, w = size
    small = rng.integers(0, 256, size=(int(rng.integers(3, 9)), int(rng.integers(3, 9)), 3))
    field = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    sigma = float(rng.uniform(4.0, 18.0))
    return cv2.GaussianBlur(field, (0, 0), sigma).astype(np.float32)


def bg_noise_photo(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Sharp per-pixel noise over a smooth base: worst case for edge detectors."""
    h, w = size
    base = bg_blurred_field(rng, size)
    grain = rng.normal(0.0, rng.uniform(8.0, 30.0), size=(h, w, 3)).astype(np.float32)
    return np.clip(base + grain, 0.0, 255.0)


BACKGROUND_MAKERS: dict[str, MakerFn] = {
    "linear_gradient": bg_linear_gradient,
    "radial_gradient": bg_radial_gradient,
    "fbm": bg_fbm,
    "stripes": bg_stripes,
    "blurred_field": bg_blurred_field,
    "noise_photo": bg_noise_photo,
}


# ------------------------------------------------------------------- shading


def shade_foreground(
    rng: np.random.Generator,
    mask: np.ndarray,
    *,
    texture: bool = True,
) -> np.ndarray:
    """Give a mask a plausible surface: base colour, lighting ramp, texture.

    Without this the foreground is a flat colour patch and the task degenerates to
    colour thresholding. The directional ramp plus fBm modulation means interior
    pixels vary as much as the background does, so a model has to use shape.
    """
    h, w = mask.shape[:2]
    base = _random_color(rng, lo=20, hi=235)

    # Directional lighting ramp across the object.
    angle = rng.uniform(0.0, 2.0 * math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (xx / w) * math.cos(angle) + (yy / h) * math.sin(angle)
    ramp = (ramp - ramp.min()) / max(float(ramp.max() - ramp.min()), 1e-6)
    strength = float(rng.uniform(0.15, 0.55))
    shading = 1.0 - strength * 0.5 + strength * ramp

    rgb = base[None, None, :] * shading[..., None]

    if texture:
        tex = fbm(rng, (h, w), octaves=3, base_cells=int(rng.integers(3, 10)))
        amount = float(rng.uniform(0.05, 0.3))
        rgb = rgb * (1.0 - amount + 2.0 * amount * tex[..., None])

    # A rim highlight along the boundary, as a real lit object would have.
    if rng.random() < 0.6:
        binary = (mask > 0.5).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        rim = (binary - cv2.erode(binary, k)).astype(np.float32)
        rim = cv2.GaussianBlur(rim, (0, 0), 1.5)
        rgb = rgb + rim[..., None] * float(rng.uniform(20.0, 70.0))

    return np.clip(rgb, 0.0, 255.0).astype(np.float32)
