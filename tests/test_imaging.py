"""Preprocessing, letterboxing and compositing.

These are the functions where an off-by-one silently degrades every model in the
repository, so the assertions here are exact rather than approximate wherever the maths
allows it. The letterbox round-trip is the important one: if it is wrong, masks come back
shifted by a pixel or two and *every* accuracy number in ``docs/benchmarks.md`` is wrong
by a small, plausible-looking amount.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from cutoutml.core.imaging import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    composite_blurred_background,
    composite_over_color,
    composite_over_image,
    cover_resize,
    decode_image,
    denormalize,
    encode_image,
    encode_mask,
    letterbox,
    normalize,
    to_uint8_alpha,
    unletterbox_mask,
    unmultiply_alpha,
)


def _gradient(height: int, width: int) -> np.ndarray:
    """A deterministic image whose every pixel is distinguishable."""
    yy, xx = np.mgrid[0:height, 0:width]
    return np.dstack(
        [
            (xx * 255 // max(1, width - 1)).astype(np.uint8),
            (yy * 255 // max(1, height - 1)).astype(np.uint8),
            np.full((height, width), 128, dtype=np.uint8),
        ]
    )


# ----------------------------------------------------------------- letterboxing


@pytest.mark.parametrize(
    ("src", "target"),
    [
        ((100, 100), (64, 64)),  # square into square: no padding at all
        ((100, 50), (64, 64)),  # wide: pad top/bottom
        ((50, 100), (64, 64)),  # tall: pad left/right
        ((37, 91), (128, 128)),  # upscale with odd sizes
        ((64, 64), (64, 64)),  # identity
    ],
)
def test_letterbox_fills_canvas_and_preserves_aspect(src, target):
    image = _gradient(*src)
    canvas, info = letterbox(image, target)

    assert canvas.shape == (target[1], target[0], 3)
    assert info.canvas_size == target
    assert (info.orig_height, info.orig_width) == src
    # Aspect ratio preserved: the scale applied to both axes is the same number.
    assert info.resized_width == max(1, min(target[0], round(src[1] * info.scale)))
    assert info.resized_height == max(1, min(target[1], round(src[0] * info.scale)))
    # Padding is split as evenly as an odd remainder allows.
    assert abs(info.pad_left - info.pad_right) <= 1
    assert abs(info.pad_top - info.pad_bottom) <= 1


def test_letterbox_square_into_square_needs_no_padding():
    canvas, info = letterbox(_gradient(80, 80), (40, 40))
    assert (info.pad_left, info.pad_top, info.pad_right, info.pad_bottom) == (0, 0, 0, 0)
    assert canvas.shape == (40, 40, 3)


def test_letterbox_pad_value_is_used_for_the_bars():
    canvas, info = letterbox(_gradient(50, 100), (64, 64), pad_value=17)
    assert info.pad_top > 0
    assert np.all(canvas[: info.pad_top] == 17)
    assert np.all(canvas[info.pad_top + info.resized_height :] == 17)


def test_letterbox_no_upscale_keeps_original_scale():
    _, info = letterbox(_gradient(20, 20), (100, 100), allow_upscale=False)
    assert info.scale == 1.0
    assert (info.resized_width, info.resized_height) == (20, 20)


def test_letterbox_rejects_non_rgb():
    with pytest.raises(ValueError, match="RGB image"):
        letterbox(np.zeros((10, 10), dtype=np.uint8), (8, 8))


def test_letterbox_rejects_non_positive_target():
    with pytest.raises(ValueError, match="positive"):
        letterbox(_gradient(10, 10), (0, 8))


@pytest.mark.parametrize(
    ("src", "target"),
    [((100, 50), (64, 64)), ((50, 100), (64, 64)), ((91, 37), (128, 128))],
)
def test_unletterbox_mask_is_the_inverse_of_letterbox(src, target):
    """A mask painted on the canvas must come back at the original resolution.

    The tolerance is on the *area*, not per pixel: the round trip resamples twice, so a
    boundary pixel can legitimately land either side. What must not happen is a shift, and
    a shifted mask changes the centroid, which is what this checks.
    """
    image = _gradient(*src)
    _, info = letterbox(image, target)

    truth = np.zeros(src, dtype=np.float32)
    h, w = src
    truth[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0

    canvas_mask = np.zeros((target[1], target[0]), dtype=np.float32)
    import cv2

    resized = cv2.resize(
        truth, (info.resized_width, info.resized_height), interpolation=cv2.INTER_NEAREST
    )
    canvas_mask[
        info.pad_top : info.pad_top + info.resized_height,
        info.pad_left : info.pad_left + info.resized_width,
    ] = resized

    recovered = unletterbox_mask(canvas_mask, info)
    assert recovered.shape == src
    assert recovered.dtype == np.float32

    ys, xs = np.nonzero(recovered > 0.5)
    ty, tx = np.nonzero(truth > 0.5)
    assert abs(ys.mean() - ty.mean()) < 1.5
    assert abs(xs.mean() - tx.mean()) < 1.5
    assert abs(float((recovered > 0.5).mean()) - float(truth.mean())) < 0.05


def test_unletterbox_mask_resizes_a_low_resolution_mask_first():
    """Networks emit masks at their own resolution, not at the canvas resolution."""
    _, info = letterbox(_gradient(80, 120), (64, 64))
    coarse = np.ones((16, 16), dtype=np.float32)
    out = unletterbox_mask(coarse, info)
    assert out.shape == (80, 120)


def test_unletterbox_mask_rejects_3d_input():
    _, info = letterbox(_gradient(10, 10), (8, 8))
    with pytest.raises(ValueError, match="expected"):
        unletterbox_mask(np.zeros((8, 8, 1), dtype=np.float32), info)


def test_letterbox_info_serialises():
    _, info = letterbox(_gradient(10, 20), (8, 8))
    payload = info.as_dict()
    assert payload["orig_width"] == 20
    assert payload["orig_height"] == 10
    assert set(payload) >= {"scale", "pad_left", "pad_top"}


# --------------------------------------------------------------- normalisation


def test_normalize_produces_chw_float_with_imagenet_statistics():
    image = np.full((4, 6, 3), 128, dtype=np.uint8)
    out = normalize(image)
    assert out.shape == (3, 4, 6)
    assert out.dtype == np.float32
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        assert out[channel] == pytest.approx((128 / 255.0 - mean) / std, abs=1e-5)


def test_normalize_denormalize_round_trips_within_quantisation_error():
    image = _gradient(12, 15)
    recovered = denormalize(normalize(image))
    assert recovered.shape == image.shape
    assert np.abs(recovered.astype(int) - image.astype(int)).max() <= 1


def test_normalize_with_identity_statistics_is_a_plain_rescale():
    image = np.full((2, 2, 3), 51, dtype=np.uint8)
    out = normalize(image, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert out == pytest.approx(np.full((3, 2, 2), 0.2, dtype=np.float32), abs=1e-6)


# ----------------------------------------------------------- encode and decode


def test_decode_image_returns_rgb_uint8():
    data = encode_image(_gradient(9, 11), "png")
    decoded = decode_image(data)
    assert decoded.shape == (9, 11, 3)
    assert decoded.dtype == np.uint8


def test_decode_image_converts_grayscale_and_palette_to_rgb():
    for mode in ("L", "P"):
        buf = io.BytesIO()
        Image.new(mode, (8, 8), 3).save(buf, format="PNG")
        assert decode_image(buf.getvalue()).shape == (8, 8, 3)


def test_decode_image_enforces_the_pixel_budget():
    data = encode_image(_gradient(40, 40), "png")
    with pytest.raises(ValueError, match="exceeds the limit"):
        decode_image(data, max_pixels=100)


def test_decode_image_applies_exif_orientation():
    """EXIF orientation 6 means "rotate 90 degrees clockwise on display".

    Without honouring it, a phone photo is segmented sideways: the model sees a rotated
    subject and the returned alpha does not line up with what the user uploaded.
    """
    portrait = _gradient(40, 20)  # 40 tall, 20 wide
    pil = Image.fromarray(portrait)
    exif = pil.getexif()
    exif[274] = 6
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", exif=exif)

    rotated = decode_image(buf.getvalue(), apply_exif=True)
    assert rotated.shape[:2] == (20, 40)

    raw = decode_image(buf.getvalue(), apply_exif=False)
    assert raw.shape[:2] == (40, 20)


def test_encode_image_png_carries_alpha():
    image = _gradient(8, 8)
    alpha = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    data = encode_image(image, "png", alpha=alpha)
    with Image.open(io.BytesIO(data)) as decoded:
        assert decoded.mode == "RGBA"
        channel = np.asarray(decoded)[..., 3]
    assert channel[0, 0] == 0
    assert channel[-1, -1] == 255


def test_encode_image_jpeg_drops_alpha_rather_than_failing():
    """JPEG has no alpha channel. Silently producing an opaque file is the right call
    here because the caller explicitly asked for JPEG."""
    data = encode_image(_gradient(8, 8), "jpeg", alpha=np.zeros((8, 8), np.float32))
    with Image.open(io.BytesIO(data)) as decoded:
        assert decoded.mode == "RGB"


def test_encode_mask_is_single_channel_grayscale():
    alpha = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    with Image.open(io.BytesIO(encode_mask(alpha))) as decoded:
        assert decoded.mode == "L"
        arr = np.asarray(decoded)
    assert arr.tolist() == [[0, 128], [255, 64]]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0), (1.0, 255), (0.5, 128), (0.002, 1), (-1.0, 0), (2.0, 255)],
)
def test_to_uint8_alpha_rounds_and_clamps(value, expected):
    assert to_uint8_alpha(np.full((2, 2), value, dtype=np.float32))[0, 0] == expected


# ------------------------------------------------------------------ compositing


def test_composite_over_color_matches_the_alpha_blend_formula():
    image = np.full((2, 2, 3), 200, dtype=np.uint8)
    alpha = np.array([[1.0, 0.0], [0.5, 0.25]], dtype=np.float32)
    out = composite_over_color(image, alpha, (0, 100, 0))

    assert out[0, 0].tolist() == [200, 200, 200]  # fully opaque: foreground survives
    assert out[0, 1].tolist() == [0, 100, 0]  # fully transparent: background only
    assert out[1, 0].tolist() == [100, 150, 100]  # half and half
    assert out[1, 1].tolist() == [50, 125, 50]


def test_composite_over_image_cover_crops_the_background():
    image = np.full((10, 10, 3), 255, dtype=np.uint8)
    alpha = np.zeros((10, 10), dtype=np.float32)
    background = np.zeros((40, 80, 3), dtype=np.uint8)
    background[:, :] = (10, 20, 30)
    out = composite_over_image(image, alpha, background)
    assert out.shape == (10, 10, 3)
    assert np.all(out == (10, 20, 30))


def test_composite_blurred_background_keeps_the_subject_pixel_exact():
    """The subject must be untouched: a blur that bleeds inside the mask is the
    difference between portrait mode and a smeared face."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    alpha = np.zeros((48, 48), dtype=np.float32)
    alpha[16:32, 16:32] = 1.0

    out = composite_blurred_background(image, alpha, blur_sigma=6.0)
    assert np.array_equal(out[16:32, 16:32], image[16:32, 16:32])
    assert not np.array_equal(out[:8, :8], image[:8, :8])


def test_composite_blurred_background_with_zero_sigma_is_a_copy():
    image = _gradient(8, 8)
    out = composite_blurred_background(image, np.zeros((8, 8), np.float32), blur_sigma=0.0)
    assert np.array_equal(out, image)


def test_alpha_is_resized_when_it_does_not_match_the_image():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    small_alpha = np.ones((5, 5), dtype=np.float32)
    out = composite_over_color(image, small_alpha, (0, 0, 0))
    assert out.shape == (20, 20, 3)
    assert np.all(out == 255)


@pytest.mark.parametrize(
    ("src", "target"),
    [((10, 20), (5, 5)), ((20, 10), (5, 5)), ((7, 7), (13, 11))],
)
def test_cover_resize_produces_exactly_the_target_size(src, target):
    out = cover_resize(_gradient(*src), target)
    assert out.shape == (target[1], target[0], 3)


def test_unmultiply_alpha_recovers_colour_from_a_premultiplied_composite():
    colour = np.full((4, 4, 3), 200, dtype=np.float32)
    alpha = np.full((4, 4), 0.5, dtype=np.float32)
    premultiplied = (colour * alpha[..., None]).astype(np.uint8)
    recovered = unmultiply_alpha(premultiplied, alpha)
    assert np.abs(recovered.astype(int) - 200).max() <= 1


def test_unmultiply_alpha_leaves_fully_transparent_pixels_alone():
    image = np.full((2, 2, 3), 30, dtype=np.uint8)
    out = unmultiply_alpha(image, np.zeros((2, 2), np.float32))
    assert np.array_equal(out, image)
