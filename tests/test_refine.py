"""Alpha refinement stages.

Each stage is tested for the property it exists to provide rather than for exact
pixel values, because the useful contract is behavioural: the guided filter must
follow image edges, soft clip must kill the background wash, feathering must not
erode the interior. Exact outputs would just pin OpenCV's box-filter arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from cutoutml.core.refine import (
    RefineConfig,
    apply_gamma,
    ema_smooth,
    feather_edges,
    guided_filter,
    morphological_cleanup,
    refine_alpha,
    remove_small_components,
    soft_clip,
    temporal_flicker,
    temporal_median,
)


@pytest.fixture
def blocky() -> tuple[np.ndarray, np.ndarray]:
    """A hard-edged square and the RGB image whose edge coincides with it."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = (220, 210, 200)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0
    return alpha, image


# --------------------------------------------------------------- guided filter


def test_guided_filter_preserves_an_edge_that_the_guide_agrees_with(blocky):
    alpha, image = blocky
    out = guided_filter(image, alpha, radius=4, eps=1e-6)
    # Deep interior and far background stay at the rails; a plain box blur of the
    # same radius would not touch them either, so the discriminating check is the
    # edge column below.
    assert out[32, 32] == pytest.approx(1.0, abs=0.05)
    assert out[2, 2] == pytest.approx(0.0, abs=0.05)
    # One pixel outside the square, the guide says "background", so the filter must
    # not bleed foreground across: a box blur with radius 4 would leave ~0.4 here.
    assert out[32, 15] < 0.25


def test_guided_filter_upsamples_a_low_resolution_mask_to_the_guide(blocky):
    """The real use: a 64x64 network output joint-upsampled against a 256x256 photo."""
    alpha, _ = blocky
    big_image = np.zeros((256, 256, 3), dtype=np.uint8)
    big_image[64:192, 64:192] = (220, 210, 200)

    out = guided_filter(big_image, alpha, radius=8, eps=1e-4)

    assert out.shape == (256, 256)
    assert out[128, 128] > 0.9
    assert out[8, 8] < 0.1


def test_guided_filter_with_radius_below_one_is_a_pass_through(blocky):
    alpha, image = blocky
    assert np.array_equal(guided_filter(image, alpha, radius=0), alpha)


def test_guided_filter_accepts_a_grayscale_guide(blocky):
    alpha, image = blocky
    gray = image[:, :, 0]
    out = guided_filter(gray, alpha, radius=4)
    assert out.shape == alpha.shape
    assert out.dtype == np.float32


# ------------------------------------------------------------------- soft clip


def test_soft_clip_removes_the_background_wash_and_saturates_the_interior():
    """The point of the stage: alpha 0.03 over the background is a visible halo."""
    alpha = np.array([[0.0, 0.02, 0.03, 0.5, 0.97, 0.98, 1.0]], dtype=np.float32)
    out = soft_clip(alpha, 0.02, 0.98)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 0.0
    assert out[0, 4] == pytest.approx((0.97 - 0.02) / 0.96)
    assert out[0, 5] == 1.0
    assert out[0, 6] == 1.0


def test_soft_clip_is_monotonic_so_soft_edges_survive():
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :]
    out = soft_clip(ramp, 0.1, 0.9)[0]
    assert np.all(np.diff(out) >= 0.0)
    # A hard threshold would collapse the ramp to two values; this keeps a gradient.
    assert len(np.unique(out)) > 40


def test_soft_clip_with_an_inverted_range_only_clamps():
    alpha = np.array([[-0.5, 0.3, 1.7]], dtype=np.float32)
    assert np.array_equal(soft_clip(alpha, 0.9, 0.1), np.array([[0.0, 0.3, 1.0]], dtype=np.float32))


# ----------------------------------------------------------------------- gamma


def test_gamma_above_one_thins_soft_edges_and_leaves_the_rails_alone():
    alpha = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    out = apply_gamma(alpha, 2.0)
    assert out[0, 0] == 0.0
    assert out[0, 4] == 1.0
    assert out[0, 2] == pytest.approx(0.25)
    assert np.all(out[0, 1:4] < alpha[0, 1:4])


def test_gamma_of_one_is_a_pass_through():
    alpha = np.array([[0.3, 0.7]], dtype=np.float32)
    assert np.array_equal(apply_gamma(alpha, 1.0), alpha)


# --------------------------------------------------------------- morphological


def test_morphological_close_fills_a_pinhole():
    alpha = np.ones((32, 32), dtype=np.float32)
    alpha[16, 16] = 0.0
    assert morphological_cleanup(alpha, close=2, open_=0)[16, 16] == pytest.approx(1.0)


def test_morphological_open_removes_a_speckle_but_keeps_the_subject():
    alpha = np.zeros((48, 48), dtype=np.float32)
    alpha[8:40, 8:40] = 1.0
    alpha[2, 45] = 1.0

    out = morphological_cleanup(alpha, close=0, open_=2)

    assert out[2, 45] == pytest.approx(0.0)
    assert out[24, 24] == pytest.approx(1.0)


def test_morphological_cleanup_with_no_stages_is_a_pass_through():
    alpha = np.full((8, 8), 0.4, dtype=np.float32)
    assert np.array_equal(morphological_cleanup(alpha), alpha)


# ------------------------------------------------------- component filtering


def test_remove_small_components_drops_the_stray_blob():
    """Saliency models fire on a bright background detail; this is the fix."""
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[8:40, 8:40] = 1.0  # 1024 px subject
    alpha[52:56, 52:56] = 1.0  # 16 px stray, 1.5% of the subject

    out = remove_small_components(alpha, min_ratio=0.05)

    assert out[24, 24] == pytest.approx(1.0)
    assert out[53, 53] == pytest.approx(0.0)


def test_remove_small_components_keeps_islands_above_the_ratio():
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[8:40, 8:40] = 1.0
    alpha[48:60, 48:60] = 1.0  # 144 px, 14% of the subject

    out = remove_small_components(alpha, min_ratio=0.05)

    assert out[54, 54] == pytest.approx(1.0)


def test_remove_small_components_never_touches_a_single_object_image():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 1.0
    assert np.array_equal(remove_small_components(alpha, min_ratio=0.5), alpha)


def test_remove_small_components_with_ratio_zero_is_disabled():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[0, 0] = 1.0
    alpha[8:24, 8:24] = 1.0
    assert np.array_equal(remove_small_components(alpha, min_ratio=0.0), alpha)


def test_remove_small_components_preserves_soft_values_inside_kept_regions():
    """The filter gates on a threshold but must return the original soft alpha, or
    it would silently binarise every mask it passes through."""
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 0.75
    out = remove_small_components(alpha, min_ratio=0.5)
    assert out[16, 16] == pytest.approx(0.75)


# ------------------------------------------------------------------ feathering


def test_feather_edges_softens_the_boundary_without_eroding_the_interior():
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0

    out = feather_edges(alpha, radius=2, band=4)

    assert out[32, 32] == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(0.0)
    # The boundary column is now partially transparent rather than a hard step.
    assert 0.0 < out[32, 16] < 1.0


def test_feather_edges_with_radius_below_one_is_a_pass_through():
    alpha = np.zeros((16, 16), dtype=np.float32)
    alpha[4:12, 4:12] = 1.0
    assert np.array_equal(feather_edges(alpha, radius=0), alpha)


# --------------------------------------------------------------- refine_alpha


def test_refine_alpha_returns_a_fresh_array_and_does_not_mutate_its_input(blocky):
    alpha, image = blocky
    before = alpha.copy()
    out = refine_alpha(alpha, image, RefineConfig.quality())
    assert np.array_equal(alpha, before)
    assert out is not alpha


def test_refine_alpha_clamps_out_of_range_input():
    alpha = np.array([[-3.0, 0.5, 9.0]], dtype=np.float32)
    out = refine_alpha(alpha, None, RefineConfig.off())
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_refine_alpha_off_preset_only_clamps():
    alpha = np.array([[0.03, 0.5, 0.97]], dtype=np.float32)
    assert np.array_equal(refine_alpha(alpha, None, RefineConfig.off()), alpha)


def test_refine_alpha_fast_preset_skips_the_guided_filter(blocky):
    """The video pipeline depends on this: the guided filter is the per-frame cost."""
    alpha, image = blocky
    cfg = RefineConfig.fast()
    assert cfg.guided_filter is False
    # With the guide disabled, passing an unrelated image cannot change the result.
    noise = np.random.default_rng(0).integers(0, 255, image.shape, dtype=np.uint8)
    assert np.array_equal(refine_alpha(alpha, image, cfg), refine_alpha(alpha, noise, cfg))


def test_refine_alpha_without_an_image_skips_the_guided_filter_rather_than_failing():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 1.0
    out = refine_alpha(alpha, None, RefineConfig.quality())
    assert out.shape == alpha.shape


def test_refine_alpha_full_stack_removes_a_wash_a_speckle_and_a_pinhole():
    """One image exercising the whole quality preset end to end."""
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[24:72, 24:72] = (200, 190, 180)

    alpha = np.full((96, 96), 0.015, dtype=np.float32)  # background wash
    alpha[24:72, 24:72] = 0.99
    alpha[48, 48] = 0.0  # pinhole
    alpha[88:90, 88:90] = 0.99  # stray speckle

    out = refine_alpha(alpha, image, RefineConfig.quality())

    assert out[4, 4] == pytest.approx(0.0)
    assert out[48, 48] > 0.9
    # The speckle never reaches the component filter: the guided filter averages a
    # 2x2 blob over a 25x25 window, which drops it below the 0.5 threshold that
    # remove_small_components binarises on. What is left is bleed, not an island,
    # and it is far below the ~0.02 at which a halo becomes visible when composited.
    assert out[88, 88] < 0.02


def test_refine_config_presets_are_distinct_and_serialisable():
    assert RefineConfig.fast().as_dict() != RefineConfig.quality().as_dict()
    assert RefineConfig.off().as_dict()["soft_clip"] is False
    assert set(RefineConfig().as_dict()) >= {"guided_filter", "soft_clip", "feather_radius"}


# ------------------------------------------------------------------- temporal


def test_ema_smooth_passes_the_first_frame_through_unchanged():
    frame = np.full((4, 4), 0.8, dtype=np.float32)
    assert np.array_equal(ema_smooth(None, frame, 0.5), frame)


def test_ema_smooth_lerps_towards_the_current_frame():
    previous = np.zeros((2, 2), dtype=np.float32)
    current = np.ones((2, 2), dtype=np.float32)
    assert np.allclose(ema_smooth(previous, current, 0.25), 0.25)


def test_ema_weight_of_one_disables_smoothing():
    previous = np.zeros((2, 2), dtype=np.float32)
    current = np.ones((2, 2), dtype=np.float32)
    assert np.array_equal(ema_smooth(previous, current, 1.0), current)


def test_ema_attenuates_a_dropout_but_does_not_remove_it():
    """The documented difference from the median smoother."""
    good = np.ones((4, 4), dtype=np.float32)
    dropout = np.zeros((4, 4), dtype=np.float32)
    out = ema_smooth(good, dropout, 0.65)
    assert 0.0 < out.mean() < 1.0


def test_temporal_median_removes_a_single_frame_dropout_entirely():
    good = np.ones((4, 4), dtype=np.float32)
    dropout = np.zeros((4, 4), dtype=np.float32)
    assert np.array_equal(temporal_median([good, dropout, good]), good)


def test_temporal_median_of_one_frame_is_that_frame():
    frame = np.full((3, 3), 0.4, dtype=np.float32)
    assert np.array_equal(temporal_median([frame]), frame)


def test_temporal_median_rejects_an_empty_window():
    with pytest.raises(ValueError, match="non-empty"):
        temporal_median([])


def test_temporal_flicker_is_zero_for_a_frozen_sequence_and_positive_when_it_moves():
    frame = np.zeros((8, 8), dtype=np.float32)
    frame[2:6, 2:6] = 1.0
    assert temporal_flicker([frame, frame.copy(), frame.copy()]) == 0.0

    moved = np.zeros((8, 8), dtype=np.float32)
    moved[3:7, 2:6] = 1.0
    assert temporal_flicker([frame, moved]) > 0.0


def test_temporal_flicker_of_fewer_than_two_frames_is_zero():
    assert temporal_flicker([]) == 0.0
    assert temporal_flicker([np.ones((2, 2), dtype=np.float32)]) == 0.0


def test_temporal_flicker_is_the_mean_absolute_frame_difference():
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.full((2, 2), 0.5, dtype=np.float32)
    c = np.full((2, 2), 0.25, dtype=np.float32)
    # |b-a| = 0.5, |c-b| = 0.25 -> mean 0.375
    assert temporal_flicker([a, b, c]) == pytest.approx(0.375)
