"""Segmentation metrics, checked against values computed by hand.

Every published accuracy number in this repository comes out of this module, so the
assertions here are arithmetic identities rather than regression snapshots: a 3x3 mask
whose true positives, false positives and false negatives can be counted on paper. If
these definitions drift, the drift is a silent revision of every row in
``docs/benchmarks.md``.

The convention checks matter as much as the arithmetic. ``F-beta`` with ``beta^2 = 0.3``
weights precision above recall, and getting that backwards would make every number look
*better* on a model that over-segments.
"""

from __future__ import annotations

import numpy as np
import pytest

from cutoutml.core.metrics import (
    FBETA_SQ,
    MaskMetrics,
    aggregate,
    ber,
    boundary_f1,
    compute_all,
    dice,
    f_beta,
    f_beta_curve,
    iou,
    mae,
    pixel_accuracy,
    precision_recall,
    s_measure,
    soft_iou,
)


@pytest.fixture
def hand_case() -> tuple[np.ndarray, np.ndarray]:
    """A 3x3 pair with counts that are trivial to verify.

    Ground truth foreground: the top-left 2x2 block (4 pixels).
    Prediction: the 2x2 block shifted one column right.

    Overlap is the middle column of the top two rows, so:
      TP = 2, FP = 2, FN = 2, TN = 3.
    """
    gt = np.zeros((3, 3), dtype=np.float32)
    gt[0:2, 0:2] = 1.0
    pred = np.zeros((3, 3), dtype=np.float32)
    pred[0:2, 1:3] = 1.0
    return pred, gt


# ------------------------------------------------------------------------ IoU


def test_iou_matches_the_hand_count(hand_case):
    pred, gt = hand_case
    # intersection 2, union 6
    assert iou(pred, gt) == pytest.approx(2 / 6)


def test_iou_is_one_for_an_exact_match():
    mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert iou(mask, mask) == 1.0


def test_iou_is_zero_for_disjoint_masks():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    assert iou(a, b) == 0.0


def test_iou_of_two_empty_masks_is_defined_as_one():
    """0/0 has to be *chosen*. Defining it as 1.0 means "correctly found nothing", which
    is what a background-only image deserves; defining it as 0 would drag the mean down on
    every dataset containing one."""
    empty = np.zeros((4, 4), dtype=np.float32)
    assert iou(empty, empty) == 1.0


def test_iou_threshold_is_applied_to_both_sides():
    pred = np.full((2, 2), 0.4, dtype=np.float32)
    gt = np.ones((2, 2), dtype=np.float32)
    assert iou(pred, gt, threshold=0.5) == 0.0
    assert iou(pred, gt, threshold=0.3) == 1.0


# ------------------------------------------------------------------------ MAE


def test_mae_is_the_mean_absolute_difference():
    pred = np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
    gt = np.array([[0.0, 0.0], [1.00, 0.75]], dtype=np.float32)
    # |0| + |1| + |0.75| + |0| = 1.75 over 4 pixels
    assert mae(pred, gt) == pytest.approx(1.75 / 4)


def test_mae_sees_soft_alpha_that_iou_cannot():
    """A confident-but-wrong model and a hedging model can share an IoU and differ
    enormously in MAE. That is exactly why both are reported for matting."""
    gt = np.ones((4, 4), dtype=np.float32)
    confident = np.ones((4, 4), dtype=np.float32)
    hedging = np.full((4, 4), 0.51, dtype=np.float32)
    assert iou(confident, gt) == iou(hedging, gt) == 1.0
    assert mae(confident, gt) == 0.0
    assert mae(hedging, gt) == pytest.approx(0.49, abs=1e-6)


def test_mae_clamps_out_of_range_inputs():
    assert mae(np.full((2, 2), 3.0), np.ones((2, 2))) == 0.0


# ----------------------------------------------------------------- dice / F1


def test_dice_matches_the_hand_count(hand_case):
    pred, gt = hand_case
    # 2 * 2 / (4 + 4)
    assert dice(pred, gt) == pytest.approx(0.5)


def test_dice_is_always_at_least_iou():
    rng = np.random.default_rng(3)
    for _ in range(20):
        pred = (rng.random((8, 8)) > 0.5).astype(np.float32)
        gt = (rng.random((8, 8)) > 0.5).astype(np.float32)
        assert dice(pred, gt) >= iou(pred, gt) - 1e-9


def test_soft_iou_uses_continuous_values():
    pred = np.full((2, 2), 0.5, dtype=np.float32)
    gt = np.ones((2, 2), dtype=np.float32)
    # inter = 4*0.5 = 2, union = 2 + 4 - 2 = 4
    assert soft_iou(pred, gt) == pytest.approx(0.5)


def test_soft_iou_of_empty_inputs_is_one():
    assert soft_iou(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0


# -------------------------------------------------------- precision and recall


def test_precision_and_recall_match_the_hand_count(hand_case):
    pred, gt = hand_case
    precision, recall = precision_recall(pred, gt)
    assert precision == pytest.approx(2 / 4)  # TP / (TP + FP)
    assert recall == pytest.approx(2 / 4)  # TP / (TP + FN)


def test_precision_is_one_when_nothing_is_predicted():
    """No positive predictions means no false positives. Defining precision as 1.0 there
    is the standard convention; recall carries the penalty instead."""
    precision, recall = precision_recall(np.zeros((2, 2)), np.ones((2, 2)))
    assert precision == 1.0
    assert recall == 0.0


# ---------------------------------------------------------------------- F-beta


def test_f_beta_weights_precision_above_recall():
    """With beta^2 = 0.3 the harmonic weighting favours precision. A mask that is
    precise-but-incomplete must therefore score higher than its mirror image."""
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[:, :5] = 1.0

    precise = np.zeros((10, 10), dtype=np.float32)
    precise[:, :3] = 1.0  # 30 TP, 0 FP, 20 FN -> P=1.00, R=0.60

    greedy = np.zeros((10, 10), dtype=np.float32)
    greedy[:, :8] = 1.0  # 50 TP, 30 FP, 0 FN -> P=0.625, R=1.00

    assert f_beta(precise, gt) > f_beta(greedy, gt)


def test_f_beta_matches_the_closed_form(hand_case):
    pred, gt = hand_case
    precision = recall = 0.5
    expected = (1 + FBETA_SQ) * precision * recall / (FBETA_SQ * precision + recall)
    assert f_beta(pred, gt) == pytest.approx(expected)


def test_f_beta_curve_has_one_entry_per_threshold():
    rng = np.random.default_rng(11)
    pred = rng.random((32, 32)).astype(np.float32)
    gt = (rng.random((32, 32)) > 0.6).astype(np.float32)
    curve = f_beta_curve(pred, gt, steps=255)
    assert curve.shape == (255,)
    assert np.all((curve >= 0.0) & (curve <= 1.0))


def test_f_beta_curve_agrees_with_the_scalar_at_the_same_threshold():
    """The curve is computed from cumulative histograms for speed; this pins it to the
    straightforward implementation so the optimisation cannot drift."""
    rng = np.random.default_rng(5)
    pred = rng.random((64, 64)).astype(np.float32)
    gt = (rng.random((64, 64)) > 0.5).astype(np.float32)
    steps = 255
    curve = f_beta_curve(pred, gt, steps=steps)
    for index in (10, 128, 200):
        threshold = (index + 1) / steps
        assert curve[index] == pytest.approx(f_beta(pred, gt, threshold), abs=2e-2)


def test_f_beta_curve_max_is_at_least_the_fixed_threshold_score():
    rng = np.random.default_rng(9)
    pred = rng.random((48, 48)).astype(np.float32)
    gt = (rng.random((48, 48)) > 0.5).astype(np.float32)
    assert f_beta_curve(pred, gt).max() >= f_beta(pred, gt) - 1e-9


# -------------------------------------------------------------------- S-measure


def test_s_measure_is_one_for_a_perfect_prediction():
    gt = np.zeros((32, 32), dtype=np.float32)
    gt[8:24, 8:24] = 1.0
    assert s_measure(gt, gt) == pytest.approx(1.0, abs=1e-3)


def test_s_measure_degenerate_ground_truth_falls_back_to_the_mean():
    empty = np.zeros((8, 8), dtype=np.float32)
    assert s_measure(np.full((8, 8), 0.25, dtype=np.float32), empty) == pytest.approx(0.75)
    full = np.ones((8, 8), dtype=np.float32)
    assert s_measure(np.full((8, 8), 0.25, dtype=np.float32), full) == pytest.approx(0.25)


def test_s_measure_punishes_the_right_area_in_the_wrong_place():
    """The point of S-measure: two masks with identical area, one structurally correct."""
    gt = np.zeros((32, 32), dtype=np.float32)
    gt[4:20, 4:20] = 1.0
    displaced = np.zeros((32, 32), dtype=np.float32)
    displaced[12:28, 12:28] = 1.0
    assert s_measure(displaced, gt) < s_measure(gt, gt)


# ----------------------------------------------------------------- boundary F1


def test_boundary_f1_is_one_for_identical_contours():
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    assert boundary_f1(mask, mask) == pytest.approx(1.0)


def test_boundary_f1_tolerates_a_shift_inside_the_tolerance_band():
    gt = np.zeros((60, 60), dtype=np.float32)
    gt[20:40, 20:40] = 1.0
    shifted = np.zeros((60, 60), dtype=np.float32)
    shifted[21:41, 21:41] = 1.0
    assert boundary_f1(shifted, gt, tolerance=3) > 0.9
    assert boundary_f1(shifted, gt, tolerance=0) < 0.9


def test_boundary_f1_separates_contour_quality_from_region_overlap():
    """Speckle inside the object barely moves IoU and collapses boundary F1. This is why
    both are reported: IoU answers "is the right region selected", boundary F1 answers
    "does the cutout have clean edges", and the two disagree on exactly this failure."""
    gt = np.zeros((96, 96), dtype=np.float32)
    gt[24:72, 24:72] = 1.0

    # Pinholes on a spaced grid well inside the object. Each costs one pixel of area but
    # adds a whole 3x3 contour ring nowhere near the true contour, so precision drops
    # while the region overlap stays above 0.97.
    speckled = gt.copy()
    speckled[32:69:6, 32:69:6] = 0.0

    assert iou(speckled, gt) > 0.95
    assert boundary_f1(gt, gt, tolerance=3) == pytest.approx(1.0)
    assert boundary_f1(speckled, gt, tolerance=3) < 0.75


def test_boundary_f1_of_two_empty_masks_is_one_and_of_one_empty_is_zero():
    empty = np.zeros((16, 16), dtype=np.float32)
    solid = np.ones((16, 16), dtype=np.float32)
    filled = np.zeros((16, 16), dtype=np.float32)
    filled[4:12, 4:12] = 1.0
    assert boundary_f1(empty, empty) == 1.0
    assert boundary_f1(empty, filled) == 0.0
    # An all-ones mask has no interior contour either, since the gradient is zero.
    assert boundary_f1(solid, solid) == 1.0


# ------------------------------------------------------------------------- BER


def test_ber_matches_the_hand_count(hand_case):
    pred, gt = hand_case
    # sensitivity = TP/(TP+FN) = 2/4, specificity = TN/(TN+FP) = 3/5
    assert ber(pred, gt) == pytest.approx(1.0 - 0.5 * (0.5 + 0.6))


def test_ber_does_not_reward_predicting_all_background():
    """The whole reason BER is reported: on a mask that is 96% background, predicting
    nothing scores 0.96 accuracy and 0.5 BER."""
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[:2, :2] = 1.0
    nothing = np.zeros((10, 10), dtype=np.float32)
    assert pixel_accuracy(nothing, gt) == pytest.approx(0.96)
    assert ber(nothing, gt) == pytest.approx(0.5)


def test_ber_is_zero_for_a_perfect_prediction():
    gt = np.zeros((8, 8), dtype=np.float32)
    gt[2:6, 2:6] = 1.0
    assert ber(gt, gt) == pytest.approx(0.0)


# ---------------------------------------------------------------- aggregation


def test_compute_all_returns_every_field_populated(hand_case):
    pred, gt = hand_case
    metrics = compute_all(pred, gt)
    assert isinstance(metrics, MaskMetrics)
    payload = metrics.as_dict()
    assert set(payload) == {
        "mae",
        "iou",
        "dice",
        "f_beta",
        "f_beta_max",
        "f_beta_mean",
        "s_measure",
        "boundary_f1",
        "ber",
        "precision",
        "recall",
        "accuracy",
    }
    assert all(isinstance(value, float) for value in payload.values())
    assert payload["iou"] == pytest.approx(2 / 6)


def test_aggregate_averages_each_field_and_records_the_sample_count():
    gt = np.zeros((8, 8), dtype=np.float32)
    gt[2:6, 2:6] = 1.0
    perfect = compute_all(gt, gt)
    empty = compute_all(np.zeros((8, 8), dtype=np.float32), gt)

    summary = aggregate([perfect, empty])
    assert summary["n_samples"] == 2.0
    assert summary["iou"] == pytest.approx((perfect.iou + empty.iou) / 2)


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {}


# ---------------------------------------------------------------- input guards


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shape"):
        iou(np.zeros((4, 4)), np.zeros((4, 5)))


def test_empty_arrays_are_an_error():
    with pytest.raises(ValueError, match="empty"):
        mae(np.zeros((0, 0)), np.zeros((0, 0)))


def test_trailing_channel_dimension_is_squeezed():
    gt = np.zeros((4, 4, 1), dtype=np.float32)
    gt[1:3, 1:3, 0] = 1.0
    assert iou(gt, gt) == 1.0
