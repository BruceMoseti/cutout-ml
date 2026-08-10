"""Training losses.

These are the losses the shipped ``cutoutnet-small`` checkpoint was actually trained
with, so a silent change here invalidates the weights rather than just failing a test.
They are also pure functions of two tensors, which makes them cheap to check against
values worked out by hand - and hand-worked values are the only way to catch a loss that
is wrong but still decreases, which trains a worse model without ever looking broken.

The composite's docstring makes three specific claims about *why* BCE, IoU and edge are
combined. Each has a test here, because a rationale nobody verifies is how a term ends up
carrying no weight for a year without anyone noticing.
"""

from __future__ import annotations

import math

import pytest
import torch

from cutoutml.training.losses import (
    LossWeights,
    SegmentationLoss,
    _sobel,
    bce_loss,
    dice_loss,
    edge_loss,
    soft_iou_loss,
    ssim_loss,
)

LOG2 = math.log(2.0)


def logits_for(probability: float, shape: tuple[int, ...]) -> torch.Tensor:
    """Logits whose sigmoid is exactly ``probability``."""
    value = math.log(probability / (1.0 - probability))
    return torch.full(shape, value, dtype=torch.float32)


# ============================================================ binary cross-entropy


def test_bce_at_a_zero_logit_is_log_two_whatever_the_target():
    """sigmoid(0) = 0.5, so every pixel contributes -log(0.5) regardless of its label.
    The one value in this file that can be checked without a calculator."""
    logits = torch.zeros(2, 1, 4, 4)
    for target_value in (0.0, 0.5, 1.0):
        target = torch.full_like(logits, target_value)
        assert bce_loss(logits, target).item() == pytest.approx(LOG2, abs=1e-6)


def test_bce_is_hand_computable_for_a_confident_correct_pixel():
    """p = 0.9 against target 1 costs -log(0.9); against target 0 it costs -log(0.1)."""
    logits = logits_for(0.9, (1, 1, 1, 1))
    assert bce_loss(logits, torch.ones_like(logits)).item() == pytest.approx(
        -math.log(0.9), abs=1e-6
    )
    assert bce_loss(logits, torch.zeros_like(logits)).item() == pytest.approx(
        -math.log(0.1), abs=1e-6
    )


def test_bce_survives_a_logit_that_would_overflow_a_hand_rolled_sigmoid():
    """The documented reason every loss here takes logits rather than probabilities:
    ``binary_cross_entropy_with_logits`` fuses the sigmoid in log space, whereas
    ``log(sigmoid(x))`` underflows to ``-inf`` well before |x| = 100."""
    logits = torch.tensor([[[[-100.0, 100.0]]]])
    target = torch.tensor([[[[0.0, 1.0]]]])

    loss = bce_loss(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    wrong = bce_loss(logits, 1.0 - target)
    assert torch.isfinite(wrong)
    assert wrong.item() == pytest.approx(100.0, rel=1e-3)


# ========================================================================= soft IoU


def test_soft_iou_of_a_confident_perfect_prediction_is_almost_zero():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    logits = (target * 2.0 - 1.0) * 30.0
    assert soft_iou_loss(logits, target).item() == pytest.approx(0.0, abs=1e-4)


def test_soft_iou_at_a_zero_logit_against_a_full_target_is_one_half():
    """Every probability is 0.5 and the target is all ones, so over N pixels the
    intersection is N/2 and the union is N/2 + N - N/2 = N. IoU = 0.5."""
    logits = torch.zeros(1, 1, 6, 6)
    target = torch.ones_like(logits)
    assert soft_iou_loss(logits, target).item() == pytest.approx(0.5, abs=1e-5)


def test_soft_iou_is_reduced_per_sample_so_a_small_object_is_not_drowned_out():
    """The documented reason for per-sample reduction. Sample 0 is a 1-pixel object
    predicted perfectly; sample 1 is a large object missed entirely. Per sample the
    losses are ~0 and ~1, averaging ~0.5. Pooling the intersection over the batch would
    let the large object set the loss almost by itself, which is precisely the class
    imbalance a region loss was added to fix."""
    target = torch.zeros(2, 1, 10, 10)
    target[0, 0, 0, 0] = 1.0
    target[1, 0, 2:8, 2:8] = 1.0

    logits = torch.full((2, 1, 10, 10), -30.0)
    logits[0, 0, 0, 0] = 30.0

    per_sample = soft_iou_loss(logits, target).item()
    assert per_sample == pytest.approx(0.5, abs=0.01)

    probs = torch.sigmoid(logits)
    inter = (probs * target).sum()
    pooled = 1.0 - (inter / (probs.sum() + target.sum() - inter)).item()
    assert pooled > 0.97, "batch-pooled IoU is set by the large object alone"


def test_soft_iou_of_an_empty_target_predicted_empty_is_zero_not_undefined():
    """Backgrounds with no foreground at all occur in any real dataset. Both the
    intersection and the union are zero, so without the epsilon this would be 0/0 and one
    NaN would poison the whole training run."""
    logits = torch.full((1, 1, 4, 4), -30.0)
    target = torch.zeros_like(logits)
    loss = soft_iou_loss(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_soft_iou_gradients_nearly_vanish_without_overlap_which_is_why_bce_is_kept():
    """The documented failure mode of IoU alone - "IoU loss won't start training". With a
    confidently wrong prediction the region loss is saturated and pushes almost nothing,
    while BCE still has a strong per-pixel gradient to get training moving."""
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :4] = 1.0
    base = torch.full((1, 1, 8, 8), -20.0)

    iou_input = base.clone().requires_grad_(True)
    soft_iou_loss(iou_input, target).backward()
    assert iou_input.grad is not None

    bce_input = base.clone().requires_grad_(True)
    bce_loss(bce_input, target).backward()
    assert bce_input.grad is not None

    assert iou_input.grad.abs().max() < bce_input.grad.abs().max()


# ============================================================================ Dice


def test_dice_at_a_zero_logit_against_a_full_target_is_one_third():
    """Intersection N/2, total N/2 + N; Dice = 2(N/2)/(3N/2) = 2/3, so the loss is 1/3."""
    logits = torch.zeros(1, 1, 6, 6)
    target = torch.ones_like(logits)
    assert dice_loss(logits, target).item() == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_dice_is_gentler_than_iou_on_the_same_prediction():
    """The stated reason it is offered as an alternative. Dice double-counts the
    intersection, so it penalises a partly-right prediction less."""
    logits = torch.zeros(1, 1, 6, 6)
    target = torch.ones_like(logits)
    assert dice_loss(logits, target).item() < soft_iou_loss(logits, target).item()


# ============================================================================ edge


def test_the_sobel_magnitude_is_offset_blind_away_from_the_frame_border():
    """The claim the edge term rests on: it compares gradients, so a flat region
    contributes nothing at any brightness, leaving the loss free to spend its capacity on
    boundaries."""
    for level in (0.05, 0.5, 0.95):
        interior = _sobel(torch.full((1, 1, 10, 10), level))[..., 1:-1, 1:-1]
        assert interior.abs().max().item() == pytest.approx(0.0, abs=1e-2)


def test_the_zero_padded_sobel_reads_the_frame_border_as_an_edge():
    """A real consequence of ``padding=1`` with zeros, recorded rather than smoothed
    over: a bright mask has a gradient at the frame border against the implicit black
    outside it. For a subject cropped by the frame that is arguably correct - there *is* a
    silhouette edge there - but it does mean the edge term is not completely blind to a
    constant offset, only blind to one in the interior."""
    bright = _sobel(torch.ones(1, 1, 10, 10))

    assert bright[..., 0, :].abs().max().item() > 0.5, "the top row sees the padding"
    assert bright[..., 1:-1, 1:-1].abs().max().item() < 1e-2

    dark = torch.full((1, 1, 10, 10), -3.0)
    light = torch.full((1, 1, 10, 10), 3.0)
    target = torch.full((1, 1, 10, 10), 0.5)
    assert edge_loss(dark, target).item() != pytest.approx(edge_loss(light, target).item())
    assert bce_loss(light, target).item() > 1.0, "BCE sees the offset everywhere, as it should"


def test_edge_loss_grows_when_the_boundary_moves():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12] = 1.0

    aligned = (target * 2.0 - 1.0) * 10.0
    shifted = torch.roll(aligned, shifts=4, dims=-1)

    assert edge_loss(aligned, target).item() < edge_loss(shifted, target).item()


def test_edge_loss_is_zero_between_a_mask_and_itself():
    target = torch.zeros(1, 1, 12, 12)
    target[..., 3:9, 3:9] = 1.0
    perfect = (target * 2.0 - 1.0) * 40.0
    assert edge_loss(perfect, target).item() == pytest.approx(0.0, abs=1e-3)


# ============================================================================ SSIM


def test_ssim_loss_of_a_perfect_prediction_is_almost_zero():
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:24] = 1.0
    perfect = (target * 2.0 - 1.0) * 30.0
    assert ssim_loss(perfect, target).item() == pytest.approx(0.0, abs=1e-2)


def test_ssim_sees_wrong_local_structure_that_a_region_loss_cannot():
    """The documented reason SSIM is available at all: a checkerboard and a solid block
    can have nearly the same mean alpha, so a region loss barely separates them while
    SSIM does, because it compares local variance and covariance."""
    target = torch.zeros(1, 1, 32, 32)
    target[..., :16, :] = 1.0

    checker = torch.zeros(1, 1, 32, 32)
    checker[..., ::2, ::2] = 1.0
    checker[..., 1::2, 1::2] = 1.0

    scrambled = (checker * 2.0 - 1.0) * 30.0
    assert torch.sigmoid(scrambled).mean().item() == pytest.approx(0.5, abs=0.01)
    assert target.mean().item() == pytest.approx(0.5, abs=0.01)

    assert ssim_loss(scrambled, target).item() > 0.5


# ================================================================ composite loss


def test_only_the_weighted_terms_are_computed_and_reported():
    """A zero weight has to mean "not computed", not "computed and multiplied by zero":
    SSIM costs about 30% of a training step, so the skip is the reason it is affordable
    to leave in the code at all."""
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.ones(1, 1, 8, 8)

    loss = SegmentationLoss(LossWeights(bce=1.0, iou=0.0, edge=0.0, ssim=0.0))
    total, parts = loss(logits, target)

    assert set(parts) == {"bce", "total"}
    assert total.item() == pytest.approx(bce_loss(logits, target).item(), abs=1e-6)


def test_the_total_is_the_weighted_sum_of_its_reported_parts():
    logits = torch.randn(2, 1, 16, 16, generator=torch.Generator().manual_seed(7))
    target = (torch.rand(2, 1, 16, 16, generator=torch.Generator().manual_seed(8)) > 0.5).float()
    weights = LossWeights(bce=1.0, iou=2.0, edge=0.5, ssim=0.25)

    total, parts = SegmentationLoss(weights)(logits, target)

    expected = (
        weights.bce * parts["bce"]
        + weights.iou * parts["iou"]
        + weights.edge * parts["edge"]
        + weights.ssim * parts["ssim"]
    )
    assert total.item() == pytest.approx(expected, rel=1e-5)
    assert parts["total"] == pytest.approx(total.item(), rel=1e-6)


def test_a_bare_tensor_and_a_one_element_list_are_the_same_input():
    """Architectures here return either a tensor or a tuple of side outputs, and the
    caller should not have to normalise that."""
    logits = torch.zeros(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)
    loss = SegmentationLoss(LossWeights(bce=1.0, iou=0.0, edge=0.0, ssim=0.0))

    assert loss(logits, target)[0].item() == pytest.approx(loss([logits], target)[0].item())


def test_a_target_without_a_channel_axis_is_accepted():
    """Datasets hand back ``(N, H, W)`` masks; requiring the caller to unsqueeze is the
    kind of detail that silently broadcasts into a wrong loss instead of an error."""
    logits = torch.zeros(2, 1, 8, 8)
    squeezed = torch.ones(2, 8, 8)
    total, _ = SegmentationLoss(LossWeights(bce=1.0, iou=0.0, edge=0.0, ssim=0.0))(logits, squeezed)
    assert total.item() == pytest.approx(LOG2, abs=1e-6)


def test_side_outputs_are_weighted_by_the_side_factor():
    """Deep supervision: with three identical outputs the total is
    ``1.0 * L + side * L + side * L``."""
    logits = torch.zeros(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)
    weights = LossWeights(bce=1.0, iou=0.0, edge=0.0, ssim=0.0, side=0.4)

    total, parts = SegmentationLoss(weights)([logits, logits, logits], target)

    assert total.item() == pytest.approx(LOG2 * (1.0 + 0.4 + 0.4), abs=1e-5)
    assert parts["bce"] == pytest.approx(LOG2, abs=1e-6), (
        "the reported terms describe the primary output only"
    )


def test_a_side_output_at_a_lower_resolution_is_upsampled_to_the_target():
    """Side outputs come off intermediate decoder stages, so they are smaller by
    construction. Without the resize this is a shape error at the first step."""
    primary = torch.zeros(1, 1, 16, 16)
    coarse = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 16, 16)
    weights = LossWeights(bce=1.0, iou=0.0, edge=0.0, ssim=0.0, side=0.5)

    total, _ = SegmentationLoss(weights)([primary, coarse], target)
    assert total.item() == pytest.approx(LOG2 * 1.5, abs=1e-5)


def test_the_gradient_head_is_supervised_against_the_edges_not_the_mask():
    """BiRefNet's gradient output predicts an edge map. Supervising it against the mask
    would train it to produce something it is not for, and the mistake is invisible in
    the loss curve because both targets are in [0, 1]."""
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1.0
    primary = (target * 2.0 - 1.0) * 20.0

    loss = SegmentationLoss(LossWeights(), gradient_output_index=1)
    _, parts = loss([primary, primary], target)

    assert "gradient" in parts
    assert "bce" in parts, "the primary output is still supervised normally"

    # Supervised against the mask instead, a mask-shaped prediction would score near
    # zero. Against a thin edge target it cannot, which is what distinguishes the two.
    assert parts["gradient"] > 0.1


def test_the_gradient_term_replaces_rather_than_adds_to_the_mask_terms_for_that_output():
    """One output cannot be supervised as both a mask and an edge map."""
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1.0
    logits = torch.zeros(1, 1, 16, 16)

    with_gradient = SegmentationLoss(LossWeights(), gradient_output_index=1)(
        [logits, logits], target
    )[1]
    assert "gradient" in with_gradient

    without = SegmentationLoss(LossWeights())([logits, logits], target)[1]
    assert "gradient" not in without


def test_the_reported_parts_are_plain_floats_so_the_metrics_log_is_json_safe():
    """They are written to ``training/runs/*.json``. A detached tensor serialises as
    something unreadable, or not at all."""
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    total, parts = SegmentationLoss()(logits, torch.ones(1, 1, 8, 8))

    assert all(type(value) is float for value in parts.values())
    assert total.requires_grad, "the total is what gets backpropagated"


def test_the_composite_is_differentiable_end_to_end():
    logits = torch.randn(1, 1, 16, 16, requires_grad=True)
    total, _ = SegmentationLoss(LossWeights(bce=1.0, iou=1.0, edge=0.5, ssim=0.5))(
        logits, torch.ones(1, 1, 16, 16)
    )
    total.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_the_weights_round_trip_to_a_dict_for_the_run_record():
    weights = LossWeights(bce=1.0, iou=2.0, edge=0.5, ssim=0.0, gradient=0.3, side=0.4)
    assert weights.as_dict() == {
        "bce": 1.0,
        "iou": 2.0,
        "edge": 0.5,
        "ssim": 0.0,
        "gradient": 0.3,
        "side": 0.4,
    }


def test_the_default_recipe_is_bce_plus_iou_plus_edge_with_ssim_off():
    """Pinned because it is the recipe the committed checkpoint was trained with, and
    every accuracy number in the docs is downstream of it."""
    defaults = LossWeights()
    assert (defaults.bce, defaults.iou, defaults.edge) == (1.0, 1.0, 0.5)
    assert defaults.ssim == 0.0

    _, parts = SegmentationLoss()(torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8))
    assert set(parts) == {"bce", "iou", "edge", "total"}
