"""Tests for the published-weights path: ONNX conversion, preprocessing, batch guards.

The conversion tests need the downloaded ONNX graphs, which are not committed, so they are
marked ``integration`` and skip when the artefacts are absent. The rest build a small
BN-folded graph on the fly and need nothing external.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cutoutml.core.imaging import normalize
from cutoutml.models.base import SegmentationModel
from cutoutml.models.onnx_adapter import OnnxAdapter
from cutoutml.models.u2net.arch import RSU, u2net_full, u2net_lite
from cutoutml.models.u2net.from_onnx import (
    _NAMED_TAIL,
    ConversionError,
    convert,
    materialize_state_dict,
    sha256_file,
    torch_conv_order,
    verify_parity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
U2NET_ONNX = REPO_ROOT / "models" / "u2net" / "u2net.onnx"
U2NETP_ONNX = REPO_ROOT / "models" / "u2net" / "u2netp.onnx"


def _rsu(blocks: nn.ModuleList, index: int) -> RSU:
    """Indexing an ``nn.ModuleList`` is typed as ``Tensor | Module``; narrow it once."""
    block = blocks[index]
    assert isinstance(block, RSU)
    return block


def _conv(module: nn.Module, attribute: str) -> nn.Conv2d:
    conv = getattr(module, attribute)
    assert isinstance(conv, nn.Conv2d)
    return conv


# --------------------------------------------------------------------- architecture


def test_full_variant_matches_the_published_parameter_count():
    """44.15M against the ~44M the paper reports.

    A wrong decoder width table moves this number by millions of parameters while leaving
    the model perfectly trainable, which is why it is asserted rather than trusted.
    """
    total = sum(p.numel() for p in u2net_full().parameters())
    assert 43_500_000 < total < 44_500_000, total


def test_lite_variant_matches_the_published_parameter_count():
    total = sum(p.numel() for p in u2net_lite().parameters())
    assert 1_050_000 < total < 1_200_000, total


def test_the_decoder_widths_are_the_published_ones_not_the_encoder_widths():
    """stage4d emits 256 channels, not the 512 of the encoder stage it pairs with.

    Pinned against the reference implementation's ``RSU4(1024,128,256)`` because the two
    readings differ only in the full variant, and only when loading published weights.
    """
    net = u2net_full()
    # decoders is ordered deepest first: stage5d, 4d, 3d, 2d, 1d.
    stage4d = _rsu(net.decoders, 1)
    entry = _conv(stage4d.rebnconvin, "conv_s1")
    assert entry.in_channels == 1024
    assert entry.out_channels == 256
    assert _conv(stage4d.encoder[0], "conv_s1").out_channels == 128, "stage4d bottleneck is 128"

    stage1d = _rsu(net.decoders, 4)
    assert _conv(stage1d.encoder[0], "conv_s1").out_channels == 16, (
        "stage1d uses a 16-channel bottleneck where its paired encoder stage uses 32"
    )
    side_inputs = [head.in_channels for head in net.side if isinstance(head, nn.Conv2d)]
    assert side_inputs == [64, 64, 128, 256, 512, 512]


def test_conv_execution_order_is_recovered_by_running_the_module():
    net = u2net_lite()
    order = torch_conv_order(net)
    assert len(order) == 119
    # The tail must be the six side heads then the fusion convolution: that ordering is
    # what makes the positional mapping to a folded graph verifiable.
    assert order[-7:] == [name for name, _ in _NAMED_TAIL]


def test_conv_order_hooks_are_removed_and_training_mode_is_restored():
    """Introspection must not leave the module altered."""
    net = u2net_lite()
    net.train()
    torch_conv_order(net)
    assert net.training, "training mode was not restored"
    conv = _conv(_rsu(net.encoders, 0).rebnconvin, "conv_s1")
    assert not conv._forward_hooks, "forward hooks were left attached"


# ------------------------------------------------------------------- preprocessing


def test_max_division_stretches_a_dim_image_to_full_range():
    """The whole point of U^2-Net's extra divisor: an image peaking at 128 is rescaled."""
    dim = np.full((4, 4, 3), 128, dtype=np.uint8)
    scaled = normalize(dim, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), scale_by=128 / 255)
    assert np.allclose(scaled, 1.0, atol=1e-6)


def test_normalize_without_a_divisor_is_unchanged():
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    plain = normalize(image, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert np.allclose(plain, 128 / 255, atol=1e-6)
    assert np.allclose(plain, normalize(image, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), scale_by=None))


def test_a_zero_divisor_is_ignored_rather_than_dividing_by_zero():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    assert np.all(np.isfinite(normalize(image, scale_by=0.0)))


def test_u2net_divides_by_the_image_maximum_and_survives_a_black_image():
    from cutoutml.models.u2net.adapter import U2NetAdapter

    adapter = U2NetAdapter(variant="lite", device="cpu")
    assert adapter.intensity_divisor(np.full((8, 8, 3), 255, dtype=np.uint8)) == pytest.approx(1.0)
    assert adapter.intensity_divisor(np.full((8, 8, 3), 128, dtype=np.uint8)) == pytest.approx(
        128 / 255
    )
    assert adapter.intensity_divisor(np.zeros((8, 8, 3), dtype=np.uint8)) is None


def test_most_of_the_eval_set_peaks_below_full_intensity():
    """The measurement that makes the max division worth implementing.

    Skipping it is defensible for photographs, where a saturated pixel is near-universal
    and the divisor is 1.0. It is not defensible here, and this pins the number
    ``docs/models.md`` quotes: a generator change that saturated the eval set would make
    that paragraph wrong, and nothing else in the suite would notice.
    """
    from cutoutml.datasets.synthetic import SyntheticSegmentationDataset

    dataset = SyntheticSegmentationDataset(count=64, split="test")
    peaks = [int(np.asarray(dataset.sample(index)[0]).max()) for index in range(64)]

    assert sum(1 for peak in peaks if peak < 255) == 40
    assert min(peaks) == 155


def test_the_divisor_is_taken_before_letterboxing_so_padding_cannot_change_it():
    """A non-square image is padded; the padding must not become the maximum.

    Constructed so the padding colour would dominate a divisor computed after padding: the
    image content peaks at 60 while letterbox padding is brighter.
    """

    class Probe(SegmentationModel):
        input_size = (16, 16)

        def _load(self) -> None: ...

        def predict(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            raise NotImplementedError

        def metadata(self):  # pragma: no cover
            raise NotImplementedError

        def intensity_divisor(self, image):
            seen.append(float(np.asarray(image).max()))
            return

    seen: list[float] = []
    probe = Probe(name="probe", input_size=(16, 16), device="cpu")
    probe.preprocess([np.full((4, 16, 3), 60, dtype=np.uint8)])
    assert seen == [60.0]


# -------------------------------------------------------------------- onnx adapter


def _folded_graph(path: Path, *, batch: int | str, sigmoid: bool, size: int = 8) -> Path:
    """Export a one-convolution graph, optionally with a baked-in sigmoid.

    The weights are set explicitly rather than left to the default initialiser. A random
    convolution can land its output near zero, where ``sigmoid(x)`` is ~0.5 and the
    double-sigmoid distortion the test below looks for becomes unmeasurable - so the test
    would fail for some seeds and pass for others. The bias dominates, which puts the
    logit firmly away from zero for any input.
    """

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, padding=1)
            nn.init.constant_(self.conv.weight, 0.01)
            assert self.conv.bias is not None
            nn.init.constant_(self.conv.bias, 2.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv(x)
            return torch.sigmoid(out) if sigmoid else out

    dynamic = {"input": {0: "batch"}, "logits": {0: "batch"}} if batch == "dynamic" else None
    torch.onnx.export(
        Tiny().eval(),
        (torch.zeros(1, 3, size, size),),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=dynamic,
        dynamo=False,
    )
    return path


def test_a_sigmoid_graph_is_not_squashed_by_a_second_sigmoid(tmp_path: Path):
    """The failure this guards against is silent: alpha stays in range but goes flat.

    A logits-declared graph that actually emits probabilities produces
    sigmoid(sigmoid(x)), which compresses everything towards 0.5. Comparing the two
    declarations on one graph shows the distortion rather than asserting a magic number.
    """
    graph = _folded_graph(tmp_path / "sig.onnx", batch="dynamic", sigmoid=True)
    image = np.full((8, 8, 3), 200, dtype=np.uint8)

    honest = OnnxAdapter(onnx_path=graph, output_activation="sigmoid", device="cpu").load()
    wrong = OnnxAdapter(onnx_path=graph, output_activation="logits", device="cpu").load()

    correct_alpha = honest.infer([image])[0]
    squashed_alpha = wrong.infer([image])[0]

    assert np.abs(squashed_alpha - 0.5).max() < np.abs(correct_alpha - 0.5).max()
    assert np.allclose(squashed_alpha, 1.0 / (1.0 + np.exp(-correct_alpha)), atol=1e-5)


def test_an_unknown_output_activation_is_rejected_at_construction(tmp_path: Path):
    with pytest.raises(ValueError, match="output_activation"):
        OnnxAdapter(onnx_path=tmp_path / "absent.onnx", output_activation="softmax")


def test_an_unknown_intensity_scaling_is_rejected_at_construction(tmp_path: Path):
    with pytest.raises(ValueError, match="intensity_scaling"):
        OnnxAdapter(onnx_path=tmp_path / "absent.onnx", intensity_scaling="imagenet")


def test_a_static_batch_graph_is_detected_and_refuses_a_different_batch(tmp_path: Path):
    graph = _folded_graph(tmp_path / "static.onnx", batch=1, sigmoid=False)
    adapter = OnnxAdapter(onnx_path=graph, device="cpu")
    adapter.load()
    assert adapter.static_batch == 1

    single = torch.zeros(1, 3, 8, 8)
    assert adapter.predict(single).shape[0] == 1

    with pytest.raises(ValueError, match="fixed batch size of 1"):
        adapter.predict(torch.zeros(4, 3, 8, 8))


def test_a_dynamic_batch_graph_reports_no_static_batch_and_accepts_any_size(tmp_path: Path):
    graph = _folded_graph(tmp_path / "dynamic.onnx", batch="dynamic", sigmoid=False)
    adapter = OnnxAdapter(onnx_path=graph, device="cpu")
    adapter.load()
    assert adapter.static_batch is None
    assert adapter.predict(torch.zeros(3, 3, 8, 8)).shape[0] == 3


def test_intensity_scaling_is_carried_across_the_export_boundary(tmp_path: Path):
    graph = _folded_graph(tmp_path / "scaled.onnx", batch="dynamic", sigmoid=False)
    plain = OnnxAdapter(onnx_path=graph, intensity_scaling="none", device="cpu")
    scaled = OnnxAdapter(onnx_path=graph, intensity_scaling="max", device="cpu")
    dim = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert plain.intensity_divisor(dim) is None
    assert scaled.intensity_divisor(dim) == pytest.approx(128 / 255)


# ---------------------------------------------------------------------- conversion


def test_a_graph_with_unbiased_convolutions_is_rejected_as_unfolded(tmp_path: Path):
    """A graph whose convolutions have no bias was exported without folding."""

    class Unbiased(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    path = tmp_path / "unbiased.onnx"
    torch.onnx.export(
        Unbiased().eval(),
        (torch.zeros(1, 3, 8, 8),),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    with pytest.raises(ConversionError, match="BatchNorms folded"):
        materialize_state_dict(path, "lite")


def test_a_graph_of_the_wrong_size_is_rejected_by_convolution_count(tmp_path: Path):
    graph = _folded_graph(tmp_path / "tiny.onnx", batch=1, sigmoid=False)
    with pytest.raises(ConversionError, match="convolutions"):
        materialize_state_dict(graph, "lite")


@pytest.mark.integration
@pytest.mark.skipif(not U2NETP_ONNX.is_file(), reason="u2netp.onnx not downloaded")
def test_the_lite_conversion_agrees_with_onnxruntime():
    """The check that proves the positional mapping: identical numerics, not just shapes."""
    module = u2net_lite()
    module.load_state_dict(materialize_state_dict(U2NETP_ONNX, "lite"), strict=True)
    assert verify_parity(U2NETP_ONNX, module, samples=1) < 1e-4


@pytest.mark.integration
@pytest.mark.skipif(not U2NETP_ONNX.is_file(), reason="u2netp.onnx not downloaded")
def test_the_batchnorms_are_exact_identities_after_conversion():
    """var = 1 - eps, so sqrt(var + eps) is exactly 1 and the layer is a true no-op."""
    state = materialize_state_dict(U2NETP_ONNX, "lite")
    module = u2net_lite()
    key = "encoders.0.rebnconvin.bn_s1"
    norm = _rsu(module.encoders, 0).rebnconvin.bn_s1
    assert isinstance(norm, nn.BatchNorm2d)
    eps = norm.eps
    assert torch.allclose(state[f"{key}.weight"], torch.ones_like(state[f"{key}.weight"]))
    assert torch.allclose(state[f"{key}.bias"], torch.zeros_like(state[f"{key}.bias"]))
    assert torch.allclose(state[f"{key}.running_mean"], torch.zeros_like(state[f"{key}.weight"]))
    denominator = (state[f"{key}.running_var"] + eps).sqrt()
    assert torch.allclose(denominator, torch.ones_like(denominator), atol=1e-7)


@pytest.mark.integration
@pytest.mark.skipif(not U2NETP_ONNX.is_file(), reason="u2netp.onnx not downloaded")
def test_converting_the_wrong_variant_fails_rather_than_loading_garbage():
    with pytest.raises(ConversionError):
        materialize_state_dict(U2NETP_ONNX, "full")


@pytest.mark.integration
@pytest.mark.skipif(not U2NET_ONNX.is_file(), reason="u2net.onnx not downloaded")
def test_the_full_conversion_agrees_with_onnxruntime():
    module = u2net_full()
    module.load_state_dict(materialize_state_dict(U2NET_ONNX, "full"), strict=True)
    assert verify_parity(U2NET_ONNX, module, samples=1) < 1e-4


@pytest.mark.integration
@pytest.mark.skipif(not U2NETP_ONNX.is_file(), reason="u2netp.onnx not downloaded")
def test_two_conversions_of_one_graph_produce_byte_identical_checkpoints(tmp_path: Path):
    """The benchmark suite records the SHA-256 of the weights behind every accuracy row,
    and the archive index reads a changed digest as changed weights. Anything varying in
    the serialised checkpoint - a conversion timestamp being the obvious candidate - turns
    that inference into a false one."""
    # Same file name in two directories, not two names in one: ``torch.save`` writes a zip
    # whose internal archive name comes from the destination path, so a checkpoint's digest
    # is only comparable against one written to the same name. That is a property of the
    # format rather than of this converter, and it is why the suite compares a row against
    # the same weights file rather than against a copy.
    first = tmp_path / "a" / "u2netp.pt"
    second = tmp_path / "b" / "u2netp.pt"
    convert(U2NETP_ONNX, first, variant="lite")
    convert(U2NETP_ONNX, second, variant="lite")

    assert sha256_file(first) == sha256_file(second)


@pytest.mark.integration
@pytest.mark.skipif(not U2NETP_ONNX.is_file(), reason="u2netp.onnx not downloaded")
def test_the_conversion_record_carries_what_the_docs_quote(tmp_path: Path):
    """The weights are not committed, so a parity figure in the docs is only checkable
    from a clone if the conversion leaves a record behind."""
    result = convert(
        U2NETP_ONNX,
        tmp_path / "u2netp.pt",
        variant="lite",
        record_path=tmp_path / "record.json",
    )
    record = json.loads((tmp_path / "record.json").read_text())

    assert record["parity_max_abs_diff"] == result.parity_max_abs_diff
    assert record["source_sha256"] == result.source_sha256
    assert record["convolutions"] == 119
    assert record["license"].startswith("Apache-2.0")
    assert record["checkpoint_sha256"] == sha256_file(tmp_path / "u2netp.pt")
    # In the record, which describes an event, but never in the checkpoint, whose digest
    # has to identify the weights and nothing else.
    assert "converted_at" in record
    payload = torch.load(tmp_path / "u2netp.pt", map_location="cpu", weights_only=False)
    assert "converted_at" not in payload["provenance"]
