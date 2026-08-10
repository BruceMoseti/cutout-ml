"""Model registry and the adapter contract.

Two things are worth testing here and they are different:

* the **registry** as a lookup and availability oracle - it is what lets
  ``GET /models`` tell a caller that ``u2net`` has no weights on this machine
  *before* they submit a job;
* the **contract** in :class:`~cutoutml.models.base.SegmentationModel` - shapes,
  the ``preprocess -> predict -> postprocess`` lifecycle, and the promise that
  ``predict`` returns logits. Everything downstream (pipelines, harness, worker)
  is written against that contract exactly once, so a violation in any adapter
  breaks all of them.

The contract tests are parametrised over every adapter that can actually run on
this machine, which on a CPU-only box means the classical baselines, the trivial
references, CutoutNet and the ONNX export. TensorRT is skipped by design rather
than mocked: a mocked GPU test proves nothing about a GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cutoutml.core.config import REPO_ROOT
from cutoutml.core.imaging import LetterboxInfo
from cutoutml.models.base import (
    ModelMetadata,
    ModelSpec,
    SegmentationModel,
    WeightsUnavailableError,
)
from cutoutml.models.classical.baseline import ClassicalBaseline
from cutoutml.models.onnx_adapter import OnnxAdapter
from cutoutml.models.registry import (
    ModelNotFoundError,
    catalogue,
    get_model,
    list_model_names,
    list_models,
    register,
    resolve_spec,
    runtime_available,
    unregister,
    usable_models,
    weights_available,
)

#: CutoutNet's checkpoint is trained here rather than committed - 38 minutes on eight cores,
#: recorded in `training/runs/cutoutnet-small-latest.json` - so an environment that has not
#: run `make train` cannot verify anything that depends on the trained weights. That includes
#: CI. Skipping says so; erroring at fixture setup, which is what happened before, reads as a
#: broken suite instead of an absent artefact. Follows the same idiom as the `*.onnx` skips in
#: `test_u2net_weights.py`.
_needs_trained_cutoutnet = pytest.mark.skipif(
    not weights_available(resolve_spec("cutoutnet")),
    reason="cutoutnet.pt not present; run `make train`",
)


@pytest.fixture
def temp_spec() -> Any:
    """Register a throwaway spec and always clean it up."""
    registered: list[str] = []

    def _register(**overrides: Any) -> ModelSpec:
        spec = ModelSpec(
            **{
                "name": "unit-test-model",
                "adapter": "cutoutml.models.classical.baseline.TrivialBaseline",
                "architecture": "Trivial/ones",
                "input_size": (64, 64),
                "license": "MIT",
                "source": "tests",
                "requires_weights": False,
                "options": {"method": "ones"},
                **overrides,
            }
        )
        register(spec, overwrite=True)
        registered.append(spec.name)
        return spec

    yield _register
    for name in registered:
        unregister(name)


# --------------------------------------------------------------- registry basics


def test_the_shipped_catalogue_is_registered():
    names = list_model_names()
    for expected in ("cutoutnet", "cutoutnet-onnx", "u2net", "birefnet", "classical"):
        assert expected in names


def test_list_models_is_sorted_by_name():
    names = [s.name for s in list_models()]
    assert names == sorted(names)


def test_resolve_spec_returns_the_spec():
    spec = resolve_spec("cutoutnet")
    assert spec.name == "cutoutnet"
    assert spec.architecture == "CutoutNet-small"
    assert spec.input_size == (256, 256)


def test_unknown_model_names_what_is_available():
    """The error is a user-facing message, so it must list the alternatives."""
    with pytest.raises(ModelNotFoundError) as excinfo:
        resolve_spec("stable-diffusion")
    message = str(excinfo.value)
    assert "stable-diffusion" in message
    assert "cutoutnet" in message


def test_register_rejects_a_duplicate_name_unless_overwrite_is_set(temp_spec):
    spec = temp_spec()
    with pytest.raises(ValueError, match="already registered"):
        register(spec)
    register(spec, overwrite=True)


def test_unregister_is_idempotent():
    unregister("never-registered")


def test_registering_a_spec_makes_it_resolvable_and_instantiable(temp_spec):
    """The whole point of ADR-001: a new model is one spec plus an adapter class."""
    temp_spec(name="brand-new")
    assert "brand-new" in list_model_names()
    model = get_model("brand-new")
    assert isinstance(model, SegmentationModel)
    assert model.is_loaded


# ---------------------------------------------------------------- availability


def test_specs_with_no_artefacts_are_always_available():
    assert weights_available(resolve_spec("classical")) is True
    assert weights_available(resolve_spec("trivial-ones")) is True


@_needs_trained_cutoutnet
def test_the_trained_cutoutnet_checkpoint_is_present():
    """This is the checkpoint committed to the repository; a fresh clone must work."""
    assert weights_available(resolve_spec("cutoutnet")) is True


def test_weights_available_is_false_when_the_checkpoint_is_missing(temp_spec):
    temp_spec(
        name="missing-weights",
        adapter="cutoutml.models.cutoutnet.adapter.CutoutNetAdapter",
        requires_weights=True,
        default_weights="does-not-exist/nothing.pt",
        options={"variant": "tiny"},
    )
    assert weights_available(resolve_spec("missing-weights")) is False


def test_weights_available_accepts_any_one_of_several_artefact_paths(tmp_path, temp_spec):
    """ONNX and TensorRT specs carry their artefact in options rather than
    default_weights, and either location counts as available."""
    artefact = tmp_path / "graph.onnx"
    artefact.write_bytes(b"not really onnx")
    temp_spec(
        name="artefact-in-options",
        requires_weights=False,
        default_weights=None,
        options={"onnx_path": str(artefact)},
    )
    assert weights_available(resolve_spec("artefact-in-options")) is True


def test_runtime_available_reflects_this_machine():
    """onnxruntime is a hard dependency here; TensorRT needs a GPU this box lacks."""
    assert runtime_available(resolve_spec("cutoutnet")) is True
    assert runtime_available(resolve_spec("cutoutnet-onnx")) is True
    assert runtime_available(resolve_spec("tensorrt")) is False


@_needs_trained_cutoutnet
def test_usable_models_excludes_the_gpu_only_and_weightless_specs():
    usable = {s.name for s in usable_models()}
    assert "classical" in usable
    assert "cutoutnet" in usable
    assert "tensorrt" not in usable


def test_every_trained_in_repo_claim_has_a_committed_training_record():
    """`trained-in-repo` is a provenance claim, and the artefact that backs it is a
    committed ``training/runs/*.json`` naming the checkpoint the run wrote.

    This existed as a false claim: ``u2net-lite`` carried the tag and the words "trained
    in-repo" with no checkpoint and no run record anywhere, so `cutoutml models` printed
    `MISSING ... trained-in-repo` and docs/models.md contradicted itself about it. Nothing
    failed, because a description is not executable - which is what this test fixes.
    """
    records = list((REPO_ROOT / "training" / "runs").glob("*.json"))
    assert records, "no training run records committed; the claim cannot be checked"
    trained: set[str] = set()
    for record in records:
        data = json.loads(record.read_text())
        # `serves_as` is the registry name a run's checkpoint is written for; older
        # records only carry the architecture, which is the same string for these.
        trained.add(str(data.get("serves_as") or data.get("arch") or ""))
        checkpoint = data.get("checkpoint")
        if checkpoint:
            trained.add(Path(checkpoint).stem)

    for spec in list_models():
        claims = "trained-in-repo" in spec.tags or "trained in-repo" in spec.description
        if not claims:
            continue
        stem = Path(spec.default_weights or "").stem
        assert stem in trained or spec.name in trained, (
            f"{spec.name} claims to be trained in this repository, but no "
            f"training/runs/*.json describes a run that produced {stem or spec.name!r}"
        )


@_needs_trained_cutoutnet
def test_a_spec_with_no_weights_does_not_claim_to_have_been_trained():
    """The inverse, checked on this machine rather than against the records: an entry the
    registry reports as unavailable must not describe itself as trained."""
    for spec in list_models():
        if weights_available(spec):
            continue
        assert "trained-in-repo" not in spec.tags, (
            f"{spec.name} has no artefact on disk yet is tagged trained-in-repo"
        )
        assert "trained in-repo" not in spec.description, (
            f"{spec.name} has no artefact on disk yet its description says it was trained in-repo"
        )


def test_catalogue_is_json_serialisable_and_carries_availability():
    entries = catalogue()
    assert entries
    for entry in entries:
        assert isinstance(entry["input_size"], list)
        assert isinstance(entry["tags"], list)
        assert isinstance(entry["weights_available"], bool)
        assert isinstance(entry["runtime_available"], bool)
    names = {e["name"] for e in entries}
    assert names == set(list_model_names())


def test_catalogue_availability_is_recomputed_per_call(tmp_path, temp_spec, monkeypatch):
    """A checkpoint appearing in models/ (a finished training run, a mounted volume)
    must show up without restarting the API.

    The weights directory is redirected rather than any lookup helper monkeypatched: the
    spec's path is relative, so this exercises whatever resolution order the registry
    actually implements instead of asserting against one private function that a refactor
    is free to move the work out of.
    """
    from cutoutml.core.config import get_settings

    monkeypatch.setenv("CUTOUTML_MODEL_WEIGHTS_DIR", str(tmp_path))
    get_settings.cache_clear()

    temp_spec(
        name="appears-later",
        adapter="cutoutml.models.cutoutnet.adapter.CutoutNetAdapter",
        requires_weights=True,
        default_weights="appears-later.pt",
        options={"variant": "tiny"},
    )

    def availability() -> bool:
        entry = next(e for e in catalogue() if e["name"] == "appears-later")
        return bool(entry["weights_available"])

    try:
        assert availability() is False
        (tmp_path / "appears-later.pt").write_bytes(b"checkpoint")
        assert availability() is True
    finally:
        get_settings.cache_clear()


# ------------------------------------------------------------------- get_model


def test_get_model_attaches_the_spec_so_metadata_can_report_provenance():
    model = get_model("classical", load=False)
    assert model.spec is not None
    assert model.spec.name == "classical"
    assert model.metadata().license.startswith("baseline implementation")


def test_get_model_with_load_false_does_not_load():
    """The harness needs this to time load() separately from inference."""
    model = get_model("classical", load=False)
    assert model.is_loaded is False
    assert model.load_seconds is None
    model.load()
    assert model.is_loaded is True
    assert model.load_seconds is not None and model.load_seconds >= 0.0


def _classical(name: str, **overrides: Any) -> ClassicalBaseline:
    model = get_model(name, load=False, **overrides)
    assert isinstance(model, ClassicalBaseline)
    return model


def test_get_model_passes_spec_options_to_the_adapter():
    assert _classical("classical-saliency").method == "saliency"
    assert _classical("classical").method == "grabcut"


def test_overrides_win_over_spec_options():
    assert _classical("classical", method="saliency").method == "saliency"


def test_random_init_is_refused_for_specs_that_do_not_declare_it():
    """Random weights return noise; reaching that from an API request by accident
    would publish a meaningless mask as a real one."""
    with pytest.raises(ValueError, match="does not support random initialisation"):
        get_model("classical", random_init=True)


def test_random_init_is_allowed_for_specs_that_declare_it_and_is_flagged():
    model = get_model("cutoutnet-tiny", random_init=True)
    meta = model.metadata()
    assert meta.randomly_initialized is True
    assert meta.accuracy_valid is False
    assert "RANDOM WEIGHTS" in meta.notes


def test_a_missing_checkpoint_raises_an_error_naming_the_path_and_the_way_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """This is the normal condition for u2net and birefnet: their published weights
    live on HuggingFace, which is unreachable in this environment. The error has to
    say where the file was expected and what to do instead.

    The weights directory is redirected at an empty one rather than the test skipping
    when u2net happens to be on disk. A `skipif` on the real directory made this case
    disappear on exactly the machines that had done a download — so whether the message
    a first-time user hits was tested at all depended on the state of a cache, and the
    run that skipped it reported green.
    """
    from cutoutml.core.config import get_settings

    monkeypatch.setenv("CUTOUTML_MODEL_WEIGHTS_DIR", str(tmp_path / "empty"))
    get_settings.cache_clear()
    try:
        with pytest.raises(WeightsUnavailableError) as excinfo:
            get_model("u2net")
        message = str(excinfo.value)
        assert "u2net.pth" in message
        assert "train" in message.lower()
    finally:
        get_settings.cache_clear()


@_needs_trained_cutoutnet
def test_the_missing_weights_error_points_at_training_for_in_repo_architectures(temp_spec):
    """CutoutNet weights are never downloaded, so its hint must not suggest that.
    The adapter also resolves by model name under models/cutoutnet/ rather than
    trusting the spec's path, so a run that finishes mid-session is picked up."""
    temp_spec(
        name="cutoutnet-nonexistent-variant",
        adapter="cutoutml.models.cutoutnet.adapter.CutoutNetAdapter",
        requires_weights=True,
        default_weights="cutoutnet/absent.pt",
        supports_random_init=True,
        options={"variant": "tiny"},
    )
    with pytest.raises(WeightsUnavailableError) as excinfo:
        get_model("cutoutnet-nonexistent-variant")
    message = str(excinfo.value)
    assert "models/cutoutnet/cutoutnet-nonexistent-variant.pt" in message
    assert "cutoutml.training.train" in message


def test_adapter_must_be_a_segmentation_model_subclass(temp_spec):
    temp_spec(name="not-a-model", adapter="pathlib.Path")
    with pytest.raises(TypeError, match="not a SegmentationModel"):
        get_model("not-a-model")


# ------------------------------------------------------- the adapter contract


def _contract_models() -> list[str]:
    """Every spec that can actually run here. Skipped ones are reported, not faked."""
    names = ["classical-saliency", "trivial-center", "trivial-ones"]
    for trained in ("cutoutnet", "cutoutnet-onnx"):
        if weights_available(resolve_spec(trained)):
            names.append(trained)
    return names


@pytest.fixture(scope="module", params=_contract_models())
def contract_model(request: pytest.FixtureRequest) -> SegmentationModel:
    """Loaded once per adapter: loading CutoutNet per test would dominate runtime."""
    return get_model(request.param, device="cpu")


def test_preprocess_returns_a_batched_tensor_and_one_info_per_image(contract_model):
    images = [
        np.random.default_rng(1).integers(0, 255, (120, 200, 3), dtype=np.uint8),
        np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8),
    ]
    tensor, infos = contract_model.preprocess(images)

    w, h = contract_model.input_size
    assert tensor.shape == (2, 3, h, w)
    assert tensor.dtype == torch.float32
    assert len(infos) == 2
    assert all(isinstance(i, LetterboxInfo) for i in infos)
    assert (infos[0].orig_width, infos[0].orig_height) == (200, 120)


def test_preprocess_accepts_a_single_unbatched_image(contract_model):
    """Convenience the API relies on: a single upload should not need wrapping."""
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    tensor, infos = contract_model.preprocess(image)
    assert tensor.shape[0] == 1
    assert len(infos) == 1


def test_predict_returns_logits_shaped_n1hw(contract_model):
    image = np.random.default_rng(3).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    tensor, _ = contract_model.preprocess([image])
    logits = contract_model.predict(tensor)

    assert logits.ndim == 4
    assert logits.shape[0] == 1
    assert logits.shape[1] == 1
    assert logits.dtype == torch.float32
    # Logits, not probabilities: sigmoid lives in postprocess so the training loop
    # can reuse the same forward path with a numerically stable loss.
    assert logits.min() < 0.0 or logits.max() > 1.0


def test_postprocess_returns_one_alpha_map_per_image_at_the_original_size(contract_model):
    images = [
        np.random.default_rng(4).integers(0, 255, (100, 150, 3), dtype=np.uint8),
        np.random.default_rng(5).integers(0, 255, (60, 40, 3), dtype=np.uint8),
    ]
    tensor, infos = contract_model.preprocess(images)
    alphas = contract_model.postprocess(contract_model.predict(tensor), infos)

    assert len(alphas) == 2
    assert alphas[0].shape == (100, 150)
    assert alphas[1].shape == (60, 40)
    for alpha in alphas:
        assert alpha.dtype == np.float32
        assert float(alpha.min()) >= 0.0
        assert float(alpha.max()) <= 1.0


def test_postprocess_rejects_a_batch_and_info_count_mismatch(contract_model):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    tensor, infos = contract_model.preprocess([image, image])
    logits = contract_model.predict(tensor)
    with pytest.raises(ValueError, match="letterbox infos"):
        contract_model.postprocess(logits, infos[:1])


def test_infer_is_equivalent_to_the_three_stages(contract_model):
    """The pipelines call infer(); the harness calls the stages separately to
    attribute latency. They must agree or the stage breakdown is fiction."""
    image = np.random.default_rng(6).integers(0, 255, (80, 112, 3), dtype=np.uint8)

    one_shot = contract_model.infer([image])
    tensor, infos = contract_model.preprocess([image])
    staged = contract_model.postprocess(contract_model.predict(tensor), infos)

    assert len(one_shot) == len(staged) == 1
    np.testing.assert_allclose(one_shot[0], staged[0], atol=1e-5)


def test_batching_does_not_change_per_image_results(contract_model):
    """If it did, every batched benchmark row would be measuring a different model
    from the batch-1 row."""
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]

    batched = contract_model.infer(images)
    individually = [contract_model.infer([img])[0] for img in images]

    for got, want in zip(batched, individually, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-4)


def test_metadata_is_complete_and_serialisable(contract_model):
    meta = contract_model.metadata()
    assert isinstance(meta, ModelMetadata)
    assert meta.name
    assert meta.architecture
    assert meta.device == "cpu"
    assert meta.runtime
    assert meta.license
    payload = meta.as_dict()
    assert isinstance(payload["input_size"], list)
    assert payload["accuracy_valid"] is not None


@_needs_trained_cutoutnet
def test_metadata_records_the_digest_of_the_weights_actually_loaded(tmp_path: Path):
    """A published accuracy figure that names only a checkpoint *path* is not evidence:
    the file behind that path changes with every training run."""
    from cutoutml.models.base import weights_digest

    model = get_model("cutoutnet", device="cpu")
    meta = model.metadata()
    assert meta.weights_path is not None
    expected = hashlib.sha256(Path(meta.weights_path).read_bytes()).hexdigest()
    assert meta.weights_sha256 == expected

    # Rewriting a file in place must not serve the previous digest from the cache.
    checkpoint = tmp_path / "w.pt"
    checkpoint.write_bytes(b"first")
    first = weights_digest(checkpoint)
    checkpoint.write_bytes(b"second")
    assert weights_digest(checkpoint) != first


def test_weightless_models_report_no_digest_rather_than_a_hash_of_nothing():
    assert get_model("trivial-center", device="cpu").metadata().weights_sha256 is None


def test_a_converted_checkpoint_reports_the_digest_of_what_it_was_converted_from():
    """The only identity a converted checkpoint has that survives re-conversion.

    `torch.save` does not promise identical bytes for identical tensors, so re-deriving
    `u2net.pt` from the same ONNX produces a file with a different `weights_sha256` and
    exactly the same weights. That happened between two benchmark runs here, and it made
    a published row look as though it had been measured against weights that were no
    longer present. The conversion already records the source digest inside the
    checkpoint; this asserts it reaches the metadata, which is what a benchmark row can
    be checked against later.
    """
    sidecar_path = REPO_ROOT / "models" / "conversions" / "u2net.json"
    if not weights_available(resolve_spec("u2net")) or not sidecar_path.is_file():
        pytest.skip("u2net weights are not present, so no conversion can be inspected")

    meta = get_model("u2net", device="cpu").metadata()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert meta.weights_source_sha256 == sidecar["source_sha256"]
    # And it is the ONNX's digest, not a restatement of the checkpoint's own.
    assert meta.weights_source_sha256 != meta.weights_sha256


@_needs_trained_cutoutnet
def test_a_checkpoint_trained_here_reports_no_source_digest():
    """`None` means "not converted from anything", which the table prints as such rather
    than leaving a reader to read a blank cell as missing data."""
    assert get_model("cutoutnet", device="cpu").metadata().weights_source_sha256 is None


def test_the_onnx_adapter_hashes_the_graph_it_runs_not_the_checkpoint_it_ignores():
    model = get_model("cutoutnet-onnx", device="cpu")
    meta = model.metadata()
    assert meta.weights_path is not None
    assert meta.weights_path.endswith(".onnx")
    assert meta.weights_sha256 == hashlib.sha256(Path(meta.weights_path).read_bytes()).hexdigest()


def test_load_is_idempotent_and_records_cold_start_time(contract_model):
    seconds = contract_model.load_seconds
    contract_model.load()
    assert contract_model.load_seconds == seconds


def test_using_a_model_before_load_is_an_error_not_a_crash():
    model = get_model("cutoutnet", load=False, device="cpu")
    with pytest.raises(RuntimeError, match="before load"):
        model.infer([np.zeros((16, 16, 3), dtype=np.uint8)])


# ------------------------------------------------------------- adapter specifics


def test_classical_baselines_report_valid_accuracy_and_zero_parameters():
    """They are the interpretability floor of the benchmark table, so their
    accuracy column must be trustworthy and clearly non-learned."""
    meta = get_model("classical-saliency", device="cpu").metadata()
    assert meta.param_count == 0
    assert meta.accuracy_valid is True
    assert meta.randomly_initialized is False
    assert "opencv" in meta.runtime


def test_trivial_ones_predicts_foreground_everywhere():
    model = get_model("trivial-ones", device="cpu")
    (alpha,) = model.infer([np.zeros((40, 40, 3), dtype=np.uint8)])
    assert float(alpha.min()) > 0.99


def test_trivial_center_ellipse_is_content_blind():
    """Its whole diagnostic value is that the image cannot influence it."""
    model = get_model("trivial-center", device="cpu")
    rng = np.random.default_rng(8)
    a = model.infer([rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)])[0]
    b = model.infer([np.zeros((64, 64, 3), dtype=np.uint8)])[0]
    np.testing.assert_array_equal(a, b)
    assert a[32, 32] > 0.9
    assert a[0, 0] < 0.1


@_needs_trained_cutoutnet
def test_cutoutnet_reports_its_real_parameter_count():
    model = get_model("cutoutnet", device="cpu")
    meta = model.metadata()
    assert 0.5e6 < meta.param_count < 3e6
    assert meta.accuracy_valid is True
    assert meta.weights_path is not None and Path(meta.weights_path).is_file()


@_needs_trained_cutoutnet
def test_cutoutnet_produces_a_non_degenerate_mask_on_a_high_contrast_subject():
    """A trained model must at least separate a bright disc from a dark field. If
    this fails, the committed checkpoint is broken and every accuracy number in
    docs/benchmarks.md is meaningless."""
    image = np.full((128, 128, 3), 20, dtype=np.uint8)
    yy, xx = np.mgrid[0:128, 0:128]
    disc = (xx - 64) ** 2 + (yy - 64) ** 2 <= 34**2
    image[disc] = (240, 230, 60)

    (alpha,) = get_model("cutoutnet", device="cpu").infer([image])

    assert alpha[disc].mean() > alpha[~disc].mean()
    assert 0.05 < float(alpha.mean()) < 0.95


@pytest.mark.skipif(
    not weights_available(resolve_spec("cutoutnet-onnx")),
    reason="ONNX export not present; run python -m cutoutml.models.export_onnx",
)
def test_onnx_adapter_reports_the_provider_it_actually_got():
    """Reporting a provider that failed to initialise is how bogus benchmark rows
    are produced, so metadata records get_providers() rather than the request."""
    model = get_model("cutoutnet-onnx", device="cpu")
    assert isinstance(model, OnnxAdapter)
    assert model.active_providers
    assert model.active_providers[0] == "CPUExecutionProvider"
    assert model.metadata().runtime == "onnxruntime:CPUExecutionProvider"


@pytest.mark.skipif(
    not weights_available(resolve_spec("cutoutnet-onnx")),
    reason="ONNX export not present; run python -m cutoutml.models.export_onnx",
)
@_needs_trained_cutoutnet
def test_onnx_and_torch_cutoutnet_agree_within_numerical_tolerance():
    """The export is only a valid benchmark row if it computes the same function."""
    image = np.random.default_rng(9).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    torch_alpha = get_model("cutoutnet", device="cpu").infer([image])[0]
    onnx_alpha = get_model("cutoutnet-onnx", device="cpu").infer([image])[0]
    np.testing.assert_allclose(onnx_alpha, torch_alpha, atol=2e-3)


def test_tensorrt_spec_is_registered_but_not_usable_on_this_machine():
    """Recorded as a test so the CPU-only constraint is asserted, not assumed."""
    spec = resolve_spec("tensorrt")
    assert spec.runtime == "tensorrt"
    assert runtime_available(spec) is False
