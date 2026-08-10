"""Declarative model registry.

Callers ask for a model *by name*; nothing outside this module knows which class
implements it, what its input size is or where its weights live. Adding a model is
one :class:`~cutoutml.models.base.ModelSpec` entry plus an adapter class - no
change to the API, the pipelines, the worker or the benchmark harness. The
rationale is written up in ``docs/decisions/ADR-001-model-registry.md``.

Adapters are referenced by dotted path and imported lazily, so listing the
catalogue (which ``GET /models`` does on every call) never imports TensorRT or
even instantiates a network.
"""

from __future__ import annotations

import importlib
import importlib.util
import threading
from pathlib import Path
from typing import Any

from cutoutml.core.config import REPO_ROOT, get_settings
from cutoutml.core.devices import Precision
from cutoutml.core.logging import get_logger
from cutoutml.models.base import ModelSpec, SegmentationModel

log = get_logger(__name__)

_REGISTRY: dict[str, ModelSpec] = {}
_LOCK = threading.RLock()


class ModelNotFoundError(KeyError):
    """Raised for an unknown model name, listing what *is* available."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(
            f"unknown model {name!r}; available models: {', '.join(sorted(available))}"
        )


def register(spec: ModelSpec, *, overwrite: bool = False) -> ModelSpec:
    """Add a spec to the registry. Raises on duplicate names unless ``overwrite``."""
    with _LOCK:
        if spec.name in _REGISTRY and not overwrite:
            raise ValueError(f"model {spec.name!r} is already registered")
        _REGISTRY[spec.name] = spec
    return spec


def unregister(name: str) -> None:
    """Remove a spec (used by tests that register temporary models)."""
    with _LOCK:
        _REGISTRY.pop(name, None)


def resolve_spec(name: str) -> ModelSpec:
    """Look up a spec, raising :class:`ModelNotFoundError` when absent."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ModelNotFoundError(name, list(_REGISTRY)) from None


def list_models() -> list[ModelSpec]:
    """Every registered spec, sorted by name."""
    return sorted(_REGISTRY.values(), key=lambda s: s.name)


def list_model_names() -> list[str]:
    return sorted(_REGISTRY)


def _import_adapter(dotted: str) -> type[SegmentationModel]:
    module_path, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, SegmentationModel):
        raise TypeError(f"{dotted} is not a SegmentationModel subclass")
    return cls  # type: ignore[no-any-return]


def _resolve_weights(spec: ModelSpec) -> Path | None:
    """Turn a spec's relative ``default_weights`` into an absolute path if present."""
    if not spec.default_weights:
        return None
    candidate = Path(spec.default_weights)
    if not candidate.is_absolute():
        candidate = get_settings().model_weights_dir / candidate
    return candidate


def _artifact_candidates(spec: ModelSpec) -> list[Path]:
    """Every on-disk artefact a spec could load from."""
    candidates: list[Path] = []
    resolved = _resolve_weights(spec)
    if resolved is not None:
        candidates.append(resolved)
    for key in ("onnx_path", "engine_path"):
        raw = spec.options.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        candidates.append(path if path.is_absolute() else REPO_ROOT / path)
    return candidates


def weights_available(spec: ModelSpec) -> bool:
    """Whether this spec's artefacts exist on disk.

    Specs that need nothing (the classical baselines) are always available. Everything
    else needs at least one of its candidate paths to exist - which is why
    ``GET /models`` can tell a caller *before* they submit a job that ``u2net`` has no
    weights here, instead of failing the job asynchronously.
    """
    candidates = _artifact_candidates(spec)
    if not candidates:
        return True
    return any(path.is_file() for path in candidates)


def runtime_available(spec: ModelSpec) -> bool:
    """Whether the runtime this spec needs is importable/usable on this machine."""
    if spec.runtime == "onnxruntime":
        return importlib.util.find_spec("onnxruntime") is not None
    if spec.runtime == "tensorrt":
        from cutoutml.core.devices import cuda_available

        return importlib.util.find_spec("tensorrt") is not None and cuda_available()
    return True


def usable_models() -> list[ModelSpec]:
    """Specs that could actually serve a request right now."""
    return [s for s in list_models() if weights_available(s) and runtime_available(s)]


def get_model(
    name: str,
    *,
    device: str | None = None,
    precision: Precision | None = None,
    weights_path: Path | str | None = None,
    random_init: bool = False,
    load: bool = True,
    **overrides: Any,
) -> SegmentationModel:
    """Instantiate a registered model.

    Parameters
    ----------
    device, precision:
        Default to the process settings. Both are *requests*: the adapter resolves
        them against what the machine actually supports (see
        :func:`cutoutml.core.devices.resolve_device`).
    random_init:
        Build the architecture with random weights. Only allowed for specs that
        declare ``supports_random_init``, because it produces meaningless masks and
        must never be reachable by accident from an API request.
    load:
        Set ``False`` to construct without paying the load cost - the benchmark
        harness uses this to time ``load()`` separately.
    overrides:
        Passed to the adapter constructor, merged over ``spec.options``.
    """
    spec = resolve_spec(name)
    settings = get_settings()

    if random_init and not spec.supports_random_init:
        raise ValueError(
            f"model {name!r} does not support random initialisation; it exists to "
            "produce real masks, and random weights would silently return noise"
        )

    cls = _import_adapter(spec.adapter)
    resolved_weights = Path(weights_path) if weights_path else _resolve_weights(spec)

    kwargs: dict[str, Any] = {
        "name": spec.name,
        "input_size": spec.input_size,
        "device": device if device is not None else settings.device,
        "precision": precision if precision is not None else settings.precision,
        "random_init": random_init,
        **spec.options,
        **overrides,
    }
    # Runtime adapters (ONNX, TensorRT) locate their artefact through onnx_path /
    # engine_path and have no state dict to load, so passing weights_path would only
    # give them a second, contradictory source of truth.
    serialised_runtime = "onnx_path" in kwargs or "engine_path" in kwargs
    if spec.requires_weights and not serialised_runtime:
        kwargs["weights_path"] = resolved_weights

    model = cls(**kwargs)
    model.spec = spec
    if load:
        model.load()
    return model


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

register(
    ModelSpec(
        name="cutoutnet",
        adapter="cutoutml.models.cutoutnet.adapter.CutoutNetAdapter",
        architecture="CutoutNet-small",
        input_size=(256, 256),
        license="MIT (original architecture and weights, this repository)",
        source="https://github.com/your-org/cutout-ml",
        default_weights="cutoutnet/cutoutnet-small.pt",
        supports_random_init=True,
        description=(
            "Original ~1.1M-parameter depthwise-separable encoder + FPN-lite decoder "
            "with a learned alpha refinement head. Trained in-repo on the synthetic "
            "dataset; this is the default model."
        ),
        tags=("default", "fast", "trained-in-repo"),
        options={"variant": "small"},
    )
)

register(
    ModelSpec(
        name="cutoutnet-tiny",
        adapter="cutoutml.models.cutoutnet.adapter.CutoutNetAdapter",
        architecture="CutoutNet-tiny",
        input_size=(256, 256),
        license="MIT (original architecture, this repository)",
        source="https://github.com/your-org/cutout-ml",
        default_weights="cutoutnet/cutoutnet-tiny.pt",
        supports_random_init=True,
        description="0.3M-parameter CutoutNet for measuring the latency floor.",
        tags=("fast",),
        options={"variant": "tiny"},
    )
)

register(
    ModelSpec(
        name="cutoutnet-base",
        adapter="cutoutml.models.cutoutnet.adapter.CutoutNetAdapter",
        architecture="CutoutNet-base",
        input_size=(256, 256),
        license="MIT (original architecture and weights, this repository)",
        source="https://github.com/BruceMoseti/cutout-ml",
        default_weights="cutoutnet/cutoutnet-base.pt",
        supports_random_init=True,
        description=(
            "4.3M-parameter CutoutNet. Trained on exactly the same data budget as "
            "cutoutnet-small, so the pair isolates capacity from everything else."
        ),
        tags=("high-accuracy", "trained-in-repo"),
        options={"variant": "base"},
    )
)

register(
    ModelSpec(
        name="cutoutnet-onnx",
        adapter="cutoutml.models.onnx_adapter.OnnxAdapter",
        architecture="CutoutNet-small (ONNX)",
        input_size=(256, 256),
        license="MIT (original architecture and weights, this repository)",
        source="exported from the cutoutnet checkpoint",
        default_weights="cutoutnet/cutoutnet-small.onnx",
        runtime="onnxruntime",
        requires_weights=False,
        description=(
            "The trained CutoutNet exported to ONNX and run through onnxruntime "
            "with TensorRT -> CUDA -> CPU provider fallback."
        ),
        tags=("onnx", "trained-in-repo"),
        options={
            "onnx_path": "models/cutoutnet/cutoutnet-small.onnx",
            "normalization": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        },
    )
)

register(
    ModelSpec(
        name="u2net",
        adapter="cutoutml.models.u2net.adapter.U2NetAdapter",
        architecture="U2Net-full",
        input_size=(320, 320),
        license="Apache-2.0 (upstream architecture and published weights)",
        source="https://github.com/xuebinqin/U-2-Net",
        default_weights="u2net/u2net.pth",
        weights_url=None,
        supports_random_init=True,
        description=(
            "44M-parameter nested U-structure. Independent implementation, "
            "shape-compatible with official u2net.pth (keys are remapped on load). "
            "Weights are not bundled."
        ),
        tags=("high-accuracy", "needs-weights"),
        options={"variant": "full"},
    )
)

register(
    ModelSpec(
        name="u2net-lite",
        adapter="cutoutml.models.u2net.adapter.U2NetAdapter",
        architecture="U2Net-lite",
        input_size=(256, 256),
        license=(
            "Architecture: Apache-2.0 (Qin et al., independently reimplemented). "
            "Weights: MIT, trained in this repository on the synthetic dataset."
        ),
        source="https://github.com/xuebinqin/U-2-Net",
        default_weights="u2net/u2net-lite.pt",
        supports_random_init=True,
        description=(
            "U^2-Net-P, the 1.1M-parameter nested-U variant, trained in-repo. Almost "
            "the same parameter count as cutoutnet-small with a very different "
            "compute profile, which is what makes the comparison informative. The "
            "official u2netp.pth also loads here (keys are remapped)."
        ),
        tags=("trained-in-repo",),
        options={"variant": "lite"},
    )
)

register(
    ModelSpec(
        name="birefnet",
        adapter="cutoutml.models.birefnet.adapter.BiRefNetAdapter",
        architecture="BiRefNetCompact",
        input_size=(512, 512),
        license=(
            "MIT for this reimplementation. Official BiRefNet code is MIT; official "
            "weights are on HuggingFace and SOME third-party fine-tunes are "
            "non-commercial - verify before use."
        ),
        source="https://github.com/ZhengPeng7/BiRefNet",
        default_weights="birefnet/birefnet-compact.pt",
        supports_random_init=True,
        description=(
            "Architecture-inspired reimplementation of the bilateral-reference "
            "design: localization module + reconstruction module with inner (source "
            "pixel) and outer (gradient) references. Not weight-compatible with "
            "official BiRefNet checkpoints."
        ),
        tags=("high-resolution", "needs-weights"),
        options={"variant": "compact"},
    )
)

register(
    ModelSpec(
        name="classical",
        adapter="cutoutml.models.classical.baseline.ClassicalBaseline",
        architecture="Classical/grabcut",
        input_size=(320, 320),
        license="MIT (this implementation)",
        source="Rother et al. 2004 (GrabCut)",
        requires_weights=False,
        description=(
            "GrabCut seeded from a centred rectangle. The strongest zero-training "
            "baseline in this repo and the number every learned model must beat."
        ),
        tags=("baseline", "no-weights"),
        options={"method": "grabcut"},
    )
)

register(
    ModelSpec(
        name="classical-saliency",
        adapter="cutoutml.models.classical.baseline.ClassicalBaseline",
        architecture="Classical/saliency",
        input_size=(320, 320),
        license="MIT (this implementation)",
        source="Hou & Zhang 2007",
        requires_weights=False,
        description="Spectral-residual saliency + Otsu only. The cheapest baseline.",
        tags=("baseline", "no-weights", "fast"),
        options={"method": "saliency"},
    )
)

register(
    ModelSpec(
        name="classical-saliency-grabcut",
        adapter="cutoutml.models.classical.baseline.ClassicalBaseline",
        architecture="Classical/saliency+grabcut",
        input_size=(320, 320),
        license="MIT (this implementation)",
        source="Hou & Zhang 2007 + Rother et al. 2004",
        requires_weights=False,
        description=(
            "Saliency builds a trimap that GrabCut refines. Included because it is "
            "the obvious combination and because it demonstrates a real failure "
            "mode: a bad saliency seed makes GrabCut worse than a plain rectangle."
        ),
        tags=("baseline", "no-weights"),
        options={"method": "saliency+grabcut"},
    )
)

register(
    ModelSpec(
        name="trivial-center",
        adapter="cutoutml.models.classical.baseline.TrivialBaseline",
        architecture="Trivial/center_ellipse",
        input_size=(320, 320),
        license="MIT (this implementation)",
        source="metric calibration reference",
        requires_weights=False,
        description=(
            "A fixed centred ellipse, ignoring the image entirely. Calibrates the "
            "accuracy column and exposes centre-prior bias in the eval set."
        ),
        tags=("reference", "no-weights"),
        options={"method": "center_ellipse"},
    )
)

register(
    ModelSpec(
        name="trivial-ones",
        adapter="cutoutml.models.classical.baseline.TrivialBaseline",
        architecture="Trivial/ones",
        input_size=(320, 320),
        license="MIT (this implementation)",
        source="metric calibration reference",
        requires_weights=False,
        description=(
            "Predicts foreground everywhere. Its IoU equals the mean foreground "
            "coverage of the eval set - the absolute floor for any real method."
        ),
        tags=("reference", "no-weights"),
        options={"method": "ones"},
    )
)

register(
    ModelSpec(
        name="tensorrt",
        adapter="cutoutml.models.tensorrt_adapter.TensorRTAdapter",
        architecture="CutoutNet-small (TensorRT)",
        input_size=(256, 256),
        license="MIT (original architecture and weights, this repository)",
        source="built from the exported CutoutNet ONNX graph",
        runtime="tensorrt",
        requires_weights=False,
        description=(
            "Builds/loads a TensorRT engine from the CutoutNet ONNX export. Raises "
            "RuntimeError('TensorRT unavailable') without a CUDA GPU + TensorRT."
        ),
        tags=("gpu-only", "tensorrt"),
        options={"onnx_path": "models/cutoutnet/cutoutnet-small.onnx", "max_batch": 8},
    )
)


def catalogue() -> list[dict[str, Any]]:
    """JSON-serialisable catalogue, used by ``GET /models``.

    Availability is computed per call rather than cached: a checkpoint appearing in
    ``models/`` (a finished training run, a mounted volume) should show up without a
    restart.
    """
    return [
        spec.as_dict()
        | {
            "weights_available": weights_available(spec),
            "runtime_available": runtime_available(spec),
        }
        for spec in list_models()
    ]
