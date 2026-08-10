"""Segmentation model adapters and the registry that resolves them by name.

Import the registry rather than a concrete adapter: ``get_model("cutoutnet")``
keeps callers decoupled from architectures and runtimes
(``docs/decisions/ADR-001-model-registry.md``).
"""

from cutoutml.models.base import (
    ModelMetadata,
    ModelSpec,
    SegmentationModel,
    TorchSegmentationModel,
    WeightsUnavailableError,
)
from cutoutml.models.registry import (
    ModelNotFoundError,
    get_model,
    list_models,
    register,
    resolve_spec,
)

__all__ = [
    "ModelMetadata",
    "ModelNotFoundError",
    "ModelSpec",
    "SegmentationModel",
    "TorchSegmentationModel",
    "WeightsUnavailableError",
    "get_model",
    "list_models",
    "register",
    "resolve_spec",
]
