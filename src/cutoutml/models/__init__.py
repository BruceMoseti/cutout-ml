"""Segmentation model adapters and the registry that resolves them by name.

Import the registry rather than a concrete adapter: ``get_model("cutoutnet")``
keeps callers decoupled from architectures and runtimes
(``docs/decisions/ADR-001-model-registry.md``).

The re-exports are lazy (PEP 562) for the same reason the registry is torch-free:
``from cutoutml.models.registry import resolve_spec`` executes this module first, and an
eager ``from cutoutml.models.base import ...`` here dragged torch and OpenCV into the API
process purely as a side effect of the package's own convenience imports.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import for type checkers only; see the module docstring
    from cutoutml.models.base import SegmentationModel, TorchSegmentationModel
    from cutoutml.models.registry import (
        ModelNotFoundError,
        get_model,
        list_models,
        register,
        resolve_spec,
    )
    from cutoutml.models.spec import ModelMetadata, ModelSpec, WeightsUnavailableError

#: Attribute name -> defining submodule. ``ModelSpec`` and friends resolve to the
#: torch-free :mod:`cutoutml.models.spec`, not to :mod:`cutoutml.models.base`, so that
#: reading the catalogue stays cheap.
_EXPORTS: dict[str, str] = {
    "ModelMetadata": "cutoutml.models.spec",
    "ModelNotFoundError": "cutoutml.models.registry",
    "ModelSpec": "cutoutml.models.spec",
    "SegmentationModel": "cutoutml.models.base",
    "TorchSegmentationModel": "cutoutml.models.base",
    "WeightsUnavailableError": "cutoutml.models.spec",
    "get_model": "cutoutml.models.registry",
    "list_models": "cutoutml.models.registry",
    "register": "cutoutml.models.registry",
    "resolve_spec": "cutoutml.models.registry",
}

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


def __getattr__(name: str) -> Any:
    try:
        module_path = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    # importlib rather than `from . import base`: the latter reaches this same hook to
    # resolve the name and recurses until the stack runs out.
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
