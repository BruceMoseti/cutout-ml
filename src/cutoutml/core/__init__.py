"""Framework-agnostic primitives shared by every CutoutML component.

The re-exports below are resolved lazily (PEP 562). Eager ones were a quiet
correctness problem: ``from cutoutml.core.config import Settings`` runs this module
first, and when it imported :mod:`cutoutml.core.devices` at the top, reading a
configuration value pulled in torch - about 0.7 s and several hundred megabytes - for
every process, including the API, which never runs a model. Attribute access still
works exactly as before; the import is simply deferred to the first use.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import for type checkers only; see the module docstring
    from cutoutml.core.config import Settings, get_settings
    from cutoutml.core.devices import (
        DeviceInfo,
        autocast_context,
        resolve_device,
        resolve_precision,
    )
    from cutoutml.core.logging import bind_request_id, configure_logging, get_logger
    from cutoutml.core.precision import Precision

#: Attribute name -> defining submodule. Kept explicit rather than derived so that a
#: typo surfaces as an ``AttributeError`` here instead of an import of the wrong module.
_EXPORTS: dict[str, str] = {
    "DeviceInfo": "cutoutml.core.devices",
    "Precision": "cutoutml.core.precision",
    "Settings": "cutoutml.core.config",
    "autocast_context": "cutoutml.core.devices",
    "bind_request_id": "cutoutml.core.logging",
    "configure_logging": "cutoutml.core.logging",
    "get_logger": "cutoutml.core.logging",
    "get_settings": "cutoutml.core.config",
    "resolve_device": "cutoutml.core.devices",
    "resolve_precision": "cutoutml.core.devices",
}

__all__ = [
    "DeviceInfo",
    "Precision",
    "Settings",
    "autocast_context",
    "bind_request_id",
    "configure_logging",
    "get_logger",
    "get_settings",
    "resolve_device",
    "resolve_precision",
]


def __getattr__(name: str) -> Any:
    try:
        module_path = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    # importlib rather than `from . import devices`: the latter reaches this same hook to
    # resolve the name and recurses until the stack runs out.
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
