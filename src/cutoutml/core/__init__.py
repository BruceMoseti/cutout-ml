"""Framework-agnostic primitives shared by every CutoutML component."""

from cutoutml.core.config import Settings, get_settings
from cutoutml.core.devices import (
    DeviceInfo,
    Precision,
    autocast_context,
    resolve_device,
    resolve_precision,
)
from cutoutml.core.logging import bind_request_id, configure_logging, get_logger

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
