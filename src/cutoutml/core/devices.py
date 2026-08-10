"""Device and precision resolution.

CutoutML is developed and benchmarked on CPU-only machines but is meant to run
on CUDA GPUs in production, so *nothing* may hardcode ``cuda``. Every model
adapter routes through :func:`resolve_device`, which honours an explicit request
when it is actually available and otherwise degrades to CPU with a warning.

Precision handling is equally defensive: fp16 is only useful on CUDA, while
bf16 is safe on both modern CPUs (AVX512-BF16/AMX) and Ampere+ GPUs. Requesting
an unusable precision downgrades rather than crashing.
"""

from __future__ import annotations

import contextlib
import dataclasses
import platform
from collections.abc import Iterator
from typing import Literal

import torch

from cutoutml.core.logging import get_logger

log = get_logger(__name__)

Precision = Literal["fp32", "fp16", "bf16"]

_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceInfo:
    """A snapshot of the compute device actually in use."""

    type: str
    index: int | None
    name: str
    total_memory_bytes: int | None
    capability: str | None

    @property
    def torch_device(self) -> torch.device:
        if self.index is None:
            return torch.device(self.type)
        return torch.device(f"{self.type}:{self.index}")

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def cuda_available() -> bool:
    """True when a usable CUDA device is present."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:  # pragma: no cover - defensive: broken driver installs  # noqa: BLE001 - driver probes raise assorted vendor errors
        return False


def mps_available() -> bool:
    """True on Apple Silicon with a working Metal backend."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    """Map a user request onto a device that exists on this machine.

    ``"auto"`` prefers CUDA, then MPS, then CPU. An explicit but unavailable
    request (e.g. ``"cuda"`` on this CPU-only box) logs a warning and falls back
    to CPU instead of raising, which keeps the same container image usable in
    both environments.
    """
    if isinstance(requested, torch.device):
        requested = str(requested)
    req = (requested or "auto").strip().lower()

    if req in {"auto", ""}:
        if cuda_available():
            return torch.device("cuda:0")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if req.startswith("cuda"):
        if cuda_available():
            return torch.device(req)
        log.warning("cuda_requested_but_unavailable", requested=req, fallback="cpu")
        return torch.device("cpu")

    if req == "mps":
        if mps_available():
            return torch.device("mps")
        log.warning("mps_requested_but_unavailable", requested=req, fallback="cpu")
        return torch.device("cpu")

    return torch.device(req)


def describe_device(device: torch.device) -> DeviceInfo:
    """Human-readable description of a resolved device, for run metadata."""
    if device.type == "cuda" and cuda_available():
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        return DeviceInfo(
            type="cuda",
            index=index,
            name=props.name,
            total_memory_bytes=int(props.total_memory),
            capability=f"{props.major}.{props.minor}",
        )
    if device.type == "mps":
        return DeviceInfo("mps", None, platform.processor() or "apple-silicon", None, None)
    return DeviceInfo(
        type="cpu",
        index=None,
        name=cpu_name(),
        total_memory_bytes=None,
        capability=None,
    )


def cpu_name() -> str:
    """Best-effort CPU model string (``/proc/cpuinfo`` on Linux)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:  # noqa: PTH123 - procfs
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown-cpu"


def bf16_supported(device: torch.device) -> bool:
    """Whether bf16 autocast is meaningful on this device."""
    if device.type == "cuda":
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:  # pragma: no cover  # noqa: BLE001 - driver probes raise assorted vendor errors
            return False
    # torch.autocast('cpu') supports bf16 everywhere; it is only *fast* with
    # AVX512-BF16/AMX, but correctness does not depend on that. Any other device type
    # (mps, xpu) has no bf16 autocast path we have tested.
    return device.type == "cpu"


def resolve_precision(requested: Precision, device: torch.device) -> Precision:
    """Downgrade a precision request to something valid for ``device``.

    * fp16 autocast on CPU is supported by recent PyTorch but is usually slower
      than fp32 and numerically fragile, so we refuse it and use bf16 instead.
    * bf16 on a device without support falls back to fp32.
    """
    if requested == "fp32":
        return "fp32"
    if requested == "fp16":
        if device.type == "cuda":
            return "fp16"
        log.warning("fp16_unsupported_on_device", device=str(device), fallback="bf16")
        return "bf16" if bf16_supported(device) else "fp32"
    if requested == "bf16":
        if bf16_supported(device):
            return "bf16"
        log.warning("bf16_unsupported_on_device", device=str(device), fallback="fp32")
        return "fp32"
    return "fp32"


def torch_dtype(precision: Precision) -> torch.dtype:
    """The ``torch.dtype`` matching a precision label."""
    return _DTYPES[precision]


@contextlib.contextmanager
def autocast_context(device: torch.device, precision: Precision) -> Iterator[None]:
    """Enter ``torch.autocast`` when the precision requires it.

    fp32 yields a no-op context so callers never need to branch. Note that
    autocast is deliberately *not* combined with ``model.half()``: keeping master
    weights in fp32 and letting autocast cast per-op is both more accurate and
    what the CUDA kernels are tuned for.
    """
    if precision == "fp32":
        yield
        return
    dtype = torch_dtype(precision)
    if device.type not in {"cuda", "cpu"}:
        # MPS autocast support is incomplete; run fp32 there.
        yield
        return
    with torch.autocast(device_type=device.type, dtype=dtype):
        yield


def configure_threads(num_threads: int) -> None:
    """Pin PyTorch's intra-op thread count (0 keeps the default).

    Benchmarks set this explicitly so that CPU latency numbers are comparable
    between runs on the same machine.
    """
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)


def synchronize(device: torch.device) -> None:
    """Block until queued work on ``device`` has completed.

    CUDA kernel launches are asynchronous, so timing code that does not
    synchronize measures launch overhead rather than execution.
    """
    if device.type == "cuda" and cuda_available():
        torch.cuda.synchronize(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    """Peak allocated VRAM since the last reset, or ``None`` off-CUDA."""
    if device.type == "cuda" and cuda_available():
        return int(torch.cuda.max_memory_allocated(device))
    return None


def reset_peak_memory(device: torch.device) -> None:
    """Reset the CUDA peak-memory counter (no-op elsewhere)."""
    if device.type == "cuda" and cuda_available():
        torch.cuda.reset_peak_memory_stats(device)
