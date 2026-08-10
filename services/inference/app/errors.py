"""Failure classification.

The single most useful thing a job queue can do is tell *retryable* apart from
*non-retryable*, because getting it wrong is expensive in both directions:

* Retrying a non-retryable failure burns a worker slot N times to produce the same
  error. A corrupt upload does not become valid on the third attempt, and a queue full
  of poison messages starves real work.
* Failing a retryable failure permanently turns a two-second Postgres restart into a
  batch of user-visible errors that nobody needs to see.

So classification is explicit and testable rather than "retry everything three times".
Three outcomes:

``RETRYABLE``
    Transient infrastructure: a dropped connection, a storage 503, a lock timeout.
    Retried with exponential backoff and jitter.
``OOM``
    Out of device memory. Retryable, but only if the *inputs shrink* - retrying the
    same batch size against the same GPU reproduces it exactly. The handler halves the
    batch size, records that it did so on the run row, and tries again.
``NON_RETRYABLE``
    The request itself is wrong: corrupt file, unsupported codec, unknown model,
    missing weights, a validation error. Fail immediately with a specific code.

CUDA OOM detection deliberately checks both the exception type and the message.
``torch.cuda.OutOfMemoryError`` exists in modern PyTorch, but OOM also surfaces as a
plain ``RuntimeError`` from cuDNN/cuBLAS paths, from ``pin_memory``, and as a
``MemoryError`` when it is the *host* that ran out.
"""

from __future__ import annotations

import dataclasses
import enum
import random
import re


class FailureKind(enum.StrEnum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    OOM = "oom"


@dataclasses.dataclass(frozen=True, slots=True)
class Classification:
    """How to react to one exception."""

    kind: FailureKind
    code: str
    message: str

    @property
    def retryable(self) -> bool:
        return self.kind in {FailureKind.RETRYABLE, FailureKind.OOM}

    @property
    def is_oom(self) -> bool:
        return self.kind is FailureKind.OOM

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


_OOM_PATTERNS = re.compile(
    r"(cuda out of memory|out of memory|cudnn_status_alloc_failed|cublas_status_alloc_failed"
    r"|failed to allocate|cannot allocate memory|no space left on device.*shm)",
    re.IGNORECASE,
)

_TRANSIENT_PATTERNS = re.compile(
    r"(connection reset|connection refused|connection aborted|broken pipe|timed out|timeout"
    r"|temporarily unavailable|server closed the connection|deadlock detected"
    r"|could not serialize access|too many connections|503|slowdown|throttl)",
    re.IGNORECASE,
)


def classify(exc: BaseException) -> Classification:
    """Map an exception onto a :class:`Classification`.

    Type checks come first and pattern matching second: a type is a contract, whereas a
    message is a string that upstream can reword in a patch release. Pattern matching is
    only the fallback for the libraries that signal with bare ``RuntimeError``.
    """
    name = type(exc).__name__
    text = str(exc)

    # ------------------------------------------------------------------ OOM first
    # Checked before the generic RuntimeError branch, since CUDA OOM *is* a
    # RuntimeError subclass.
    if name in {"OutOfMemoryError", "CudaOutOfMemoryError"} or isinstance(exc, MemoryError):
        return Classification(FailureKind.OOM, "out_of_memory", text or name)
    if _OOM_PATTERNS.search(text):
        return Classification(FailureKind.OOM, "out_of_memory", text)

    # ---------------------------------------------------- definitely the request
    from cutoutml.models.base import WeightsUnavailableError
    from cutoutml.models.registry import ModelNotFoundError
    from cutoutml.pipelines.ffmpeg import UnsupportedCodecError
    from cutoutml.storage.base import ObjectNotFoundError

    if isinstance(exc, UnsupportedCodecError):
        return Classification(FailureKind.NON_RETRYABLE, "unsupported_codec", text)
    if isinstance(exc, ModelNotFoundError):
        return Classification(FailureKind.NON_RETRYABLE, "unknown_model", text)
    if isinstance(exc, WeightsUnavailableError):
        return Classification(FailureKind.NON_RETRYABLE, "weights_unavailable", text)
    if isinstance(exc, ObjectNotFoundError):
        # The asset row exists but its bytes do not. Either the upload never completed
        # or storage lost it; neither is fixed by trying again.
        return Classification(FailureKind.NON_RETRYABLE, "object_missing", text)
    if isinstance(exc, FileNotFoundError):
        return Classification(FailureKind.NON_RETRYABLE, "file_missing", text)
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return Classification(FailureKind.NON_RETRYABLE, "invalid_request", f"{name}: {text}")

    # ------------------------------------------------------- transient by default
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return Classification(FailureKind.RETRYABLE, "connection_error", f"{name}: {text}")
    if name in {"OperationalError", "InterfaceError", "DBAPIError", "RedisError", "ConnectionError"}:
        return Classification(FailureKind.RETRYABLE, "infrastructure_error", f"{name}: {text}")
    if _TRANSIENT_PATTERNS.search(text):
        return Classification(FailureKind.RETRYABLE, "transient_error", f"{name}: {text}")
    if isinstance(exc, OSError):
        return Classification(FailureKind.RETRYABLE, "io_error", f"{name}: {text}")

    # Unknown exceptions are treated as non-retryable on purpose. An unrecognised bug
    # retried three times is three times the log noise and three times the latency for
    # the same failure; the classification table is the place to fix it once diagnosed.
    return Classification(FailureKind.NON_RETRYABLE, "internal_error", f"{name}: {text}")


def retry_delay(attempt: int, *, base: float = 2.0, cap: float = 120.0, jitter: float = 0.25) -> float:
    """Exponential backoff with jitter, in seconds.

    Jitter is not decoration: without it, N tasks that failed together because one
    dependency blipped all retry at the same instant and knock it over again. This is
    "full jitter"'s milder cousin - deterministic growth, +/-25% spread.
    """
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    spread = delay * jitter
    return max(0.1, delay + random.uniform(-spread, spread))


__all__ = [
    "Classification",
    "FailureKind",
    "classify",
    "retry_delay",
]
