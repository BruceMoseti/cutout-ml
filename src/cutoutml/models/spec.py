"""What a model *is*, separated from what a model *does*.

A registry entry, the metadata it reports and the digest of the weights behind it are
plain data: dataclasses, a string and a hash. Running the model needs torch, cv2 and an
architecture. Both halves used to live in :mod:`cutoutml.models.base`, which meant that
asking "what models exist and can they run here?" imported a deep-learning framework.

That question is exactly what the API asks, on `GET /v1/models` and on every job
submission, and the API is meant to load no models at all - see ``docs/architecture.md``.
Splitting the declarative half out is what makes that boundary real rather than
aspirational. :mod:`cutoutml.models.base` re-exports everything here, so adapters and
pipelines are unaffected.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from pathlib import Path
from typing import Any


@functools.lru_cache(maxsize=32)
def _digest(path: str, size: int, mtime_ns: int) -> str:  # noqa: ARG001 - cache-key only
    """Hash a file, memoised on its identity rather than its name alone.

    ``size`` and ``mtime_ns`` are part of the key so that retraining a checkpoint in place
    invalidates the entry instead of serving the previous run's digest - which is exactly
    the case where a stale hash would be most misleading. Neither is read in the body.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_digest(path: Path | str | None) -> str | None:
    """SHA-256 of a weights file, or ``None`` if there is nothing to hash.

    Recorded in every benchmark row. Without it a published accuracy figure names a
    checkpoint path, and a path is not evidence: the file behind it changes with each
    training run, so a reader cannot tell whether the number in the docs came from the
    weights currently on disk.
    """
    if path is None:
        return None
    file = Path(path)
    try:
        stat = file.stat()
    except OSError:
        return None
    if not file.is_file():
        return None
    return _digest(str(file.resolve()), stat.st_size, stat.st_mtime_ns)


class WeightsUnavailableError(RuntimeError):
    """Raised when a model needs pretrained weights that are not on disk.

    The message is deliberately actionable: it names the expected path and the
    command that would fetch it. Several architectures in this repo can only get
    real weights from HuggingFace, which is not always reachable, so this error
    is a normal operating condition rather than a bug.
    """

    def __init__(self, model: str, expected_path: Path | str, hint: str = "") -> None:
        self.model = model
        self.expected_path = Path(expected_path)
        message = (
            f"No weights found for model '{model}'. Expected a checkpoint at {self.expected_path}."
        )
        if hint:
            message += f" {hint}"
        super().__init__(message)


@dataclasses.dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Everything a caller (or a benchmark record) needs to know about a model."""

    name: str
    architecture: str
    param_count: int
    trainable_param_count: int
    input_size: tuple[int, int]
    precision: str
    device: str
    device_name: str
    runtime: str
    license: str
    source: str
    weights_path: str | None
    weights_sha256: str | None
    randomly_initialized: bool
    accuracy_valid: bool
    #: For a checkpoint converted from another artefact, the digest of that artefact.
    #: ``weights_sha256`` names one conversion of it and does not survive re-conversion,
    #: so this is the digest a benchmark row can be checked against later.
    weights_source_sha256: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["input_size"] = list(self.input_size)
        return d


@dataclasses.dataclass(frozen=True, slots=True)
class ModelSpec:
    """Declarative registry entry. See :mod:`cutoutml.models.registry`."""

    name: str
    adapter: str
    architecture: str
    input_size: tuple[int, int]
    license: str
    source: str
    default_weights: str | None = None
    #: Further paths this spec will load from, tried in order after ``default_weights``.
    #: Exists because one architecture can have several equally legitimate artefacts -
    #: U^2-Net accepts both the checkpoint converted from the redistributed ONNX graph
    #: and the authors' own ``.pth`` - and a model should not report itself unavailable
    #: because the weights present are the second kind.
    alt_weights: tuple[str, ...] = ()
    weights_url: str | None = None
    runtime: str = "pytorch"
    supports_random_init: bool = False
    requires_weights: bool = True
    description: str = ""
    tags: tuple[str, ...] = ()
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["input_size"] = list(self.input_size)
        d["tags"] = list(self.tags)
        d["alt_weights"] = list(self.alt_weights)
        return d
