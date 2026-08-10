"""Dataset manifests.

A benchmark number is only meaningful if the reader can regenerate the exact
inputs. Rather than committing thousands of images, this repository commits a
manifest: the generator version, the master seed, the per-split sample counts and
every generation parameter. ``make eval-data`` replays it byte-for-byte.

The manifest also carries a **content fingerprint** - a hash over the first N
generated samples - so a change in NumPy's or OpenCV's resampling behaviour is
detected loudly instead of silently shifting the accuracy column.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "1.0.0"
"""Bump on any change that alters generated pixels; invalidates fingerprints."""


@dataclasses.dataclass(slots=True)
class SplitSpec:
    """One split of a generated dataset."""

    name: str
    count: int
    seed_offset: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class DatasetManifest:
    """Everything needed to reproduce a synthetic dataset exactly."""

    dataset_id: str
    generator: str
    generator_version: str
    master_seed: int
    resolution: tuple[int, int]
    splits: list[SplitSpec]
    config: dict[str, Any]
    fingerprint: str | None = None
    fingerprint_samples: int = 0
    notes: str = ""

    def split(self, name: str) -> SplitSpec:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(f"no split named {name!r}; have {[s.name for s in self.splits]}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "generator": self.generator,
            "generator_version": self.generator_version,
            "master_seed": self.master_seed,
            "resolution": list(self.resolution),
            "splits": [s.as_dict() for s in self.splits],
            "config": self.config,
            "fingerprint": self.fingerprint,
            "fingerprint_samples": self.fingerprint_samples,
            "notes": self.notes,
        }

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")
        return out

    @classmethod
    def load(cls, path: Path | str) -> DatasetManifest:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        return cls(
            dataset_id=data["dataset_id"],
            generator=data["generator"],
            generator_version=data["generator_version"],
            master_seed=int(data["master_seed"]),
            resolution=(int(data["resolution"][0]), int(data["resolution"][1])),
            splits=[SplitSpec(**s) for s in data["splits"]],
            config=data.get("config", {}),
            fingerprint=data.get("fingerprint"),
            fingerprint_samples=int(data.get("fingerprint_samples", 0)),
            notes=data.get("notes", ""),
        )


def fingerprint_samples(samples: list[tuple[Any, Any]]) -> str:
    """SHA-256 over the raw bytes of ``(image, alpha)`` pairs.

    Alpha is quantised to 8 bits before hashing: float32 alpha differs in the last
    mantissa bit across BLAS builds, which would make the fingerprint useless as a
    portability check.
    """
    digest = hashlib.sha256()
    for image, alpha in samples:
        digest.update(image.tobytes())
        quantised = (alpha * 255.0 + 0.5).astype("uint8")
        digest.update(quantised.tobytes())
    return digest.hexdigest()
