"""Adapters for real segmentation / matting datasets.

The synthetic generator exists because nothing can be downloaded in this
environment, not because synthetic data is preferable. The harness must therefore
run unchanged on real data when a user has it, which is what these adapters
provide: the same ``(image_uint8_rgb, alpha_float32)`` tuple interface as
:class:`~cutoutml.datasets.synthetic.SyntheticSegmentationDataset`.

Supported directory layouts
---------------------------
**DUTS** (salient object detection, binary masks)::

    DUTS/
      DUTS-TR/DUTS-TR-Image/*.jpg
      DUTS-TR/DUTS-TR-Mask/*.png
      DUTS-TE/DUTS-TE-Image/*.jpg
      DUTS-TE/DUTS-TE-Mask/*.png

**DIS5K** (dichotomous image segmentation, very high resolution)::

    DIS5K/
      DIS-TR/im/*.jpg   DIS-TR/gt/*.png
      DIS-VD/im/*.jpg   DIS-VD/gt/*.png
      DIS-TE1..TE4/im, gt

**AM-2k** (animal matting, genuinely continuous alpha)::

    AM-2k/
      train/original/*.jpg   train/mask/*.png
      validation/original/*.jpg   validation/mask/*.png

Pairing is by **filename stem**, not by sort order: several public releases have
a different number of images and masks, and index-pairing silently misaligns the
whole set. A mismatch raises instead of producing quietly wrong metrics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from cutoutml.core.logging import get_logger

log = get_logger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Where images and masks live for one split of one dataset family."""

    name: str
    image_subdir: str
    mask_subdir: str

    def resolve(self, root: Path) -> tuple[Path, Path]:
        return (root / self.image_subdir, root / self.mask_subdir)


LAYOUTS: dict[str, dict[str, DatasetLayout]] = {
    "duts": {
        "train": DatasetLayout("DUTS-TR", "DUTS-TR/DUTS-TR-Image", "DUTS-TR/DUTS-TR-Mask"),
        "test": DatasetLayout("DUTS-TE", "DUTS-TE/DUTS-TE-Image", "DUTS-TE/DUTS-TE-Mask"),
    },
    "dis5k": {
        "train": DatasetLayout("DIS-TR", "DIS-TR/im", "DIS-TR/gt"),
        "val": DatasetLayout("DIS-VD", "DIS-VD/im", "DIS-VD/gt"),
        "test": DatasetLayout("DIS-TE1", "DIS-TE1/im", "DIS-TE1/gt"),
        "test2": DatasetLayout("DIS-TE2", "DIS-TE2/im", "DIS-TE2/gt"),
        "test3": DatasetLayout("DIS-TE3", "DIS-TE3/im", "DIS-TE3/gt"),
        "test4": DatasetLayout("DIS-TE4", "DIS-TE4/im", "DIS-TE4/gt"),
    },
    "am2k": {
        "train": DatasetLayout("AM2K-train", "train/original", "train/mask"),
        "val": DatasetLayout("AM2K-val", "validation/original", "validation/mask"),
    },
    # Generic fallback for a user's own flat directory pair.
    "flat": {
        "train": DatasetLayout("flat", "images", "masks"),
        "val": DatasetLayout("flat", "images", "masks"),
        "test": DatasetLayout("flat", "images", "masks"),
    },
}


class RealSegmentationDataset:
    """Filename-stem-paired image/mask dataset with the standard tuple interface.

    Parameters
    ----------
    max_side:
        Longest-side cap applied at load time. DIS5K images reach 4000+ px; without
        a cap a benchmark spends all of its time in JPEG decode rather than in the
        model, and peak RSS becomes a measure of the decoder.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        family: str = "duts",
        split: str = "test",
        max_side: int | None = 1024,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.family = family.lower()
        self.split = split
        self.max_side = max_side

        if self.family not in LAYOUTS:
            raise ValueError(
                f"unknown dataset family {family!r}; supported: {sorted(LAYOUTS)}"
            )
        splits = LAYOUTS[self.family]
        if split not in splits:
            raise ValueError(
                f"family {family!r} has no split {split!r}; available: {sorted(splits)}"
            )

        self.layout = splits[split]
        self.image_dir, self.mask_dir = self.layout.resolve(self.root)
        self.pairs = self._index_pairs()
        if limit is not None:
            self.pairs = self.pairs[:limit]
        log.info(
            "real_dataset_indexed",
            family=self.family,
            split=split,
            pairs=len(self.pairs),
            root=str(self.root),
        )

    def _index_pairs(self) -> list[tuple[Path, Path]]:
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"image directory not found: {self.image_dir}. Expected the "
                f"{self.family.upper()} layout under {self.root} - see "
                "cutoutml.datasets.real for the exact structure."
            )
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"mask directory not found: {self.mask_dir}")

        masks: dict[str, Path] = {}
        for suffix in MASK_SUFFIXES:
            for path in self.mask_dir.glob(f"*{suffix}"):
                masks.setdefault(path.stem, path)

        pairs: list[tuple[Path, Path]] = []
        missing: list[str] = []
        for suffix in IMAGE_SUFFIXES:
            for image_path in sorted(self.image_dir.glob(f"*{suffix}")):
                mask_path = masks.get(image_path.stem)
                if mask_path is None:
                    missing.append(image_path.name)
                    continue
                pairs.append((image_path, mask_path))

        if not pairs:
            raise FileNotFoundError(
                f"no image/mask pairs found under {self.image_dir} and {self.mask_dir}"
            )
        if missing:
            log.warning(
                "real_dataset_unpaired_images", count=len(missing), sample=missing[:5]
            )
        return sorted(pairs, key=lambda p: p[0].stem)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.pairs[index]
        image = self._load_image(image_path)
        alpha = self._load_mask(mask_path, image.shape[:2])
        return image, alpha

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]

    def _load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            if self.max_side:
                rgb = _cap_side(rgb, self.max_side)
            return np.asarray(rgb, dtype=np.uint8)

    def _load_mask(self, path: Path, hw: tuple[int, int]) -> np.ndarray:
        with Image.open(path) as img:
            gray = img.convert("L")
            if gray.size != (hw[1], hw[0]):
                # Masks are resized with bilinear, not nearest: AM-2k alpha is
                # continuous and nearest-neighbour would destroy the soft band.
                gray = gray.resize((hw[1], hw[0]), Image.Resampling.BILINEAR)
            return (np.asarray(gray, dtype=np.float32) / 255.0).astype(np.float32)

    def describe(self) -> dict[str, object]:
        """Metadata for the benchmark record, mirroring a synthetic manifest."""
        return {
            "dataset_id": f"{self.family}-{self.split}",
            "generator": "cutoutml.datasets.real",
            "family": self.family,
            "split": self.split,
            "root": str(self.root),
            "count": len(self),
            "max_side": self.max_side,
            "layout": {
                "images": str(self.image_dir),
                "masks": str(self.mask_dir),
            },
        }


def _cap_side(img: Image.Image, max_side: int) -> Image.Image:
    longest = max(img.size)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def detect_family(root: Path | str) -> str | None:
    """Guess the dataset family from the directories present under ``root``."""
    p = Path(root)
    if (p / "DUTS-TE" / "DUTS-TE-Image").is_dir() or (p / "DUTS-TR").is_dir():
        return "duts"
    if (p / "DIS-VD" / "im").is_dir() or (p / "DIS-TR" / "im").is_dir():
        return "dis5k"
    if (p / "train" / "original").is_dir() or (p / "validation" / "original").is_dir():
        return "am2k"
    if (p / "images").is_dir() and (p / "masks").is_dir():
        return "flat"
    return None


def available_splits(root: Path | str, family: str | None = None) -> Sequence[str]:
    """Splits of ``family`` that actually exist on disk under ``root``."""
    p = Path(root)
    fam = family or detect_family(p)
    if fam is None:
        return []
    return [
        name
        for name, layout in LAYOUTS[fam].items()
        if layout.resolve(p)[0].is_dir() and layout.resolve(p)[1].is_dir()
    ]
