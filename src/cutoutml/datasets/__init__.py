"""Dataset generation and loading.

Two interchangeable sources, both yielding ``(uint8 RGB image, float32 alpha)``:

* :class:`~cutoutml.datasets.synthetic.SyntheticSegmentationDataset` - procedural,
  reproducible from a single seed, used for the committed benchmark numbers;
* :class:`~cutoutml.datasets.real.RealSegmentationDataset` - DUTS / DIS5K / AM-2k
  on disk, for when you have the real thing.
"""

from cutoutml.datasets.manifest import DatasetManifest, SplitSpec
from cutoutml.datasets.real import LAYOUTS, RealSegmentationDataset, detect_family
from cutoutml.datasets.synthetic import (
    DEFAULT_SEED,
    SyntheticConfig,
    SyntheticSegmentationDataset,
    build_splits,
    generate_sample,
)

__all__ = [
    "DEFAULT_SEED",
    "LAYOUTS",
    "DatasetManifest",
    "RealSegmentationDataset",
    "SplitSpec",
    "SyntheticConfig",
    "SyntheticSegmentationDataset",
    "build_splits",
    "detect_family",
    "generate_sample",
]
