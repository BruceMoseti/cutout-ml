"""The synthetic (image, alpha) dataset.

Composition per sample
----------------------
1. Pick a background maker and render it.
2. Pick 1-2 shape makers, shade them, and composite with **fractional alpha** so
   boundary pixels really are partially transparent.
3. Optionally add a *distractor*: a second shaded shape composited into the
   background and **excluded from the ground truth**. This is the single most
   important element of the design - without distractors a model can score well by
   segmenting "the thing that isn't the backdrop", and saliency baselines look
   artificially strong.
4. Global augmentation: brightness/contrast, blur, JPEG re-encode, noise. Blur is
   applied to the *composite and the alpha together* so the pair stays consistent -
   a blurred photograph genuinely has a blurred alpha.

Determinism
-----------
``sample(index)`` is a pure function of ``(master_seed, split_offset, index)``.
Two processes, two machines and two epochs all produce identical bytes, which is
what lets the manifest fingerprint mean something.
"""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np
from PIL import Image

from cutoutml.datasets.manifest import (
    GENERATOR_VERSION,
    DatasetManifest,
    SplitSpec,
    fingerprint_samples,
)
from cutoutml.datasets.procedural import (
    BACKGROUND_MAKERS,
    SHAPE_MAKERS,
    shade_foreground,
)

DEFAULT_SEED = 20240817


@dataclasses.dataclass(slots=True)
class SyntheticConfig:
    """Generation parameters. Serialised verbatim into the manifest."""

    resolution: tuple[int, int] = (256, 256)
    shapes: tuple[str, ...] = ("blob", "superellipse", "star", "rounded_rect", "glyph", "layered")
    backgrounds: tuple[str, ...] = tuple(BACKGROUND_MAKERS)
    max_objects: int = 2
    distractor_prob: float = 0.45
    max_distractors: int = 2
    soft_edge_prob: float = 0.5
    soft_edge_sigma: tuple[float, float] = (0.6, 2.4)
    blur_prob: float = 0.35
    blur_sigma: tuple[float, float] = (0.4, 1.8)
    jpeg_prob: float = 0.35
    jpeg_quality: tuple[int, int] = (35, 92)
    noise_prob: float = 0.4
    noise_sigma: tuple[float, float] = (2.0, 12.0)
    brightness_jitter: float = 0.22
    contrast_jitter: float = 0.25
    rotation_degrees: float = 30.0
    scale_range: tuple[float, float] = (0.62, 1.18)
    # Objects are translated by up to +-27% of the frame. Deliberately large: centring
    # every object instead lifts a fixed centred ellipse from 0.4382 to 0.5948 IoU and
    # GrabCut seeded from a centred rectangle from 0.6493 to 0.8306, so a tighter range
    # would mostly measure how well a model has learned a centre prior. Measured by
    # benchmarks/center_prior.py; see benchmarks/results/experiments/center-prior-*.json.
    translation_frac: float = 0.27
    color_collision_prob: float = 0.3
    min_alpha_coverage: float = 0.03
    max_alpha_coverage: float = 0.75

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for key, value in list(d.items()):
            if isinstance(value, tuple):
                d[key] = list(value)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyntheticConfig:
        fields = {f.name: f for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in fields:
                continue
            kwargs[key] = tuple(value) if isinstance(value, list) else value
        return cls(**kwargs)


def _rng_for(master_seed: int, split_offset: int, index: int) -> np.random.Generator:
    """A generator whose stream depends only on the three integers.

    ``SeedSequence`` with a spawn key is used rather than ``seed + index`` so that
    adjacent indices are not correlated: naive additive seeding makes sample 5 and
    sample 6 share large parts of their stream on some bit generators.
    """
    seq = np.random.SeedSequence(entropy=master_seed, spawn_key=(split_offset, index))
    return np.random.default_rng(seq)


class SyntheticSegmentationDataset:
    """Deterministic procedural dataset of ``(image_uint8_rgb, alpha_float32)``.

    Implements ``__len__``/``__getitem__`` so it works as a ``torch.utils.data``
    dataset without importing torch here (keeping the module usable from the
    benchmark harness with no training dependencies).
    """

    def __init__(
        self,
        *,
        count: int,
        split: str = "train",
        seed: int = DEFAULT_SEED,
        seed_offset: int | None = None,
        config: SyntheticConfig | None = None,
    ) -> None:
        self.count = count
        self.split = split
        self.seed = seed
        self.config = config or SyntheticConfig()
        self.seed_offset = seed_offset if seed_offset is not None else _split_offset(split)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= index < self.count:
            raise IndexError(f"index {index} out of range for {self.count} samples")
        return self.sample(index)

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample ``index`` deterministically."""
        rng = _rng_for(self.seed, self.seed_offset, index)
        return generate_sample(rng, self.config)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for i in range(self.count):
            yield self.sample(i)

    def manifest(
        self, splits: dict[str, int] | None = None, fingerprint_n: int = 8
    ) -> DatasetManifest:
        """Build a manifest describing this dataset (and sibling splits)."""
        split_counts = splits or {self.split: self.count}
        specs = [
            SplitSpec(name=name, count=count, seed_offset=_split_offset(name))
            for name, count in sorted(split_counts.items())
        ]
        samples = [self.sample(i) for i in range(min(fingerprint_n, self.count))]
        return DatasetManifest(
            dataset_id=f"synthetic-v{GENERATOR_VERSION}-seed{self.seed}",
            generator="cutoutml.datasets.synthetic",
            generator_version=GENERATOR_VERSION,
            master_seed=self.seed,
            resolution=self.config.resolution,
            splits=specs,
            config=self.config.as_dict(),
            fingerprint=fingerprint_samples(samples) if samples else None,
            fingerprint_samples=len(samples),
            notes=(
                "Procedurally generated. Accuracy measured on this set is NOT "
                "comparable to published DUTS/DIS5K numbers; use "
                "cutoutml.datasets.real for those."
            ),
        )


_SPLIT_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000, "bench": 3_000_000}


def _split_offset(split: str) -> int:
    """Disjoint seed ranges per split, so train and test never overlap."""
    if split in _SPLIT_OFFSETS:
        return _SPLIT_OFFSETS[split]
    # Stable hash for user-defined split names.
    return 4_000_000 + (abs(hash(split)) % 1_000_000)


def build_splits(
    counts: dict[str, int],
    *,
    seed: int = DEFAULT_SEED,
    config: SyntheticConfig | None = None,
) -> dict[str, SyntheticSegmentationDataset]:
    """Create several splits sharing one seed and config."""
    cfg = config or SyntheticConfig()
    return {
        name: SyntheticSegmentationDataset(count=count, split=name, seed=seed, config=cfg)
        for name, count in counts.items()
    }


# --------------------------------------------------------------------- sampling


def generate_sample(
    rng: np.random.Generator, config: SyntheticConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one ``(image, alpha)`` pair from an explicit generator.

    Retries up to 6 times if the sampled foreground coverage is degenerate (nearly
    empty or nearly full) - both make per-image IoU statistically useless.
    """
    cfg = config or SyntheticConfig()
    h, w = cfg.resolution[1], cfg.resolution[0]

    for _ in range(6):
        image, alpha = _compose(rng, cfg, (h, w))
        coverage = float((alpha > 0.5).mean())
        if cfg.min_alpha_coverage <= coverage <= cfg.max_alpha_coverage:
            return image, alpha
    return image, alpha


def _compose(
    rng: np.random.Generator, cfg: SyntheticConfig, hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = hw
    bg_name = str(rng.choice(list(cfg.backgrounds)))
    background = BACKGROUND_MAKERS[bg_name](rng, (h, w))

    # Distractors go in first so the real foreground can occlude them.
    canvas = background.copy()
    if rng.random() < cfg.distractor_prob:
        for _ in range(int(rng.integers(1, cfg.max_distractors + 1))):
            d_mask = _make_shape(rng, cfg, (h, w), scale_bias=0.7)
            d_rgb = shade_foreground(rng, d_mask)
            canvas = _blend(canvas, d_rgb, d_mask * float(rng.uniform(0.7, 1.0)))

    # Foreground: 1..max_objects shapes, unioned into the ground-truth alpha.
    alpha = np.zeros((h, w), dtype=np.float32)
    n_objects = int(rng.integers(1, cfg.max_objects + 1))
    for _ in range(n_objects):
        mask = _make_shape(rng, cfg, (h, w))
        rgb = shade_foreground(rng, mask)
        if rng.random() < cfg.color_collision_prob:
            # Deliberately pull the foreground toward the background's mean colour
            # so colour alone cannot separate them.
            bg_mean = background.reshape(-1, 3).mean(axis=0)
            blend = float(rng.uniform(0.35, 0.7))
            rgb = rgb * (1 - blend) + bg_mean[None, None, :] * blend
        canvas = _blend(canvas, rgb, mask)
        alpha = np.maximum(alpha, mask)

    image = np.clip(canvas, 0, 255)
    return _augment(rng, cfg, image, alpha)


def _make_shape(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    hw: tuple[int, int],
    *,
    scale_bias: float = 1.0,
) -> np.ndarray:
    """Sample a shape mask, then apply affine jitter and optional edge softening."""
    h, w = hw
    name = str(rng.choice(list(cfg.shapes)))
    mask = SHAPE_MAKERS[name](rng, (h, w))

    scale = float(rng.uniform(*cfg.scale_range)) * scale_bias
    angle = float(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees))
    tx = float(rng.uniform(-cfg.translation_frac, cfg.translation_frac)) * w
    ty = float(rng.uniform(-cfg.translation_frac, cfg.translation_frac)) * h
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)

    if rng.random() < cfg.soft_edge_prob:
        # Motion blur / shallow depth of field: a genuinely soft matte, not just
        # an anti-aliased one. This is what makes MAE a more informative metric
        # than IoU on this dataset.
        sigma = float(rng.uniform(*cfg.soft_edge_sigma))
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _blend(background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    return background * (1.0 - a) + foreground * a


def _augment(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    image: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Photometric + lens + codec augmentation.

    Order matters and mirrors a real capture pipeline: optical blur happens before
    the sensor, noise at the sensor, tone mapping in the ISP, JPEG last. Applying
    JPEG before noise would let the noise hide the very artefacts we are trying to
    simulate.
    """
    img = image.astype(np.float32)

    if rng.random() < cfg.blur_prob:
        sigma = float(rng.uniform(*cfg.blur_sigma))
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        # The matte must be blurred by the same optics, or the label is wrong.
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)

    if rng.random() < cfg.noise_prob:
        sigma = float(rng.uniform(*cfg.noise_sigma))
        img = img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)

    brightness = 1.0 + float(rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter))
    contrast = 1.0 + float(rng.uniform(-cfg.contrast_jitter, cfg.contrast_jitter))
    mean = float(img.mean())
    img = (img - mean) * contrast + mean * brightness

    out = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < cfg.jpeg_prob:
        quality = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
        out = _jpeg_roundtrip(out, quality)

    return out, np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Real JPEG encode/decode, so blocking artefacts are authentic.

    Simulating JPEG with additive noise is a common shortcut that produces the
    wrong artefact structure: JPEG error is 8x8 block-correlated and concentrated
    at edges, which is exactly where a matting model is most sensitive.
    """
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)
