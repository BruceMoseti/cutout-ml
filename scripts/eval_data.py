#!/usr/bin/env python3
"""Write, verify and materialise the committed evaluation-set manifest.

The eval set is procedurally generated, so the repository commits a *manifest*
(``datasets/synthetic-eval.json``) instead of thousands of images. This script is the
other half of that bargain:

    scripts/eval_data.py --write            # regenerate the manifest (bump the generator
                                            #   version first; this changes the eval set)
    scripts/eval_data.py --verify           # regenerate and compare the fingerprint
    scripts/eval_data.py --dump artifacts/  # materialise PNG image/mask pairs to look at

``--verify`` is the interesting one and it runs in CI. It regenerates the first N
samples and compares a SHA-256 over their bytes against the committed fingerprint. A
mismatch means the pixels changed - a different OpenCV resampling default, a NumPy RNG
change, an accidental edit to the generator - and therefore that every accuracy number
in ``docs/benchmarks.md`` was measured on a different dataset than the one this
checkout produces. That is worth failing a build over, because the alternative is an
accuracy column that silently drifts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cutoutml.core.imaging import encode_image, encode_mask  # noqa: E402
from cutoutml.datasets.manifest import DatasetManifest, fingerprint_samples  # noqa: E402
from cutoutml.datasets.synthetic import (  # noqa: E402
    DEFAULT_SEED,
    SyntheticConfig,
    SyntheticSegmentationDataset,
)

MANIFEST_PATH = REPO_ROOT / "datasets" / "synthetic-eval.json"

#: Must match ``BenchmarkConfig`` so that the manifest describes the set the committed
#: benchmark numbers were actually measured on.
EVAL_SPLIT = "test"
EVAL_COUNT = 64
EVAL_RESOLUTION = (256, 256)
FINGERPRINT_SAMPLES = 8

SPLIT_COUNTS = {"train": 2048, "val": 192, "test": EVAL_COUNT}


def build_dataset(resolution: tuple[int, int] = EVAL_RESOLUTION) -> SyntheticSegmentationDataset:
    return SyntheticSegmentationDataset(
        count=EVAL_COUNT,
        split=EVAL_SPLIT,
        seed=DEFAULT_SEED,
        config=SyntheticConfig(resolution=resolution),
    )


def cmd_write(path: Path) -> int:
    manifest = build_dataset().manifest(SPLIT_COUNTS, fingerprint_n=FINGERPRINT_SAMPLES)
    manifest.save(path)
    print(f"wrote {path}")
    print(f"  dataset_id  {manifest.dataset_id}")
    print(f"  fingerprint {manifest.fingerprint}")
    return 0


def cmd_verify(path: Path) -> int:
    if not path.is_file():
        print(f"no manifest at {path}; run --write first", file=sys.stderr)
        return 2

    committed = DatasetManifest.load(path)
    dataset = SyntheticSegmentationDataset(
        count=max(FINGERPRINT_SAMPLES, committed.fingerprint_samples),
        split=EVAL_SPLIT,
        seed=committed.master_seed,
        config=SyntheticConfig.from_dict(committed.config),
    )
    samples = [dataset.sample(i) for i in range(committed.fingerprint_samples)]
    actual = fingerprint_samples(samples)

    if actual != committed.fingerprint:
        print(
            "eval-set fingerprint MISMATCH\n"
            f"  committed: {committed.fingerprint}\n"
            f"  regenerated: {actual}\n"
            "The generator no longer produces the dataset the committed benchmark\n"
            "numbers were measured on. Either revert the generator change, or bump\n"
            "GENERATOR_VERSION, re-run --write and re-run the benchmarks.",
            file=sys.stderr,
        )
        return 1

    print(f"eval set matches the committed manifest ({committed.dataset_id})")
    print(f"  fingerprint {actual} over {committed.fingerprint_samples} samples")
    return 0


def cmd_dump(path: Path, out_dir: Path, limit: int) -> int:
    manifest = DatasetManifest.load(path) if path.is_file() else None
    config = SyntheticConfig.from_dict(manifest.config) if manifest else SyntheticConfig()
    dataset = SyntheticSegmentationDataset(
        count=limit,
        split=EVAL_SPLIT,
        seed=manifest.master_seed if manifest else DEFAULT_SEED,
        config=config,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(limit):
        image, alpha = dataset.sample(i)
        (out_dir / f"{i:04d}_image.png").write_bytes(encode_image(image, fmt="png"))
        (out_dir / f"{i:04d}_alpha.png").write_bytes(encode_mask(alpha))
    print(f"wrote {limit * 2} files to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="regenerate the manifest")
    mode.add_argument("--verify", action="store_true", help="check the committed fingerprint")
    mode.add_argument("--dump", type=Path, metavar="DIR", help="materialise image/mask PNGs")
    p.add_argument("--dump-limit", type=int, default=16)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write:
        return cmd_write(args.manifest)
    if args.dump:
        return cmd_dump(args.manifest, args.dump, args.dump_limit)
    return cmd_verify(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
