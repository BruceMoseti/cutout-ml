#!/usr/bin/env python3
"""Measure how much of the eval set's difficulty comes from *not* centring the object.

    python benchmarks/center_prior.py                       # the committed experiment
    python benchmarks/center_prior.py --fractions 0 0.27    # two rungs only
    python benchmarks/center_prior.py --samples 128         # a wider set

Why this exists
---------------
``SyntheticConfig.translation_frac`` decides how far an object may sit from the centre of
the frame, and the shipped eval set uses 0.27 - objects move by up to +-27% of the frame.
That choice is the difference between a benchmark that measures segmentation and one that
measures a centre prior: if every object sits in the middle, a fixed centred ellipse that
never looks at the image scores well, and so does GrabCut seeded from a centred rectangle.

ADR-004 asserted the effect with two numbers that no committed artefact produced. This
script measures it instead. Only accuracy is computed, never latency, which is why it is
safe to run on a busy machine: IoU is a function of the pixels and the model, not of what
else the CPU is doing.

The baselines are chosen to have no learned content at all:

- ``trivial-center`` is a fixed centred ellipse. It is pure centre prior.
- ``trivial-ones`` predicts foreground everywhere. It moves only because tightening the
  translation range changes which samples survive coverage rejection, so it doubles as a
  control on that: without it, a shift in the other rows could be a change in the set
  rather than a change in difficulty.
- ``classical`` is GrabCut seeded from a centred rectangle - an algorithm, but one whose
  initialisation is a centre prior.

Refinement is off for every rung. The suite's ``classical`` row is measured with
refinement on and therefore reads slightly higher there; the comparison that matters here
is between rungs of this sweep, so the setting is held fixed rather than matched to the
suite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow running without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cutoutml.benchmarks.environment import capture  # noqa: E402
from cutoutml.core.logging import configure_logging, get_logger  # noqa: E402
from cutoutml.core.metrics import aggregate, compute_all  # noqa: E402
from cutoutml.datasets.synthetic import (  # noqa: E402
    DEFAULT_SEED,
    SyntheticConfig,
    SyntheticSegmentationDataset,
)
from cutoutml.models.registry import get_model  # noqa: E402

log = get_logger("benchmarks.center_prior")

#: Models with no learned content, whose score is a property of the set rather than of
#: training. See the module docstring for why each one is here.
BASELINES: tuple[str, ...] = ("trivial-ones", "trivial-center", "classical")

#: Translation ranges to measure, as a fraction of the frame. 0.27 is what the shipped
#: eval set uses and is the reference rung; 0.0 is the pathological case where every
#: object is centred.
FRACTIONS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.135, 0.2, 0.27)

#: The rung the shipped `datasets/synthetic-eval.json` was generated with.
SHIPPED_FRACTION = 0.27


def measure_rung(fraction: float, samples: int, resolution: int, seed: int) -> dict[str, Any]:
    """Generate the eval split at ``fraction`` and score every baseline on it."""
    config = SyntheticConfig(resolution=(resolution, resolution), translation_frac=fraction)
    dataset = SyntheticSegmentationDataset(count=samples, split="test", seed=seed, config=config)
    pairs = [dataset[i] for i in range(samples)]
    images = [image for image, _ in pairs]

    scores: dict[str, dict[str, float]] = {}
    for name in BASELINES:
        alphas = get_model(name).infer(images)
        metrics = [
            compute_all(alpha, truth) for (_, truth), alpha in zip(pairs, alphas, strict=True)
        ]
        agg = aggregate(metrics)
        scores[name] = {"iou": round(float(agg["iou"]), 4), "mae": round(float(agg["mae"]), 4)}
        log.warning("rung_measured", fraction=fraction, model=name, iou=scores[name]["iou"])

    return {
        "translation_frac": fraction,
        "foreground_coverage": round(float(np.mean([truth.mean() for _, truth in pairs])), 4),
        "scores": scores,
    }


def table(rungs: list[dict[str, Any]]) -> str:
    """Markdown table, with the shipped rung marked."""
    header = (
        "| Translation range | " + " | ".join(f"`{n}` IoU" for n in BASELINES) + " | Coverage |"
    )
    rows = [header, "|---" * (len(BASELINES) + 2) + "|"]
    for rung in rungs:
        frac = rung["translation_frac"]
        label = f"+-{frac * 100:.1f}%" + (" (shipped)" if frac == SHIPPED_FRACTION else "")
        cells = " | ".join(f"{rung['scores'][n]['iou']:.4f}" for n in BASELINES)
        rows.append(f"| {label} | {cells} | {rung['foreground_coverage'] * 100:.1f}% |")
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    p.add_argument(
        "--samples",
        type=int,
        default=64,
        help="eval samples per rung; 64 is the shipped test split's size",
    )
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--log-format", default="console", choices=["console", "json"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level="WARNING", fmt=args.log_format)

    started = time.perf_counter()
    rungs = [
        measure_rung(fraction, args.samples, args.resolution, args.seed)
        for fraction in sorted(args.fractions)
    ]

    shipped = next((r for r in rungs if r["translation_frac"] == SHIPPED_FRACTION), None)
    centred = next((r for r in rungs if r["translation_frac"] == 0.0), None)
    lift = (
        {
            name: round(
                centred["scores"][name]["iou"] - shipped["scores"][name]["iou"],
                4,
            )
            for name in BASELINES
        }
        if shipped and centred
        else None
    )

    report = {
        "experiment": "center_prior",
        "schema_version": 1,
        "run_id": f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "question": (
            "How much of each learning-free baseline's IoU comes from objects being near "
            "the centre of the frame?"
        ),
        "method": {
            "samples_per_rung": args.samples,
            "resolution": [args.resolution, args.resolution],
            "split": "test",
            "master_seed": args.seed,
            "baselines": list(BASELINES),
            "refinement": "off",
            "shipped_translation_frac": SHIPPED_FRACTION,
            "metrics_are_load_independent": True,
        },
        "environment": capture().as_dict(),
        "rungs": rungs,
        "iou_lift_from_centring_everything": lift,
    }

    target_dir = args.output_dir or (REPO_ROOT / "benchmarks" / "results" / "experiments")
    target_dir.mkdir(parents=True, exist_ok=True)
    # Not in benchmarks/results/ itself: the renderer treats every *.json directly in that
    # directory as a suite report and would pick this up as the latest.
    path = target_dir / f"center-prior-{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(table(rungs))
    if lift:
        print()
        for name, delta in lift.items():
            print(f"centring everything moves {name} by {delta:+.4f} IoU")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
