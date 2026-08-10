#!/usr/bin/env python3
"""Measure how much a case's *position in the process* changes its latency.

    python benchmarks/order_effect.py                  # the committed experiment
    python benchmarks/order_effect.py --repetitions 50 # tighter
    python benchmarks/order_effect.py --prelude u2netp # one arm, in this process

Why this exists
---------------
The benchmark suite measures ``cutoutnet`` eager at batch 1 and one intra-op thread
twice: once in the main table and once as the one-thread rung of the thread-scaling
sweep. On a quiet machine the two disagree by about 1.5x - roughly 31 ms against 20 ms -
with a per-case standard deviation under 0.2 ms on both. Neither is noise, and machine
contention does not explain it: the harness samples external CPU demand before every
timing loop and both rows recorded an idle machine.

The variable is what ran *earlier in the same process*. This script isolates it: it times
one fixed configuration after each of several preludes, one prelude per subprocess so
that nothing leaks between arms, and records the load snapshot alongside each figure so
that a contended arm cannot be mistaken for an ordering effect.

The result is not a curiosity. It means two rows of the published table are only
comparable when they sat at comparable points in the run, which is why
``docs/benchmarks.md`` compares within the sweep rather than across it, and why the
generated repeatability note cites this experiment instead of blaming the machine.

The mechanism is deliberately *not* asserted here. Two candidates were tested and
rejected: pre-faulting up to 1 GiB of heap before the timed loop changes nothing, and
running a compiled case first changes nothing. What this script reports is the effect,
reproducibly and with its provenance, which is worth more than a guess at the cause.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow running without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cutoutml.benchmarks.environment import capture  # noqa: E402
from cutoutml.benchmarks.harness import (  # noqa: E402
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkHarness,
)
from cutoutml.core.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("benchmarks.order_effect")

#: The configuration whose cost is being measured. Held fixed across every arm.
TARGET_MODEL = "cutoutnet"

#: Models run before the target, one per arm. ``none`` is the control. The rest were
#: chosen to separate "anything heavy" from "this model in particular": ``birefnet`` costs
#: an order of magnitude more per pass than the target and ``classical`` is pure OpenCV,
#: so if either moved the number, the cause would be generic rather than specific.
PRELUDES: tuple[str, ...] = ("none", "classical", "cutoutnet-onnx", "birefnet", "u2netp")

#: ``birefnet`` has no obtainable checkpoint here, so it is timed with random weights.
#: Accuracy is never measured in this script, so that costs nothing.
RANDOM_INIT_PRELUDES = frozenset({"birefnet"})


def measure_one(prelude: str, config: BenchmarkConfig) -> dict[str, Any]:
    """Run ``prelude`` (if any) then the target, and return the target's measurement.

    Both cases skip the accuracy loop: this script is about timing, and the eval loop
    would add minutes per arm to recompute an IoU that the suite already publishes.
    """
    harness = BenchmarkHarness(config)

    if prelude != "none":
        harness.run_case(
            BenchmarkCase(
                model=prelude,
                device="cpu",
                label=f"prelude-{prelude}",
                measure_accuracy=False,
                random_init=prelude in RANDOM_INIT_PRELUDES,
            )
        )

    result = harness.run_case(
        BenchmarkCase(
            model=TARGET_MODEL,
            device="cpu",
            label="target",
            measure_accuracy=False,
            threads=config.threads,
        )
    )
    if result.latency is None:
        raise RuntimeError(f"the target case did not produce a timing: {result.error}")

    return {
        "prelude": prelude,
        "p50_ms": round(result.latency.p50_ms, 4),
        "p95_ms": round(result.latency.p95_ms, 4),
        "stddev_ms": round(result.latency.stddev_ms, 4),
        "threads": result.latency.threads,
        "repetitions": result.latency.repetitions,
        "load": result.load.as_dict() if result.load else None,
        "quiet": bool(result.load.quiet) if result.load else None,
    }


def run_arm_in_subprocess(prelude: str, config: BenchmarkConfig) -> dict[str, Any]:
    """Run one arm in a fresh interpreter.

    A subprocess per arm is the whole method: the effect under study is process state
    accumulated by earlier work, so arms measured in one process would each contaminate
    the next and the control would only be a control the first time.
    """
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as handle:
        out_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--prelude",
                prelude,
                "--emit-to",
                str(out_path),
                "--warmup",
                str(config.warmup),
                "--repetitions",
                str(config.repetitions),
                "--threads",
                str(config.threads),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"arm {prelude!r} exited {completed.returncode}: {completed.stderr[-2000:]}"
            )
        return json.loads(out_path.read_text())  # type: ignore[no-any-return]
    finally:
        out_path.unlink(missing_ok=True)


def table(arms: list[dict[str, Any]]) -> str:
    """Markdown table, with the control's figure as the reference."""
    control = next((a["p50_ms"] for a in arms if a["prelude"] == "none"), None)
    rows = [
        f"| Ran before `{TARGET_MODEL}` | p50 ms | stddev ms | vs control | Machine |",
        "|---|---|---|---|---|",
    ]
    for arm in arms:
        ratio = f"{control / arm['p50_ms']:.2f}x" if control and arm["p50_ms"] else "n/a"
        rows.append(
            "| {prelude} | {p50:.2f} | {sd:.2f} | {ratio} | {quiet} |".format(
                prelude="nothing (control)" if arm["prelude"] == "none" else f"`{arm['prelude']}`",
                p50=arm["p50_ms"],
                sd=arm["stddev_ms"],
                ratio=ratio,
                quiet="quiet" if arm["quiet"] else "**contended**",
            )
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--threads", type=int, default=1, help="intra-op threads for every arm")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--prelude",
        default=None,
        help="run a single arm in this process and print its JSON (used by the driver)",
    )
    p.add_argument(
        "--emit-to", type=Path, default=None, help="with --prelude, write the arm's JSON here"
    )
    p.add_argument("--log-format", default="console", choices=["console", "json"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level="WARNING", fmt=args.log_format)

    config = BenchmarkConfig(
        warmup=args.warmup,
        repetitions=args.repetitions,
        threads=args.threads,
        # The eval set is still constructed (the timed batch comes from it) but no
        # accuracy is computed, so a handful of samples is enough.
        accuracy_samples=8,
    )

    if args.prelude is not None:
        arm = measure_one(args.prelude, config)
        payload = json.dumps(arm, indent=2, sort_keys=True)
        if args.emit_to:
            args.emit_to.write_text(payload + "\n")
        else:
            print(payload)
        return 0

    started = time.perf_counter()
    arms = [run_arm_in_subprocess(prelude, config) for prelude in PRELUDES]
    for arm in arms:
        log.warning("arm_measured", prelude=arm["prelude"], p50_ms=arm["p50_ms"])

    quiet = [a for a in arms if a["quiet"]]
    spread = (
        max(a["p50_ms"] for a in quiet) / min(a["p50_ms"] for a in quiet)
        if len(quiet) > 1
        else None
    )
    report = {
        "experiment": "order_effect",
        "schema_version": 1,
        "run_id": f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "question": (
            f"Does running another model before {TARGET_MODEL} in the same process change "
            f"{TARGET_MODEL}'s measured latency?"
        ),
        "target": {"model": TARGET_MODEL, "batch_size": 1, "threads": args.threads},
        "config": config.as_dict(),
        "environment": capture().as_dict(),
        "arms": arms,
        "spread_among_quiet_arms": round(spread, 3) if spread else None,
        "arms_contended": sum(1 for a in arms if a["quiet"] is False),
    }

    target_dir = args.output_dir or (REPO_ROOT / "benchmarks" / "results" / "experiments")
    target_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately not in benchmarks/results/ itself: the renderer treats every *.json
    # directly in that directory as a suite report and would pick this up as the latest.
    path = target_dir / f"order-effect-{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(table(arms))
    print()
    if spread:
        print(f"Spread among arms measured on a quiet machine: {spread:.2f}x")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
