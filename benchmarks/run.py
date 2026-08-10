#!/usr/bin/env python3
"""Benchmark entry point.

    python benchmarks/run.py                       # the standard committed suite
    python benchmarks/run.py --quick               # fewer reps, for a smoke check
    python benchmarks/run.py --models cutoutnet    # a subset
    python benchmarks/run.py --dataset-root /data/DUTS --dataset-family duts
    python benchmarks/run.py --no-render           # write JSON but leave docs alone

The default suite is defined in :func:`default_cases` and is exactly what the
committed ``benchmarks/results/*.json`` contains. Keeping the suite in code rather than
in a config file means the reproduction command is a single argument-free invocation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow running without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cutoutml.benchmarks.harness import (  # noqa: E402
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkHarness,
    save_report,
)
from cutoutml.benchmarks.render_report import render  # noqa: E402
from cutoutml.core.logging import configure_logging, get_logger  # noqa: E402
from cutoutml.core.refine import RefineConfig  # noqa: E402

log = get_logger("benchmarks.run")


def default_cases(
    *, include_batches: bool = True, include_compile: bool = True
) -> list[BenchmarkCase]:
    """The committed suite.

    Ordering is deliberate: calibration references first, then the non-learned
    baselines, then the models actually trained in this repository, then the runtime
    comparison for one of them, then the architectures that can only be measured for
    latency. Reading the table top to bottom tells the story of what each layer of
    sophistication buys.

    A case whose checkpoint is absent comes back as ``status="skipped"`` with the
    reason attached rather than silently falling back to random weights, so a missing
    training run can never be misread as a bad accuracy number.
    """
    cases: list[BenchmarkCase] = [
        # --- content-blind calibration references
        BenchmarkCase(model="trivial-ones", device="cpu", label="trivial-ones"),
        BenchmarkCase(model="trivial-center", device="cpu", label="trivial-center"),
        # --- zero-training baselines
        BenchmarkCase(model="classical-saliency", device="cpu", label="classical-saliency"),
        BenchmarkCase(model="classical", device="cpu", label="classical-grabcut"),
        BenchmarkCase(
            model="classical-saliency-grabcut", device="cpu", label="classical-saliency+grabcut"
        ),
        # --- the capacity sweep: identical data budget, three sizes
        BenchmarkCase(model="cutoutnet-tiny", device="cpu", label="cutoutnet-tiny-fp32"),
        BenchmarkCase(model="cutoutnet", precision="fp32", device="cpu", label="cutoutnet-fp32"),
        BenchmarkCase(model="cutoutnet-base", device="cpu", label="cutoutnet-base-fp32"),
        # --- a different architecture at a comparable parameter count
        BenchmarkCase(model="u2net-lite", device="cpu", label="u2net-lite-fp32"),
        # --- same weights, different runtime
        BenchmarkCase(
            model="cutoutnet-onnx", precision="fp32", device="cpu", label="cutoutnet-onnx-cpu"
        ),
        # --- architectures whose pretrained weights are not downloadable here and
        #     which are too expensive to train on this box: latency only.
        BenchmarkCase(
            model="u2net",
            precision="fp32",
            device="cpu",
            random_init=True,
            label="u2net-full-randominit",
        ),
        BenchmarkCase(
            model="birefnet",
            precision="fp32",
            device="cpu",
            random_init=True,
            label="birefnet-compact-randominit",
        ),
    ]

    if include_compile:
        # Same weights, same batch, same resolution as the eager row above: the only
        # difference is the Inductor backend, so the delta is attributable to it.
        cases += [
            BenchmarkCase(
                model="cutoutnet", device="cpu", compile=True, label="cutoutnet-fp32-compiled"
            ),
            BenchmarkCase(
                model="cutoutnet",
                device="cpu",
                batch_size=8,
                compile=True,
                label="cutoutnet-fp32-b8-compiled",
            ),
        ]

    if include_batches:
        # Batch scaling for the trained model: shows the throughput/latency trade-off
        # with real numbers instead of an assertion that batching helps.
        cases += [
            BenchmarkCase(
                model="cutoutnet", device="cpu", batch_size=4, label="cutoutnet-fp32-b4"
            ),
            BenchmarkCase(
                model="cutoutnet", device="cpu", batch_size=8, label="cutoutnet-fp32-b8"
            ),
            BenchmarkCase(
                model="cutoutnet-onnx", device="cpu", batch_size=8, label="cutoutnet-onnx-b8"
            ),
        ]
    return cases


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=None, help="restrict to these model names")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--accuracy-samples", type=int, default=64)
    p.add_argument("--resolution", type=int, default=256, help="eval dataset resolution")
    p.add_argument("--torch-threads", type=int, default=0, help="0 = PyTorch default")
    p.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    p.add_argument("--no-batches", action="store_true", help="skip batch-scaling cases")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile cases (they add a minute of codegen per case)",
    )
    p.add_argument("--no-render", action="store_true", help="do not regenerate markdown")
    p.add_argument("--no-refine", action="store_true", help="measure accuracy without refinement")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dataset-root", type=Path, default=None, help="use a real dataset instead")
    p.add_argument(
        "--dataset-family", default=None, choices=["duts", "dis5k", "am2k", "flat"]
    )
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--log-format", default="console", choices=["console", "json"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(fmt=args.log_format)

    if args.quick:
        args.warmup, args.repetitions, args.accuracy_samples = 1, 3, 8
        # Inductor codegen costs tens of seconds per case regardless of how few
        # repetitions follow it, which defeats the purpose of a smoke check.
        args.no_compile = True

    config = BenchmarkConfig(
        warmup=args.warmup,
        repetitions=args.repetitions,
        accuracy_samples=args.accuracy_samples,
        dataset_resolution=(args.resolution, args.resolution),
        dataset_split=args.dataset_split,
        torch_threads=args.torch_threads,
        refine=RefineConfig.off() if args.no_refine else RefineConfig.fast(),
    )

    dataset = None
    dataset_description = None
    if args.dataset_root is not None:
        from cutoutml.datasets.real import RealSegmentationDataset, detect_family

        family = args.dataset_family or detect_family(args.dataset_root)
        if family is None:
            log.error("dataset_family_undetected", root=str(args.dataset_root))
            return 2
        dataset = RealSegmentationDataset(
            args.dataset_root,
            family=family,
            split=args.dataset_split,
            limit=args.accuracy_samples,
        )
        dataset_description = dataset.describe()
        log.info("using_real_dataset", family=family, samples=len(dataset))

    cases = default_cases(
        include_batches=not args.no_batches, include_compile=not args.no_compile
    )
    if args.models:
        wanted = set(args.models)
        cases = [c for c in cases if c.model in wanted or (c.label or "") in wanted]
        if not cases:
            log.error("no_matching_cases", requested=sorted(wanted))
            return 2

    harness = BenchmarkHarness(
        config, dataset=dataset, dataset_description=dataset_description
    )
    log.info(
        "benchmark_start",
        cases=len(cases),
        hardware=harness.environment.hardware,
        gpu=harness.environment.gpu,
    )

    report = harness.run(cases)
    path = save_report(report, args.output_dir)
    print(f"\nwrote {path}")
    print(f"summary: {report['summary']}")

    if not args.no_render:
        docs, readme = render(path)
        print(f"rendered {docs}")
        if readme:
            print(f"updated {readme}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
