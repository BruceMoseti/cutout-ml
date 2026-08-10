"""``cutoutml`` command line interface.

One entry point for everything that does not need the API::

    cutoutml models                              # what is registered and usable
    cutoutml segment photo.jpg -o out/           # cut out a still
    cutoutml video clip.mp4 -o out.mp4 --mode composite
    cutoutml export-onnx cutoutnet -o models/cutoutnet/cutoutnet-small.onnx
    cutoutml train --arch cutoutnet-base --epochs 14
    cutoutml benchmark --quick
    cutoutml doctor                              # what works on this machine

Subcommands delegate to the library rather than reimplementing it, so the CLI and the
worker cannot drift: ``segment`` runs the same :class:`~cutoutml.pipelines.image.
ImagePipeline` the Celery task does, and ``train``/``benchmark`` forward straight to
their existing argument parsers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cutoutml.core.config import get_settings
from cutoutml.core.logging import configure_logging, get_logger

log = get_logger("cutoutml.cli")


# --------------------------------------------------------------------- models


def cmd_models(args: argparse.Namespace) -> int:
    """List the registry, marking which entries can actually run here."""
    from cutoutml.models.registry import list_models

    settings = get_settings()
    rows: list[dict[str, object]] = []
    for spec in list_models():
        weights = spec.default_weights
        available = True
        if spec.requires_weights and weights:
            candidate = Path(weights)
            if not candidate.is_absolute():
                candidate = settings.model_weights_dir / candidate
            available = candidate.is_file()
        rows.append(
            {
                "name": spec.name,
                "architecture": spec.architecture,
                "runtime": spec.runtime,
                "input": f"{spec.input_size[0]}x{spec.input_size[1]}",
                "weights": "yes" if available else "MISSING",
                "tags": ",".join(spec.tags),
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = "  ".join(k.ljust(widths[k]) for k in widths)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[k]).ljust(widths[k]) for k in widths))
    return 0


# -------------------------------------------------------------------- segment


def cmd_segment(args: argparse.Namespace) -> int:
    """Segment one still image and write the requested outputs."""
    from cutoutml.core.imaging import decode_image
    from cutoutml.models.registry import get_model
    from cutoutml.pipelines.image import ImagePipeline, ImageRequest

    settings = get_settings()
    source = Path(args.input)
    if not source.is_file():
        log.error("input_missing", path=str(source))
        return 2

    background = decode_image(Path(args.background).read_bytes()) if args.background else None
    model = get_model(args.model or settings.default_model, device=args.device, precision=args.precision)
    request = ImageRequest(
        outputs=tuple(args.outputs),
        background_color=tuple(args.background_color),  # type: ignore[arg-type]
        background_image=background,
        blur_sigma=args.blur_sigma,
    )

    result = ImagePipeline(model).process_bytes(source.read_bytes(), request)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    for kind, payload in result.outputs.items():
        extension = "webp" if kind.endswith("webp") else "png"
        path = outdir / f"{source.stem}.{kind}.{extension}"
        path.write_bytes(payload)
        print(f"wrote {path} ({len(payload)} bytes)")

    print(json.dumps(result.summary(), indent=2))
    return 0


# ---------------------------------------------------------------------- video


def cmd_video(args: argparse.Namespace) -> int:
    """Segment a video clip."""
    from cutoutml.core.imaging import decode_image
    from cutoutml.models.registry import get_model
    from cutoutml.pipelines.video import VideoPipeline, VideoProgress, VideoRequest

    settings = get_settings()
    source = Path(args.input)
    if not source.is_file():
        log.error("input_missing", path=str(source))
        return 2

    background = decode_image(Path(args.background).read_bytes()) if args.background else None
    model = get_model(args.model or settings.default_model, device=args.device, precision=args.precision)
    request = VideoRequest(
        mode=args.mode,
        container=args.container,
        background_color=tuple(args.background_color),  # type: ignore[arg-type]
        background_image=background,
        blur_background=args.blur_background,
        smoothing=args.smoothing,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        crf=args.crf,
        measure_flicker=args.measure_flicker,
        frame_limit=settings.max_video_frames,
    )

    def report(progress: VideoProgress) -> None:
        pct = progress.fraction * 100.0
        sys.stderr.write(
            f"\r{progress.stage}: {progress.frames_done}/{progress.frames_total} "
            f"({pct:5.1f}%) {progress.fps:.1f} fps"
        )
        sys.stderr.flush()

    try:
        result = VideoPipeline(model).process(
            source,
            Path(args.output),
            request,
            on_progress=None if args.quiet else report,
            ffmpeg=settings.ffmpeg_binary,
            ffprobe=settings.ffprobe_binary,
        )
    except ValueError as exc:
        # Impossible requests (MP4 + alpha, a frame count over the guard) are rejected
        # up front with an actionable message; a traceback would bury it.
        sys.stderr.write(f"\n{exc}\n")
        return 2

    sys.stderr.write("\n")
    print(json.dumps(result.summary(), indent=2))
    return 0


# ----------------------------------------------------------------- export-onnx


def cmd_export_onnx(args: argparse.Namespace) -> int:
    """Export a trained PyTorch model to ONNX."""
    from cutoutml.models.base import TorchSegmentationModel
    from cutoutml.models.registry import get_model

    model = get_model(args.model, device="cpu")
    if not isinstance(model, TorchSegmentationModel):
        log.error("not_exportable", model=args.model, reason="not a PyTorch adapter")
        return 2
    path = model.to_onnx(args.output, opset=args.opset, dynamic_batch=not args.static_batch)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


# ---------------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what this machine can actually do.

    Exists because most of the "why is this slow / why did that fail" questions in this
    project have the same three answers - no GPU, no ffmpeg, no onnxruntime - and it is
    faster to print them than to explain them.
    """
    import shutil

    import torch

    from cutoutml.core.devices import cpu_name, cuda_available, describe_device, resolve_device
    from cutoutml.db.session import check_database
    from cutoutml.models.registry import list_model_names

    settings = get_settings()
    device = resolve_device("auto")
    db_ok, db_detail = check_database(settings)

    try:
        import onnxruntime

        onnx_detail = f"{onnxruntime.__version__} providers={onnxruntime.get_available_providers()}"
    except ImportError:
        onnx_detail = "not installed (pip install onnxruntime)"

    try:
        from cutoutml.pipelines.ffmpeg import working_alpha_containers

        alpha = working_alpha_containers(settings.ffmpeg_binary)
        alpha_detail = ", ".join(alpha) if alpha else "none (transparent video unavailable)"
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
        alpha_detail = f"probe failed: {type(exc).__name__}"

    report = {
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu": cpu_name(),
        "device_resolved": str(device),
        "device_name": describe_device(device).name,
        "cuda_available": cuda_available(),
        "onnxruntime": onnx_detail,
        "ffmpeg": shutil.which(settings.ffmpeg_binary) or "NOT FOUND",
        "ffprobe": shutil.which(settings.ffprobe_binary) or "NOT FOUND",
        "alpha_capable_containers": alpha_detail,
        "database": db_detail if db_ok else f"UNAVAILABLE: {db_detail}",
        "storage_backend": settings.storage_backend,
        "storage_root": str(settings.storage_root),
        "models_registered": len(list_model_names()),
        "default_model": settings.default_model,
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        width = max(len(k) for k in report)
        for key, value in report.items():
            print(f"{key.ljust(width)}  {value}")
    return 0


# ------------------------------------------------------------------ delegating


def cmd_train(argv: list[str]) -> int:
    from cutoutml.training.train import main as train_main

    return train_main(argv)


def cmd_benchmark(argv: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from run import main as benchmark_main  # type: ignore[import-not-found]

    return int(benchmark_main(argv))


# ------------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cutoutml",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-format", default="console", choices=["console", "json"])
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="list registered models")
    p_models.add_argument("--json", action="store_true")
    p_models.set_defaults(func=cmd_models)

    p_seg = sub.add_parser("segment", help="segment a single image")
    p_seg.add_argument("input")
    p_seg.add_argument("-o", "--output", default="out")
    p_seg.add_argument("-m", "--model", default=None)
    p_seg.add_argument("--device", default="auto")
    p_seg.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p_seg.add_argument(
        "--outputs",
        nargs="+",
        default=["transparent_png", "mask_png"],
        choices=[
            "transparent_png",
            "transparent_webp",
            "mask_png",
            "color_composite",
            "background_composite",
            "blurred_background",
        ],
    )
    p_seg.add_argument("--background", default=None, help="background image for compositing")
    p_seg.add_argument(
        "--background-color", nargs=3, type=int, default=[255, 255, 255], metavar=("R", "G", "B")
    )
    p_seg.add_argument("--blur-sigma", type=float, default=12.0)
    p_seg.set_defaults(func=cmd_segment)

    p_vid = sub.add_parser("video", help="segment a video")
    p_vid.add_argument("input")
    p_vid.add_argument("-o", "--output", required=True)
    p_vid.add_argument("-m", "--model", default=None)
    p_vid.add_argument("--device", default="auto")
    p_vid.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p_vid.add_argument(
        "--mode", default="composite", choices=["composite", "transparent", "frames", "mask"]
    )
    p_vid.add_argument("--container", default="mp4", choices=["mp4", "webm", "mov", "qtrle"])
    p_vid.add_argument(
        "--background-color", nargs=3, type=int, default=[0, 177, 64], metavar=("R", "G", "B")
    )
    p_vid.add_argument("--background", default=None)
    p_vid.add_argument("--blur-background", action="store_true")
    p_vid.add_argument("--smoothing", default="ema", choices=["none", "ema", "median"])
    p_vid.add_argument("--batch-size", type=int, default=4)
    p_vid.add_argument("--max-frames", type=int, default=None)
    p_vid.add_argument("--crf", type=int, default=23)
    p_vid.add_argument("--measure-flicker", action="store_true")
    p_vid.add_argument("--quiet", action="store_true")
    p_vid.set_defaults(func=cmd_video)

    p_onnx = sub.add_parser("export-onnx", help="export a model to ONNX")
    p_onnx.add_argument("model")
    p_onnx.add_argument("-o", "--output", required=True)
    p_onnx.add_argument("--opset", type=int, default=17)
    p_onnx.add_argument("--static-batch", action="store_true")
    p_onnx.set_defaults(func=cmd_export_onnx)

    p_doctor = sub.add_parser("doctor", help="report hardware and dependency status")
    p_doctor.add_argument("--json", action="store_true")
    p_doctor.set_defaults(func=cmd_doctor)

    # `train` and `benchmark` own rich flag sets of their own; forwarding the remainder
    # keeps one canonical definition of each instead of two that drift.
    p_train = sub.add_parser(
        "train", help="train a model (arguments forwarded to cutoutml.training.train)"
    )
    p_train.add_argument("rest", nargs=argparse.REMAINDER)
    p_train.set_defaults(delegate=cmd_train)

    p_bench = sub.add_parser(
        "benchmark", help="run the benchmark suite (arguments forwarded to benchmarks/run.py)"
    )
    p_bench.add_argument("rest", nargs=argparse.REMAINDER)
    p_bench.set_defaults(delegate=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level, fmt=args.log_format)

    delegate = getattr(args, "delegate", None)
    if delegate is not None:
        return int(delegate(list(args.rest)))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
