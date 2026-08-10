"""End-to-end video pipeline.

::

    ffmpeg decode (rgb24 pipe)
        -> bounded frame batches           <- never more than batch_size frames in RAM
        -> batched model inference
        -> temporal alpha smoothing        <- optional, measured, not assumed
        -> composite / RGBA
        -> ffmpeg encode (or frame files)

Streaming discipline
--------------------
The pipeline never materialises the whole video. Frames arrive from an ffmpeg pipe in
batches of ``batch_size`` and leave immediately after encoding, so peak memory is
``batch_size x width x height x 3`` plus whatever the smoother holds (1 frame for EMA,
``window`` frames for median). A 60 s 4K clip is ~1.5 GB of raw RGB; at
``batch_size=4`` this pipeline touches ~100 MB. Frame-sequence output writes straight
to a temp directory rather than accumulating buffers.

Temporal smoothing
------------------
Per-frame segmentation flickers: a pixel near the decision boundary flips between
frames even when nothing moved, which reads as a crawling edge. Two smoothers are
provided and **both are optional and measured** - :func:`estimate_flicker` reports the
mean absolute frame-to-frame alpha difference with and without smoothing, so the
trade-off is a number rather than an opinion. Smoothing always costs some
responsiveness on fast motion; the default EMA weight of 0.65 is a compromise, and
``docs/benchmarks.md`` shows the measured effect.

Output modes and the alpha problem
----------------------------------
``composite`` burns a background into an ordinary H.264 MP4 - universally playable,
no transparency. ``transparent`` needs a container that actually carries an alpha
plane, which in practice means WebM/VP9 (``yuva420p``), ProRes 4444 or QuickTime RLE;
:func:`cutoutml.pipelines.ffmpeg.encoder_args` refuses MP4 + alpha rather than
silently producing an opaque file. ``frames`` sidesteps codecs entirely by writing an
RGBA PNG per frame, optionally zipped - lossless alpha, largest output, and the only
option a video editor will always accept. ``docs/decisions/ADR-003-video-output.md``
works through the trade-off.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cutoutml.core.imaging import (
    composite_blurred_background,
    composite_over_color,
    composite_over_image,
    encode_image,
    to_uint8_alpha,
)
from cutoutml.core.logging import get_logger
from cutoutml.core.refine import RefineConfig, ema_smooth, refine_alpha, temporal_median
from cutoutml.models.base import SegmentationModel
from cutoutml.pipelines.ffmpeg import (
    FrameReader,
    FrameWriter,
    VideoInfo,
    alpha_roundtrip_works,
    container_supports_alpha,
    encoder_args,
    probe,
    working_alpha_containers,
)

log = get_logger(__name__)

SmoothingMode = Literal["none", "ema", "median"]
VideoOutputMode = Literal["composite", "transparent", "frames", "mask"]

ProgressCallback = Callable[["VideoProgress"], None]


@dataclasses.dataclass(slots=True)
class VideoProgress:
    """A progress event. Emitted at most every ``progress_interval`` frames."""

    stage: str
    frames_done: int
    frames_total: int
    seconds_elapsed: float
    fps: float
    message: str = ""

    @property
    def fraction(self) -> float:
        if self.frames_total <= 0:
            return 0.0
        return min(1.0, self.frames_done / self.frames_total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "fraction": round(self.fraction, 4),
            "seconds_elapsed": round(self.seconds_elapsed, 3),
            "fps": round(self.fps, 3),
            "message": self.message,
        }


@dataclasses.dataclass(slots=True)
class VideoRequest:
    """Parameters for one video job."""

    mode: VideoOutputMode = "composite"
    container: str = "mp4"
    background_color: tuple[int, int, int] = (0, 177, 64)
    background_image: np.ndarray | None = None
    blur_background: bool = False
    blur_sigma: float = 12.0
    smoothing: SmoothingMode = "ema"
    ema_weight: float = 0.65
    median_window: int = 3
    batch_size: int = 4
    refine: RefineConfig = dataclasses.field(default_factory=RefineConfig.fast)
    max_frames: int | None = None
    scale_to: tuple[int, int] | None = None
    crf: int = 23
    preset: str = "medium"
    frame_format: Literal["png", "webp"] = "png"
    keep_audio: bool = True
    progress_interval: int = 15
    measure_flicker: bool = False
    #: For ``mode="frames"``, collect the sequence into a single zip archive. A
    #: thousand-file directory is not a deliverable a client can download.
    archive_frames: bool = True
    frame_limit: int = 18_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "container": self.container,
            "background_color": list(self.background_color),
            "has_background_image": self.background_image is not None,
            "blur_background": self.blur_background,
            "smoothing": self.smoothing,
            "ema_weight": self.ema_weight,
            "median_window": self.median_window,
            "batch_size": self.batch_size,
            "refine": self.refine.as_dict(),
            "max_frames": self.max_frames,
            "crf": self.crf,
            "preset": self.preset,
            "measure_flicker": self.measure_flicker,
            "archive_frames": self.archive_frames,
        }


@dataclasses.dataclass(slots=True)
class VideoResult:
    """Outcome of a video job."""

    output_path: Path
    frame_paths: list[Path]
    info: VideoInfo
    frames_processed: int
    seconds: float
    fps: float
    mode: str
    container: str
    smoothing: str
    flicker_raw: float | None = None
    flicker_smoothed: float | None = None
    output_bytes: int = 0
    archive_path: Path | None = None
    has_alpha: bool = False

    @property
    def deliverable(self) -> Path:
        """The single path a client should be handed.

        For frame-sequence output that is the zip, not the directory; every other
        mode produces one file already.
        """
        return self.archive_path or self.output_path

    def summary(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "deliverable": str(self.deliverable),
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "frame_count": len(self.frame_paths) or self.frames_processed,
            "frames_processed": self.frames_processed,
            "seconds": round(self.seconds, 3),
            "fps": round(self.fps, 3),
            "mode": self.mode,
            "container": self.container,
            "smoothing": self.smoothing,
            "has_alpha": self.has_alpha,
            "flicker_raw": self.flicker_raw,
            "flicker_smoothed": self.flicker_smoothed,
            "output_bytes": self.output_bytes,
            "source": self.info.as_dict(),
        }


class TemporalSmoother:
    """Stateful alpha smoother with a bounded buffer.

    ``median`` buffers ``window`` frames and therefore introduces ``window // 2``
    frames of latency; it is the only way to remove a single-frame dropout entirely.
    ``ema`` has one frame of state and no latency but attenuates rather than removes.
    """

    def __init__(self, mode: SmoothingMode, *, ema_weight: float = 0.65, window: int = 3) -> None:
        self.mode = mode
        self.ema_weight = ema_weight
        self.window = max(1, window | 1)  # force odd so the median has a middle
        self._previous: np.ndarray | None = None
        self._buffer: list[np.ndarray] = []

    def __call__(self, alpha: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return alpha
        if self.mode == "ema":
            out = ema_smooth(self._previous, alpha, self.ema_weight)
            self._previous = out
            return out
        self._buffer.append(alpha)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return temporal_median(self._buffer)

    def reset(self) -> None:
        self._previous = None
        self._buffer.clear()


class VideoPipeline:
    """Streaming video segmentation bound to one loaded model."""

    def __init__(self, model: SegmentationModel) -> None:
        if not model.is_loaded:
            model.load()
        self.model = model

    # ------------------------------------------------------------------- process

    def process(
        self,
        source: Path | str,
        destination: Path | str,
        request: VideoRequest | None = None,
        *,
        on_progress: ProgressCallback | None = None,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
    ) -> VideoResult:
        """Segment ``source`` and write the result to ``destination``.

        For ``mode="frames"``, ``destination`` names either the directory to fill with
        PNGs or the ``.zip`` to deliver; see :func:`_frame_destinations`.
        """
        req = request or VideoRequest()
        src = Path(source)
        dst = Path(destination)
        if not src.is_file():
            raise FileNotFoundError(f"video not found: {src}")

        frame_dir, archive_target = (
            _frame_destinations(dst) if req.mode == "frames" else (dst, None)
        )

        info = probe(src, ffprobe=ffprobe)
        self._validate(req, info, ffmpeg)

        out_w, out_h = req.scale_to or (info.width, info.height)
        total = req.max_frames or info.frame_count
        started = time.perf_counter()

        alphas_raw: list[np.ndarray] = []
        alphas_out: list[np.ndarray] = []
        smoother = TemporalSmoother(
            req.smoothing, ema_weight=req.ema_weight, window=req.median_window
        )

        frame_paths: list[Path] = []
        frames_done = 0

        writer_ctx = self._make_writer(frame_dir, req, info, (out_w, out_h), src, ffmpeg)

        with (
            FrameReader(
                src,
                width=info.width,
                height=info.height,
                ffmpeg=ffmpeg,
                scale=req.scale_to,
                max_frames=req.max_frames,
            ) as reader,
            writer_ctx as writer,
        ):
            for batch in _batched(_decode_frames(reader, out_h, out_w), req.batch_size):
                alphas = self.model.infer(batch)
                for frame, alpha in zip(batch, alphas, strict=True):
                    refined = refine_alpha(alpha, frame, req.refine)
                    if req.measure_flicker:
                        alphas_raw.append(refined)
                    smoothed = smoother(refined)
                    if req.measure_flicker:
                        alphas_out.append(smoothed)

                    if writer is not None:
                        writer.write(self._render(frame, smoothed, req).tobytes())
                    else:
                        frame_paths.append(
                            self._write_frame_file(frame_dir, frames_done, frame, smoothed, req)
                        )
                    frames_done += 1

                if on_progress and frames_done % max(1, req.progress_interval) < req.batch_size:
                    elapsed = time.perf_counter() - started
                    on_progress(
                        VideoProgress(
                            stage="inference",
                            frames_done=frames_done,
                            frames_total=total,
                            seconds_elapsed=elapsed,
                            fps=frames_done / max(elapsed, 1e-6),
                        )
                    )

        archive: Path | None = None
        if frame_paths and req.archive_frames and archive_target is not None:
            archive = archive_frames(frame_paths, archive_target)

        elapsed = time.perf_counter() - started
        output_bytes = (
            archive.stat().st_size
            if archive is not None
            else (
                sum(p.stat().st_size for p in frame_paths)
                if frame_paths
                else (dst.stat().st_size if dst.is_file() else 0)
            )
        )

        result = VideoResult(
            output_path=frame_dir,
            frame_paths=frame_paths,
            info=info,
            frames_processed=frames_done,
            seconds=elapsed,
            fps=frames_done / max(elapsed, 1e-6),
            mode=req.mode,
            # A frame sequence has no container, and reporting the request's unused
            # default ("mp4") next to an RGBA PNG zip reads as a contradiction.
            container="png-sequence" if req.mode == "frames" else req.container,
            smoothing=req.smoothing,
            flicker_raw=estimate_flicker(alphas_raw) if req.measure_flicker else None,
            flicker_smoothed=estimate_flicker(alphas_out) if req.measure_flicker else None,
            output_bytes=output_bytes,
            archive_path=archive,
            has_alpha=req.mode in {"transparent", "frames"},
        )

        if on_progress:
            on_progress(
                VideoProgress(
                    stage="complete",
                    frames_done=frames_done,
                    frames_total=total or frames_done,
                    seconds_elapsed=elapsed,
                    fps=result.fps,
                    message="done",
                )
            )
        log.info("video_processed", **result.summary())
        return result

    # ------------------------------------------------------------------ internals

    def _validate(self, req: VideoRequest, info: VideoInfo, ffmpeg: str) -> None:
        """Reject impossible requests before spawning ffmpeg or decoding frames.

        Every check here would otherwise surface either as a confusing ffmpeg stderr
        dump several seconds in, or - worse, in the alpha case - as a silently opaque
        "transparent" video.
        """
        if info.width <= 0 or info.height <= 0:
            raise ValueError(f"invalid video dimensions {info.width}x{info.height}")

        if req.mode == "transparent":
            if not container_supports_alpha(req.container):
                raise ValueError(
                    f"container {req.container!r} cannot carry alpha. Use 'mov' "
                    "(ProRes 4444), 'qtrle', 'webm' (VP9), switch to mode='frames' "
                    "for an RGBA PNG sequence, or mode='composite' to burn in a "
                    "background. See docs/decisions/ADR-003-video-output.md."
                )
            if not alpha_roundtrip_works(req.container, ffmpeg):
                usable = working_alpha_containers(ffmpeg)
                raise ValueError(
                    f"container {req.container!r} is specified to carry alpha but this "
                    f"ffmpeg build drops it (verified by encode/decode round-trip). "
                    f"Containers that do preserve alpha here: "
                    f"{', '.join(usable) if usable else 'none'}. mode='frames' always "
                    "works. See docs/decisions/ADR-003-video-output.md."
                )
        requested = req.max_frames or info.frame_count
        if req.frame_limit > 0 and requested > req.frame_limit:
            raise ValueError(
                f"video has {requested} frames, above the configured limit of "
                f"{req.frame_limit}. Raise CUTOUTML_MAX_VIDEO_FRAMES or pass "
                "max_frames to process a prefix."
            )

    def _make_writer(
        self,
        dst: Path,
        req: VideoRequest,
        info: VideoInfo,
        size: tuple[int, int],
        source: Path,
        ffmpeg: str,
    ) -> Any:
        """A ``FrameWriter`` context, or a null context for frame-sequence output."""
        if req.mode == "frames":
            dst.mkdir(parents=True, exist_ok=True)
            return _NullWriter()

        alpha = req.mode == "transparent"
        return FrameWriter(
            dst,
            width=size[0],
            height=size[1],
            fps=info.fps,
            pix_fmt="rgba" if alpha else "rgb24",
            encoder_arguments=encoder_args(
                req.container, crf=req.crf, preset=req.preset, alpha=alpha
            ),
            ffmpeg=ffmpeg,
            # Audio is only copied when the source has a stream and the output is not
            # a frame sequence; ffmpeg would otherwise fail on the missing map.
            audio_source=source if (req.keep_audio and info.has_audio) else None,
        )

    def _render(self, frame: np.ndarray, alpha: np.ndarray, req: VideoRequest) -> np.ndarray:
        """Produce the pixel buffer for one output frame."""
        if req.mode == "transparent":
            return np.dstack([frame, to_uint8_alpha(alpha)])
        if req.mode == "mask":
            return np.repeat(to_uint8_alpha(alpha)[:, :, None], 3, axis=2)
        if req.blur_background:
            return composite_blurred_background(frame, alpha, blur_sigma=req.blur_sigma)
        if req.background_image is not None:
            return composite_over_image(frame, alpha, req.background_image)
        return composite_over_color(frame, alpha, req.background_color)

    def _write_frame_file(
        self, directory: Path, index: int, frame: np.ndarray, alpha: np.ndarray, req: VideoRequest
    ) -> Path:
        path = directory / f"frame_{index:06d}.{req.frame_format}"
        if req.mode in {"frames", "transparent"}:
            data = encode_image(frame, req.frame_format, alpha=alpha)
        else:
            data = encode_image(self._render(frame, alpha, req), req.frame_format)
        path.write_bytes(data)
        return path

    # ------------------------------------------------------------ smoothing study

    def compare_smoothing(
        self,
        source: Path | str,
        *,
        modes: Sequence[SmoothingMode] = ("none", "ema", "median"),
        max_frames: int = 48,
        batch_size: int = 4,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
    ) -> dict[str, dict[str, float]]:
        """Measure the effect of each smoothing mode on flicker and on cost.

        Decodes once, caches the per-frame alpha maps, then applies each smoother to
        the same sequence - so the comparison isolates smoothing from any
        frame-to-frame variation in inference. Returns flicker and the smoothing
        overhead per frame in milliseconds.
        """
        info = probe(source, ffprobe=ffprobe)
        cached: list[np.ndarray] = []
        with FrameReader(
            source, width=info.width, height=info.height, ffmpeg=ffmpeg, max_frames=max_frames
        ) as reader:
            for batch in _batched(_decode_frames(reader, info.height, info.width), batch_size):
                alphas = self.model.infer(batch)
                cached.extend(
                    refine_alpha(a, f, RefineConfig.fast())
                    for a, f in zip(alphas, batch, strict=True)
                )

        out: dict[str, dict[str, float]] = {}
        for mode in modes:
            smoother = TemporalSmoother(mode)
            started = time.perf_counter()
            smoothed = [smoother(a) for a in cached]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            out[mode] = {
                "flicker": estimate_flicker(smoothed),
                "overhead_ms_per_frame": elapsed_ms / max(1, len(cached)),
                "frames": float(len(cached)),
            }
        out["_baseline"] = {"flicker": estimate_flicker(cached), "frames": float(len(cached))}
        return out


class _NullWriter:
    """Context manager that yields ``None`` - frame-sequence mode has no encoder."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _decode_frames(reader: FrameReader, height: int, width: int) -> Iterator[np.ndarray]:
    """Wrap raw ffmpeg buffers as ``(H, W, 3)`` uint8 arrays.

    ``frombuffer`` gives a read-only view over the pipe buffer with no copy; the
    reshape is free. A copy is taken only because downstream OpenCV calls want a
    writable, aligned array.
    """
    for buf in reader.frames():
        yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()


def _batched(iterator: Iterator[np.ndarray], size: int) -> Iterator[list[np.ndarray]]:
    """Group an iterator into lists of at most ``size`` - the memory bound."""
    batch: list[np.ndarray] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= max(1, size):
            yield batch
            batch = []
    if batch:
        yield batch


def estimate_flicker(alphas: Sequence[np.ndarray]) -> float:
    """Mean absolute difference between consecutive alpha maps.

    Lower is smoother. Reported alongside IoU because a frozen mask scores a perfect
    0 here while being useless - flicker is only meaningful next to accuracy.
    """
    if len(alphas) < 2:
        return 0.0
    return float(
        np.mean(
            [
                np.abs(alphas[i].astype(np.float32) - alphas[i - 1].astype(np.float32)).mean()
                for i in range(1, len(alphas))
            ]
        )
    )


def make_test_video(
    path: Path | str,
    *,
    frames: int = 24,
    width: int = 160,
    height: int = 120,
    fps: float = 12.0,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Render a small synthetic clip - a moving blob - for tests and demos.

    Kept here rather than in the test suite so the end-to-end video path can be
    exercised from a Makefile target without any sample media in the repository.
    """
    out = Path(path)
    with FrameWriter(
        out,
        width=width,
        height=height,
        fps=fps,
        pix_fmt="rgb24",
        encoder_arguments=encoder_args("mp4", crf=20, preset="veryfast"),
        ffmpeg=ffmpeg,
    ) as writer:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        for i in range(frames):
            t = i / max(1, frames - 1)
            cx = width * (0.25 + 0.5 * t)
            cy = height * (0.5 + 0.15 * np.sin(2 * np.pi * t))
            radius = min(width, height) * 0.22
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = 30
            frame[..., 2] = 90
            frame[mask] = (250, 230, 40)
            writer.write(frame.tobytes())
    return out


def _frame_destinations(destination: Path) -> tuple[Path, Path]:
    """Split a frame-sequence destination into (directory to fill, zip to deliver).

    Callers write ``-o out.zip`` at least as often as ``-o out_frames/``, and both have
    to work. Deriving the zip with ``with_suffix(".zip")`` cannot: given ``out.zip`` it
    returns the directory itself, and the archive then fails to open. So a ``.zip``
    destination names the archive and the frames go to a sibling directory; anything
    else names the directory and the archive sits beside it.
    """
    if destination.suffix.lower() == ".zip":
        return destination.with_suffix(""), destination
    return destination, destination.parent / f"{destination.name}.zip"


def archive_frames(paths: Sequence[Path], destination: Path | str) -> Path:
    """Zip a PNG frame sequence into a single downloadable artefact.

    ``ZIP_STORED`` rather than ``ZIP_DEFLATE``: PNG is already deflate-compressed, so
    re-compressing costs CPU proportional to the whole output and typically saves
    under 1%. Frames are added by basename so the archive extracts into a flat
    directory rather than reproducing the server's temp path.
    """
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in paths:
            zf.write(path, arcname=path.name)
    log.info("frames_archived", frames=len(paths), path=str(dst), bytes=dst.stat().st_size)
    return dst


def frames_to_video(
    directory: Path | str,
    destination: Path | str,
    *,
    fps: float = 25.0,
    container: str = "webm",
    alpha: bool = True,
    pattern: str = "frame_%06d.png",
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Re-assemble a frame sequence into a video (the inverse of ``mode="frames"``)."""
    src = Path(directory)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        shutil.which(ffmpeg) or ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(src / pattern),
        *encoder_args(container, alpha=alpha),
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        from cutoutml.pipelines.ffmpeg import FFmpegError

        raise FFmpegError(
            "frame sequence assembly failed", stderr=proc.stderr, returncode=proc.returncode
        )
    return dst


def temp_workdir(prefix: str = "cutoutml-video-") -> Path:
    """Create a temp directory for intermediate frames.

    Callers are responsible for cleanup; the Celery task uses a ``finally`` block so a
    failed job does not leak a directory of full-resolution PNGs.
    """
    return Path(tempfile.mkdtemp(prefix=prefix))
