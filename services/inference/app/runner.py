"""Job execution: the part of the worker that does not know it is a worker.

``tasks.py`` owns Celery concerns (retries, acks, routing). Everything here is plain
Python operating on a database session, a storage backend and a job row, which is what
makes it testable without a broker and reusable from a script.

Idempotency
-----------
Brokers are at-least-once, so a task *will* occasionally be delivered twice - a worker
killed after finishing but before acking is the common case. Two mechanisms make that
harmless:

1. **A terminal job short-circuits.** If the job already succeeded and has a result, the
   stored result is returned untouched.
2. **Output keys are derived from the job id, not random.** Re-executing overwrites the
   same objects rather than creating a second set, so a duplicate delivery cannot leave
   orphaned bytes behind or change which key the API hands out.

Model caching
-------------
Loading a model is 50-500 ms and is pure overhead when consecutive jobs use the same
one. Models are cached per worker *process* keyed by (name, device, precision). The cache
is bounded because each entry holds device memory: on a GPU worker an unbounded cache is
an OOM waiting for the third distinct model.

CUDA OOM handling
-----------------
On OOM the batch size is halved and the job retried in-process, down to 1, and each
attempt is recorded on the run row (``oom_retry``, ``batch_size``). That turns "how often
do we OOM and at what batch size does 4K video actually fit" into a SQL query. Below
batch size 1 there is nothing left to halve, so the failure becomes permanent.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import socket
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from sqlalchemy import select
from sqlalchemy.orm import Session

from cutoutml.core.config import Settings, get_settings
from cutoutml.core.devices import describe_device, peak_memory_bytes, reset_peak_memory
from cutoutml.core.imaging import decode_image
from cutoutml.core.logging import get_logger
from cutoutml.core.refine import RefineConfig
from cutoutml.db.models import Asset, AssetKind, InferenceJob, InferenceRun, JobStatus
from cutoutml.models.base import SegmentationModel
from cutoutml.models.registry import get_model
from cutoutml.pipelines.image import ImagePipeline, ImageRequest, OutputKind
from cutoutml.pipelines.video import VideoPipeline, VideoProgress, VideoRequest
from cutoutml.storage.base import Storage
from services.inference.app.errors import Classification, FailureKind, classify

log = get_logger(__name__)

MODEL_CACHE_SIZE = 2
"""Two entries: enough for a mixed image/video worker, small enough to bound VRAM."""

PROGRESS_MIN_INTERVAL_SECONDS = 1.0
"""Throttle for progress writes. A 30 fps 2-minute video would otherwise be 3600 UPDATEs."""

CONTENT_TYPES: dict[str, str] = {
    "png": "image/png",
    "webp": "image/webp",
    "jpg": "image/jpeg",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "zip": "application/zip",
}

_OUTPUT_EXTENSIONS: dict[str, str] = {
    "transparent_png": "png",
    "transparent_webp": "webp",
    "mask_png": "png",
    "color_composite": "png",
    "background_composite": "png",
    "blurred_background": "png",
}


# ---------------------------------------------------------------------------
# model cache
# ---------------------------------------------------------------------------


class ModelCache:
    """Bounded LRU cache of loaded models, one per worker process."""

    def __init__(self, max_entries: int = MODEL_CACHE_SIZE) -> None:
        self.max_entries = max(1, max_entries)
        self._entries: OrderedDict[tuple[str, str, str], SegmentationModel] = OrderedDict()

    def get(self, name: str, *, device: str, precision: str) -> SegmentationModel:
        key = (name, device, precision)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached

        model = get_model(name, device=device, precision=precision)  # type: ignore[arg-type]
        self._entries[key] = model
        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            log.info("model_cache_evict", model=evicted.name)
            evicted.unload()
        return model

    def clear(self) -> None:
        for model in self._entries.values():
            model.unload()
        self._entries.clear()


_CACHE = ModelCache()


def model_cache() -> ModelCache:
    return _CACHE


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------


def _refine_config(params: dict[str, Any], default: RefineConfig) -> RefineConfig:
    """Build a refine config from API params, ignoring keys the dataclass lacks.

    Filtering rather than passing straight through: the API schema and the internal
    dataclass are allowed to diverge, and an unknown key should not become a 500.
    """
    raw = params.get("refine") or {}
    if not isinstance(raw, dict):
        return default
    valid = {f.name for f in dataclasses.fields(RefineConfig)}
    return dataclasses.replace(default, **{k: v for k, v in raw.items() if k in valid})


def build_image_request(
    params: dict[str, Any], *, background: np.ndarray | None = None, max_pixels: int
) -> ImageRequest:
    """Reconstruct an :class:`ImageRequest` from the stored job params."""
    image_params = params.get("image") or {}
    outputs = tuple(image_params.get("outputs") or ("transparent_png", "mask_png"))
    colour = tuple(image_params.get("background_color") or (255, 255, 255))
    return ImageRequest(
        outputs=outputs,  # type: ignore[arg-type]
        background_color=(int(colour[0]), int(colour[1]), int(colour[2])),
        background_image=background,
        blur_sigma=float(image_params.get("blur_sigma", 12.0)),
        webp_quality=int(image_params.get("webp_quality", 90)),
        refine=_refine_config(image_params, RefineConfig.quality()),
        max_pixels=max_pixels,
    )


def build_video_request(
    params: dict[str, Any],
    *,
    background: np.ndarray | None = None,
    batch_size: int | None = None,
    frame_limit: int,
) -> VideoRequest:
    """Reconstruct a :class:`VideoRequest` from the stored job params."""
    video_params = params.get("video") or {}
    colour = tuple(video_params.get("background_color") or (0, 177, 64))
    return VideoRequest(
        mode=video_params.get("mode", "composite"),
        container=video_params.get("container", "mp4"),
        background_color=(int(colour[0]), int(colour[1]), int(colour[2])),
        background_image=background,
        blur_background=bool(video_params.get("blur_background", False)),
        blur_sigma=float(video_params.get("blur_sigma", 12.0)),
        smoothing=video_params.get("smoothing", "ema"),
        ema_weight=float(video_params.get("ema_weight", 0.65)),
        median_window=int(video_params.get("median_window", 3)),
        batch_size=int(batch_size or video_params.get("batch_size", 4)),
        refine=_refine_config(video_params, RefineConfig.fast()),
        max_frames=video_params.get("max_frames"),
        crf=int(video_params.get("crf", 23)),
        keep_audio=bool(video_params.get("keep_audio", True)),
        measure_flicker=bool(video_params.get("measure_flicker", False)),
        frame_limit=frame_limit,
    )


# ---------------------------------------------------------------------------
# storage keys
# ---------------------------------------------------------------------------


def result_key(job: InferenceJob, name: str, extension: str) -> str:
    """Deterministic output key: ``results/{owner}/{job}/{name}.{ext}``.

    Deterministic rather than random, unlike upload keys. That is the whole idempotency
    guarantee: a redelivered task overwrites its own outputs instead of producing a
    second set that nothing references. The owner and job ids are UUIDs, so the key is
    still unguessable, and authorisation is enforced in the database regardless.
    """
    return f"results/{job.owner_id}/{job.id}/{name}.{extension}"


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class JobOutcome:
    """What one successful execution produced."""

    result: dict[str, Any]
    duration_seconds: float
    batch_size: int
    oom_retries: int
    frames_processed: int | None
    device: str
    device_name: str
    peak_rss_bytes: int
    peak_vram_bytes: int | None
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


class JobRunner:
    """Executes one job end to end against a database session and storage."""

    def __init__(
        self,
        session: Session,
        storage: Storage,
        *,
        settings: Settings | None = None,
        cache: ModelCache | None = None,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
    ) -> None:
        self.session = session
        self.storage = storage
        self.settings = settings or get_settings()
        self.cache = cache or model_cache()
        self.ffmpeg = ffmpeg or self.settings.ffmpeg_binary
        self.ffprobe = ffprobe or self.settings.ffprobe_binary
        self._last_progress = 0.0

    # ------------------------------------------------------------------ helpers

    def load_job(self, job_id: uuid.UUID) -> InferenceJob:
        job = self.session.get(InferenceJob, job_id)
        if job is None:
            raise FileNotFoundError(f"job {job_id} does not exist")
        return job

    def _asset(self, job: InferenceJob) -> Asset:
        asset = self.session.get(Asset, job.asset_id)
        if asset is None:
            raise FileNotFoundError(f"asset {job.asset_id} does not exist")
        return asset

    def _background(self, job: InferenceJob) -> np.ndarray | None:
        """Decode the background image referenced by the job, if any.

        The API resolved and authorised the background asset when the job was created and
        stored its storage key; the worker never looks up an asset by user-supplied id,
        so a job cannot be used to read another tenant's object.
        """
        key = job.params.get("background_storage_key")
        if not key:
            return None
        return decode_image(self.storage.get(str(key)), apply_exif=True)

    def _publish_progress(self, job: InferenceJob, fraction: float, message: str) -> None:
        """Persist coarse progress, throttled by wall-clock time."""
        now = time.monotonic()
        if now - self._last_progress < PROGRESS_MIN_INTERVAL_SECONDS and fraction < 1.0:
            return
        self._last_progress = now
        job.progress = round(min(1.0, max(0.0, fraction)), 4)
        job.progress_message = message[:255]
        self.session.commit()

    # -------------------------------------------------------------------- public

    def execute(self, job_id: uuid.UUID, *, attempt: int | None = None) -> dict[str, Any]:
        """Run a job, updating its row and returning the result manifest.

        Raises the original exception after recording it, so the Celery layer can decide
        whether to retry based on :func:`services.inference.app.errors.classify`.
        """
        job = self.load_job(job_id)

        if job.status == JobStatus.SUCCEEDED.value and job.result:
            log.info("job_already_complete", job_id=str(job.id), reason="idempotent replay")
            return dict(job.result)
        if job.status == JobStatus.CANCELLED.value:
            log.info("job_cancelled_skip", job_id=str(job.id))
            return {"cancelled": True}

        job.attempts = (job.attempts or 0) + 1
        job.status = JobStatus.RUNNING.value
        job.started_at = dt.datetime.now(dt.UTC)
        job.error_code = None
        job.error_message = None
        run = InferenceRun(
            job_id=job.id,
            attempt=attempt or job.attempts,
            status=JobStatus.RUNNING.value,
            worker_hostname=socket.gethostname()[:255],
            model_name=job.model_name,
            precision=job.precision,
        )
        self.session.add(run)
        self.session.commit()

        started = time.perf_counter()
        try:
            outcome = self._dispatch(job)
        except BaseException as exc:
            classification = classify(exc)
            self._record_failure(job, run, classification, time.perf_counter() - started)
            raise

        self._record_success(job, run, outcome)
        return outcome.result

    # ------------------------------------------------------------------ dispatch

    def _dispatch(self, job: InferenceJob) -> JobOutcome:
        if job.kind == AssetKind.VIDEO.value:
            return self._run_video(job)
        return self._run_image(job)

    # --------------------------------------------------------------------- image

    def _run_image(self, job: InferenceJob) -> JobOutcome:
        asset = self._asset(job)
        model = self.cache.get(job.model_name, device=self._device(job), precision=job.precision)
        device_info = describe_device(model.device)
        reset_peak_memory(model.device)

        request = build_image_request(
            job.params,
            background=self._background(job),
            max_pixels=self.settings.max_image_pixels,
        )
        payload = self.storage.get(asset.storage_key)

        started = time.perf_counter()
        pipeline = ImagePipeline(model)
        result = pipeline.process_bytes(payload, request)
        duration = time.perf_counter() - started

        outputs: list[dict[str, Any]] = []
        for kind, data in result.outputs.items():
            extension = _OUTPUT_EXTENSIONS.get(kind, "png")
            key = result_key(job, kind, extension)
            content_type = CONTENT_TYPES.get(extension, "application/octet-stream")
            self.storage.put(key, data, content_type=content_type)
            outputs.append(
                {
                    "kind": kind,
                    "storage_key": key,
                    "size_bytes": len(data),
                    "content_type": content_type,
                }
            )

        self._publish_progress(job, 1.0, "complete")
        return JobOutcome(
            result={
                "kind": "image",
                "model": model.name,
                "width": result.width,
                "height": result.height,
                "alpha_coverage": round(result.alpha_coverage, 5),
                "outputs": outputs,
                "timings_ms": {k: round(v, 3) for k, v in result.timings_ms.items()},
                "content_sha256": result.content_sha256,
            },
            duration_seconds=duration,
            batch_size=1,
            oom_retries=0,
            frames_processed=1,
            device=str(model.device),
            device_name=device_info.name,
            peak_rss_bytes=int(psutil.Process().memory_info().rss),
            peak_vram_bytes=peak_memory_bytes(model.device),
            metrics={"timings_ms": result.timings_ms, "outputs": len(outputs)},
        )

    # --------------------------------------------------------------------- video

    def _run_video(self, job: InferenceJob) -> JobOutcome:
        asset = self._asset(job)
        model = self.cache.get(job.model_name, device=self._device(job), precision=job.precision)
        device_info = describe_device(model.device)
        background = self._background(job)

        workdir = Path(tempfile.mkdtemp(prefix=f"cutoutml-job-{job.id.hex[:8]}-"))
        try:
            source = workdir / "source.bin"
            # Streamed, not read into memory: a 200 MB upload should not become a 200 MB
            # Python bytes object on a worker that also holds a model.
            with self.storage.open(asset.storage_key) as src, source.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)

            batch_size = int((job.params.get("video") or {}).get("batch_size", 4))
            oom_retries = 0
            reset_peak_memory(model.device)
            started = time.perf_counter()

            while True:
                request = build_video_request(
                    job.params,
                    background=background,
                    batch_size=batch_size,
                    frame_limit=self.settings.max_video_frames,
                )
                destination = self._video_destination(workdir, request)
                try:
                    result = VideoPipeline(model).process(
                        source,
                        destination,
                        request,
                        on_progress=lambda p: self._on_video_progress(job, p),
                        ffmpeg=self.ffmpeg,
                        ffprobe=self.ffprobe,
                    )
                    break
                except BaseException as exc:
                    classification = classify(exc)
                    if classification.kind is not FailureKind.OOM or batch_size <= 1:
                        raise
                    oom_retries += 1
                    batch_size = max(1, batch_size // 2)
                    log.warning(
                        "video_oom_retry",
                        job_id=str(job.id),
                        new_batch_size=batch_size,
                        attempt=oom_retries,
                        error=classification.message[:200],
                    )
                    self._free_device_memory(model)

            duration = time.perf_counter() - started
            deliverable = result.deliverable
            extension = deliverable.suffix.lstrip(".") or "bin"
            key = result_key(job, "output", extension)
            content_type = CONTENT_TYPES.get(extension, "application/octet-stream")
            with deliverable.open("rb") as fh:
                self.storage.put_stream(key, fh, content_type=content_type)

            summary = result.summary()
            self._publish_progress(job, 1.0, "complete")
            return JobOutcome(
                result={
                    "kind": "video",
                    "model": model.name,
                    "outputs": [
                        {
                            "kind": f"video_{result.mode}",
                            "storage_key": key,
                            "size_bytes": deliverable.stat().st_size,
                            "content_type": content_type,
                        }
                    ],
                    "frames_processed": result.frames_processed,
                    "fps": round(result.fps, 3),
                    "smoothing": result.smoothing,
                    "has_alpha": result.has_alpha,
                    "container": result.container,
                    "flicker_raw": result.flicker_raw,
                    "flicker_smoothed": result.flicker_smoothed,
                    "source": summary["source"],
                    "batch_size": batch_size,
                    "oom_retries": oom_retries,
                },
                duration_seconds=duration,
                batch_size=batch_size,
                oom_retries=oom_retries,
                frames_processed=result.frames_processed,
                device=str(model.device),
                device_name=device_info.name,
                peak_rss_bytes=int(psutil.Process().memory_info().rss),
                peak_vram_bytes=peak_memory_bytes(model.device),
                metrics={
                    "fps": result.fps,
                    "frames": result.frames_processed,
                    "output_bytes": result.output_bytes,
                },
            )
        finally:
            # Full-resolution intermediate frames are large; a failed job must not leave
            # them behind or a busy worker fills its disk within hours.
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)

    def _video_destination(self, workdir: Path, request: VideoRequest) -> Path:
        from cutoutml.pipelines.ffmpeg import container_extension

        if request.mode == "frames":
            return workdir / "frames"
        return workdir / f"output.{container_extension(request.container)}"

    def _on_video_progress(self, job: InferenceJob, progress: VideoProgress) -> None:
        self._publish_progress(
            job,
            progress.fraction,
            f"{progress.stage}: {progress.frames_done}/{progress.frames_total} frames "
            f"({progress.fps:.1f} fps)",
        )

    @staticmethod
    def _free_device_memory(model: SegmentationModel) -> None:
        """Release cached blocks before an OOM retry.

        PyTorch's caching allocator holds freed blocks, so the retry can OOM on memory
        that is technically free. ``empty_cache`` returns it to the driver.
        """
        import gc

        import torch

        gc.collect()
        if model.device.type == "cuda":
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------- record

    def _device(self, job: InferenceJob) -> str:
        requested = job.params.get("device")
        return str(requested) if requested else self.settings.device

    def _record_success(self, job: InferenceJob, run: InferenceRun, outcome: JobOutcome) -> None:
        now = dt.datetime.now(dt.UTC)
        job.status = JobStatus.SUCCEEDED.value
        job.result = outcome.result
        job.progress = 1.0
        job.progress_message = "complete"
        job.finished_at = now
        run.status = JobStatus.SUCCEEDED.value
        run.finished_at = now
        run.device = outcome.device
        run.device_name = outcome.device_name[:128]
        run.batch_size = outcome.batch_size
        run.oom_retry = outcome.oom_retries > 0
        run.retryable_error = None
        run.duration_seconds = round(outcome.duration_seconds, 4)
        run.frames_processed = outcome.frames_processed
        run.peak_rss_bytes = outcome.peak_rss_bytes
        run.peak_vram_bytes = outcome.peak_vram_bytes
        run.metrics = outcome.metrics
        self.session.commit()
        log.info(
            "job_succeeded",
            job_id=str(job.id),
            kind=job.kind,
            model=job.model_name,
            seconds=round(outcome.duration_seconds, 3),
            attempt=run.attempt,
            oom_retries=outcome.oom_retries,
        )

    def _record_failure(
        self,
        job: InferenceJob,
        run: InferenceRun,
        classification: Classification,
        elapsed: float,
    ) -> None:
        """Persist a failure.

        The job is only marked ``failed`` for non-retryable classifications; a retryable
        one stays ``running`` so a client polling ``GET /jobs/{id}`` sees "still working"
        rather than a failure that is about to be retried anyway. The run row records the
        attempt either way.

        The session is rolled back first: whatever raised may have left it in a failed
        transaction, and these writes have to land.
        """
        self.session.rollback()
        run = self.session.merge(run)
        job = self.session.merge(job)
        now = dt.datetime.now(dt.UTC)
        run.status = JobStatus.FAILED.value
        run.finished_at = now
        run.error_code = classification.code[:64]
        run.error_message = classification.message
        run.retryable_error = classification.retryable
        run.oom_retry = classification.is_oom
        run.duration_seconds = round(elapsed, 4)
        job.error_code = classification.code[:64]
        job.error_message = classification.message
        if not classification.retryable:
            job.status = JobStatus.FAILED.value
            job.finished_at = now
        self.session.commit()
        log.warning(
            "job_failed",
            job_id=str(job.id),
            attempt=run.attempt,
            code=classification.code,
            retryable=classification.retryable,
            error=classification.message[:500],
        )

    def mark_permanently_failed(self, job_id: uuid.UUID, classification: Classification) -> None:
        """Mark a job failed after retries are exhausted."""
        self.session.rollback()
        job = self.session.get(InferenceJob, job_id)
        if job is None:
            return
        job.status = JobStatus.FAILED.value
        job.finished_at = dt.datetime.now(dt.UTC)
        job.error_code = classification.code[:64]
        job.error_message = f"retries exhausted: {classification.message}"
        self.session.commit()


def pending_jobs(session: Session, *, limit: int = 100) -> list[InferenceJob]:
    """Jobs that were queued but never reached a terminal state.

    Used by the ``requeue-stuck`` maintenance command: a worker killed between ack and
    completion leaves rows in ``running`` that nothing will ever finish.
    """
    stmt = (
        select(InferenceJob)
        .where(InferenceJob.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]))
        .order_by(InferenceJob.created_at)
        .limit(limit)
    )
    return list(session.execute(stmt).scalars())


__all__ = [
    "CONTENT_TYPES",
    "JobOutcome",
    "JobRunner",
    "ModelCache",
    "OutputKind",
    "build_image_request",
    "build_video_request",
    "model_cache",
    "pending_jobs",
    "result_key",
]
