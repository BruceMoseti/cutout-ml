"""Celery task definitions.

Every task here is designed around one uncomfortable fact: a Redis-backed broker with
``task_acks_late`` is **at-least-once**. A task body will occasionally run twice for the
same message - after a visibility-timeout expiry, after a worker is SIGKILLed between
finishing and acking, after a broker failover. So the tasks are written to make a second
execution harmless rather than to pretend it cannot happen:

* :class:`~services.inference.app.runner.JobRunner` returns the stored manifest
  immediately when the job is already ``succeeded``, so a replay is a database read.
* Output storage keys are deterministic (``results/{owner}/{job}/{name}.{ext}``), so a
  replay that *does* re-run overwrites its own objects instead of orphaning a second set.
* The database row, not the Celery result backend, is the source of truth for job state.
  Celery results expire after a day; the job row does not.

Retry policy is explicit rather than ``autoretry_for=(Exception,)``:

``NON_RETRYABLE``
    Fail now. A corrupt upload or an unknown model does not improve on attempt three,
    and retrying it three times triples the log noise for the same outcome.
``RETRYABLE``
    Exponential backoff with jitter, capped at :data:`MAX_RETRIES`.
``OOM``
    Retried like a transient failure, but the retry carries a **halved batch size** in
    its kwargs. Re-running an identical OOM is a guaranteed second OOM; shrinking the
    working set is the only thing that can change the outcome. The reduction is recorded
    on the ``inference_runs`` row so batch sizes can be tuned from data later.
"""

from __future__ import annotations

import uuid
from typing import Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from cutoutml.core.config import get_settings
from cutoutml.core.logging import bind_request_id, get_logger
from cutoutml.db.models import InferenceJob, JobStatus
from cutoutml.db.session import session_scope
from cutoutml.storage.factory import build_storage
from services.inference.app.celery_app import celery
from services.inference.app.errors import (
    Classification,
    FailureKind,
    classify,
    retry_delay,
)
from services.inference.app.runner import JobRunner

log = get_logger(__name__)

MAX_RETRIES = 3
"""Total attempts is ``MAX_RETRIES + 1``. Beyond that a "transient" failure is not."""

MIN_BATCH_SIZE = 1


def _halve(batch_size: int | None) -> int:
    """Next batch size to try after an OOM, floored at 1."""
    if not batch_size or batch_size <= MIN_BATCH_SIZE:
        return MIN_BATCH_SIZE
    return max(MIN_BATCH_SIZE, batch_size // 2)


def _apply_batch_override(job_id: uuid.UUID, batch_size: int) -> None:
    """Persist a reduced batch size onto the job's params before retrying.

    Written to the row rather than passed only through task kwargs so that the value
    survives a worker restart mid-retry, and so ``GET /jobs/{id}`` shows what the job
    is actually being run with.
    """
    with session_scope() as session:
        job = session.get(InferenceJob, job_id)
        if job is None:
            return
        params = dict(job.params or {})
        video = dict(params.get("video") or {})
        video["batch_size"] = batch_size
        params["video"] = video
        params["batch_size_override"] = batch_size
        job.params = params


class JobTask(Task):
    """Base task that binds the job's request id into the log context.

    Correlation is the point: a request that fails in the worker should be findable
    from the ``X-Request-ID`` the client saw, and that only works if the id travels
    with the message and is bound before the first log line.
    """

    abstract = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        request_id = kwargs.get("request_id")
        bind_request_id(str(request_id) if request_id else None)
        try:
            return super().__call__(*args, **kwargs)
        finally:
            bind_request_id(None)


def _run_job(
    task: Task,
    job_id: str,
    *,
    request_id: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Shared body for the image and video tasks.

    Image and video differ only in which queue they land on and which pipeline the
    runner dispatches to; the retry/idempotency machinery is identical, so it lives in
    one place instead of being copy-pasted and drifting.
    """
    identifier = uuid.UUID(job_id)
    settings = get_settings()
    storage = build_storage(settings)

    if batch_size is not None:
        _apply_batch_override(identifier, batch_size)

    with session_scope() as session:
        runner = JobRunner(session, storage, settings=settings)
        try:
            return runner.execute(identifier)
        except SoftTimeLimitExceeded as exc:
            # The soft limit fires inside the task so temp frames get cleaned up by the
            # pipeline's context managers. It is not retried: a job that needs more than
            # an hour needs a different resolution or a shorter clip, not another hour.
            classification = Classification(
                FailureKind.NON_RETRYABLE,
                "time_limit_exceeded",
                f"job exceeded the soft time limit: {exc}",
            )
            runner.mark_permanently_failed(identifier, classification)
            raise
        except BaseException as exc:
            classification = classify(exc)
            attempt = task.request.retries + 1
            log.warning(
                "task_failure",
                job_id=job_id,
                attempt=attempt,
                classification=classification.as_dict(),
            )

            if not classification.retryable or attempt > MAX_RETRIES:
                if classification.retryable:
                    runner.mark_permanently_failed(identifier, classification)
                raise

            countdown = retry_delay(attempt)
            retry_kwargs: dict[str, Any] = {"request_id": request_id}
            if classification.is_oom:
                current = batch_size or _current_batch_size(session, identifier)
                retry_kwargs["batch_size"] = _halve(current)
                log.warning(
                    "oom_retry_scheduled",
                    job_id=job_id,
                    from_batch_size=current,
                    to_batch_size=retry_kwargs["batch_size"],
                )
            raise task.retry(
                exc=exc,
                countdown=countdown,
                max_retries=MAX_RETRIES,
                kwargs={
                    "job_id": job_id,
                    **retry_kwargs,
                },
            ) from exc


def _current_batch_size(session: Any, job_id: uuid.UUID) -> int:
    """Batch size the failing attempt used, from the job row."""
    job = session.get(InferenceJob, job_id)
    if job is None:
        return get_settings().video_batch_size
    params = job.params or {}
    video = params.get("video") or {}
    return int(video.get("batch_size") or get_settings().video_batch_size)


@celery.task(
    base=JobTask,
    bind=True,
    name="cutoutml.process_image",
    max_retries=MAX_RETRIES,
    acks_late=True,
)
def process_image(
    self: Task,
    job_id: str,
    *,
    request_id: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run an image segmentation job."""
    return _run_job(self, job_id, request_id=request_id, batch_size=batch_size)


@celery.task(
    base=JobTask,
    bind=True,
    name="cutoutml.process_video",
    max_retries=MAX_RETRIES,
    acks_late=True,
)
def process_video(
    self: Task,
    job_id: str,
    *,
    request_id: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run a video segmentation job."""
    return _run_job(self, job_id, request_id=request_id, batch_size=batch_size)


@celery.task(bind=True, name="cutoutml.requeue_stuck", acks_late=True)
def requeue_stuck(self: Task, *, older_than_seconds: int = 3600) -> dict[str, int]:  # noqa: ARG001
    """Re-dispatch jobs left ``queued``/``running`` by a worker that died.

    ``task_reject_on_worker_lost`` covers a worker killed while the broker still holds
    the message, but not the window after the broker's visibility timeout has already
    expired and the redelivery itself was lost. This is the sweeper for that case; it is
    safe precisely because the tasks are idempotent.
    """
    import datetime as dt

    from services.inference.app.runner import pending_jobs

    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=older_than_seconds)
    requeued = 0
    with session_scope() as session:
        for job in pending_jobs(session):
            updated = job.started_at or job.queued_at or job.created_at
            if updated and updated > cutoff:
                continue
            task = process_video if job.kind == "video" else process_image
            async_result = task.apply_async(kwargs={"job_id": str(job.id)}, queue=job.queue)
            job.celery_task_id = async_result.id
            job.status = JobStatus.QUEUED.value
            requeued += 1
    log.info("requeue_stuck_done", requeued=requeued)
    return {"requeued": requeued}


__all__ = ["MAX_RETRIES", "process_image", "process_video", "requeue_stuck"]
