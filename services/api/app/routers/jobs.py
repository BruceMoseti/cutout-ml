"""Job status, results, cancellation and per-output download."""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Path, Query, Response, status
from sqlalchemy import select

from cutoutml.core.logging import get_logger
from cutoutml.db.models import InferenceJob, InferenceRun, JobStatus
from cutoutml.storage.base import ObjectNotFoundError
from services.api.app.deps import (
    CurrentUser,
    OwnedJob,
    RateLimited,
    SessionDep,
    SettingsDep,
    StorageDep,
)
from services.api.app.errors import ApiError, not_found
from services.api.app.routers.assets import build_result_response
from services.api.app.schemas import (
    JobDetailResponse,
    JobResponse,
    JobRunResponse,
    ResultResponse,
)

log = get_logger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"], dependencies=[RateLimited])


@router.get("", response_model=list[JobResponse], summary="List the caller's jobs")
def list_jobs(
    session: SessionDep,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    job_status: Annotated[str | None, Query(alias="status")] = None,
) -> list[JobResponse]:
    conditions = [InferenceJob.owner_id == user.id]
    if job_status:
        if job_status not in {s.value for s in JobStatus}:
            raise ApiError(
                status.HTTP_400_BAD_REQUEST,
                "bad_request",
                f"unknown status {job_status!r}",
            )
        conditions.append(InferenceJob.status == job_status)
    rows = session.execute(
        select(InferenceJob)
        .where(*conditions)
        .order_by(InferenceJob.created_at.desc())
        .limit(limit)
    ).scalars()
    return [JobResponse.model_validate(job) for job in rows]


@router.get("/{job_id}", response_model=JobDetailResponse, summary="Job status and attempts")
def get_job(job: OwnedJob, session: SessionDep) -> JobDetailResponse:
    """Full job state including every execution attempt.

    The attempts are the interesting part: they show whether a success came on the first
    try or after an OOM retry at a smaller batch size, which is exactly what is invisible
    in a system that mutates a single row per job.
    """
    runs = session.execute(
        select(InferenceRun)
        .where(InferenceRun.job_id == job.id)
        .order_by(InferenceRun.attempt)
    ).scalars()
    detail = JobDetailResponse.model_validate(job)
    detail.runs = [JobRunResponse.model_validate(run) for run in runs]
    return detail


@router.get("/{job_id}/result", response_model=ResultResponse, summary="Job outputs")
def get_job_result(
    job: OwnedJob, storage: StorageDep, settings: SettingsDep
) -> ResultResponse:
    if job.status != JobStatus.SUCCEEDED.value or not job.result:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "job_not_complete",
            f"job is {job.status}; results are only available for succeeded jobs",
            details={"status": job.status, "progress": job.progress},
        )
    return build_result_response(job, storage, settings)


@router.get("/{job_id}/outputs/{kind}", summary="Download one output")
def download_output(
    job: OwnedJob,
    storage: StorageDep,
    kind: Annotated[str, Path(description="Output kind from the result manifest")],
) -> Response:
    """Stream one named output through the API.

    Exists so the local storage backend has a real download path, and so a deployment
    can keep result objects entirely private rather than relying on unguessable URLs.
    """
    manifest = job.result or {}
    match = next((o for o in manifest.get("outputs", []) if o.get("kind") == kind), None)
    if match is None:
        raise not_found("output", kind)
    try:
        data = storage.get(str(match["storage_key"]))
    except ObjectNotFoundError as exc:
        raise not_found("object", match["storage_key"]) from exc
    return Response(
        content=data,
        media_type=str(match.get("content_type", "application/octet-stream")),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.post("/{job_id}/cancel", response_model=JobResponse, summary="Cancel a job")
def cancel_job(job: OwnedJob, session: SessionDep) -> JobResponse:
    """Request cancellation.

    Revocation is best-effort by nature: a task already executing on a worker cannot be
    stopped safely from here, so the row is marked cancelled and the runner short-circuits
    on its next check. ``terminate=True`` is deliberately not used - SIGTERM mid-ffmpeg
    leaves partial output and temp directories behind.
    """
    if JobStatus(job.status).is_terminal:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "job_already_terminal",
            f"job is already {job.status}",
        )

    job.status = JobStatus.CANCELLED.value
    job.finished_at = dt.datetime.now(dt.UTC)
    job.progress_message = "cancelled by owner"
    session.commit()

    if job.celery_task_id:
        try:
            from services.inference.app.celery_app import celery

            celery.control.revoke(job.celery_task_id, terminate=False)
        except Exception as exc:
            log.warning("job_revoke_failed", job_id=str(job.id), error=str(exc))

    return JobResponse.model_validate(job)
