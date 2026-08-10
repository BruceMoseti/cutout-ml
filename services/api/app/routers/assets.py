"""Asset upload, inspection, processing and result retrieval.

Upload is two-phase, which is what makes large files workable:

1. ``POST /v1/assets/upload-url`` creates the row in ``awaiting_upload`` and returns
   where to send the bytes.
2. The client PUTs the bytes, then the asset flips to ``ready``.

With the S3 backend step 2 is a genuine presigned PUT straight to the bucket and the
client calls ``POST /v1/assets/{id}/complete`` afterwards, so a 200 MB video never
transits the API process. With the local backend there is no independent HTTP endpoint to
sign for, so the returned URL points back at ``PUT /v1/assets/{id}/content`` and the API
does the validation and the write itself. Either way the *client* code is the same shape,
and either way the bytes are validated before the asset is marked ready.

``POST /v1/assets`` is the single-request multipart convenience path. The browser UI uses
it because a drag-and-drop upload of a 3 MB PNG does not benefit from two round-trips.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from sqlalchemy import func, select

from cutoutml.core.logging import get_logger
from cutoutml.core.queues import select_queue
from cutoutml.db.models import Asset, AssetKind, AssetStatus, InferenceJob, JobStatus
from cutoutml.models.registry import ModelNotFoundError, resolve_spec
from cutoutml.storage.base import ObjectNotFoundError, build_storage_key
from services.api.app.deps import (
    CurrentUser,
    MetricsDep,
    OwnedAsset,
    RateLimited,
    SessionDep,
    SettingsDep,
    StorageDep,
    resolve_background_asset,
)
from services.api.app.errors import ApiError, not_found
from services.api.app.schemas import (
    AssetListResponse,
    AssetResponse,
    JobResponse,
    ProcessImageOptions,
    ProcessRequest,
    ProcessVideoOptions,
    ResultOutput,
    ResultResponse,
    UploadUrlRequest,
    UploadUrlResponse,
)
from services.api.app.uploads import (
    UploadValidationError,
    content_hash,
    safe_filename,
    validate_upload,
)

log = get_logger(__name__)

router = APIRouter(prefix="/v1/assets", tags=["assets"], dependencies=[RateLimited])

_EXTENSION_BY_KIND = {"image": "png", "video": "mp4"}


def _as_api_error(exc: UploadValidationError) -> ApiError:
    return ApiError(exc.status_code, exc.code, str(exc))


def _store_and_finalise(
    *,
    asset: Asset,
    data: bytes,
    session: SessionDep,
    storage: StorageDep,
    settings: SettingsDep,
    metrics: MetricsDep,
    declared_content_type: str | None,
) -> Asset:
    """Validate the payload, write it, and flip the asset to ``ready``.

    Shared by the multipart and the two-phase paths so there is exactly one place where
    an asset becomes usable, and therefore exactly one place validation can be skipped
    from - which is nowhere.
    """
    try:
        detected, properties = validate_upload(
            data,
            declared_content_type=declared_content_type,
            filename=asset.original_filename,
            max_image_bytes=settings.max_upload_bytes,
            max_video_bytes=settings.max_video_upload_bytes,
            max_image_pixels=settings.max_image_pixels,
            expected_kind=asset.kind,  # type: ignore[arg-type]
        )
    except UploadValidationError as exc:
        metrics.upload_rejections.labels(exc.code).inc()
        asset.status = AssetStatus.FAILED.value
        session.commit()
        raise _as_api_error(exc) from exc

    # The key was allocated with a guessed extension at reservation time; correct it to
    # the sniffed one now that the bytes are known.
    key = asset.storage_key.rsplit(".", 1)[0] + f".{detected.extension}"
    storage.put(key, data, content_type=detected.mime)

    asset.storage_key = key
    asset.status = AssetStatus.READY.value
    asset.content_type = detected.mime
    asset.content_sha256 = content_hash(data)
    asset.size_bytes = len(data)
    asset.width = properties.get("width")
    asset.height = properties.get("height")
    session.commit()

    metrics.uploads.labels(detected.kind).inc()
    log.info(
        "asset_ready",
        asset_id=str(asset.id),
        kind=detected.kind,
        mime=detected.mime,
        bytes=len(data),
    )
    return asset


# ------------------------------------------------------------------ upload


@router.post(
    "/upload-url",
    response_model=UploadUrlResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Reserve an asset and get somewhere to PUT the bytes",
)
def create_upload_url(
    payload: UploadUrlRequest,
    request: Request,
    session: SessionDep,
    storage: StorageDep,
    settings: SettingsDep,
    user: CurrentUser,
) -> UploadUrlResponse:
    limit = (
        settings.max_upload_bytes if payload.kind == "image" else settings.max_video_upload_bytes
    )
    if payload.size_bytes is not None and payload.size_bytes > limit:
        raise ApiError(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "payload_too_large",
            f"declared size {payload.size_bytes} exceeds the {payload.kind} limit of {limit}",
        )

    key = build_storage_key(
        user_id=str(user.id),
        kind="uploads",
        extension=_EXTENSION_BY_KIND[payload.kind],
    )
    asset = Asset(
        owner_id=user.id,
        kind=payload.kind,
        status=AssetStatus.AWAITING_UPLOAD.value,
        storage_backend=storage.backend,
        storage_key=key,
        original_filename=safe_filename(payload.filename),
        content_type=payload.content_type,
        size_bytes=payload.size_bytes or 0,
    )
    session.add(asset)
    session.commit()

    presigned = storage.presign_upload(
        key,
        content_type=payload.content_type,
        max_bytes=limit,
        expires_in=settings.presign_expiry_seconds,
    )
    upload_url = presigned.url
    if storage.backend == "local":
        # The local backend cannot sign a URL for an endpoint it does not own, so point
        # the client back at this API, which authorises and validates normally.
        upload_url = str(request.url_for("upload_asset_content", asset_id=asset.id))

    return UploadUrlResponse(
        asset_id=asset.id,
        storage_key=key,
        upload_url=upload_url,
        method=presigned.method,
        headers=presigned.headers,
        expires_at=presigned.expires_at,
        max_bytes=limit,
    )


@router.put(
    "/{asset_id}/content",
    response_model=AssetResponse,
    name="upload_asset_content",
    summary="Upload the bytes for a reserved asset",
)
async def upload_asset_content(
    request: Request,
    asset: OwnedAsset,
    session: SessionDep,
    storage: StorageDep,
    settings: SettingsDep,
    metrics: MetricsDep,
) -> AssetResponse:
    """Accept the raw body for an asset reserved by ``/upload-url``.

    Re-uploading a ``ready`` asset is rejected: the content hash is part of the
    idempotency key of any job already created from it, so silently swapping the bytes
    would make a completed job's result inconsistent with its input.
    """
    if asset.status == AssetStatus.READY.value:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "asset_already_uploaded",
            "this asset already has content; create a new asset instead",
        )

    limit = (
        settings.max_upload_bytes
        if asset.kind == AssetKind.IMAGE.value
        else settings.max_video_upload_bytes
    )
    body = await request.body()
    if len(body) > limit:
        metrics.upload_rejections.labels("payload_too_large").inc()
        raise ApiError(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "payload_too_large",
            f"body of {len(body)} bytes exceeds the limit of {limit}",
        )

    stored = _store_and_finalise(
        asset=asset,
        data=body,
        session=session,
        storage=storage,
        settings=settings,
        metrics=metrics,
        declared_content_type=request.headers.get("content-type"),
    )
    return AssetResponse.model_validate(stored)


@router.post(
    "",
    response_model=AssetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an asset in one multipart request",
)
async def upload_asset(
    session: SessionDep,
    storage: StorageDep,
    settings: SettingsDep,
    metrics: MetricsDep,
    user: CurrentUser,
    file: Annotated[UploadFile, File(description="The image or video to upload")],
    kind: Annotated[str, Form(description="'image' or 'video'")] = "image",
) -> AssetResponse:
    if kind not in {"image", "video"}:
        raise ApiError(status.HTTP_400_BAD_REQUEST, "bad_request", "kind must be image or video")

    limit = settings.max_upload_bytes if kind == "image" else settings.max_video_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        metrics.upload_rejections.labels("payload_too_large").inc()
        raise ApiError(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "payload_too_large",
            f"file exceeds the {kind} limit of {limit} bytes",
        )

    asset = Asset(
        owner_id=user.id,
        kind=kind,
        status=AssetStatus.AWAITING_UPLOAD.value,
        storage_backend=storage.backend,
        storage_key=build_storage_key(
            user_id=str(user.id), kind="uploads", extension=_EXTENSION_BY_KIND[kind]
        ),
        original_filename=safe_filename(file.filename),
        content_type=file.content_type,
    )
    session.add(asset)
    session.commit()

    stored = _store_and_finalise(
        asset=asset,
        data=data,
        session=session,
        storage=storage,
        settings=settings,
        metrics=metrics,
        declared_content_type=file.content_type,
    )
    return AssetResponse.model_validate(stored)


@router.post(
    "/{asset_id}/complete",
    response_model=AssetResponse,
    summary="Mark a presigned upload complete (S3 backend)",
)
def complete_upload(
    asset: OwnedAsset,
    session: SessionDep,
    storage: StorageDep,
    settings: SettingsDep,
    metrics: MetricsDep,
) -> AssetResponse:
    """Validate bytes that were PUT straight to object storage.

    The client uploaded without passing through this process, so nothing has been
    validated yet. The object is fetched back and run through exactly the same checks as
    a multipart upload - trusting a direct-to-bucket PUT is how an "image" endpoint ends
    up storing executables.
    """
    if asset.status == AssetStatus.READY.value:
        return AssetResponse.model_validate(asset)

    try:
        data = storage.get(asset.storage_key)
    except ObjectNotFoundError as exc:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "upload_incomplete",
            "no object exists at the reserved key; the upload did not complete",
        ) from exc

    stored = _store_and_finalise(
        asset=asset,
        data=data,
        session=session,
        storage=storage,
        settings=settings,
        metrics=metrics,
        declared_content_type=asset.content_type,
    )
    return AssetResponse.model_validate(stored)


# ------------------------------------------------------------------ read


@router.get("", response_model=AssetListResponse, summary="List the caller's assets")
def list_assets(
    session: SessionDep,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    kind: Annotated[str | None, Query(pattern="^(image|video)$")] = None,
) -> AssetListResponse:
    conditions = [Asset.owner_id == user.id, Asset.status != AssetStatus.DELETED.value]
    if kind:
        conditions.append(Asset.kind == kind)

    total = session.execute(select(func.count()).select_from(Asset).where(*conditions)).scalar_one()
    rows = session.execute(
        select(Asset)
        .where(*conditions)
        .order_by(Asset.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).scalars()
    return AssetListResponse(
        items=[AssetResponse.model_validate(a) for a in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@router.get("/{asset_id}", response_model=AssetResponse, summary="Asset metadata")
def get_asset(asset: OwnedAsset) -> AssetResponse:
    return AssetResponse.model_validate(asset)


@router.get("/{asset_id}/content", summary="Download the original bytes")
def get_asset_content(asset: OwnedAsset, storage: StorageDep) -> Response:
    """Stream the original upload back.

    Served through the API rather than by a public URL so that the ownership check
    applies to reads as well as writes. For a production S3 deployment this would return
    a 302 to a short-lived presigned GET instead, which keeps the bytes off the API path.
    """
    if asset.status != AssetStatus.READY.value:
        raise ApiError(status.HTTP_409_CONFLICT, "asset_not_ready", f"asset is {asset.status}")
    try:
        data = storage.get(asset.storage_key)
    except ObjectNotFoundError as exc:
        raise not_found("object", asset.storage_key) from exc
    return Response(
        content=data,
        media_type=asset.content_type or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete an asset")
def delete_asset(asset: OwnedAsset, session: SessionDep, storage: StorageDep) -> Response:
    """Soft-delete the row and hard-delete the object.

    The row is kept so that jobs referencing it still resolve and so the audit trail
    survives; the bytes go immediately, because that is what a user means by delete.
    """
    asset.status = AssetStatus.DELETED.value
    session.commit()
    try:
        storage.delete(asset.storage_key)
    except Exception as exc:  # noqa: BLE001 - the row is already marked deleted
        log.warning("asset_object_delete_failed", asset_id=str(asset.id), error=str(exc))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ------------------------------------------------------------------ process


def _idempotency_key(asset: Asset, payload: ProcessRequest) -> str:
    """Derive a stable key from the request when the client did not supply one.

    Content hash plus the exact parameter set: submitting the same bytes with the same
    options twice is almost always a double-click or a client retry, not a genuine
    request for a second identical output. A client that really wants a fresh run passes
    its own ``idempotency_key``.
    """
    if payload.idempotency_key:
        return payload.idempotency_key[:128]

    material = json.dumps(
        {
            "asset": str(asset.id),
            "sha256": asset.content_sha256,
            "body": payload.model_dump(mode="json", exclude={"idempotency_key"}),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


@router.post(
    "/{asset_id}/process",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a segmentation job",
)
def process_asset(
    payload: ProcessRequest,
    asset: OwnedAsset,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    metrics: MetricsDep,
    user: CurrentUser,
    response: Response,
) -> JobResponse:
    """Create (or return) a job and dispatch it to the right queue."""
    if asset.status != AssetStatus.READY.value:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "asset_not_ready",
            f"asset is {asset.status}; upload its content before processing",
        )

    model_name = payload.model or settings.default_model
    try:
        spec = resolve_spec(model_name)
    except ModelNotFoundError as exc:
        raise ApiError(
            status.HTTP_400_BAD_REQUEST,
            "unknown_model",
            str(exc),
            details={"model": model_name},
        ) from exc
    if not spec.requires_weights and "gpu-only" in spec.tags:
        raise ApiError(
            status.HTTP_400_BAD_REQUEST,
            "model_unavailable",
            f"model {model_name} requires hardware this deployment does not have",
        )

    options = payload.image if asset.kind == AssetKind.IMAGE.value else payload.video
    background_asset_id = getattr(options, "background_asset_id", None)
    background_key = resolve_background_asset(session, user, background_asset_id)

    params: dict[str, Any] = {"device": payload.device or settings.device}
    if background_key:
        params["background_storage_key"] = background_key
    if asset.kind == AssetKind.IMAGE.value:
        params["image"] = (payload.image or ProcessImageOptions()).model_dump(mode="json")
    else:
        params["video"] = (payload.video or ProcessVideoOptions()).model_dump(mode="json")

    key = _idempotency_key(asset, payload)
    existing = session.execute(
        select(InferenceJob).where(
            InferenceJob.owner_id == user.id, InferenceJob.idempotency_key == key
        )
    ).scalar_one_or_none()
    if existing is not None:
        metrics.idempotent_hits.inc()
        response.status_code = status.HTTP_200_OK
        log.info("job_idempotent_hit", job_id=str(existing.id))
        return JobResponse.model_validate(existing)

    queue = select_queue(asset.kind, device=payload.device)
    job = InferenceJob(
        owner_id=user.id,
        asset_id=asset.id,
        status=JobStatus.PENDING.value,
        kind=asset.kind,
        model_name=model_name,
        precision=payload.precision or settings.precision,
        queue=queue,
        idempotency_key=key,
        params=params,
    )
    session.add(job)
    session.commit()

    _dispatch(job, session, request_id=getattr(request.state, "request_id", None))
    session.commit()

    metrics.jobs_created.labels(job.kind, job.model_name, job.queue).inc()
    return JobResponse.model_validate(job)


def _dispatch(job: InferenceJob, session: SessionDep, *, request_id: str | None) -> None:
    """Send the job to Celery, or leave it pending if the broker is unreachable.

    A broker outage must not lose the request. The row is already committed, so the
    ``requeue_stuck`` maintenance task picks up anything left ``pending``; returning 202
    with ``status: pending`` is honest, whereas a 500 would make the client resubmit and
    (thanks to the idempotency key) get the same job anyway.
    """
    from services.inference.app.tasks import process_image, process_video

    task = process_video if job.kind == AssetKind.VIDEO.value else process_image
    try:
        async_result = task.apply_async(
            kwargs={"job_id": str(job.id), "request_id": request_id}, queue=job.queue
        )
    except Exception as exc:  # noqa: BLE001 - a broker outage must not lose the request
        log.error("job_dispatch_failed", job_id=str(job.id), error=str(exc))
        job.progress_message = "queued locally; broker unreachable"
        session.flush()
        return

    job.celery_task_id = async_result.id
    job.status = JobStatus.QUEUED.value
    job.queued_at = dt.datetime.now(dt.UTC)
    session.flush()
    log.info("job_queued", job_id=str(job.id), queue=job.queue, task_id=async_result.id)


# ------------------------------------------------------------------- result


@router.get(
    "/{asset_id}/result",
    response_model=ResultResponse,
    summary="The latest successful result for an asset",
)
def get_asset_result(
    asset: OwnedAsset, session: SessionDep, storage: StorageDep, settings: SettingsDep
) -> ResultResponse:
    """Return the newest succeeded job's outputs for this asset.

    404 distinguishes "no job has succeeded yet" from "no such asset" only in the error
    code, never by leaking whether an id exists for another tenant - ownership was
    already enforced by the dependency that resolved the asset.
    """
    job = session.execute(
        select(InferenceJob)
        .where(
            InferenceJob.asset_id == asset.id,
            InferenceJob.owner_id == asset.owner_id,
            InferenceJob.status == JobStatus.SUCCEEDED.value,
        )
        .order_by(InferenceJob.finished_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    if job is None or not job.result:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "result_not_available",
            "no completed job exists for this asset yet",
        )
    return build_result_response(job, storage, settings)


def build_result_response(
    job: InferenceJob, storage: StorageDep, settings: SettingsDep
) -> ResultResponse:
    """Turn a stored result manifest into a response with fetchable URLs."""
    manifest = job.result or {}
    outputs: list[ResultOutput] = []
    for item in manifest.get("outputs", []):
        key = str(item.get("storage_key", ""))
        try:
            url = storage.presign_download(key, expires_in=settings.presign_expiry_seconds)
        except Exception:  # noqa: BLE001 - fall back to the authenticated API route
            url = f"/v1/jobs/{job.id}/outputs/{item.get('kind')}"
        outputs.append(
            ResultOutput(
                kind=str(item.get("kind", "output")),
                storage_key=key,
                url=url,
                size_bytes=int(item.get("size_bytes", 0)),
                content_type=str(item.get("content_type", "application/octet-stream")),
            )
        )

    metrics = {
        k: v
        for k, v in manifest.items()
        if k
        in {
            "alpha_coverage",
            "timings_ms",
            "frames_processed",
            "fps",
            "smoothing",
            "has_alpha",
            "container",
            "flicker_raw",
            "flicker_smoothed",
            "batch_size",
            "oom_retries",
            "width",
            "height",
            "model",
        }
    }
    return ResultResponse(
        job_id=job.id,
        asset_id=job.asset_id,
        status=job.status,
        outputs=outputs,
        metrics=metrics or None,
    )


__all__ = ["build_result_response", "router"]
