"""FastAPI dependencies: settings, session, storage, authentication, rate limiting.

Authorisation is expressed as a *dependency* rather than a check inside each handler.
``owned_asset`` and ``owned_job`` resolve an id to a row **scoped by owner in the SQL
query itself**, so a handler physically cannot receive another tenant's object. That is
the difference between "we remembered to check" and "there is no code path that skips the
check": the filter is in the WHERE clause, not in an ``if`` a future edit can drop.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Path, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from cutoutml.core.config import Settings, get_settings
from cutoutml.db.models import Asset, AssetStatus, InferenceJob, User
from cutoutml.db.session import get_session
from cutoutml.storage.base import Storage
from cutoutml.storage.factory import get_storage
from services.api.app.errors import ApiError, not_found
from services.api.app.metrics import Metrics, get_metrics
from services.api.app.ratelimit import RateLimiter
from services.api.app.security import TokenError, token_subject

# ``auto_error=False`` so a missing header produces our envelope instead of
# Starlette's ``{"detail": "Not authenticated"}``.
_bearer = HTTPBearer(auto_error=False, description="JWT access token")


def settings_dep() -> Settings:
    return get_settings()


def storage_dep() -> Storage:
    return get_storage()


def metrics_dep() -> Metrics:
    return get_metrics()


def limiter_dep(request: Request) -> RateLimiter:
    """The limiter built at startup and stored on app state."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:  # pragma: no cover - startup always sets it
        raise RuntimeError("rate limiter is not configured on app state")
    return limiter  # type: ignore[no-any-return]


SettingsDep = Annotated[Settings, Depends(settings_dep)]
SessionDep = Annotated[Session, Depends(get_session)]
StorageDep = Annotated[Storage, Depends(storage_dep)]
MetricsDep = Annotated[Metrics, Depends(metrics_dep)]
LimiterDep = Annotated[RateLimiter, Depends(limiter_dep)]


def request_id(request: Request) -> str:
    """The correlation id assigned by the request-context middleware."""
    return str(getattr(request.state, "request_id", "") or "")


RequestIdDep = Annotated[str, Depends(request_id)]


# --------------------------------------------------------------------- auth


def current_user(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> User:
    """Resolve the bearer token to an active user row.

    The user is loaded from the database on every request rather than trusted from the
    token's claims. A JWT is a bearer credential valid until it expires, so a
    deactivated account would otherwise keep working for up to the token TTL; one
    indexed primary-key lookup is a cheap price for immediate revocation.
    """
    if credentials is None or not credentials.credentials:
        raise ApiError(
            status.HTTP_401_UNAUTHORIZED,
            "unauthenticated",
            "an Authorization: Bearer <token> header is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = token_subject(credentials.credentials, settings=settings)
    except TokenError as exc:
        raise ApiError(
            status.HTTP_401_UNAUTHORIZED,
            "invalid_token",
            str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = session.get(User, user_id)
    if user is None or not user.is_active:
        raise ApiError(
            status.HTTP_401_UNAUTHORIZED,
            "inactive_user",
            "the account for this token no longer exists or is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Recorded on request state so the access log line carries the user without every
    # handler having to pass it.
    request.state.user_id = str(user.id)
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def admin_user(user: CurrentUser) -> User:
    """Require an admin principal."""
    if not user.is_admin:
        raise ApiError(
            status.HTTP_403_FORBIDDEN, "forbidden", "this endpoint requires an admin account"
        )
    return user


AdminUser = Annotated[User, Depends(admin_user)]


# --------------------------------------------------------------- rate limiting


def rate_limit(
    request: Request,
    limiter: LimiterDep,
    metrics: MetricsDep,
    settings: SettingsDep,
) -> None:
    """Consume one token for the caller, keyed by user when authenticated.

    Keyed by user id where possible and by client IP otherwise. IP-keying alone is
    wrong in both directions: it throttles everyone behind one NAT together, and it lets
    one account evade the limit by rotating source addresses.

    Applied as a router-level dependency rather than as middleware so that unauthenticated
    routes (``/health``, ``/metrics``) are excluded by construction and so the limiter can
    see the resolved user, which middleware running before routing cannot.
    """
    identity = getattr(request.state, "user_id", None)
    per_user_limit: int | None = None
    if identity is None:
        client = request.client.host if request.client else "unknown"
        identity = f"ip:{client}"
    else:
        identity = f"user:{identity}"
        per_user_limit = getattr(request.state, "user_rate_limit", None)

    decision = limiter.check(identity, limit=per_user_limit or settings.rate_limit_per_minute)
    if not decision.allowed:
        metrics.rate_limited.labels(decision.backend).inc()
        raise ApiError(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate_limited",
            f"rate limit of {decision.limit} requests/minute exceeded",
            details={"retry_after_seconds": round(decision.retry_after, 2)},
            headers=decision.headers(),
        )


RateLimited = Depends(rate_limit)


def user_rate_limit(request: Request, user: CurrentUser) -> None:
    """Publish the user's personal quota so :func:`rate_limit` can pick it up.

    Ordering inside a single dependency list is not guaranteed to be useful, so the
    per-user override is stashed on request state by this dependency and read by the
    limiter; when the two run in the other order the limiter simply uses the default.
    """
    request.state.user_rate_limit = user.rate_limit_per_minute


# ----------------------------------------------------------- owned resources


def owned_asset(
    asset_id: Annotated[uuid.UUID, Path(description="Asset id")],
    session: SessionDep,
    user: CurrentUser,
) -> Asset:
    """Fetch an asset that belongs to the caller, or 404.

    ``owner_id`` is part of the query, not a post-hoc comparison. Deleted assets are
    treated as absent so a delete is not reversible by a lucky read.
    """
    stmt = select(Asset).where(
        Asset.id == asset_id,
        Asset.owner_id == user.id,
        Asset.status != AssetStatus.DELETED.value,
    )
    asset = session.execute(stmt).scalar_one_or_none()
    if asset is None:
        raise not_found("asset", asset_id)
    return asset


OwnedAsset = Annotated[Asset, Depends(owned_asset)]


def owned_job(
    job_id: Annotated[uuid.UUID, Path(description="Job id")],
    session: SessionDep,
    user: CurrentUser,
) -> InferenceJob:
    """Fetch a job that belongs to the caller, or 404."""
    stmt = select(InferenceJob).where(
        InferenceJob.id == job_id, InferenceJob.owner_id == user.id
    )
    job = session.execute(stmt).scalar_one_or_none()
    if job is None:
        raise not_found("job", job_id)
    return job


OwnedJob = Annotated[InferenceJob, Depends(owned_job)]


def resolve_background_asset(
    session: Session, user: User, background_asset_id: uuid.UUID | None
) -> str | None:
    """Authorise a background-image asset and return its storage key.

    This exists because the worker must never resolve a user-supplied asset id itself:
    if it did, a request could name another tenant's asset as its "background" and
    receive those pixels back in the composited output. The API resolves it under the
    caller's ownership and stores only the resulting key on the job.
    """
    if background_asset_id is None:
        return None
    stmt = select(Asset).where(
        Asset.id == background_asset_id,
        Asset.owner_id == user.id,
        Asset.status == AssetStatus.READY.value,
    )
    asset = session.execute(stmt).scalar_one_or_none()
    if asset is None:
        raise not_found("asset", background_asset_id)
    return asset.storage_key


__all__ = [
    "AdminUser",
    "CurrentUser",
    "LimiterDep",
    "MetricsDep",
    "OwnedAsset",
    "OwnedJob",
    "RateLimited",
    "RequestIdDep",
    "SessionDep",
    "SettingsDep",
    "StorageDep",
    "resolve_background_asset",
    "user_rate_limit",
]
