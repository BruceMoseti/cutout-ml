"""Liveness and readiness, deliberately split.

They answer different questions and a load balancer or orchestrator reacts to them
differently:

``GET /health/live``
    "Is this process running?" Touches nothing external and always answers if the event
    loop is alive. A failure here means *restart me*. Wiring dependency checks into
    liveness is a classic outage amplifier: Postgres hiccups, every replica's liveness
    probe fails, Kubernetes restarts the entire fleet, and the cold caches turn a
    30-second database blip into a ten-minute outage.

``GET /health/ready``
    "Should traffic be sent here?" Checks the database, Redis and the model registry,
    reports each one individually, and returns 503 when a hard dependency is down so the
    pod is removed from the service without being killed.

``GET /health``
    Liveness, for humans and for the many tools that assume that path exists.

Storage and ffmpeg are reported but not fatal: a missing ffmpeg only breaks video jobs,
and refusing all traffic for that would take the working image path down with it.
"""

from __future__ import annotations

import shutil
import time
from typing import Any

from fastapi import APIRouter, Request, Response, status

from cutoutml.db.session import check_database
from cutoutml.models.registry import list_model_names
from services.api.app.deps import SettingsDep
from services.api.app.schemas import HealthResponse, ReadinessCheck, ReadinessResponse

router = APIRouter(tags=["health"])

VERSION = "0.1.0"

#: Checks that must pass for the process to accept traffic. Everything else is
#: informational.
REQUIRED_CHECKS = frozenset({"database", "redis"})


def _timed(name: str, fn: Any) -> ReadinessCheck:
    started = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    return ReadinessCheck(
        name=name,
        ok=bool(ok),
        detail=str(detail)[:200],
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness")
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION, environment=settings.environment)


@router.get("/health/live", response_model=HealthResponse, summary="Liveness")
def live(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION, environment=settings.environment)


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness")
def ready(request: Request, response: Response, settings: SettingsDep) -> ReadinessResponse:
    """Probe each dependency and report them individually."""

    def _redis() -> tuple[bool, str]:
        client = getattr(request.app.state, "redis", None)
        if client is None:
            return (False, "no redis client (rate limiting is degraded to in-process)")
        client.ping()
        return (True, "ok")

    def _registry() -> tuple[bool, str]:
        names = list_model_names()
        return (bool(names), f"{len(names)} models registered")

    def _storage() -> tuple[bool, str]:
        from cutoutml.storage.factory import get_storage

        storage = get_storage()
        storage.list(limit=1)
        return (True, storage.backend)

    def _ffmpeg() -> tuple[bool, str]:
        path = shutil.which(settings.ffmpeg_binary)
        return (path is not None, path or f"{settings.ffmpeg_binary} not on PATH")

    checks = [
        _timed("database", lambda: check_database(settings)),
        _timed("redis", _redis),
        _timed("model_registry", _registry),
        _timed("storage", _storage),
        _timed("ffmpeg", _ffmpeg),
    ]

    degraded = [c for c in checks if not c.ok and c.name in REQUIRED_CHECKS]
    if degraded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="degraded" if degraded else "ready", checks=checks, version=VERSION
    )
