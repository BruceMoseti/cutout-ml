"""ASGI application factory.

Run it with::

    uvicorn services.api.app.main:app --reload

Middleware order is the part of this file that is easy to get wrong. Starlette applies
middleware outermost-first in the order added, so:

1. :class:`RequestContextMiddleware` is added first and is therefore outermost. Every log
   line produced by anything inside it - including the exception handlers - carries the
   request id.
2. :class:`AccessLogMiddleware` next, so the duration it records includes the body-size
   check and CORS handling rather than only the handler.
3. :class:`BodySizeLimitMiddleware` inside those, so an oversized body is rejected before
   routing but is still logged and still gets a request id.
4. CORS last (innermost of ours) because it must be able to short-circuit ``OPTIONS``.

The rate limiter is *not* middleware. It is a router dependency, so it runs after routing
and after authentication, which is what lets it key on the resolved user id and skip
``/health`` and ``/metrics`` by construction rather than by a path allow-list.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from cutoutml.core.config import Settings, get_settings
from cutoutml.core.logging import configure_logging, get_logger
from services.api.app.errors import install_exception_handlers
from services.api.app.metrics import get_metrics
from services.api.app.middleware import (
    AccessLogMiddleware,
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
)
from services.api.app.ratelimit import RateLimiter, build_redis_client
from services.api.app.routers import assets, auth, catalog, health, jobs
from services.api.app.schemas import ErrorResponse

log = get_logger(__name__)

DESCRIPTION = """
Image and video background removal / segmentation.

**Authentication.** All `/v1` endpoints except `/v1/auth/*` require
`Authorization: Bearer <token>`. Obtain a token from `POST /v1/auth/login`.

**Errors** always have the shape
`{"error": {"code", "message", "request_id", "details?"}}`. `code` is stable and safe to
branch on; `message` is not.

**Asynchronous processing.** `POST /v1/assets/{id}/process` returns `202` with a job id;
poll `GET /v1/jobs/{id}` for `status` and `progress`. Requests carry an idempotency key
(explicit or derived from the content hash plus parameters), so a retried submission
returns the original job with `200` instead of creating a duplicate.
"""

#: Attached to every route so the generated OpenAPI document describes the envelope
#: rather than FastAPI's default ``{"detail": ...}``.
COMMON_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Malformed request"},
    401: {"model": ErrorResponse, "description": "Missing or invalid token"},
    404: {"model": ErrorResponse, "description": "Not found, or not owned by the caller"},
    409: {"model": ErrorResponse, "description": "Conflicts with current state"},
    413: {"model": ErrorResponse, "description": "Payload too large"},
    415: {"model": ErrorResponse, "description": "Unsupported media type"},
    422: {"model": ErrorResponse, "description": "Validation failed"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Internal error"},
}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. A factory, not a module-level singleton, so tests can
    construct an isolated instance with overridden settings."""
    cfg = settings or get_settings()
    configure_logging(level=cfg.log_level, fmt=cfg.log_format)
    metrics = get_metrics()

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Acquire shared clients once per process.

        The Redis client is created here rather than per request because connection setup
        costs more than the rate-limit check it enables. It is allowed to be ``None``:
        the limiter degrades to per-process buckets and ``/health/ready`` reports the
        degradation, which is preferable to refusing to boot.
        """
        redis_client = build_redis_client(cfg.redis_url)
        application.state.redis = redis_client
        application.state.rate_limiter = RateLimiter(
            per_minute=cfg.rate_limit_per_minute,
            burst=cfg.rate_limit_burst,
            redis_client=redis_client,
        )
        application.state.settings = cfg
        log.info(
            "api_startup",
            environment=cfg.environment,
            storage=cfg.storage_backend,
            default_model=cfg.default_model,
            rate_limit_backend="redis" if redis_client else "in-memory",
        )
        try:
            yield
        finally:
            if redis_client is not None:
                with contextlib.suppress(Exception):
                    redis_client.close()
            log.info("api_shutdown")

    app = FastAPI(
        title="CutoutML API",
        version=health.VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
        responses=COMMON_RESPONSES,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware, header_name=cfg.request_id_header)
    app.add_middleware(
        AccessLogMiddleware, metrics=metrics, skip_paths=("/metrics", "/health/live")
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=max(cfg.max_upload_bytes, cfg.max_video_upload_bytes),
    )
    app.add_middleware(
        CORSMiddleware,
        # Explicit origins, not "*": the API uses bearer tokens that a browser stores,
        # and a wildcard origin with credentials is what turns XSS on any site into
        # account takeover here.
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", cfg.request_id_header],
        expose_headers=[cfg.request_id_header, "X-RateLimit-Remaining", "Retry-After"],
        max_age=600,
    )

    install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(assets.router)
    app.include_router(jobs.router)
    app.include_router(catalog.router)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        """Prometheus scrape endpoint.

        Unauthenticated on purpose - a scraper is usually not a user - and therefore
        intended to be reachable only from inside the deployment network. See
        ``docs/security.md``.
        """
        payload, content_type = metrics.render()
        return Response(content=payload, media_type=content_type)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": "CutoutML API",
            "version": health.VERSION,
            "docs": "/docs",
            "health": "/health/ready",
        }

    return app


app = create_app()
