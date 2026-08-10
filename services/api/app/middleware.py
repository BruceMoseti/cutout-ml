"""Request-scoped middleware: request IDs, structured access logs, metrics.

Ordering matters and is set in ``main.py``: the request-id middleware must be outermost
so that every log line emitted by anything inside it - including the exception handlers -
carries the id.

Metrics are recorded in the same middleware that times the request, and the route label
is read *after* the handler runs, because Starlette only populates ``scope["route"]``
during routing. Reading it before would label everything ``unmatched``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from cutoutml.core.logging import bind_request_id, get_logger
from services.api.app.metrics import Metrics, route_template

log = get_logger("cutoutml.api.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a request id and bind it into the logging context.

    An inbound id is trusted only in shape: it is truncated to 64 characters and
    non-printable characters would be rejected by the header parser upstream. Accepting a
    client-supplied id is what makes a trace span the frontend, the API and the worker;
    validating its length is what stops it becoming a log-injection vector.
    """

    def __init__(self, app: ASGIApp, *, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(self.header_name)
        request_id = (incoming or uuid.uuid4().hex)[:64]
        request.state.request_id = request_id
        bind_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            bind_request_id(None)
        response.headers[self.header_name] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Structured access log + Prometheus recording for every request."""

    def __init__(self, app: ASGIApp, *, metrics: Metrics, skip_paths: tuple[str, ...] = ()) -> None:
        super().__init__(app)
        self.metrics = metrics
        self.skip_paths = skip_paths

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if path in self.skip_paths:
            return await call_next(request)

        started = time.perf_counter()
        self.metrics.http_in_flight.inc()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            self.metrics.http_in_flight.dec()
            route = route_template(request.scope)
            self.metrics.http_duration.labels(request.method, route).observe(elapsed)
            self.metrics.http_requests.labels(
                request.method, route, f"{status_code // 100}xx"
            ).inc()
            log.info(
                "http_request",
                method=request.method,
                path=path,
                route=route,
                status=status_code,
                duration_ms=round(elapsed * 1000.0, 3),
                client=request.client.host if request.client else None,
                user_id=getattr(request.state, "user_id", None),
            )


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies from the declared ``Content-Length``.

    This is a cheap first gate, not the real check: ``Content-Length`` is client-supplied
    and absent for chunked transfers. The authoritative size check happens on the actual
    bytes in :func:`services.api.app.uploads.validate_upload`. Rejecting early still
    matters, because it avoids buffering a 2 GB body just to reject it.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": (
                            f"request body of {declared} bytes exceeds the limit of "
                            f"{self.max_bytes}"
                        ),
                    }
                },
            )
        return await call_next(request)
