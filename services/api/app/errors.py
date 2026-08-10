"""One error shape for every failure the API can produce.

A client should never have to branch on where an error came from. FastAPI natively emits
three different bodies - ``{"detail": "..."}`` for ``HTTPException``, a list of objects
for validation errors, and an HTML traceback or bare 500 for anything unhandled - so all
three are re-shaped here into::

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": {...}}}

``code`` is a stable machine-readable string that clients may switch on;
``message`` is for humans and may be reworded; ``request_id`` is what a user quotes in a
support ticket and what correlates the response with the server logs and with the worker
that later ran the job.

Unhandled exceptions deliberately return a generic message and log the traceback rather
than returning it. An exception string routinely contains a SQL fragment, a filesystem
path or a connection URL, and none of that belongs in an HTTP response.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from cutoutml.core.logging import get_logger

log = get_logger(__name__)

#: Default ``code`` per status class, used when a raiser did not supply one.
_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


class ApiError(HTTPException):
    """An ``HTTPException`` that carries a stable error code and structured details."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.code = code
        self.details = details
        super().__init__(status_code=status_code, detail=message, headers=headers)


def not_found(resource: str, identifier: Any) -> ApiError:
    """404 for a missing resource.

    Also used for resources the caller is not allowed to see, which is why the message
    is identical in both cases: returning 403 for "exists but not yours" and 404 for
    "does not exist" turns the endpoint into an existence oracle that lets one tenant
    enumerate another's ids.
    """
    return ApiError(
        status.HTTP_404_NOT_FOUND,
        f"{resource}_not_found",
        f"{resource} {identifier} was not found",
    )


def envelope(
    *,
    code: str,
    message: str,
    request_id: str | None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the response body."""
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if details:
        error["details"] = details
    return {"error": error}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _json(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=envelope(
            code=code, message=message, request_id=_request_id(request), details=details
        ),
        headers=headers,
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers for every exception class FastAPI can surface."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _json(
            request,
            exc.status_code,
            exc.code,
            str(exc.detail),
            details=exc.details,
            headers=exc.headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, f"http_{exc.status_code}")
        return _json(
            request, exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field errors are kept, but each ``loc`` is joined into a dotted path and any
        # ``ctx`` is dropped: ctx can echo the submitted value, which for a login body is
        # the password.
        fields = [
            {
                "field": ".".join(str(part) for part in err.get("loc", ())),
                "message": err.get("msg", "invalid value"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()
        ]
        return _json(
            request,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "validation_error",
            "the request body or parameters failed validation",
            details={"fields": fields},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
        )
        return _json(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "an internal error occurred; quote the request_id when reporting it",
        )


__all__ = ["ApiError", "envelope", "install_exception_handlers", "not_found"]
