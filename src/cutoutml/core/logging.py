"""Structured logging.

Everything logs through ``structlog`` so that API requests, Celery tasks and
benchmark runs emit the same machine-parseable shape. A ``request_id`` is bound
into the context so a single HTTP request can be traced across the API process
and the worker that eventually executes its job.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

_request_id: ContextVar[str | None] = ContextVar("cutoutml_request_id", default=None)
_configured = False


def _inject_request_id(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    rid = _request_id.get()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Idempotently configure stdlib logging + structlog.

    ``fmt="console"`` gives a colourised developer view; ``"json"`` is what
    containers should use so log aggregators can parse it.
    """
    global _configured

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
        if fmt == "console"
        else structlog.processors.JSONRenderer()
    )

    # The stdlib logger factory (rather than PrintLogger) is what makes
    # `add_logger_name` work and lets uvicorn/celery/SQLAlchemy log records flow
    # through the same handler and level configuration.
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            _inject_request_id,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_request_id(request_id: str | None) -> None:
    """Bind (or clear) the request id for the current async/thread context."""
    _request_id.set(request_id)


def current_request_id() -> str | None:
    """The request id bound to this context, if any."""
    return _request_id.get()
