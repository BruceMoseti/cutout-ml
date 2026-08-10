"""Engine and session management."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from cutoutml.core.config import Settings, get_settings
from cutoutml.core.logging import get_logger

log = get_logger(__name__)


def build_engine(settings: Settings | None = None) -> Engine:
    """Create the SQLAlchemy engine.

    ``pool_pre_ping`` is on because the API and the workers are long-lived while
    Postgres connections are not: a restart or an idle timeout otherwise surfaces as
    a random ``OperationalError`` on the next request instead of a transparent
    reconnect.
    """
    cfg = settings or get_settings()
    return create_engine(
        cfg.sync_database_url,
        pool_size=cfg.db_pool_size,
        max_overflow=cfg.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=cfg.db_echo,
        future=True,
    )


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide engine singleton (connection pools must not be duplicated)."""
    return build_engine()


@functools.lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    """Session factory bound to the process engine.

    ``expire_on_commit=False`` so that ORM objects stay readable after the
    transaction closes - otherwise every FastAPI handler that commits and then
    serialises its result triggers a lazy reload on a closed session.
    """
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session per request.

    Commits are the handler's responsibility; this only guarantees the session is
    rolled back and closed, so a handler that raises never leaves a half-applied
    transaction holding locks.
    """
    session = get_sessionmaker()()
    try:
        yield session
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def check_database(settings: Settings | None = None) -> tuple[bool, str]:
    """Readiness probe: can we execute a trivial query?

    Returns ``(ok, detail)`` rather than raising, because a readiness endpoint has to
    report *which* dependency is down, not just that something is.
    """
    try:
        engine = build_engine(settings) if settings else get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        return (False, f"{type(exc).__name__}: {exc}")
    return (True, "ok")


def reset_caches() -> None:
    """Drop cached engine/sessionmaker. Used by tests that change the database URL."""
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
