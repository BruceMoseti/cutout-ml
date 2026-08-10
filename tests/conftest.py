"""Shared fixtures.

Two rules this file exists to enforce:

* **Unit tests touch nothing external.** Anything needing Postgres, Redis or ffmpeg is
  marked ``integration`` and lives under ``tests/integration``, so ``pytest -m
  "not integration"`` is a complete, fast, dependency-free suite.
* **Integration tests skip rather than fail when a service is absent.** A developer
  without Postgres running should see skips, not a wall of red that hides real
  failures. CI sets the services up and additionally asserts that they were *not*
  skipped (see ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- test data


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator. Never ``np.random`` directly: a flaky test is worse than none."""
    return np.random.default_rng(20240817)


@pytest.fixture
def sample_image() -> np.ndarray:
    """A 64x48 RGB image with a bright rectangle on a dark ground.

    Deliberately not square and not a power of two: square test images hide axis-order
    bugs, which are the single most common defect in this kind of code.
    """
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[..., 2] = 40
    image[12:36, 20:44] = (240, 220, 60)
    return image


@pytest.fixture
def sample_alpha() -> np.ndarray:
    """The exact matte for :func:`sample_image`'s rectangle."""
    alpha = np.zeros((48, 64), dtype=np.float32)
    alpha[12:36, 20:44] = 1.0
    return alpha


@pytest.fixture
def png_bytes(sample_image: np.ndarray) -> bytes:
    from cutoutml.core.imaging import encode_image

    return encode_image(sample_image, "png")


# ------------------------------------------------------------------ environment


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Settings pointed at a temp storage root, with the singleton cache cleared.

    ``get_settings`` is ``lru_cache``d, so without clearing it here a test that changes
    the environment would silently receive another test's configuration.
    """
    from cutoutml.core.config import Settings, get_settings

    monkeypatch.setenv("CUTOUTML_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("CUTOUTML_ENVIRONMENT", "test")
    monkeypatch.setenv("CUTOUTML_JWT_SECRET", "test-secret-not-a-real-one")
    get_settings.cache_clear()
    try:
        yield Settings()
    finally:
        get_settings.cache_clear()


@pytest.fixture
def local_storage(tmp_path: Path) -> Any:
    from cutoutml.storage.local import LocalStorage

    return LocalStorage(root=tmp_path / "objects")


# ------------------------------------------------------- external-service gating


def _postgres_url() -> str:
    return os.environ.get(
        "CUTOUTML_TEST_DATABASE_URL",
        os.environ.get(
            "CUTOUTML_DATABASE_URL", "postgresql+psycopg://dev:dev@127.0.0.1:5432/cutoutml"
        ),
    )


def _redis_url() -> str:
    return os.environ.get(
        "CUTOUTML_TEST_REDIS_URL", os.environ.get("CUTOUTML_REDIS_URL", "redis://127.0.0.1:6379/15")
    )


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """URL of a reachable Postgres, or skip the test."""
    url = _postgres_url()
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("select 1"))
        engine.dispose()
    except Exception as exc:  # noqa: BLE001 - any connection problem means "skip"
        pytest.skip(f"Postgres not reachable at {url}: {type(exc).__name__}: {exc}")
    return url


@pytest.fixture(scope="session")
def redis_url() -> str:
    """URL of a reachable Redis, or skip the test.

    Defaults to database 15 so a test run cannot flush or collide with development data
    sitting in database 0.
    """
    url = _redis_url()
    try:
        import redis

        redis.Redis.from_url(url, socket_connect_timeout=2).ping()
    except Exception as exc:  # noqa: BLE001 - any connection problem means "skip"
        pytest.skip(f"Redis not reachable at {url}: {type(exc).__name__}: {exc}")
    return url


@pytest.fixture(scope="session")
def ffmpeg_available() -> str:
    path = shutil.which(os.environ.get("CUTOUTML_FFMPEG_BINARY", "ffmpeg"))
    if path is None:
        pytest.skip("ffmpeg is not installed")
    return path


@pytest.fixture(scope="session")
def onnxruntime_available() -> Any:
    return pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")


# ------------------------------------------------------------------ API fixtures


@pytest.fixture
def api_database(postgres_url: str) -> Iterator[str]:
    """A migrated, disposable database for one test module.

    A separate database per run rather than transactional rollback, because the API is
    exercised through its real dependency graph (which commits) and through Celery in
    eager mode (which uses its own session). Sharing a database with the developer's own
    would mean tests deleting their assets.
    """
    from sqlalchemy import create_engine, text

    name = f"cutoutml_test_{uuid.uuid4().hex[:12]}"
    admin_url = postgres_url.rsplit("/", 1)[0] + "/postgres"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))

    url = postgres_url.rsplit("/", 1)[0] + f"/{name}"
    engine = create_engine(url)
    from cutoutml.db.models import Base

    Base.metadata.create_all(engine)
    engine.dispose()
    try:
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": name},
            )
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()
