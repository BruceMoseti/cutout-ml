"""Shared fixtures.

Two things are configured here *before* anything imports ``cutoutml``, because
:func:`cutoutml.core.config.get_settings` is an ``lru_cache`` singleton and the first
caller wins:

* a dedicated ``cutoutml_test`` database, so running the suite can never truncate the
  development data;
* a per-session temporary storage root, so no test writes into the repository's
  ``storage/`` directory.

Celery runs in eager mode with exceptions propagating, which is what lets the API tests
exercise the *real* task body (and therefore the real pipeline, the real model and the
real storage writes) without a broker or a worker process. The alternative - mocking
``apply_async`` - would leave the most interesting half of the system untested.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_TEST_DB = "postgresql+psycopg://dev:dev@127.0.0.1:5432/cutoutml_test"
_ADMIN_DB = "postgresql+psycopg://dev:dev@127.0.0.1:5432/postgres"

_STORAGE_ROOT = Path(tempfile.mkdtemp(prefix="cutoutml-tests-storage-"))

os.environ.setdefault("CUTOUTML_ENVIRONMENT", "test")
os.environ.setdefault("CUTOUTML_LOG_FORMAT", "console")
os.environ.setdefault("CUTOUTML_LOG_LEVEL", "WARNING")
os.environ.setdefault("CUTOUTML_DATABASE_URL", _TEST_DB)
os.environ.setdefault("CUTOUTML_STORAGE_ROOT", str(_STORAGE_ROOT))
os.environ.setdefault("CUTOUTML_STORAGE_BACKEND", "local")
os.environ.setdefault("CUTOUTML_CELERY_TASK_ALWAYS_EAGER", "1")
os.environ.setdefault("CUTOUTML_DEVICE", "cpu")
# One thread. The suite runs many tiny forward passes and thread-pool spin-up dominates
# them; it also keeps the timing-sensitive benchmark tests from fighting pytest-xdist.
os.environ.setdefault("CUTOUTML_TORCH_NUM_THREADS", "1")
# An unreachable Redis would make every rate-limited request wait on a connect timeout.
# The limiter degrades to in-process buckets, which is what the API tests want anyway.
os.environ.setdefault("CUTOUTML_REDIS_URL", "redis://127.0.0.1:6379/15")

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- helpers


def png_bytes(width: int = 96, height: int = 64, *, seed: int = 7) -> bytes:
    """A small, genuinely decodable PNG with a bright blob on a dark field.

    Deterministic so that content hashes - and therefore derived idempotency keys - are
    stable across runs.
    """
    from cutoutml.core.imaging import encode_image

    rng = np.random.default_rng(seed)
    image = (rng.integers(10, 45, size=(height, width, 3))).astype(np.uint8)
    cy, cx = height // 2, width // 2
    ry, rx = height // 4, width // 4
    image[cy - ry : cy + ry, cx - rx : cx + rx] = (240, 210, 60)
    return encode_image(image, "png")


@pytest.fixture(scope="session")
def sample_png() -> bytes:
    return png_bytes()


@pytest.fixture
def rgb_image() -> np.ndarray:
    """A decoded ``(H, W, 3)`` uint8 RGB array."""
    from cutoutml.core.imaging import decode_image

    return decode_image(png_bytes())


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    root = tmp_path / "storage"
    root.mkdir()
    return root


@pytest.fixture
def local_storage(storage_root: Path) -> Any:
    from cutoutml.storage.local import LocalStorage

    return LocalStorage(storage_root)


# -------------------------------------------------------------------- database


def _postgres_available() -> bool:
    try:
        import sqlalchemy
    except ImportError:  # pragma: no cover
        return False
    try:
        engine = sqlalchemy.create_engine(_ADMIN_DB, isolation_level="AUTOCOMMIT")
        with engine.connect():
            pass
    except Exception:  # noqa: BLE001 - probing a server that may be absent
        return False
    return True


POSTGRES_AVAILABLE = _postgres_available()

requires_postgres = pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="needs a local Postgres reachable at dev:dev@127.0.0.1:5432",
)


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Any]:
    """Create ``cutoutml_test``, build the schema, and hand back the engine.

    The schema is created with ``Base.metadata.create_all`` rather than by running Alembic
    so that a schema/model divergence shows up as a *test* failure in
    ``tests/test_migrations.py`` (which does run Alembic) instead of silently making every
    other test exercise the migration's idea of the schema.
    """
    if not POSTGRES_AVAILABLE:
        pytest.skip("postgres unavailable")

    import sqlalchemy
    from sqlalchemy import text

    admin = sqlalchemy.create_engine(_ADMIN_DB, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = 'cutoutml_test'")
        ).scalar()
        if not exists:
            conn.execute(text("CREATE DATABASE cutoutml_test"))
    admin.dispose()

    from cutoutml.db.models import Base
    from cutoutml.db.session import get_engine, reset_caches

    reset_caches()
    engine = get_engine()
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    reset_caches()


@pytest.fixture
def clean_db(db_engine: Any) -> Iterator[Any]:
    """Truncate every table before each test.

    ``TRUNCATE ... CASCADE`` rather than ``DROP``/``CREATE``: it is an order of magnitude
    faster and it keeps the indexes, so a test that depends on a unique constraint still
    exercises it.
    """
    from sqlalchemy import text

    with db_engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE users, assets, inference_jobs, inference_runs, benchmark_runs "
                "RESTART IDENTITY CASCADE"
            )
        )
    yield db_engine


@pytest.fixture
def db_session(clean_db: Any) -> Iterator[Any]:
    from cutoutml.db.session import get_sessionmaker

    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ------------------------------------------------------------------------- api


@pytest.fixture
def settings() -> Any:
    from cutoutml.core.config import get_settings

    return get_settings()


@pytest.fixture
def app(clean_db: Any, storage_root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A FastAPI app wired to the test database and a per-test storage root."""
    from services.api.app.main import create_app

    from cutoutml.core.config import get_settings
    from cutoutml.storage.factory import reset_storage_cache

    cfg = get_settings()
    monkeypatch.setattr(cfg, "storage_root", storage_root, raising=False)
    reset_storage_cache()
    yield create_app(cfg)
    reset_storage_cache()


@pytest.fixture
def client(app: Any) -> Iterator[Any]:
    """A ``TestClient`` inside a lifespan context, so app state exists."""
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def auth_client(client: Any) -> Any:
    """A client with a registered user's bearer token already attached."""
    email = f"user-{uuid.uuid4().hex[:12]}@example.test"
    response = client.post(
        "/v1/auth/register", json={"email": email, "password": "correct-horse-battery"}
    )
    assert response.status_code == 201, response.text
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    client.email = email  # type: ignore[attr-defined]
    return client


# ----------------------------------------------------------------------- models


@pytest.fixture(scope="session")
def trivial_model() -> Any:
    """A loaded, content-blind model.

    Used wherever a test needs *a* model but not a specific one - pipeline plumbing,
    output encoding, batching. It has no weights and no compute cost, so it keeps those
    tests measuring the thing they are about.
    """
    from cutoutml.models.registry import get_model

    return get_model("trivial-center", device="cpu")


@pytest.fixture(scope="session")
def cutoutnet_available() -> bool:
    from cutoutml.models.registry import resolve_spec, weights_available

    return weights_available(resolve_spec("cutoutnet"))
