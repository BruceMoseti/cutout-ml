"""Central, environment-driven configuration.

Every tunable lives here so that the API, the Celery worker and the benchmark
harness all agree on paths, limits and connection strings. Values are read from
the process environment (and an optional ``.env``) with the ``CUTOUTML_``
prefix; see ``.env.example`` for the full list.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
"""Repository root, derived from this file's location (``src/cutoutml/core``)."""


class Settings(BaseSettings):
    """Runtime configuration for all CutoutML services."""

    model_config = SettingsConfigDict(
        env_prefix="CUTOUTML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------- general
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # ---------------------------------------------------------------- storage
    storage_backend: Literal["local", "s3"] = "local"
    storage_root: Path = REPO_ROOT / "storage"
    s3_bucket: str = "cutoutml"
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_force_path_style: bool = True
    presign_expiry_seconds: int = 900

    # ---------------------------------------------------------------- database
    database_url: str = "postgresql+psycopg://dev:dev@127.0.0.1:5432/cutoutml"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_echo: bool = False

    # ---------------------------------------------------------------- queue
    redis_url: str = "redis://127.0.0.1:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_always_eager: bool = False

    # ---------------------------------------------------------------- auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_ttl_seconds: int = 3600
    jwt_issuer: str = "cutoutml"

    # ---------------------------------------------------------------- limits
    max_upload_bytes: int = 32 * 1024 * 1024
    max_video_upload_bytes: int = 256 * 1024 * 1024
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 30
    max_image_pixels: int = 64_000_000

    # ---------------------------------------------------------------- models
    default_model: str = "cutoutnet"
    model_weights_dir: Path = REPO_ROOT / "models"
    device: str = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    torch_num_threads: int = 0  # 0 => leave PyTorch's default alone

    # ---------------------------------------------------------------- video
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    video_frame_queue_size: int = 8
    video_batch_size: int = 4
    max_video_frames: int = 18_000

    # ---------------------------------------------------------------- misc
    benchmark_results_dir: Path = REPO_ROOT / "benchmarks" / "results"
    dataset_cache_dir: Path = REPO_ROOT / "artifacts" / "datasets"
    request_id_header: str = "X-Request-ID"

    @field_validator("storage_root", "model_weights_dir", "benchmark_results_dir", "dataset_cache_dir")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return Path(value).expanduser()

    @property
    def broker_url(self) -> str:
        """Celery broker, defaulting to the shared Redis instance."""
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        """Celery result backend, defaulting to the shared Redis instance."""
        return self.celery_result_backend or self.redis_url

    @property
    def sync_database_url(self) -> str:
        """A psycopg3 (sync) SQLAlchemy URL, normalising bare ``postgresql://``."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()


def field_names() -> list[str]:
    """Names of every configurable field (used by ``docs`` and ``.env`` checks)."""
    return sorted(Settings.model_fields)
