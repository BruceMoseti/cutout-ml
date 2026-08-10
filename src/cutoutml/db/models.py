"""SQLAlchemy 2.0 ORM models.

Schema shape and the reasoning behind it lives in ``docs/data-model.md``. The parts
worth knowing while reading this file:

* **Two-level job model.** ``inference_jobs`` is the *user-visible unit of work*;
  ``inference_runs`` is one *attempt* at executing it. A retry after a CUDA OOM adds
  a run, it does not mutate the job. That separation is what makes "how often do we
  OOM and what batch size did the retry succeed at" answerable from SQL instead of
  from log grep.
* **Idempotency keys.** ``inference_jobs.idempotency_key`` is unique per user. A
  re-delivered Celery message (which *will* happen - brokers are at-least-once)
  finds the existing job and returns its result rather than producing a second set
  of outputs. See ``docs/decisions/ADR-002-queues.md``.
* **``benchmark_runs`` stores provenance, not just numbers.** Git SHA, hardware
  description, library versions, dataset id and full config. A benchmark number
  without those is an anecdote.
* Enums are stored as ``VARCHAR`` with a CHECK-style Python enum rather than a
  Postgres ``ENUM`` type: adding a new job status to a native enum requires
  ``ALTER TYPE``, which is awkward inside a transaction-wrapped migration and needs
  a lock on every table using it.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base with a JSON-friendly ``as_dict``."""

    def as_dict(self, *, exclude: set[str] | None = None) -> dict[str, Any]:
        skip = exclude or set()
        out: dict[str, Any] = {}
        for column in self.__table__.columns:
            if column.name in skip:
                continue
            value = getattr(self, column.name)
            if isinstance(value, dt.datetime):
                value = value.isoformat()
            elif isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, enum.Enum):
                value = value.value
            out[column.name] = value
        return out


class JobStatus(enum.StrEnum):
    """Lifecycle of an inference job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


class AssetKind(enum.StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class AssetStatus(enum.StrEnum):
    """Uploads are two-phase: a row exists before the bytes do."""

    AWAITING_UPLOAD = "awaiting_upload"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


TimestampTZ = DateTime(timezone=True)


class User(Base):
    """An API principal.

    Passwords are bcrypt hashes with a per-hash salt; the plaintext never leaves the
    request handler. ``is_active`` is checked on every authenticated request so
    revocation takes effect without waiting for token expiry.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(
        TimestampTZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        TimestampTZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    assets: Mapped[list[Asset]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    jobs: Mapped[list[InferenceJob]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Asset(Base):
    """An uploaded image or video.

    ``storage_key`` is server-generated and random (see
    :func:`cutoutml.storage.base.build_storage_key`); ``original_filename`` is kept
    only for display and is never used to build a path.

    ``content_sha256`` enables deduplication and, more importantly, lets a job be
    keyed by content so re-processing the same bytes with the same parameters is
    detectably idempotent.
    """

    __tablename__ = "assets"
    __table_args__ = (
        Index("ix_assets_owner_created", "owner_id", "created_at"),
        Index("ix_assets_owner_status", "owner_id", "status"),
        UniqueConstraint("storage_key", name="uq_assets_storage_key"),
        CheckConstraint("size_bytes >= 0", name="ck_assets_size_nonnegative"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default=AssetKind.IMAGE.value)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=AssetStatus.AWAITING_UPLOAD.value
    )
    storage_backend: Mapped[str] = mapped_column(String(16), nullable=False, default="local")
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(127))
    content_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        TimestampTZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        TimestampTZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    owner: Mapped[User] = relationship(back_populates="assets")
    jobs: Mapped[list[InferenceJob]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class InferenceJob(Base):
    """A unit of work requested by a user.

    ``result`` holds the output manifest (storage keys, sizes, per-output metadata)
    so ``GET /assets/{id}/result`` needs one query and no storage round-trip.
    """

    __tablename__ = "inference_jobs"
    __table_args__ = (
        UniqueConstraint("owner_id", "idempotency_key", name="uq_jobs_owner_idempotency"),
        Index("ix_jobs_owner_created", "owner_id", "created_at"),
        Index("ix_jobs_status_queue", "status", "queue"),
        CheckConstraint("attempts >= 0", name="ck_jobs_attempts_nonnegative"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=JobStatus.PENDING.value, index=True
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default=AssetKind.IMAGE.value)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    precision: Mapped[str] = mapped_column(String(8), nullable=False, default="fp32")
    queue: Mapped[str] = mapped_column(String(32), nullable=False, default="cpu")
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), index=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    progress_message: Mapped[str | None] = mapped_column(String(255))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    queued_at: Mapped[dt.datetime | None] = mapped_column(TimestampTZ)
    started_at: Mapped[dt.datetime | None] = mapped_column(TimestampTZ)
    finished_at: Mapped[dt.datetime | None] = mapped_column(TimestampTZ)
    created_at: Mapped[dt.datetime] = mapped_column(
        TimestampTZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        TimestampTZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    owner: Mapped[User] = relationship(back_populates="jobs")
    asset: Mapped[Asset] = relationship(back_populates="jobs")
    runs: Mapped[list[InferenceRun]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="InferenceRun.attempt"
    )


class InferenceRun(Base):
    """One execution attempt of a job.

    Records the batch size actually used and whether the attempt was an OOM retry,
    which is the data needed to tune batch sizes empirically rather than by feel.
    """

    __tablename__ = "inference_runs"
    __table_args__ = (
        UniqueConstraint("job_id", "attempt", name="uq_runs_job_attempt"),
        Index("ix_runs_job_created", "job_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.RUNNING.value)
    worker_hostname: Mapped[str | None] = mapped_column(String(255))
    device: Mapped[str | None] = mapped_column(String(32))
    device_name: Mapped[str | None] = mapped_column(String(128))
    model_name: Mapped[str | None] = mapped_column(String(64))
    precision: Mapped[str | None] = mapped_column(String(8))
    batch_size: Mapped[int | None] = mapped_column(Integer)
    oom_retry: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retryable_error: Mapped[bool | None] = mapped_column(Boolean)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    frames_processed: Mapped[int | None] = mapped_column(Integer)
    peak_rss_bytes: Mapped[int | None] = mapped_column(BigInteger)
    peak_vram_bytes: Mapped[int | None] = mapped_column(BigInteger)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        TimestampTZ, nullable=False, server_default=func.now()
    )
    finished_at: Mapped[dt.datetime | None] = mapped_column(TimestampTZ)

    job: Mapped[InferenceJob] = relationship(back_populates="runs")


class BenchmarkRun(Base):
    """One benchmark execution, with everything needed to trust the numbers.

    Deliberately denormalised and append-only. A benchmark table in a README is only
    credible if a reader can see the commit, the hardware, the library versions and
    the dataset that produced it, so all four are columns rather than a comment.
    """

    __tablename__ = "benchmark_runs"
    __table_args__ = (
        Index("ix_benchmark_created", "created_at"),
        Index("ix_benchmark_model", "model_name", "runtime", "precision"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.PENDING.value)
    git_commit: Mapped[str | None] = mapped_column(String(40))
    git_dirty: Mapped[bool | None] = mapped_column(Boolean)
    hardware: Mapped[str | None] = mapped_column(String(255))
    cpu_model: Mapped[str | None] = mapped_column(String(255))
    cpu_count: Mapped[int | None] = mapped_column(Integer)
    total_ram_bytes: Mapped[int | None] = mapped_column(BigInteger)
    gpu_name: Mapped[str | None] = mapped_column(String(128))
    os_description: Mapped[str | None] = mapped_column(String(255))
    library_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    dataset_id: Mapped[str | None] = mapped_column(String(128), index=True)
    dataset_manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    model_name: Mapped[str | None] = mapped_column(String(64))
    runtime: Mapped[str | None] = mapped_column(String(64))
    precision: Mapped[str | None] = mapped_column(String(8))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    results_path: Mapped[str | None] = mapped_column(String(512))
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        TimestampTZ, nullable=False, server_default=func.now()
    )
    finished_at: Mapped[dt.datetime | None] = mapped_column(TimestampTZ)


__all__ = [
    "Asset",
    "AssetKind",
    "AssetStatus",
    "Base",
    "BenchmarkRun",
    "InferenceJob",
    "InferenceRun",
    "JobStatus",
    "User",
]
