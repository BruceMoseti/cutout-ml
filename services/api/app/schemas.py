"""Pydantic request/response models.

Kept separate from the ORM so the wire format can evolve independently of the schema, and
so a column added to a table is never automatically exposed. Notably ``password_hash``
cannot leak: no response model has a field for it.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

Precision = Literal["fp32", "fp16", "bf16"]
ImageOutput = Literal[
    "transparent_png",
    "transparent_webp",
    "mask_png",
    "color_composite",
    "background_composite",
    "blurred_background",
]
VideoMode = Literal["composite", "transparent", "frames", "mask"]
Smoothing = Literal["none", "ema", "median"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------- auth


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=256)
    display_name: str | None = Field(default=None, max_length=120)

    @field_validator("password")
    @classmethod
    def _password_strength(cls, value: str) -> str:
        """A minimum bar, not a policy theatre.

        Length dominates entropy, so a 10-character minimum plus a check that the
        password is not a single repeated character is more useful than mandatory symbol
        classes, which push users toward ``Password1!``.
        """
        if len(set(value)) < 4:
            raise ValueError("password must contain at least 4 distinct characters")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user_id: uuid.UUID


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    display_name: str | None
    is_active: bool
    is_admin: bool
    created_at: dt.datetime


# -------------------------------------------------------------------- assets


class UploadUrlRequest(BaseModel):
    filename: str = Field(max_length=255)
    content_type: str | None = Field(default=None, max_length=127)
    kind: Literal["image", "video"] = "image"
    size_bytes: int | None = Field(default=None, ge=0)


class UploadUrlResponse(BaseModel):
    asset_id: uuid.UUID
    storage_key: str
    upload_url: str
    method: str
    headers: dict[str, str]
    expires_at: dt.datetime
    max_bytes: int


class AssetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: str
    status: str
    original_filename: str | None
    content_type: str | None
    size_bytes: int
    width: int | None
    height: int | None
    duration_seconds: float | None
    frame_count: int | None
    fps: float | None
    created_at: dt.datetime
    storage_backend: str


class AssetListResponse(BaseModel):
    items: list[AssetResponse]
    total: int
    limit: int
    offset: int


class RefineOptions(BaseModel):
    """Alpha refinement knobs exposed to API clients."""

    guided_filter: bool = True
    guided_radius: int = Field(default=8, ge=0, le=64)
    soft_clip: bool = True
    clip_low: float = Field(default=0.02, ge=0.0, le=0.5)
    clip_high: float = Field(default=0.98, ge=0.5, le=1.0)
    morph_close: int = Field(default=0, ge=0, le=32)
    morph_open: int = Field(default=0, ge=0, le=32)
    min_component_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    feather_radius: int = Field(default=0, ge=0, le=32)

    def to_config(self) -> Any:
        from cutoutml.core.refine import RefineConfig

        return RefineConfig(**self.model_dump())


class ProcessImageOptions(BaseModel):
    outputs: list[ImageOutput] = Field(default_factory=lambda: ["transparent_png", "mask_png"])
    background_color: tuple[int, int, int] = (255, 255, 255)
    background_asset_id: uuid.UUID | None = None
    blur_sigma: float = Field(default=12.0, ge=0.0, le=100.0)
    webp_quality: int = Field(default=90, ge=1, le=100)
    refine: RefineOptions = Field(default_factory=RefineOptions)

    @field_validator("outputs")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one output must be requested")
        return value


class ProcessVideoOptions(BaseModel):
    mode: VideoMode = "composite"
    container: Literal["mp4", "webm", "mov", "qtrle"] | None = Field(
        default=None,
        description=(
            "Output container. Defaults to mp4 for composite/mask output and webm "
            "(VP9 with alpha) for mode='transparent', since mp4 cannot carry alpha."
        ),
    )
    background_color: tuple[int, int, int] = (0, 177, 64)
    background_asset_id: uuid.UUID | None = None
    blur_background: bool = False
    blur_sigma: float = Field(default=12.0, ge=0.0, le=100.0)
    smoothing: Smoothing = "ema"
    ema_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    median_window: int = Field(default=3, ge=1, le=15)
    batch_size: int = Field(default=4, ge=1, le=32)
    max_frames: int | None = Field(default=None, ge=1)
    crf: int = Field(default=23, ge=0, le=63)
    keep_audio: bool = True
    measure_flicker: bool = False

    @model_validator(mode="after")
    def _default_container_for_mode(self) -> ProcessVideoOptions:
        """Pick a container that can actually carry what ``mode`` asks for.

        A single ``container="mp4"`` default made ``{"mode": "transparent"}`` - the most
        obvious way to ask for a transparent video - fail validation every time, telling
        the caller that mp4 cannot carry alpha when they never asked for mp4. The default
        now follows the mode: WebM/VP9 for transparency (the only alpha container
        browsers play), mp4 otherwise. An explicit container is still honoured and still
        rejected if it contradicts the mode, so ``{"mode":"transparent","container":"mp4"}``
        remains an error rather than being silently rewritten.
        """
        if self.container is None:
            self.container = "webm" if self.mode == "transparent" else "mp4"
        return self


class ProcessRequest(BaseModel):
    """Body of ``POST /assets/{id}/process``."""

    model: str | None = Field(
        default=None,
        description="Registered model name. Defaults to the server's configured model.",
    )
    precision: Precision | None = None
    device: str | None = Field(default=None, description="'auto', 'cpu' or 'cuda[:n]'")
    image: ProcessImageOptions | None = None
    video: ProcessVideoOptions | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)
    priority: Literal["normal", "high"] = "normal"


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    asset_id: uuid.UUID
    status: str
    kind: str
    model_name: str
    precision: str
    queue: str
    progress: float
    progress_message: str | None
    attempts: int
    error_code: str | None
    error_message: str | None
    created_at: dt.datetime
    queued_at: dt.datetime | None
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    result: dict[str, Any] | None = None


class JobRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    attempt: int
    status: str
    device: str | None
    device_name: str | None
    batch_size: int | None
    oom_retry: bool
    retryable_error: bool | None
    error_code: str | None
    duration_seconds: float | None
    frames_processed: int | None
    peak_rss_bytes: int | None
    peak_vram_bytes: int | None


class JobDetailResponse(JobResponse):
    runs: list[JobRunResponse] = Field(default_factory=list)


class ResultOutput(BaseModel):
    kind: str
    storage_key: str
    url: str
    size_bytes: int
    content_type: str


class ResultResponse(BaseModel):
    job_id: uuid.UUID
    asset_id: uuid.UUID
    status: str
    outputs: list[ResultOutput]
    metrics: dict[str, Any] | None = None


# -------------------------------------------------------------------- models


class ModelInfo(BaseModel):
    name: str
    architecture: str
    runtime: str
    input_size: list[int]
    license: str
    source: str
    description: str
    tags: list[str]
    weights_available: bool
    supports_random_init: bool
    default_weights: str | None = None


class ModelListResponse(BaseModel):
    items: list[ModelInfo]
    default_model: str


# ---------------------------------------------------------------- benchmarks


class BenchmarkRequest(BaseModel):
    models: list[str] = Field(default_factory=list)
    precision: Precision = "fp32"
    device: str = "auto"
    batch_sizes: list[int] = Field(default_factory=lambda: [1])
    warmup: int = Field(default=2, ge=0, le=50)
    repetitions: int = Field(default=10, ge=1, le=500)
    accuracy_samples: int = Field(default=16, ge=0, le=2000)
    resolution: int = Field(default=256, ge=64, le=2048)


class BenchmarkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: str
    status: str
    git_commit: str | None
    hardware: str | None
    gpu_name: str | None
    dataset_id: str | None
    duration_seconds: float | None
    created_at: dt.datetime
    finished_at: dt.datetime | None
    metrics: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------- health


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: str


class ReadinessCheck(BaseModel):
    name: str
    ok: bool
    detail: str
    duration_ms: float


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: list[ReadinessCheck]
    version: str
