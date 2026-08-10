"""``GET /v1/models`` and ``GET /v1/benchmarks``.

Both are read-only catalogue endpoints, and both deliberately report *availability*
rather than pretending everything listed works here. ``GET /v1/models`` says whether a
checkpoint is actually on disk, so a client can grey out a selector entry instead of
queueing a job that is guaranteed to fail with ``weights_unavailable``.

``GET /v1/benchmarks`` reads the JSON files that ``benchmarks/run.py`` wrote. The API
never computes a benchmark number on request: a measurement taken while serving traffic
is a measurement of a contended machine, and it would have no provenance attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from cutoutml.benchmarks.harness import load_report
from cutoutml.core.logging import get_logger
from cutoutml.models.registry import catalogue, list_models
from services.api.app.deps import RateLimited, SettingsDep
from services.api.app.errors import ApiError
from services.api.app.schemas import ModelInfo, ModelListResponse

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["catalog"], dependencies=[RateLimited])


@router.get("/models", response_model=ModelListResponse, summary="Registered models")
def list_available_models(settings: SettingsDep) -> ModelListResponse:
    items: list[ModelInfo] = []
    for spec in list_models():
        weights = spec.default_weights
        available = True
        if spec.requires_weights and weights:
            candidate = Path(weights)
            if not candidate.is_absolute():
                candidate = settings.model_weights_dir / candidate
            available = candidate.is_file()
        elif spec.runtime in {"onnxruntime", "tensorrt"}:
            artefact = spec.options.get("onnx_path") or spec.options.get("engine_path")
            available = bool(artefact) and Path(str(artefact)).is_file()

        items.append(
            ModelInfo(
                name=spec.name,
                architecture=spec.architecture,
                runtime=spec.runtime,
                input_size=list(spec.input_size),
                license=spec.license,
                source=spec.source,
                description=spec.description,
                tags=list(spec.tags),
                weights_available=available,
                supports_random_init=spec.supports_random_init,
                default_weights=spec.default_weights,
            )
        )
    return ModelListResponse(items=items, default_model=settings.default_model)


@router.get("/models/catalogue", summary="Raw registry specs")
def raw_catalogue() -> list[dict[str, Any]]:
    """The registry verbatim, for tooling that wants the unfiltered spec."""
    return catalogue()


def _result_files(directory: Path) -> list[Path]:
    """Benchmark result files, newest first.

    Filenames are timestamp-prefixed, so a name sort is a chronological sort and does not
    need a stat() per file.
    """
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"), reverse=True)


@router.get("/benchmarks", summary="Recorded benchmark runs")
def list_benchmarks(
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
    summary_only: Annotated[bool, Query(description="Omit per-case detail")] = False,
) -> dict[str, Any]:
    """Return recorded benchmark runs, newest first.

    Reads from ``benchmarks/results/`` rather than from the ``benchmark_runs`` table so
    that a fresh clone with no database still serves the committed numbers, which is what
    the frontend's dashboard needs on first boot. A run that was executed through the
    worker is written to both.
    """
    directory = settings.benchmark_results_dir
    runs: list[dict[str, Any]] = []
    for path in _result_files(directory)[:limit]:
        try:
            report = load_report(path)
        except (OSError, ValueError) as exc:
            log.warning("benchmark_result_unreadable", path=str(path), error=str(exc))
            continue
        if summary_only:
            report = {
                key: report.get(key)
                for key in (
                    "schema_version",
                    "run_id",
                    "created_at",
                    "duration_seconds",
                    "environment",
                    "summary",
                )
            }
        runs.append(report)

    return {
        "items": runs,
        "total_files": len(_result_files(directory)),
        "results_dir": str(directory),
    }


@router.get("/benchmarks/{run_id}", summary="One benchmark run")
def get_benchmark(run_id: str, settings: SettingsDep) -> dict[str, Any]:
    # ``run_id`` is used as a filename, so it is matched against the directory listing
    # rather than concatenated into a path: joining a user string onto a directory is how
    # ``../../etc/passwd`` gets read.
    for path in _result_files(settings.benchmark_results_dir):
        if path.stem == run_id:
            return load_report(path)
    raise ApiError(status.HTTP_404_NOT_FOUND, "benchmark_not_found", f"no benchmark run {run_id!r}")
