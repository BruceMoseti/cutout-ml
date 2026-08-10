"""Latency and accuracy benchmark harness.

Methodology, and why each decision matters
------------------------------------------
**Warmup is discarded.** The first few forward passes pay for lazy kernel selection
(oneDNN picks an algorithm and caches it), memory-pool growth, and on CUDA the context
creation and autotuning. Including them inflates the mean and destroys p50. Warmup
iterations are run and thrown away, and their cost is reported separately as
cold-start.

**Many repetitions, and percentiles rather than a mean.** A single timing is
meaningless: CPU frequency scaling, other tenants on a cloud VM, and page faults
produce a long right tail. p50 describes the typical request, p95/p99 describe what a
user actually complains about, and stddev shows whether the machine was quiet. A mean
alone hides all three. ``docs/benchmarks.md`` expands on this.

**CUDA is synchronised.** Kernel launches are asynchronous; timing them without
``torch.cuda.synchronize()`` measures the launch, not the work, and produces
impossibly fast numbers. The harness synchronises before starting and before stopping
the clock for every repetition.

**Accuracy and latency are measured in separate loops.** The accuracy loop needs
ground truth and per-image metrics (which cost more than the inference itself for
cheap models); mixing them would contaminate the latency distribution.

**Random weights are never allowed to produce an accuracy number.** A model built with
``random_init=True`` gets ``accuracy_valid=False`` and its metrics are recorded as
``None`` with an explicit note, so a latency-only row can never be misread as an
accuracy claim.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import statistics
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from cutoutml.core.config import get_settings
from cutoutml.core.devices import (
    Precision,
    peak_memory_bytes,
    reset_peak_memory,
    resolve_device,
    synchronize,
)
from cutoutml.core.logging import get_logger
from cutoutml.core.metrics import MaskMetrics, aggregate, compute_all
from cutoutml.core.refine import RefineConfig, refine_alpha
from cutoutml.benchmarks.environment import Environment, capture
from cutoutml.datasets.synthetic import SyntheticConfig, SyntheticSegmentationDataset
from cutoutml.models.base import SegmentationModel, WeightsUnavailableError
from cutoutml.models.registry import get_model

log = get_logger(__name__)

RESULTS_SCHEMA_VERSION = 2


@dataclasses.dataclass(slots=True)
class BenchmarkCase:
    """One (model, runtime, precision, resolution, batch size) configuration."""

    model: str
    precision: Precision = "fp32"
    device: str = "auto"
    batch_size: int = 1
    resolution: tuple[int, int] | None = None
    random_init: bool = False
    label: str | None = None
    refine: bool = True
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        suffix = "-randominit" if self.random_init else ""
        res = f"@{self.resolution[0]}x{self.resolution[1]}" if self.resolution else ""
        return f"{self.model}{suffix}{res}-{self.precision}-b{self.batch_size}"

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["resolution"] = list(self.resolution) if self.resolution else None
        d["name"] = self.name
        return d


@dataclasses.dataclass(slots=True)
class LatencyStats:
    """Timing distribution for one case, all values in milliseconds."""

    repetitions: int
    warmup: int
    batch_size: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    per_image_p50_ms: float
    per_image_mean_ms: float
    throughput_images_per_second: float
    throughput_frames_per_second: float
    cold_start_seconds: float | None
    first_inference_ms: float | None
    peak_rss_bytes: int
    peak_vram_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class CaseResult:
    """Everything measured for one case."""

    case: BenchmarkCase
    status: str
    model_metadata: dict[str, Any] | None
    latency: LatencyStats | None
    accuracy: dict[str, float] | None
    accuracy_valid: bool
    model_size_bytes: int | None
    stage_timings_ms: dict[str, float] | None
    error: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.as_dict(),
            "status": self.status,
            "model_metadata": self.model_metadata,
            "latency": self.latency.as_dict() if self.latency else None,
            "accuracy": self.accuracy,
            "accuracy_valid": self.accuracy_valid,
            "model_size_bytes": self.model_size_bytes,
            "stage_timings_ms": self.stage_timings_ms,
            "error": self.error,
            "notes": self.notes,
        }


@dataclasses.dataclass(slots=True)
class BenchmarkConfig:
    """Harness-wide settings."""

    warmup: int = 3
    repetitions: int = 20
    accuracy_samples: int = 64
    dataset_seed: int = 20240817
    dataset_split: str = "test"
    dataset_resolution: tuple[int, int] = (256, 256)
    torch_threads: int = 0
    refine: RefineConfig = dataclasses.field(default_factory=RefineConfig.fast)
    gc_between_cases: bool = True

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["dataset_resolution"] = list(self.dataset_resolution)
        d["refine"] = self.refine.as_dict()
        return d


class BenchmarkHarness:
    """Runs benchmark cases and produces a JSON-serialisable report."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        *,
        dataset: Any = None,
        dataset_description: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig()
        if self.config.torch_threads > 0:
            torch.set_num_threads(self.config.torch_threads)

        if dataset is None:
            data_cfg = SyntheticConfig(resolution=self.config.dataset_resolution)
            synthetic = SyntheticSegmentationDataset(
                count=self.config.accuracy_samples,
                split=self.config.dataset_split,
                seed=self.config.dataset_seed,
                config=data_cfg,
            )
            self.dataset: Any = synthetic
            self.dataset_description = synthetic.manifest(
                {self.config.dataset_split: self.config.accuracy_samples}
            ).as_dict()
        else:
            self.dataset = dataset
            self.dataset_description = dataset_description or (
                dataset.describe() if hasattr(dataset, "describe") else {}
            )

        self.environment: Environment = capture()
        self._samples: list[tuple[np.ndarray, np.ndarray]] | None = None

    # ------------------------------------------------------------------ dataset

    def samples(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Materialise the eval set once and reuse it for every case.

        Generating it per case would be both slow and, for a real dataset, would put
        JPEG decode time inside the measurement loop.
        """
        if self._samples is None:
            limit = min(self.config.accuracy_samples, len(self.dataset))
            self._samples = [self.dataset[i] for i in range(limit)]
        return self._samples

    # --------------------------------------------------------------------- run

    def run_case(self, case: BenchmarkCase) -> CaseResult:
        """Load a model, measure latency then accuracy, and unload it."""
        log.info("benchmark_case_start", case=case.name)
        try:
            model = self._load_model(case)
        except WeightsUnavailableError as exc:
            log.warning("benchmark_case_skipped", case=case.name, reason=str(exc))
            return CaseResult(
                case=case,
                status="skipped",
                model_metadata=None,
                latency=None,
                accuracy=None,
                accuracy_valid=False,
                model_size_bytes=None,
                stage_timings_ms=None,
                error=str(exc),
                notes="weights unavailable in this environment",
            )
        except Exception as exc:
            log.warning("benchmark_case_failed", case=case.name, error=str(exc))
            return CaseResult(
                case=case,
                status="failed",
                model_metadata=None,
                latency=None,
                accuracy=None,
                accuracy_valid=False,
                model_size_bytes=None,
                stage_timings_ms=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        try:
            latency = self._measure_latency(model, case)
            accuracy_valid = not case.random_init
            accuracy = self._measure_accuracy(model, case) if accuracy_valid else None
            stages = self._measure_stages(model, case)
            metadata = model.metadata().as_dict()
            return CaseResult(
                case=case,
                status="ok",
                model_metadata=metadata,
                latency=latency,
                accuracy=accuracy,
                accuracy_valid=accuracy_valid,
                model_size_bytes=self._model_size(model),
                stage_timings_ms=stages,
                notes=(
                    "accuracy: n/a - random weights (latency only)"
                    if case.random_init
                    else ""
                ),
            )
        finally:
            model.unload()
            del model
            if self.config.gc_between_cases:
                gc.collect()

    def run(self, cases: Sequence[BenchmarkCase]) -> dict[str, Any]:
        """Run every case and assemble the report."""
        started = time.perf_counter()
        results = [self.run_case(case) for case in cases]
        elapsed = time.perf_counter() - started

        run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        return {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": round(elapsed, 2),
            "environment": self.environment.as_dict(),
            "config": self.config.as_dict(),
            "dataset": self.dataset_description,
            "cases": [r.as_dict() for r in results],
            "summary": _summarise(results),
        }

    # ----------------------------------------------------------------- internals

    def _load_model(self, case: BenchmarkCase) -> SegmentationModel:
        overrides = dict(case.options)
        if case.resolution:
            overrides["input_size"] = case.resolution
        return get_model(
            case.model,
            device=case.device,
            precision=case.precision,
            random_init=case.random_init,
            load=True,
            **overrides,
        )

    def _measure_latency(self, model: SegmentationModel, case: BenchmarkCase) -> LatencyStats:
        """Timed loop with warmup discarded and CUDA synchronisation."""
        device = resolve_device(case.device)
        samples = self.samples()
        batch = [samples[i % len(samples)][0] for i in range(case.batch_size)]

        tensor, infos = model.preprocess(batch)

        process = psutil.Process()
        reset_peak_memory(device)

        # Warmup, discarded. Its first iteration is also recorded as the "cold" number
        # so the cost of lazy initialisation is visible rather than hidden.
        first_ms: float | None = None
        for i in range(max(0, self.config.warmup)):
            synchronize(device)
            t0 = time.perf_counter()
            model.predict(tensor)
            synchronize(device)
            if i == 0:
                first_ms = (time.perf_counter() - t0) * 1000.0

        timings: list[float] = []
        for _ in range(max(1, self.config.repetitions)):
            synchronize(device)
            t0 = time.perf_counter()
            logits = model.predict(tensor)
            synchronize(device)
            timings.append((time.perf_counter() - t0) * 1000.0)
            del logits

        model.postprocess(model.predict(tensor), infos)  # exercise the full path once
        peak_rss = int(process.memory_info().rss)

        timings.sort()
        mean = statistics.fmean(timings)
        p50 = _percentile(timings, 50)
        return LatencyStats(
            repetitions=len(timings),
            warmup=self.config.warmup,
            batch_size=case.batch_size,
            p50_ms=p50,
            p95_ms=_percentile(timings, 95),
            p99_ms=_percentile(timings, 99),
            mean_ms=mean,
            stddev_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
            min_ms=timings[0],
            max_ms=timings[-1],
            per_image_p50_ms=p50 / case.batch_size,
            per_image_mean_ms=mean / case.batch_size,
            throughput_images_per_second=1000.0 * case.batch_size / mean if mean > 0 else 0.0,
            # Video frames go through the identical path, so frames/s equals images/s;
            # both are reported because readers look for one or the other.
            throughput_frames_per_second=1000.0 * case.batch_size / mean if mean > 0 else 0.0,
            cold_start_seconds=model.load_seconds,
            first_inference_ms=first_ms,
            peak_rss_bytes=peak_rss,
            peak_vram_bytes=peak_memory_bytes(device),
        )

    def _measure_accuracy(
        self, model: SegmentationModel, case: BenchmarkCase
    ) -> dict[str, float]:
        """Full metric bundle over the eval set, batched at the case's batch size."""
        samples = self.samples()
        metrics: list[MaskMetrics] = []
        refine_cfg = self.config.refine if case.refine else RefineConfig.off()

        for start in range(0, len(samples), max(1, case.batch_size)):
            chunk = samples[start : start + max(1, case.batch_size)]
            images = [img for img, _ in chunk]
            alphas = model.infer(images)
            for (image, truth), alpha in zip(chunk, alphas, strict=True):
                refined = refine_alpha(alpha, image, refine_cfg)
                metrics.append(compute_all(refined, truth))

        out = aggregate(metrics)
        out["refinement_enabled"] = 1.0 if case.refine else 0.0
        return out

    def _measure_stages(
        self, model: SegmentationModel, case: BenchmarkCase
    ) -> dict[str, float]:
        """Per-stage breakdown, so a slow row can be attributed to a cause.

        Frequently the answer is not the model: for the classical baseline at 320 px,
        postprocessing plus refinement can exceed inference, and for a fast model on a
        large image, resizing dominates.
        """
        samples = self.samples()
        images = [samples[i % len(samples)][0] for i in range(max(1, case.batch_size))]
        device = resolve_device(case.device)

        t0 = time.perf_counter()
        tensor, infos = model.preprocess(images)
        t1 = time.perf_counter()
        synchronize(device)
        logits = model.predict(tensor)
        synchronize(device)
        t2 = time.perf_counter()
        alphas = model.postprocess(logits, infos)
        t3 = time.perf_counter()
        for alpha, image in zip(alphas, images, strict=True):
            refine_alpha(alpha, image, self.config.refine)
        t4 = time.perf_counter()

        n = len(images)
        return {
            "preprocess": (t1 - t0) * 1000.0 / n,
            "inference": (t2 - t1) * 1000.0 / n,
            "postprocess": (t3 - t2) * 1000.0 / n,
            "refine": (t4 - t3) * 1000.0 / n,
        }

    @staticmethod
    def _model_size(model: SegmentationModel) -> int | None:
        """On-disk size of the weights/graph, or the in-memory parameter size."""
        path = getattr(model, "weights_path", None)
        if isinstance(path, Path) and path.is_file():
            return path.stat().st_size
        onnx_path = getattr(model, "onnx_path", None)
        if isinstance(onnx_path, Path) and onnx_path.is_file():
            return onnx_path.stat().st_size
        module = getattr(model, "module", None)
        if module is not None:
            return int(sum(p.numel() * p.element_size() for p in module.parameters()))
        return None


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list.

    ``statistics.quantiles`` would need n>=2 and produces fixed cut points; this works
    for any n and any percentile, which the p99 of a 20-repetition run needs.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _summarise(results: Sequence[CaseResult]) -> dict[str, Any]:
    ok = [r for r in results if r.status == "ok"]
    with_accuracy = [r for r in ok if r.accuracy_valid and r.accuracy]
    best = max(with_accuracy, key=lambda r: r.accuracy["iou"], default=None) if with_accuracy else None
    fastest = min(
        (r for r in ok if r.latency),
        key=lambda r: r.latency.per_image_p50_ms,  # type: ignore[union-attr]
        default=None,
    )
    return {
        "cases_total": len(results),
        "cases_ok": len(ok),
        "cases_skipped": sum(1 for r in results if r.status == "skipped"),
        "cases_failed": sum(1 for r in results if r.status == "failed"),
        "best_iou_case": best.case.name if best else None,
        "best_iou": round(best.accuracy["iou"], 5) if best else None,
        "fastest_case": fastest.case.name if fastest else None,
        "fastest_per_image_p50_ms": (
            round(fastest.latency.per_image_p50_ms, 4) if fastest and fastest.latency else None
        ),
    }


def save_report(report: dict[str, Any], directory: Path | str | None = None) -> Path:
    """Write a report to ``benchmarks/results/<timestamp>-<id>.json``."""
    target_dir = Path(directory) if directory else get_settings().benchmark_results_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log.info("benchmark_report_saved", path=str(path))
    return path


def load_report(path: Path | str) -> dict[str, Any]:
    """Read a saved report."""
    return json.loads(Path(path).read_text())  # type: ignore[no-any-return]


def latest_report(directory: Path | str | None = None) -> dict[str, Any] | None:
    """Most recent report in ``directory`` by filename (timestamp-prefixed)."""
    target = Path(directory) if directory else get_settings().benchmark_results_dir
    files = sorted(target.glob("*.json"))
    if not files:
        return None
    return load_report(files[-1])
