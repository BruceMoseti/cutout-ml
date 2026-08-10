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

from cutoutml.benchmarks.contention import LoadSnapshot, sample
from cutoutml.benchmarks.environment import Environment, capture
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
from cutoutml.datasets.synthetic import SyntheticConfig, SyntheticSegmentationDataset
from cutoutml.models.base import (
    SegmentationModel,
    TorchSegmentationModel,
    WeightsUnavailableError,
)
from cutoutml.models.registry import get_model, resolve_spec
from cutoutml.models.torch_compile import CompileOutcome, compile_module, not_attempted

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
    #: Wrap the module in ``torch.compile``. Ignored for non-PyTorch runtimes, and a
    #: failed compile is recorded rather than silently falling back to eager.
    compile: bool = False
    compile_mode: str = "default"
    #: Intra-op threads for this case, overriding :attr:`BenchmarkConfig.threads`.
    #: Only the thread-scaling sweep sets it; every other case inherits the run-wide
    #: value so that a runtime comparison is not secretly also a thread comparison.
    threads: int | None = None
    #: Set ``False`` for a case that exists only to time something, where the accuracy
    #: is already established by another row. The eval loop is the expensive half of a
    #: case, and re-running it across a thread sweep would add minutes to produce
    #: numbers identical to the row already in the table. A skipped accuracy is
    #: recorded as ``None`` with a reason, never as a zero.
    measure_accuracy: bool = True
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        suffix = "-randominit" if self.random_init else ""
        res = f"@{self.resolution[0]}x{self.resolution[1]}" if self.resolution else ""
        compiled = "-compiled" if self.compile else ""
        return f"{self.model}{suffix}{res}{compiled}-{self.precision}-b{self.batch_size}"

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
    #: Intra-op threads the runtime was actually given. Recorded per case because CPU
    #: latency is meaningless without it, and because a mismatched thread count between
    #: two rows invalidates any comparison drawn between them.
    threads: int
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
    compile: CompileOutcome = dataclasses.field(default_factory=not_attempted)
    #: External CPU demand while this case's latency was measured. Attached per case
    #: rather than per run because a suite takes many minutes and a neighbouring
    #: workload can start or finish partway through it.
    load: LoadSnapshot | None = None
    error: str | None = None
    notes: str = ""

    @property
    def runtime(self) -> str:
        """Runtime label for the report table.

        Taken from the model's own metadata for non-PyTorch backends (onnxruntime
        reports which execution provider it actually got) and from the compile
        outcome otherwise.
        """
        declared = (self.model_metadata or {}).get("runtime", "")
        if declared and not declared.startswith("pytorch"):
            return str(declared)
        return self.compile.runtime_label

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.as_dict(),
            "status": self.status,
            "runtime": self.runtime,
            "model_metadata": self.model_metadata,
            "latency": self.latency.as_dict() if self.latency else None,
            "accuracy": self.accuracy,
            "accuracy_valid": self.accuracy_valid,
            "model_size_bytes": self.model_size_bytes,
            "stage_timings_ms": self.stage_timings_ms,
            "compile": self.compile.as_dict(),
            "load": self.load.as_dict() if self.load else None,
            "latency_trustworthy": self.load.quiet if self.load else None,
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
    #: Intra-op threads given to every runtime; ``0`` means "the runtime's own default",
    #: which is one thread per core.
    #:
    #: The default is 1, which looks perverse for a throughput-oriented project and is
    #: the single most important measurement decision in this harness. Intra-op
    #: parallelism only pays off if the runtime's worker threads are actually resident
    #: on cores. When they are not - because the machine has more runnable threads than
    #: cores - every one of the ~100 parallel regions in a U-Net forward pass ends in a
    #: barrier that waits for a descheduled thread, and the barriers dominate. Measured
    #: on this 8-vCPU box while it carried other tenants: cutoutnet at 320x320 took
    #: 46.7 ms with one thread and 2854 ms with eight, a 61x *penalty* for eight times
    #: the threads. Single-threaded numbers contain no barriers, so they are the only
    #: CPU latency figures that are reproducible on a shared machine, and they are
    #: honest about per-core cost. Use ``--threads 0`` on a quiet, dedicated box, and
    #: read ``thread_scaling_cases()`` for the scaling curve itself.
    threads: int = 1
    refine: RefineConfig = dataclasses.field(default_factory=RefineConfig.fast)
    gc_between_cases: bool = True
    #: Seconds spent sampling CPU contention before each case's timing loop.
    load_sample_seconds: float = 1.0

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
        self._default_torch_threads = torch.get_num_threads()
        self._apply_threads(self.config.threads)

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
        self._apply_threads(self._case_threads(case))
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
        except Exception as exc:  # noqa: BLE001 - one bad case must not abort the suite
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
            compile_outcome = self._maybe_compile(model, case)
            load = sample(self.config.load_sample_seconds)
            latency = self._measure_latency(model, case)
            accuracy_valid = not case.random_init and case.measure_accuracy
            accuracy = self._measure_accuracy(model, case) if accuracy_valid else None
            stages = self._measure_stages(model, case)
            metadata = model.metadata().as_dict()
            notes: list[str] = []
            if case.random_init:
                notes.append("accuracy: n/a - random weights (latency only)")
            elif not case.measure_accuracy:
                notes.append(
                    "accuracy: not remeasured - this case varies only the runtime "
                    "configuration, so it is identical to the row measured at the "
                    "run-wide setting"
                )
            if compile_outcome.attempted and not compile_outcome.succeeded:
                notes.append("torch.compile failed; timings are eager-mode")
            if not load.quiet:
                notes.append(
                    f"latency measured under contention ({load.external_busy_cores:.1f} of "
                    f"{load.logical_cpus} cores busy externally); accuracy is unaffected"
                )
                log.warning("benchmark_case_contended", case=case.name, load=load.summary)
            return CaseResult(
                case=case,
                status="ok",
                model_metadata=metadata,
                latency=latency,
                accuracy=accuracy,
                accuracy_valid=accuracy_valid,
                model_size_bytes=self._model_size(model),
                stage_timings_ms=stages,
                compile=compile_outcome,
                load=load,
                notes="; ".join(notes),
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

    def _apply_threads(self, threads: int) -> int:
        """Set PyTorch's intra-op width, returning the count now in force.

        ``0`` restores the count PyTorch chose at import, so a sweep can return to the
        default rather than pinning whatever value the previous case happened to use.
        """
        effective = threads if threads > 0 else self._default_torch_threads
        torch.set_num_threads(effective)
        return effective

    def _case_threads(self, case: BenchmarkCase) -> int:
        return case.threads if case.threads is not None else self.config.threads

    def _effective_threads(self, model: SegmentationModel, case: BenchmarkCase) -> int:
        """Ask the runtime that actually ran, rather than trusting the request.

        A requested count can be ignored - ONNX Runtime resolves 0 to the core count -
        and a row whose thread count is wrong is worse than one with none at all.
        """
        threads = getattr(model, "effective_intra_op_threads", None)
        if isinstance(threads, int) and threads > 0:
            return threads
        if isinstance(model, TorchSegmentationModel):
            return int(torch.get_num_threads())
        # Pure numpy/OpenCV baselines: BLAS may still thread, but nothing in the
        # harness controls it, so report the request rather than inventing a number.
        return self._case_threads(case) or int(torch.get_num_threads())

    def _load_model(self, case: BenchmarkCase) -> SegmentationModel:
        overrides = dict(case.options)
        if case.resolution:
            overrides["input_size"] = case.resolution
        # ONNX Runtime sizes its own thread pool at session creation and ignores
        # torch.set_num_threads, so the count has to be handed to the adapter. Without
        # this, a "PyTorch vs ONNX" row pair silently compares 1 thread against 8.
        spec = resolve_spec(case.model)
        if spec.runtime == "onnxruntime" and "intra_op_threads" not in overrides:
            overrides["intra_op_threads"] = self._case_threads(case)
        return get_model(
            case.model,
            device=case.device,
            precision=case.precision,
            random_init=case.random_init,
            load=True,
            **overrides,
        )

    def _maybe_compile(self, model: SegmentationModel, case: BenchmarkCase) -> CompileOutcome:
        """Compile the module in place when the case asks for it.

        Compilation happens *before* the warmup loop and drives one forward pass, so
        the timing loop never includes codegen. Non-PyTorch adapters have no module to
        compile and are reported as not-attempted rather than as a failure.
        """
        if not case.compile:
            return not_attempted()
        if not isinstance(model, TorchSegmentationModel) or model.module is None:
            log.info("compile_skipped_non_torch", case=case.name, runtime=type(model).__name__)
            return not_attempted()

        samples = self.samples()
        batch = [samples[i % len(samples)][0] for i in range(max(1, case.batch_size))]
        example, _ = model.preprocess(batch)
        example = example.to(model.device)
        if model.use_channels_last:
            example = example.contiguous(memory_format=torch.channels_last)

        compiled, outcome = compile_module(model.module, example, mode=case.compile_mode)
        model.module = compiled
        return outcome

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
            threads=self._effective_threads(model, case),
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

    def _measure_accuracy(self, model: SegmentationModel, case: BenchmarkCase) -> dict[str, float]:
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

    def _measure_stages(self, model: SegmentationModel, case: BenchmarkCase) -> dict[str, float]:
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
    """Headline figures for the report.

    ``best_iou`` only ever considers cases with ``accuracy_valid`` set. A randomly
    initialised model produces a real IoU number that means nothing, and letting one win
    "best" would put a fabricated accuracy at the top of the report.

    The scores are paired with their results up front rather than read inside a ``key=``
    lambda, so the "accuracy is present" check and the use of that accuracy are the same
    expression instead of two that have to be kept in agreement.
    """
    ok = [r for r in results if r.status == "ok"]

    scored: list[tuple[CaseResult, float]] = [
        (r, float(r.accuracy["iou"]))
        for r in ok
        if r.accuracy_valid and r.accuracy and "iou" in r.accuracy
    ]
    timed: list[tuple[CaseResult, float]] = [
        (r, r.latency.per_image_p50_ms) for r in ok if r.latency is not None
    ]

    best = max(scored, key=lambda pair: pair[1], default=None)
    fastest = min(timed, key=lambda pair: pair[1], default=None)

    # Surfaced at the top of the report so a reader does not have to open every case to
    # learn that the timings came off a busy machine. Accuracy is deterministic and is
    # therefore never qualified by this flag.
    measured = [r.load for r in ok if r.load is not None]
    contended = [snapshot for snapshot in measured if not snapshot.quiet]

    return {
        "cases_total": len(results),
        "cases_ok": len(ok),
        "cases_skipped": sum(1 for r in results if r.status == "skipped"),
        "cases_failed": sum(1 for r in results if r.status == "failed"),
        "best_iou_case": best[0].case.name if best else None,
        "best_iou": round(best[1], 5) if best else None,
        "fastest_case": fastest[0].case.name if fastest else None,
        "fastest_per_image_p50_ms": round(fastest[1], 4) if fastest else None,
        "cases_measured_under_contention": len(contended),
        "latency_trustworthy": not contended,
        "peak_external_busy_cores": (
            round(max(s.external_busy_cores for s in measured), 3) if measured else None
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
