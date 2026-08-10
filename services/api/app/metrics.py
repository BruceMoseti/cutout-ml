"""Prometheus metrics.

Metric design notes:

* **Route templates, not raw paths, as labels.** ``/assets/{asset_id}`` is one label
  value; using the concrete URL would create unbounded cardinality (one time series per
  UUID) and take the metrics endpoint - and eventually Prometheus - down. This is the
  single most common way a metrics integration causes an outage.
* **Histograms, not gauges, for latency.** A gauge of "last request duration" cannot
  produce a percentile. Bucket boundaries are chosen for this workload: image inference
  is tens to hundreds of milliseconds, video is seconds to minutes.
* **A dedicated registry** rather than the global default, so tests can build an
  isolated set of collectors and the app can be instantiated more than once in one
  process without ``Duplicated timeseries`` errors.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

# Latency buckets in seconds. Deliberately dense in the 20-500 ms band where image
# requests live, with a long tail for video jobs.
HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
INFERENCE_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0, 300.0)


class Metrics:
    """Application metrics bound to one registry."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.http_requests = Counter(
            "cutoutml_http_requests_total",
            "HTTP requests by method, templated route and status class",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "cutoutml_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "route"],
            buckets=HTTP_BUCKETS,
            registry=self.registry,
        )
        self.http_in_flight = Gauge(
            "cutoutml_http_requests_in_flight",
            "In-flight HTTP requests",
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "cutoutml_rate_limited_total",
            "Requests rejected by the rate limiter",
            ["backend"],
            registry=self.registry,
        )
        self.uploads = Counter(
            "cutoutml_uploads_total",
            "Accepted uploads by kind",
            ["kind"],
            registry=self.registry,
        )
        self.upload_rejections = Counter(
            "cutoutml_upload_rejections_total",
            "Rejected uploads by reason code",
            ["code"],
            registry=self.registry,
        )
        self.jobs_created = Counter(
            "cutoutml_jobs_created_total",
            "Inference jobs created",
            ["kind", "model", "queue"],
            registry=self.registry,
        )
        self.jobs_completed = Counter(
            "cutoutml_jobs_completed_total",
            "Inference jobs by terminal status",
            ["kind", "model", "status"],
            registry=self.registry,
        )
        self.job_duration = Histogram(
            "cutoutml_job_duration_seconds",
            "End-to-end job execution time",
            ["kind", "model"],
            buckets=INFERENCE_BUCKETS,
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "cutoutml_inference_duration_seconds",
            "Model forward-pass duration",
            ["model", "device", "precision"],
            buckets=INFERENCE_BUCKETS,
            registry=self.registry,
        )
        self.frames_processed = Counter(
            "cutoutml_video_frames_processed_total",
            "Video frames processed",
            ["model"],
            registry=self.registry,
        )
        self.oom_retries = Counter(
            "cutoutml_oom_retries_total",
            "CUDA OOM retries with a halved batch size",
            ["model"],
            registry=self.registry,
        )
        self.model_loads = Counter(
            "cutoutml_model_loads_total",
            "Model load events (cold starts)",
            ["model", "runtime"],
            registry=self.registry,
        )
        self.idempotent_hits = Counter(
            "cutoutml_idempotent_hits_total",
            "Requests that matched an existing job via its idempotency key",
            registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        """Serialised metrics and the content type for the HTTP response."""
        return (generate_latest(self.registry), CONTENT_TYPE_LATEST)


_METRICS: Metrics | None = None


def get_metrics() -> Metrics:
    """Process-wide metrics singleton."""
    global _METRICS
    if _METRICS is None:
        _METRICS = Metrics()
    return _METRICS


def reset_metrics() -> Metrics:
    """Replace the singleton with a fresh registry (tests)."""
    global _METRICS
    _METRICS = Metrics()
    return _METRICS


def route_template(scope: dict[str, Any]) -> str:
    """The templated path for a request, or ``"unmatched"``.

    Starlette puts the matched ``route`` in the ASGI scope; reading ``path_format``
    from it is what keeps label cardinality bounded.
    """
    route = scope.get("route")
    path_format = getattr(route, "path_format", None) or getattr(route, "path", None)
    if path_format:
        return str(path_format)
    return "unmatched"
