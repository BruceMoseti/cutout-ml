"""Celery application.

Run one worker per queue rather than one worker for everything::

    celery -A services.inference.app.celery_app:celery worker -Q cpu       -c 4 -n cpu@%h
    celery -A services.inference.app.celery_app:celery worker -Q image-gpu -c 2 -n img@%h
    celery -A services.inference.app.celery_app:celery worker -Q video-gpu -c 1 -n vid@%h

Configuration choices that matter, and why
------------------------------------------
``task_acks_late=True``
    Acknowledge *after* the task finishes, so a worker killed mid-job (OOM-killer,
    spot-instance reclaim, ``docker stop``) returns the message to the broker instead of
    losing it. The cost is that a task can be delivered twice, which is exactly why
    every task in :mod:`services.inference.app.tasks` is idempotent.

``worker_prefetch_multiplier=1``
    The default of 4 lets one worker reserve four messages before starting. With
    minutes-long video jobs that means three jobs sitting idle in a worker's local
    buffer while another worker is free - invisible queueing that no dashboard shows.
    Prefetch 1 keeps the queue authoritative.

``task_reject_on_worker_lost=True``
    A hard-killed worker (SIGKILL, notably the Linux OOM killer on a GPU box) requeues
    rather than silently vanishing.

``task_time_limit`` / ``soft_time_limit``
    Soft limit raises ``SoftTimeLimitExceeded`` inside the task so it can clean up temp
    frames; the hard limit kills it. Without a limit, one pathological 4-hour video
    holds a GPU forever.

``broker_transport_options`` visibility timeout
    Redis has no real ack: it re-delivers anything not completed within the visibility
    timeout. If that is shorter than a video job, the job is redelivered *while still
    running* and processed twice. It is set above the hard time limit deliberately.

``worker_max_tasks_per_child``
    Recycle processes periodically. CUDA caching allocators and some decoders fragment
    over thousands of tasks; a bounded child lifetime turns a slow leak into a
    non-issue.
"""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.signals import setup_logging, worker_process_init
from kombu import Queue

from cutoutml.core.config import get_settings
from cutoutml.core.logging import configure_logging, get_logger
from cutoutml.core.queues import ALL_QUEUES, TASK_ROUTES

log = get_logger(__name__)

HARD_TIME_LIMIT = 3600
SOFT_TIME_LIMIT = 3480


def build_celery(**overrides: Any) -> Celery:
    """Construct the Celery app from settings."""
    settings = get_settings()
    app = Celery(
        "cutoutml",
        broker=settings.broker_url,
        backend=settings.result_backend,
        include=["services.inference.app.tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=64,
        task_track_started=True,
        task_time_limit=HARD_TIME_LIMIT,
        task_soft_time_limit=SOFT_TIME_LIMIT,
        result_expires=86400,
        task_default_queue="cpu",
        # kombu.Queue objects, not dicts: Celery reads ``.name`` off each entry when it
        # resolves the destination in apply_async, so a plain dict fails there with
        # "'dict' object has no attribute 'name'" -- at dispatch time, not at startup,
        # which makes it look like a broker outage rather than a config error.
        task_queues=tuple(Queue(q) for q in ALL_QUEUES),
        task_routes=TASK_ROUTES,
        broker_transport_options={
            "visibility_timeout": HARD_TIME_LIMIT + 600,
            "socket_keepalive": True,
        },
        # Eager mode is how the integration tests exercise the real task body without a
        # broker. Exceptions propagate so a test failure is a test failure.
        task_always_eager=settings.celery_task_always_eager,
        task_eager_propagates=settings.celery_task_always_eager,
        **overrides,
    )
    return app


celery = build_celery()


@setup_logging.connect
def _configure_worker_logging(**_: Any) -> None:
    """Replace Celery's logging setup with ours so worker logs are JSON too.

    Connecting to ``setup_logging`` (rather than ``after_setup_logger``) prevents Celery
    from installing its own handlers at all; otherwise every line is emitted twice, once
    structured and once not.
    """
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)


@worker_process_init.connect
def _init_worker_process(**_: Any) -> None:
    """Per-child initialisation.

    Thread count is pinned per process because Celery's prefork pool gives each child
    the full core count by default: 4 children x 8 threads on an 8-core box is 32
    threads fighting over 8 cores, which is measurably *slower* than 4x2. PyTorch cannot
    change this after the first parallel region, so it has to happen here.
    """
    import torch

    settings = get_settings()
    if settings.torch_num_threads > 0:
        torch.set_num_threads(settings.torch_num_threads)
    log.info(
        "worker_process_init",
        torch_threads=torch.get_num_threads(),
        device_default=settings.device,
    )


__all__ = ["HARD_TIME_LIMIT", "SOFT_TIME_LIMIT", "build_celery", "celery"]
