"""Queue names and the routing rule that maps a job onto one.

Three queues, not one
---------------------
``cpu``, ``image-gpu`` and ``video-gpu`` exist because the three workloads have
incompatible service characteristics and a single queue makes them fight:

* A video job occupies a worker for **minutes**. An image job expects to finish in
  **tens of milliseconds**. Behind one queue, a handful of videos ahead of an image
  request turn a 40 ms operation into a several-minute wait - head-of-line blocking
  that no amount of extra prefetch tuning fixes.
* GPU memory is the scarce resource and it is *not* divisible by concurrency the way
  CPU cores are. An image worker can run ``concurrency=4`` on one GPU; a 4K video
  worker at the same concurrency OOMs. Concurrency is a per-worker setting, so
  different concurrencies require different workers, which requires different queues.
* CPU-only work (the classical baseline, thumbnailing, a GPU-less deployment) should
  not idle on a GPU worker, and should keep working when no GPU exists at all.

The full argument, including the alternative we rejected (one queue with priorities),
is in ``docs/decisions/ADR-002-queues.md``.

This module lives in the shared package rather than in the worker so the API can
route a job without importing Celery or any model code.
"""

from __future__ import annotations

from typing import Final, Literal

QueueName = Literal["cpu", "image-gpu", "video-gpu"]

CPU: Final[QueueName] = "cpu"
IMAGE_GPU: Final[QueueName] = "image-gpu"
VIDEO_GPU: Final[QueueName] = "video-gpu"

ALL_QUEUES: Final[tuple[QueueName, ...]] = (CPU, IMAGE_GPU, VIDEO_GPU)

#: Default Celery routing key per task name. The worker applies this; the API sets an
#: explicit queue on ``apply_async`` so a job's queue is recorded in the database and
#: never depends on broker-side configuration drift.
TASK_ROUTES: Final[dict[str, dict[str, str]]] = {
    "cutoutml.process_image": {"queue": CPU},
    "cutoutml.process_video": {"queue": CPU},
    "cutoutml.run_benchmark": {"queue": CPU},
}


def select_queue(
    kind: str, *, device: str | None = None, gpu_available: bool | None = None
) -> QueueName:
    """Choose the queue for a job.

    ``device`` is the *request*, ``gpu_available`` is what the cluster can actually
    offer. When ``gpu_available`` is omitted the local machine is probed, which is
    right for a single-node deployment and wrong for a split one - in a cluster the
    API has no GPU, so pass the value explicitly (``CUTOUTML_GPU_WORKERS=1``).

    An explicit ``device="cpu"`` always wins: a caller who asked for CPU gets CPU,
    because that request usually means "I am comparing against the CPU number".
    """
    requested = (device or "auto").strip().lower()
    if requested.startswith("cpu"):
        return CPU

    if gpu_available is None:
        from cutoutml.core.devices import cuda_available

        gpu_available = cuda_available()

    if not gpu_available:
        # An explicit cuda request with no GPU anywhere still has to run somewhere;
        # the adapters fall back to CPU, so route it to the queue that exists.
        return CPU

    return VIDEO_GPU if kind == "video" else IMAGE_GPU


def is_gpu_queue(queue: str) -> bool:
    return queue in {IMAGE_GPU, VIDEO_GPU}


def validate_queue(queue: str) -> QueueName:
    """Narrow a string to a known queue name, raising on anything else."""
    if queue not in ALL_QUEUES:
        raise ValueError(f"unknown queue {queue!r}; expected one of {', '.join(ALL_QUEUES)}")
    return queue  # type: ignore[return-value]


__all__ = [
    "ALL_QUEUES",
    "CPU",
    "IMAGE_GPU",
    "TASK_ROUTES",
    "VIDEO_GPU",
    "QueueName",
    "is_gpu_queue",
    "select_queue",
    "validate_queue",
]
