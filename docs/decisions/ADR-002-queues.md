# ADR-002: Redis broker with three queues and separate CPU/GPU workers

Status: Accepted

## Context

Segmentation is not something to do inside an HTTP request. A 4K still takes hundreds of
milliseconds; a two-minute clip at 30 fps is 3,600 forward passes and takes minutes.
Holding a connection open for that means client timeouts, no retry story, no visibility,
and a deployment where a rolling restart drops in-flight work.

So the work goes on a queue. That decision is easy. Three harder ones follow.

**What broker.** The system already needs Redis for rate limiting, so a Redis broker is
one fewer moving part. But Redis as a Celery broker is *at-least-once* at best: with
`task_acks_late` a message returns to the queue when a worker dies, which means a task
body can run twice.

**How many queues.** The workloads have incompatible service characteristics:

- A video job occupies a worker for minutes; an image job expects tens of milliseconds.
  Behind a single queue, a handful of videos ahead of an image request turns a 40 ms
  operation into a multi-minute wait. This is head-of-line blocking and no amount of
  prefetch tuning fixes it.
- GPU memory is the scarce resource and, unlike CPU cores, it does not divide by
  concurrency. An image worker can run `concurrency=4` on one GPU; a 4K video worker at
  the same concurrency runs out of memory. Concurrency is a per-worker flag, so different
  concurrency means different worker processes, which means different queues.
- CPU-only work — the classical baselines, a GPU-less deployment, this repository's own
  CI — must keep working when there is no GPU at all.

**Where the GPU lives.** Loading a model onto a GPU costs seconds and pins hundreds of
megabytes of VRAM. Doing that per task is unaffordable; doing it in the API process
means every API replica needs a GPU.

## Decision

**Redis as the broker**, with `task_acks_late=True`, `task_reject_on_worker_lost=True`
and `worker_prefetch_multiplier=1`. Duplicate delivery is treated as a fact of life and
handled by making every task idempotent (see ADR-007) rather than by pretending
exactly-once exists.

`worker_prefetch_multiplier=1` matters more than it looks. The default of 4 lets one
worker reserve four messages before starting any of them, so three jobs can sit in a
worker's local buffer while another worker is idle. That is invisible queueing: no
dashboard shows it, and the only symptom is unexplained latency.

**Three queues**, defined in `src/cutoutml/core/queues.py` — which lives in the shared
library, not in the worker, so the API can route a job without importing Celery or any
model code:

| Queue | Work | Typical worker |
|---|---|---|
| `cpu` | anything with `device=cpu`, and everything when no GPU exists | `-c 2` |
| `image-gpu` | still images on CUDA | `-c 2`, one GPU |
| `video-gpu` | video on CUDA | `-c 1`, one GPU |

`select_queue(kind, device=..., gpu_available=...)` is a pure function. An explicit
`device="cpu"` always wins, because a caller who asked for CPU is usually comparing
against a CPU number. The chosen queue is written to the `inference_jobs.queue` column,
so a job's routing is a recorded fact rather than a consequence of broker-side
configuration that might have changed since.

**A separate worker process per queue class.** The model is loaded once per worker
process — `worker_process_init` warms it — and reused across tasks. The API never loads a
model and never needs a GPU.

**Time limits are explicit**: a 3,600 s hard limit with a 3,480 s soft limit, so a job
that will not finish gets a catchable `SoftTimeLimitExceeded` two minutes before it is
killed and can record why it failed. The broker's `visibility_timeout` is set to the hard
limit plus ten minutes; if it were lower than the task duration, Redis would redeliver a
message that is still being worked on, and the job would run twice concurrently.

## Alternatives considered

**RabbitMQ.** A real broker with real acknowledgement semantics, per-message priorities
and publisher confirms. Rejected because Redis is already a dependency for rate limiting
and this system does not need per-message priorities — the queue split already encodes
the priority structure. The honest trade is that RabbitMQ would give stronger delivery
guarantees; idempotent tasks give most of that benefit for none of the operational cost.
If job volume ever justified priority queues within a class, this is the first thing to
revisit.

**One queue with Celery priorities.** Kombu supports priorities on Redis by fanning out
to `queue`, `queue\x061`, `queue\x062`… under the hood — the priority is emulated with
multiple lists, and the worker's prefetch still crosses them. It also does not solve the
real problem, which is not ordering but *concurrency*: video and image work need
different `-c` values, and one queue cannot serve two concurrencies.

**Postgres as the queue (`SELECT … FOR UPDATE SKIP LOCKED`).** Genuinely attractive: one
fewer service, transactional enqueue with the job row, and exactly-once handoff. Rejected
because it means writing the worker loop, the visibility timeout, the retry backoff and
the scheduling — all of which Celery already has and all of which are easy to get subtly
wrong. Redis is also already present.

**`asyncio` background tasks in the API process.** Rejected: work is lost on deploy, the
API scales on CPU-bound model inference rather than on connections, and every API replica
would need a GPU.

## Consequences

Good:

- A slow video job cannot delay an image job.
- CPU-only and GPU deployments run the same code and the same image; the difference is
  which queues a worker consumes.
- Worker fleets scale independently. Video backlog is a `video-gpu` replica count.
- The queue split is visible in SQL: `GROUP BY queue, status` answers "where is the
  backlog" without a dashboard.

Bad, and accepted:

- Redis is a single point of failure for dispatch. Mitigated, not solved: the job row is
  committed before dispatch is attempted, and a dispatch failure leaves the job `pending`
  with a message saying so, for a maintenance pass to requeue. The API returns 202 with
  `status: pending` rather than a 500, because the request was in fact accepted.
- Redis persistence is weaker than a real broker's. A broker restart can lose queued
  messages; the `pending`/`queued` rows in Postgres are what makes that recoverable.
- Three queues is three things to monitor, and a misrouted job waits forever on a queue
  nobody consumes. `validate_queue()` rejects unknown names at the API boundary.
- Idempotency is now load-bearing. It is exercised by tests rather than assumed.
