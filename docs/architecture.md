# Architecture

CutoutML separates *deciding what to do* from *doing it*. The API is a stateless control
plane that validates, persists and enqueues; the worker is where models are loaded and
pixels are touched. Everything a model needs to be interchangeable lives in one shared
library that both import.

## Components

```
                 ┌──────────────────────────────────────────────┐
   browser ─────▶│  apps/web  (Next.js 15, App Router)          │
                 │  /  /models  /benchmarks  /video             │
                 │  proxies /api/* to the API (no CORS in dev)  │
                 └───────────────────────┬──────────────────────┘
                                         │ HTTP + Bearer JWT
                 ┌───────────────────────▼──────────────────────┐
                 │  services/api  (FastAPI, uvicorn)            │
                 │  ────────────────────────────────────────────│
                 │  middleware: request-id → access log →       │
                 │              body-size limit → CORS          │
                 │  routers: health auth assets jobs catalog    │
                 │  rate limiter as a router dependency         │
                 │  NO model imports, NO GPU, stateless         │
                 └──┬──────────────┬───────────────┬────────────┘
                    │              │               │
        ┌───────────▼──┐   ┌───────▼──────┐   ┌────▼──────────────┐
        │  PostgreSQL  │   │    Redis     │   │  Object storage   │
        │  users       │   │  broker      │   │  local FS or any  │
        │  assets      │   │  results     │   │  S3-compatible    │
        │  inference_  │   │  rate-limit  │   │  (MinIO in the    │
        │   jobs/runs  │   │  buckets     │   │   compose stack)  │
        │  benchmark_  │   └───────┬──────┘   └────▲──────────────┘
        │   runs       │           │               │
        └───────▲──────┘           │ cpu | image-gpu | video-gpu
                │                  │               │
                │        ┌─────────▼───────────────┴────────────┐
                └────────┤  services/inference  (Celery)        │
                         │  ────────────────────────────────────│
                         │  tasks.py    retries, acks, routing  │
                         │  runner.py   pure job execution      │
                         │  bounded LRU model cache (2 entries) │
                         │  one worker per queue class          │
                         └─────────────────┬────────────────────┘
                                           │
                         ┌─────────────────▼────────────────────┐
                         │  src/cutoutml  (shared library)      │
                         │  ────────────────────────────────────│
                         │  models/     registry + adapters     │
                         │  pipelines/  image, video, ffmpeg    │
                         │  core/       imaging, refine,        │
                         │              metrics, devices,       │
                         │              queues, config, logging │
                         │  storage/    local + s3 behind one   │
                         │              narrow interface        │
                         │  datasets/   synthetic + real        │
                         │  training/   architecture-agnostic   │
                         │  benchmarks/ harness + renderer      │
                         └──────────────────────────────────────┘
```

The dependency direction is strict: `services/*` imports `cutoutml`, never the reverse,
and the API imports neither torch nor any adapter. That is what keeps the API image small
and its boot time independent of the model catalogue.

## Request lifecycle

A still image, from upload to download:

```
POST /v1/assets  (multipart)
  │
  ├─ middleware: assign request id, start access-log timer, reject on Content-Length
  ├─ auth: decode JWT, load user, check is_active
  ├─ rate limit: token bucket keyed on the resolved user id
  ├─ sniff the real content type from magic bytes (not the client's claim)
  ├─ validate size, decode header, check pixel count against the bomb limit
  ├─ generate a random storage key, stream bytes to storage
  └─ INSERT assets (status=ready, content_sha256, width, height)      → 201

POST /v1/assets/{id}/process
  │
  ├─ validate the model name against the registry, reject unusable models
  ├─ derive an idempotency key (sha256 of asset id + content hash + params)
  ├─ SELECT ... WHERE (owner_id, idempotency_key)  → hit? return the job, 200
  ├─ select_queue(kind, device, gpu_available)
  ├─ INSERT inference_jobs (status=pending) and COMMIT
  └─ apply_async to the chosen queue; on success status=queued              → 202
     on dispatch failure the row stays pending and a maintenance pass retries it

worker: cutoutml.process_image
  │
  ├─ SELECT job FOR UPDATE; already succeeded? return the stored manifest
  ├─ INSERT inference_runs (attempt=N, worker, device, precision, batch)
  ├─ model = cache.get(name, device, precision)      ← loaded once per process
  ├─ ImagePipeline: decode → EXIF orient → letterbox → normalise
  │                 → predict → sigmoid → un-letterbox → refine alpha
  │                 → encode only the requested outputs
  ├─ PUT each output at results/{owner}/{job}/{name}.{ext}   ← deterministic
  └─ UPDATE job status=succeeded, result={manifest}, and the run row with
     duration, peak RSS/VRAM, frames

GET /v1/jobs/{id}                → status, progress, attempts
GET /v1/jobs/{id}/outputs/{kind} → streamed from storage after a DB ownership check
```

Video follows the same path through `process_video`, with the pipeline streaming frames
from an ffmpeg pipe in batches and writing progress back to the job row at most once per
second — a 30 fps two-minute clip would otherwise be 3,600 `UPDATE`s.

## Why the boundaries are where they are

**The API never loads a model.** Loading is 50–500 ms and pins memory; on a GPU it pins
VRAM. Doing it per request is unaffordable, and doing it once per API process means every
API replica needs a GPU. So the API's knowledge of models is limited to the registry's
declarative specs, which import nothing.

**`runner.py` does not know it is a worker.** `tasks.py` owns Celery — retries, acks,
routing, time limits. `runner.py` is plain Python over a database session, a storage
backend and a job row. That split is why job execution is testable without a broker and
reusable from a script.

**Pipelines own no model-specific knowledge.** `ImagePipeline` takes a
`SegmentationModel` and calls `preprocess` / `predict` / `postprocess`. The same code
serves CutoutNet, an ONNX graph and GrabCut. Adding a model does not touch a pipeline.

**Refinement happens at full resolution.** The model runs at its letterboxed input size
(256×256 or 320×320); the alpha map is un-letterboxed to the original dimensions *before*
guided-filter refinement. Refining the small mask and then upsampling reintroduces exactly
the stair-stepping the filter exists to remove.

**Three queues, not one.** A video job holds a worker for minutes and an image job expects
tens of milliseconds; behind one queue that is head-of-line blocking. GPU memory does not
divide by concurrency the way CPU cores do, so image and video workers need different `-c`
values, which requires different worker processes, which requires different queues. See
[ADR-002](decisions/ADR-002-queues.md).

## Middleware order

Starlette applies middleware outermost-first in the order added, and the order here is
deliberate:

1. `RequestContextMiddleware` — outermost, so every log line produced by anything inside
   it, including the exception handlers, carries the request id.
2. `AccessLogMiddleware` — so the duration it records includes the body-size check and
   CORS handling rather than only the handler.
3. `BodySizeLimitMiddleware` — inside those, so an oversized body is rejected before
   routing but is still logged and still gets a request id.
4. `CORSMiddleware` — innermost of ours, because it must be able to short-circuit
   `OPTIONS`.

The rate limiter is deliberately **not** middleware. It is a router dependency, so it runs
after routing and after authentication. That is what lets it key on the resolved user id
and skip `/health` and `/metrics` by construction rather than by a path allow-list that
someone will forget to update.

## Failure handling

| Failure | Behaviour |
|---|---|
| Redis down at dispatch | Job row is already committed; it stays `pending` with a message saying it has not been dispatched, and the API returns 202. A maintenance pass requeues it. |
| Worker killed mid-job | `task_acks_late` returns the message to the broker. The replay finds a non-terminal job and re-runs it; deterministic output keys mean it overwrites its own objects. |
| Duplicate delivery of a finished job | `JobRunner` sees `succeeded` and returns the stored manifest. One `SELECT`. |
| CUDA OOM | Batch size halved and retried, down to 1, with `oom_retry` and the batch size recorded per attempt. Below 1 there is nothing to halve, so it becomes permanent. |
| Corrupt upload, unknown model | Classified non-retryable. Fails immediately rather than three times. |
| Postgres blip | `/health/live` still passes (it touches nothing external), so the fleet is not restarted. `/health/ready` returns 503 and the replica leaves the load balancer. |
| ffmpeg missing | Reported by `/health/ready` and `cutoutml doctor` as informational, not fatal — it only breaks video, and refusing all traffic would take the working image path down too. |

## Observability

- **Structured logs** (`structlog`, JSON in production) with a request id bound for the
  duration of the request and propagated into the Celery task via the task kwargs.
- **Prometheus metrics** at `/metrics`: HTTP request counter, duration histogram and
  in-flight gauge; rate-limit rejections; uploads and upload rejections by reason; jobs
  created and completed; job and inference duration histograms; frames processed; OOM
  retries; model loads; idempotent hits. Latency is a histogram rather than a gauge,
  because a gauge of "last request duration" cannot produce a percentile.
- **`inference_runs`** is the durable record. Metrics expire, logs are sampled, and Celery
  results are gone after a day; the run rows are what answer "how often do we OOM and at
  what batch size did it succeed" months later.

## Where to look

| Question | File |
|---|---|
| What models exist and what can run here | `src/cutoutml/models/registry.py` |
| How an image becomes a cutout | `src/cutoutml/pipelines/image.py` |
| How a video is streamed and encoded | `src/cutoutml/pipelines/video.py`, `pipelines/ffmpeg.py` |
| Alpha edge quality | `src/cutoutml/core/refine.py` |
| Metric definitions | `src/cutoutml/core/metrics.py` |
| Job execution | `services/inference/app/runner.py` |
| Retry and failure classification | `services/inference/app/errors.py`, `tasks.py` |
| Request validation and sniffing | `services/api/app/uploads.py` |
| Benchmark methodology | `src/cutoutml/benchmarks/harness.py`, [docs/benchmarks.md](benchmarks.md) |
