# CutoutML

Image and video background removal served as a job queue: a FastAPI control plane, Celery
workers that hold models in memory, a pluggable model registry spanning PyTorch, ONNX
Runtime and TensorRT, and a benchmark harness that measures every claim in this README.

## Why I built it

I wanted to know what it actually takes to put a segmentation model behind an HTTP API and
have it survive contact with reality. The model is the easy part. The parts that are not:

- **Alpha is where quality lives.** A binary mask upscaled from 256×256 looks like a
  ransom note. Getting a soft, correctly-aligned edge means un-letterboxing the mask to the
  source resolution *before* refinement, running a guided filter against the source image,
  and measuring boundary quality separately from region quality — because IoU cannot see
  the difference and a human immediately can.
- **Video is a container problem, not a model problem.** MP4 cannot carry alpha, and if you
  ask ffmpeg for RGBA in an MP4 it will silently hand back an opaque file. Whether a given
  libvpx build writes a real alpha plane varies, and reading it back without an
  alpha-aware decoder returns 255 everywhere — so a naive round-trip test can report
  success on an opaque file. This is probed at runtime rather than assumed.
- **A benchmark number without provenance is a lie by omission.** So the harness records
  the commit, whether the tree was dirty, the CPU model, the OS, every library version, the
  dataset fingerprint, the batch size and the precision, and the tables in this README are
  *generated* from that JSON. There is no way to type a number in here by hand.
- **At-least-once delivery is not a corner case.** A Redis broker with late acks will run a
  task twice. The interesting engineering is making the second run harmless and making the
  retry history queryable in SQL rather than greppable in logs.

Every design decision that was expensive to make is written down in
[`docs/decisions/`](docs/decisions/) with the alternatives that were rejected.

## Benchmarks

All numbers below were measured by `benchmarks/run.py` on the machine described in the
table. Four caveats, stated before the numbers rather than after them:

- **There is no GPU on that machine, so every figure is CPU-only.** TensorRT and CUDA rows
  are absent rather than estimated.
- **Latency is single-threaded, so it is a per-core cost and pessimistic.** A dedicated
  machine given one thread per core would be faster; by how much this environment cannot
  say, so no multi-threaded headline number is published. That choice is measured, not
  assumed — on a contended box PyTorch's thread scaling *inverts*, and
  [the curve](docs/benchmarks.md#thread-scaling) is in the suite.
- **Rows marked `†` were timed while another workload held the CPU.** The harness measures
  external CPU demand per case, so this is evidence rather than a disclaimer; those
  latencies are upper bounds on this hardware's cost. Accuracy is deterministic in the
  weights and the eval set and is unaffected — so the accuracy columns carry no caveat.
- **The eval set is synthetic, and the pretrained models were trained on photographs.**
  U²-Net therefore scores *below* a 1.1M-parameter model trained in-repo on the eval set's
  own distribution. That is a domain-shift measurement, not a quality ranking, and it is
  the most useful thing in the table: it is what the synthetic eval set costs.

See [docs/benchmarks.md](docs/benchmarks.md) for the full methodology, the per-case load
table and exactly what was and was not measured.

<!-- BENCHMARKS:BEGIN -->
| Model | Runtime | IoU | MAE | p50 latency | Throughput | Size |
|---|---|---|---|---|---|---|
| **trivial-ones** | numpy | 0.3590 | 0.6410 | 0.1 ms | 8772.9 img/s | n/a |
| **trivial-center** | numpy | 0.4382 | 0.2644 | 0.6 ms | 1650.9 img/s | n/a |
| **classical-saliency** | opencv+numpy | 0.1508 | 0.3772 | 0.9 ms | 1055.3 img/s | n/a |
| **classical** | opencv+numpy | 0.6614 | 0.1455 | 263.2 ms | 3.8 img/s | n/a |
| **classical-saliency-grabcut** | opencv+numpy | 0.1570 | 0.3693 | 397.0 ms | 2.6 img/s | n/a |
| **cutoutnet-tiny** | pytorch-eager | 0.8241 | 0.0693 | 10.0 ms | 98.2 img/s | 0.5 MiB |
| **cutoutnet** | pytorch-eager | 0.8544 | 0.0573 | 31.4 ms | 31.6 img/s | 4.5 MiB |
| **cutoutnet-base** | pytorch-eager | 0.8615 | 0.0508 | 40.2 ms | 24.8 img/s | 16.8 MiB |
| **cutoutnet-onnx ONNX/CPU** | onnxruntime:CPUExecutionProvider | 0.8544 | 0.0573 | 16.7 ms | 60.0 img/s | 4.4 MiB |
| **u2netp** | pytorch-eager | 0.6380 | 0.1388 | 276.7 ms | 3.6 img/s | 4.6 MiB |
| **u2netp-onnx ONNX/CPU** | onnxruntime:CPUExecutionProvider | 0.6380 | 0.1388 | 250.1 ms | 4.0 img/s | 4.4 MiB |
| **u2net** | pytorch-eager | 0.6974 | 0.1221 | 612.6 ms | 1.6 img/s | 168.2 MiB |
| **u2net-onnx ONNX/CPU** | onnxruntime:CPUExecutionProvider | 0.6974 | 0.1221 | 594.4 ms | 1.7 img/s | 167.8 MiB |
| **birefnet random-init** | pytorch-eager | n/a * | n/a * | 224.0 ms | 4.5 img/s | 11.8 MiB |
| **cutoutnet compiled** | pytorch-compile:inductor:default | 0.8544 | 0.0573 | 18.9 ms | 52.9 img/s | 4.5 MiB |

**Benchmark environment**: Intel(R) Xeon(R) Processor, 8 vCPU (8 physical cores), 47 GB RAM, no GPU (CPU-only). GPU: **none**. Every number above was measured by `benchmarks/run.py` on this machine - none are copied from a paper or estimated.

**Latency is single-threaded**, so these are per-core costs and a dedicated machine would beat them. That is deliberate: this box runs other tenants, and multi-threaded timings on it are dominated by barrier waits rather than by the model. The measured curve, and the reasoning, are in [docs/benchmarks.md](docs/benchmarks.md#thread-scaling).

`n/a *` = the network ran with **random weights** (no checkpoint exists that this architecture can load), so its latency is real but accuracy is not measurable.

Source data: [`benchmarks/results/20260810T071109Z-e0c34c05.json`](benchmarks/results/20260810T071109Z-e0c34c05.json) - regenerate with `make bench`. Full methodology and the per-case load table: [docs/benchmarks.md](docs/benchmarks.md).
<!-- BENCHMARKS:END -->

Methodology, the full metric set, the per-stage timing breakdown and the reasons the
synthetic eval set is not comparable to published DUTS/DIS5K numbers are all in
[docs/benchmarks.md](docs/benchmarks.md).

## Features

- **Images**: transparent PNG/WebP, mask-only, composite over a colour, over an image, or
  over a blurred copy of the source. Outputs are requested per job, because encoding a
  4000 px PNG nobody asked for is not free.
- **Video**: composited MP4, genuinely transparent WebM/VP9, ProRes 4444 or QuickTime RLE,
  RGBA PNG sequences, or a mask track. Frames stream through an ffmpeg pipe in bounded
  batches, so peak memory depends on batch size rather than clip length.
- **Temporal smoothing** (EMA or median) with `estimate_flicker()` to report the
  frame-to-frame alpha difference with and without it, so the responsiveness trade-off is a
  measurement rather than a default nobody questions.
- **Sixteen registered models**: three original CutoutNet widths trained in-repo, a
  U²-Net reimplementation that loads the authors' published weights at both sizes plus a
  U²-Net-P trained here, three ONNX Runtime paths, a TensorRT path, a BiRefNet-inspired
  architecture, three classical baselines and two trivial calibration references. The
  trivial pair exists to calibrate the others: a model that cannot beat "predict
  foreground everywhere" has not learned anything.
- **Async API**: JWT auth, two-phase and presigned uploads, idempotency keys, job/run
  history, cancellation, Prometheus metrics, split liveness and readiness.
- **Training**: an architecture-agnostic trainer over a deterministic procedural dataset,
  with every run's hyperparameters and per-epoch metrics committed as JSON.
- **Web console**: Next.js App Router — an upload studio with a before/after slider, a
  video console, the model registry, and the benchmark dashboard.
- **CLI**: `cutoutml models | segment | video | export-onnx | train | benchmark | doctor`.

## Architecture

```
   browser ──▶ apps/web (Next.js 15)          proxies /api/* to the API
                    │
                    ▼
        services/api (FastAPI, stateless, no model imports, no GPU)
          request-id ▸ access log ▸ body-size limit ▸ CORS
          routers: health auth assets jobs catalog
          rate limiter as a router dependency, not middleware
             │              │                  │
      ┌──────▼─────┐  ┌─────▼──────┐   ┌───────▼────────┐
      │ PostgreSQL │  │   Redis    │   │ Object storage │
      │ users      │  │ broker     │   │ local FS or    │
      │ assets     │  │ results    │   │ S3-compatible  │
      │ jobs/runs  │  │ rate limit │   │ (MinIO in the  │
      │ benchmarks │  └─────┬──────┘   │  compose file) │
      └──────▲─────┘        │          └───────▲────────┘
             │      cpu │ image-gpu │ video-gpu│
             │              │                  │
             │   ┌──────────▼──────────────────┴──┐
             └───┤ services/inference (Celery)    │
                 │ tasks.py   retries, acks       │
                 │ runner.py  pure execution      │
                 │ bounded LRU model cache        │
                 └──────────────┬─────────────────┘
                                ▼
                 src/cutoutml — models/ pipelines/ core/
                 storage/ datasets/ training/ benchmarks/
```

The dependency direction is strict: `services/*` imports `cutoutml`, never the reverse, and
the API imports neither torch nor any model adapter. That is what keeps the API image small
and its boot time independent of the model catalogue. Full detail in
[docs/architecture.md](docs/architecture.md).

## Technical highlights

**The model registry is declarative and lazily imported.** A model is a `ModelSpec` — name,
dotted adapter path, input size, licence, weights path, runtime, options. Listing the
catalogue imports nothing, so `GET /v1/models` cannot be broken by a missing TensorRT and
costs no model load. Availability is computed per call, so a checkpoint appearing in
`models/` shows up without a restart, and a client learns a model is unusable *before*
submitting a job to it. ([ADR-001](docs/decisions/ADR-001-model-registry.md))

**Warmup is discarded and reported separately.** The first forward pass pays for lazy
oneDNN algorithm selection, memory-pool growth and, on CUDA, context creation and
autotuning — routinely 2–50× steady state. The harness runs warmup iterations, throws them
away, and reports the first iteration and the model load as `first_inference_ms` and
`cold_start_seconds` instead of letting them corrupt the mean.

**CUDA is synchronised around every timed region.** Kernel launches are asynchronous;
timing `model(x)` without `torch.cuda.synchronize()` measures the launch and produces
impossibly fast numbers. The code path is identical on CPU, where it is a no-op.

**Percentiles, not a mean.** p50/p95/p99/mean/stddev/min/max per case. The stddev is the
number to read first: if it is large relative to p50, the machine was not quiet and nothing
else in the row should be trusted.

**Intra-op threads are pinned, and the reason is measured rather than asserted.** A CPU
latency figure without a thread count is not a measurement, and the omission fails silently:
`torch.set_num_threads` does not reach ONNX Runtime, which sizes its own pool at one thread
per core, so an uncontrolled "PyTorch vs ONNX" comparison differs by eight threads before it
differs by a runtime. One count now reaches both, and is read back *from the runtime* rather
than from the request. The suite then runs single-threaded, which looks perverse for a
throughput project and is the most consequential measurement decision in it: intra-op
parallelism only pays if the worker threads are resident on cores, and a U-Net forward pass
is ~100 parallel regions each ending in a barrier that cannot retire until every worker has
been scheduled. On a machine with more runnable threads than cores, eight threads cost two
orders of magnitude more than one — the
[measured curve](docs/benchmarks.md#thread-scaling) inverts, while ONNX Runtime's does not,
because it fuses the graph into far fewer barriers. Single-threaded timings have no barriers
to lose, so they are the only figures here that reproduce, and they understate the hardware
rather than flattering it.

**The harness measures the machine it is running on, and marks the rows that invalidates.**
Busy cores attributable to processes *outside* this process tree are sampled before every
timing loop, in cores rather than as a load average so the threshold means the same thing on
4 cores and 64. A contended row is published with the evidence attached and marked `†`
wherever it appears — never quietly dropped, never scaled to what a quiet machine would have
done. Accuracy is explicitly *not* qualified, because it is deterministic in the weights and
the eval set. The thread sweep doubles as a repeatability check against the rows it
duplicates, and where the two disagree the renderer says so: on this box the same model at
the same thread count came out 1.7× apart minutes later, which is what the marks mean in
practice.

**Random weights can never produce an accuracy number.** An architecture with no loadable
checkpoint is benchmarked for latency with random initialisation and marked
`accuracy_valid=false`, rendering as `n/a` with a footnote. A latency-only row cannot be
misread as an accuracy claim, and `random_init` is refused for any model that does not
explicitly opt in — so it is unreachable from an API request. Only BiRefNet is in that
position now, and for a structural reason rather than a network one: its published weights
target a Swin backbone whose shapes this reimplementation cannot load.

**The published U²-Net weights were recovered from a BatchNorm-folded ONNX graph, and the
recovery is proved rather than asserted.** The authors' Apache-2.0 checkpoints live on
Google Drive and HuggingFace, and HuggingFace is blocked here; what is reachable is an ONNX
export of the same weights. It was exported with constant folding on, so every
`Conv → BatchNorm` pair has collapsed into one biased convolution and the parameter names of
112 of the 119 convolutions are gone — they appear as numeric temporaries. `from_onnx` pairs
ONNX `Conv` nodes with the module's convolutions positionally, recovering the PyTorch
execution order by running the module under forward hooks rather than trusting construction
order, then verifies the result three ways: pairwise shapes, the seven convolutions that
kept their names landing where their names say, and **numerical parity against onnxruntime —
1.4e-7 for the 44M model**, about three orders of magnitude finer than one 8-bit alpha
level. The BatchNorms become exact identities, which makes the checkpoint equivalent in
`eval()` and unusable for fine-tuning; that limitation is recorded inside the file.

**That conversion found a bug no test could have.** It refused to convert, because the full
U²-Net's decoder widths had been derived from the encoder table instead of transcribed from
the paper — so the architecture was not shape-compatible with the official checkpoint that
three docstrings claimed it was. The failure would have been silent in three separate ways:
the `lite` variant is uniformly 64 wide so both readings coincide there, a from-scratch
training run learns whatever shapes it is given, and the adapter loads with `strict=False`
to tolerate upstream key renaming — so loading the real checkpoint would have skipped the
mismatched tensors and run inference on random weights with nothing but a log line.

**Preprocessing is part of the model.** U²-Net's reference pipeline divides each image by
its own maximum intensity before normalising. That was skipped here on the reasonable-sounding
grounds that it is a no-op for any image containing a saturated pixel — true of most
photographs, false of 9 of 16 images in this eval set. Since preprocessing is not part of an
ONNX artefact, the ONNX registry entries carry the requirement explicitly, alongside the fact
that those graphs bake in their own sigmoid; applying a second one costs several IoU points
and raises no error.

**Calibration rows are mandatory.** `trivial-ones` predicts foreground everywhere and
`trivial-center` draws a fixed ellipse. IoU is only interpretable against what predicting
nothing achieves; any model that does not clearly beat those has learned nothing.

**Alpha refinement is a measured stack, not a magic function.** Guided filtering against
the source image, morphological cleanup, and edge feathering — each stage independently
switchable and unit-tested, applied at full resolution because refining the small mask and
upsampling reintroduces exactly the stair-stepping the filter exists to remove.

**Three queues and separate workers.** Video holds a worker for minutes; images expect tens
of milliseconds. Behind one queue that is head-of-line blocking. GPU memory does not divide
by concurrency the way CPU cores do, so image and video workers need different `-c` values,
which requires different processes. ([ADR-002](docs/decisions/ADR-002-queues.md))

**OOM retries halve the batch size and record it.** Re-running an identical OOM is a
guaranteed second OOM. The reduction is persisted on the job row, so it survives a worker
restart and appears in `GET /v1/jobs/{id}`, and the attempt is a row in `inference_runs` —
which turns "at what batch size does 4K video actually fit" into a SQL query.

**Uploads are validated in cost order and nothing the client says is trusted.** Content
type is sniffed from magic bytes and cross-checked against both the declared type and the
extension; size is checked against the real byte count, not `Content-Length`; and the pixel
count is checked from the image header *before* decoding, because a 20 KB PNG can declare
50,000 × 50,000 pixels.

**Storage keys are server-generated and random.** The client's filename never touches a
path, so traversal is structurally impossible rather than mitigated. Local writes are
atomic via `os.replace`, because an API polling for a result a worker is writing must never
read a partial file. ([ADR-006](docs/decisions/ADR-006-storage-layout.md))

**The ONNX export is verified numerically or it is not an export.** Parity is checked to
1e-3 on post-sigmoid probabilities — under one 8-bit alpha level, so the difference cannot
survive quantisation to a PNG. And the execution provider onnxruntime *actually chose* is
recorded, not the one requested, because a row labelled GPU that ran on CPU is worse than
no row. ([ADR-005](docs/decisions/ADR-005-onnx-runtime.md))

**The eval set is a fingerprint, not a folder.** The dataset is procedurally generated from
a seed; `datasets/synthetic-eval.json` commits the generator version, every parameter and a
SHA-256 over the first samples. CI regenerates and compares, so a change in OpenCV's
resampling defaults fails the build instead of silently shifting every accuracy number.
([ADR-004](docs/decisions/ADR-004-synthetic-dataset.md))

## Tech stack

| Layer | Choice |
|---|---|
| Models | PyTorch 2.x, ONNX Runtime, TensorRT adapter (unmeasured here) |
| Vision | OpenCV (headless), Pillow, NumPy, SciPy |
| API | FastAPI, uvicorn, Pydantic v2 / pydantic-settings |
| Queue | Celery 5 on Redis 7 |
| Database | PostgreSQL 16, SQLAlchemy 2.0 ORM, Alembic |
| Storage | Local filesystem or any S3-compatible endpoint (boto3); MinIO in compose |
| Auth | bcrypt (cost 12, SHA-256 pre-hash), PyJWT HS256 |
| Video | ffmpeg 6 via streaming subprocess pipes |
| Observability | structlog (JSON), prometheus-client |
| Frontend | Next.js 15 App Router, React 19, TypeScript, Tailwind, Vitest |
| Tooling | ruff, mypy, pytest, GitHub Actions, Docker Compose |

## Running locally

### Docker Compose

```bash
cp .env.example .env
# CUTOUTML_JWT_SECRET is required; compose refuses to start without it.
python -c 'import secrets; print("CUTOUTML_JWT_SECRET=" + secrets.token_urlsafe(48))' >> .env

docker compose up -d --build          # postgres, redis, minio, api, 3 workers, web
docker compose run --rm migrate       # one-shot; not an API entrypoint step
open http://localhost:3000            # console      http://localhost:8000/docs — API
```

Migrations are a separate one-shot service on purpose: running them from the API means N
replicas racing to migrate the same database on every deploy.

The compose file's **schema is validated** (`docker compose config`, which needs no
daemon) and its images are built by the `docker` job in CI. It has never been brought
`up` — the machine this was developed on had no Docker daemon, and saying otherwise would
be exactly the kind of claim this project avoids making.

### Manual

Needs Python 3.12+, Node 20+, ffmpeg, PostgreSQL 15+, Redis 7+.

```bash
make venv install install-web
cp .env.example .env
make migrate
make doctor          # what actually works here: GPU, ffmpeg, onnxruntime, database
```

Four processes, four terminals:

```bash
make api             # uvicorn on :8000
make worker          # celery, cpu queue
make web             # next dev on :3000
# make worker-gpu    # image-gpu + video-gpu, needs CUDA
```

The learned models need checkpoints, which are **not committed**
([ADR-008](docs/decisions/ADR-008-no-committed-weights.md)):

```bash
make weights         # trains tiny/small/base/u2net-lite, then re-exports ONNX
```

Until that finishes, `GET /v1/models` reports the learned models as
`weights_available: false` and the classical and trivial baselines serve requests normally
— so the API, both pipelines, the CLI, the console and the whole test suite work
immediately after a clone.

### Without any services

```bash
cutoutml doctor
cutoutml models
cutoutml segment photo.jpg -o out/ --outputs transparent_png mask_png
cutoutml video clip.mp4 -o out.webm --mode transparent --container webm
cutoutml benchmark --quick
```

## API

Full reference in [docs/api.md](docs/api.md); the generated schema is at `/docs`,
`/redoc` and `/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live`, `/health/ready` | Liveness (touches nothing) and readiness (503 when Postgres or Redis is down) |
| `POST` | `/v1/auth/register`, `/v1/auth/login` | Accounts and tokens |
| `GET` | `/v1/auth/me` | The authenticated principal |
| `POST` | `/v1/assets` | Upload in one multipart request |
| `POST` | `/v1/assets/upload-url` → `PUT .../content` → `POST .../complete` | Presigned upload, direct to storage |
| `GET`/`DELETE` | `/v1/assets`, `/v1/assets/{id}`, `/v1/assets/{id}/content` | List, inspect, download, delete |
| `POST` | `/v1/assets/{id}/process` | Queue a job. 202 with a job id, or 200 with the existing job when idempotent. |
| `GET` | `/v1/jobs`, `/v1/jobs/{id}` | Status, progress, and every attempt with its device and batch size |
| `GET` | `/v1/jobs/{id}/result`, `/v1/jobs/{id}/outputs/{kind}` | Manifest and per-output download |
| `POST` | `/v1/jobs/{id}/cancel` | Cooperative cancellation |
| `GET` | `/v1/models`, `/v1/models/catalogue` | Registry with per-model availability, licence and source |
| `GET` | `/v1/benchmarks`, `/v1/benchmarks/{run_id}` | Recorded runs with full provenance |
| `GET` | `/metrics` | Prometheus. Unauthenticated; must be network-restricted. |

Errors are always `{"error": {"code", "message", "request_id", "details?}}`. `code` is
stable and safe to branch on. A resource owned by another user returns `404`, not `403`,
because a `403` confirms the id exists.

## Testing

```bash
make check                 # ruff lint + format check + mypy + unit tests
make test-integration      # needs Postgres, Redis, ffmpeg
make eval-data             # eval-set fingerprint still matches the manifest
cd apps/web && npm run lint && npm run typecheck && npm test && npm run build
```

Unit tests need no network, database or broker and run in seconds. Integration tests are
marked and skipped without their services — and CI fails if *every* integration test
skipped, because a broken service container otherwise turns the job green.

Numerical tests assert against hand-computed values rather than against the
implementation's own output, so a test cannot pass a refactor that changed the answer. CI
also re-renders the benchmark tables and diffs them, which is what makes "every number was
measured" enforceable.

## Design decisions

| ADR | Decision |
|---|---|
| [001](docs/decisions/ADR-001-model-registry.md) | Models are declared in a registry and loaded by name |
| [002](docs/decisions/ADR-002-queues.md) | Redis broker, three queues, separate CPU/GPU workers |
| [003](docs/decisions/ADR-003-video-output.md) | Four video output modes; alpha capability is probed, not assumed |
| [004](docs/decisions/ADR-004-synthetic-dataset.md) | Evaluate on a procedurally generated, fingerprinted dataset |
| [005](docs/decisions/ADR-005-onnx-runtime.md) | ONNX Runtime as a peer serving runtime, with verified export parity |
| [006](docs/decisions/ADR-006-storage-layout.md) | Server-generated random keys behind a narrow storage interface |
| [007](docs/decisions/ADR-007-idempotency.md) | Idempotency keys and a job/run split instead of exactly-once delivery |
| [008](docs/decisions/ADR-008-no-committed-weights.md) | No model weights in git |

Also: [architecture](docs/architecture.md) · [data model](docs/data-model.md) ·
[API](docs/api.md) · [benchmarks](docs/benchmarks.md) · [security](docs/security.md) ·
[models and attribution](docs/models.md) · [licensing](docs/licensing.md)

## Roadmap

Ordered by what I would do next, not by ambition:

1. **Measure a GPU.** Every fp16, `torch.compile`-on-CUDA and TensorRT code path is
   implemented and type-checked but unmeasured. The rows are absent, and they should be
   real.
2. **Re-measure on a dedicated machine.** Eight of twenty-seven latency rows are still
   marked `†`, and the suite is single-threaded throughout because that is the only figure
   a shared box can produce twice. A dedicated machine would give a trustworthy
   multi-threaded number and a thread-scaling curve that reflects the runtimes rather than
   the scheduler. The harness already records everything needed to tell the two runs apart.
3. **Evaluate on DUTS and DIS5K.** `RealSegmentationDataset` already handles both; what is
   missing is a run on hardware that can reach them, which would make the accuracy column
   comparable to published work.
4. **Webhooks.** Job completion is discovered by polling today.
5. **A reconciliation job.** The database is authoritative for storage, so an object
   without a row is garbage and a row without an object is a broken asset. Nothing detects
   either yet.
6. **Trimap-based matting.** The current models predict alpha directly. A trimap stage
   would materially improve hair and fur, which is where the synthetic eval set is
   deliberately hardest.
7. **API keys and refresh tokens.** Machine-to-machine use currently means storing a
   password.

## Licence

MIT — see [LICENSE](LICENSE). No weights are committed here at all, and the ones this
project produces or fetches fall into two clearly separated groups:

- **Trained here** (`cutoutnet-*`, `u2net-lite`): trained on data generated here, with no
  pretrained initialisation anywhere, so the weights are MIT. That is the part most
  background removers cannot say, because a research-use dataset carries its restriction
  into whatever is trained on it.
- **Published upstream** (`u2net`, `u2netp` and their ONNX pairs): the U²-Net authors'
  Apache-2.0 weights, fetched on demand. Converting them from ONNX to a PyTorch checkpoint
  does not change their licence, and each converted file carries its licence and source
  digest inside it rather than in a sidecar that can be separated from the weights.

Attribution, every downloadable weight file and its pinned SHA-256 are in
[NOTICE](NOTICE); per-architecture detail is in [docs/models.md](docs/models.md), and the
full project-level analysis — including the one LGPL dependency and why ffmpeg's GPL does
not propagate — is in [docs/licensing.md](docs/licensing.md).

One warning worth repeating outside a table: BiRefNet's *code* is MIT, but **some
third-party fine-tuned BiRefNet checkpoints are released under non-commercial terms**. That
applies to the file, not to the repository it came from. This project offers no BiRefNet
download and its reimplementation cannot load those checkpoints anyway, so none can arrive
here by accident.
