# Data model

Five tables in PostgreSQL. Definitions are in `src/cutoutml/db/models.py`; the schema is
created by Alembic (`infra/alembic/versions/`), never by `create_all` outside tests.

```
users ──1:N──▶ assets ──1:N──▶ inference_jobs ──1:N──▶ inference_runs
  │                                  ▲
  └────────────────1:N───────────────┘

benchmark_runs   (owner_id nullable, otherwise standalone and append-only)
```

## `users`

An API principal.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` PK | |
| `email` | `varchar(320)` unique | 320 = 64 local + `@` + 255 domain, the RFC maximum |
| `password_hash` | `varchar(255)` | bcrypt, per-hash salt. The plaintext never leaves the request handler. |
| `display_name` | `varchar(120)` | |
| `is_active` | `bool` | Checked on **every** authenticated request, so revocation takes effect without waiting for token expiry |
| `is_admin` | `bool` | |
| `rate_limit_per_minute` | `int` null | Per-user override of the global limit |
| `created_at`, `updated_at` | `timestamptz` | `server_default now()`, `onupdate` |

## `assets`

An uploaded image or video. Uploads are two-phase, so a row exists before the bytes do:
`awaiting_upload → ready`, or `failed`, or `deleted`.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` PK | |
| `owner_id` | `uuid` FK → `users` `ON DELETE CASCADE` | |
| `kind` | `varchar(16)` | `image` \| `video` |
| `status` | `varchar(32)` | `awaiting_upload` \| `ready` \| `failed` \| `deleted` |
| `storage_backend` | `varchar(16)` | Recorded so a migration between backends is detectable rather than guessed |
| `storage_key` | `varchar(512)` unique | Server-generated and random. See [ADR-006](decisions/ADR-006-storage-layout.md). |
| `original_filename` | `varchar(255)` | Display only. **Never** used to build a path. |
| `content_type` | `varchar(127)` | Sniffed from magic bytes, not the client's claim |
| `content_sha256` | `varchar(64)` indexed | Powers deduplication decisions and idempotency keys |
| `size_bytes` | `bigint` | `CHECK (size_bytes >= 0)` |
| `width`, `height` | `int` | |
| `duration_seconds`, `frame_count`, `fps` | `float`/`int`/`float` | Video only |
| `extra` | `jsonb` | Probe output and anything backend-specific |
| `created_at`, `updated_at` | `timestamptz` | |

Indexes: `(owner_id, created_at)` for the listing endpoint's sort, `(owner_id, status)` for
"my ready assets", plus the unique `storage_key` and the `content_sha256` lookup.

## `inference_jobs`

The **user-visible unit of work**. One row per requested job, for its whole life.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` PK | |
| `owner_id`, `asset_id` | `uuid` FK, cascade | |
| `status` | `varchar(16)` indexed | `pending` \| `queued` \| `running` \| `succeeded` \| `failed` \| `cancelled` |
| `kind` | `varchar(16)` | `image` \| `video` |
| `model_name`, `precision`, `queue` | `varchar` | The queue is recorded, so routing is a fact rather than a consequence of broker config that may since have changed |
| `idempotency_key` | `varchar(128)` | `UNIQUE (owner_id, idempotency_key)`. See [ADR-007](decisions/ADR-007-idempotency.md). |
| `celery_task_id` | `varchar(64)` indexed | For correlation and revocation only; never read as state |
| `params` | `jsonb` | The full validated request, so a job is re-runnable from its row |
| `result` | `jsonb` null | The output manifest: keys, sizes, per-output metadata |
| `progress`, `progress_message` | `float`, `varchar(255)` | Written at most once per second |
| `error_code`, `error_message` | `varchar(64)`, `text` | `error_code` is stable and safe to branch on |
| `attempts` | `int` | `CHECK (attempts >= 0)` |
| `queued_at`, `started_at`, `finished_at` | `timestamptz` null | Queue wait is `started_at - queued_at`; that is the number that tells you whether to add workers |
| `created_at`, `updated_at` | `timestamptz` | |

Indexes: `(owner_id, created_at)`, `(status, queue)` — the latter is what makes "where is
the backlog" a single grouped query.

`result` holds the manifest so `GET /v1/jobs/{id}/result` is one query and no storage
round-trip.

## `inference_runs`

One **execution attempt** of a job. A retry appends a row; it does not mutate the job.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` PK | |
| `job_id` | `uuid` FK, cascade | |
| `attempt` | `int` | `UNIQUE (job_id, attempt)` |
| `status` | `varchar(16)` | |
| `worker_hostname` | `varchar(255)` | Which replica, for correlating with a bad node |
| `device`, `device_name` | `varchar(32)`, `varchar(128)` | `cuda:0` and the resolved GPU/CPU name |
| `model_name`, `precision`, `batch_size` | | What was *actually* used, which after an OOM retry is not what the job asked for |
| `oom_retry` | `bool` | |
| `retryable_error` | `bool` null | The classifier's verdict |
| `error_code`, `error_message` | | |
| `duration_seconds`, `frames_processed` | | |
| `peak_rss_bytes`, `peak_vram_bytes` | `bigint` | |
| `metrics` | `jsonb` | Per-stage timings |
| `created_at`, `finished_at` | `timestamptz` | |

This table exists because of one question that logs answer badly: *how often do we OOM,
and what batch size did the retry succeed at?* With attempt history in the job row it is a
log grep across replicas; with a run table it is

```sql
SELECT batch_size, count(*)
FROM inference_runs
WHERE oom_retry AND status = 'succeeded'
GROUP BY batch_size ORDER BY batch_size;
```

## `benchmark_runs`

One benchmark execution. Deliberately denormalised and append-only.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` PK | |
| `run_id` | `varchar(64)` unique | The same id as the results JSON filename |
| `owner_id` | `uuid` FK null `ON DELETE SET NULL` | A benchmark outlives the user who ran it |
| `status` | `varchar(16)` | |
| `git_commit`, `git_dirty` | `varchar(40)`, `bool` | A number measured on a dirty tree is not attributable to a commit, and the flag says so |
| `hardware`, `cpu_model`, `cpu_count`, `total_ram_bytes` | | |
| `gpu_name` | `varchar(128)` | |
| `os_description` | `varchar(255)` | |
| `library_versions` | `jsonb` | torch, onnxruntime, numpy, opencv… |
| `dataset_id` indexed, `dataset_manifest` | `varchar(128)`, `jsonb` | Including the content fingerprint |
| `model_name`, `runtime`, `precision` | | |
| `config`, `metrics` | `jsonb` | Warmup/repetition counts; the measured numbers |
| `results_path` | `varchar(512)` | Path to the committed JSON |
| `error_message`, `duration_seconds` | | |
| `created_at`, `finished_at` | `timestamptz` | |

Every provenance field is a **column, not a comment**, because a benchmark number without
the commit, the hardware, the library versions and the dataset that produced it is an
anecdote. Indexed on `created_at` and on `(model_name, runtime, precision)` so a
regression can be found by comparing a model's history.

## Conventions and why

**UUID v4 primary keys.** Ids appear in URLs and are handed to clients. Sequential
integers leak volume ("you are customer 41") and invite enumeration. The cost is index
locality — random UUIDs scatter B-tree inserts — which at this scale is not measurable, and
UUIDv7 is the upgrade path if it ever is.

**Enums as `varchar` with a Python `StrEnum`, not a Postgres `ENUM`.** Adding a value to a
native enum requires `ALTER TYPE`, which is awkward inside a transaction-wrapped migration
and takes a lock on every table using it. A `varchar` plus an application-level enum makes
adding a status a code change. The trade-off is that the database will accept a typo; the
mitigation is that only one module writes these columns.

**`timestamptz` everywhere, never `timestamp`.** A naive timestamp is a bug waiting for a
worker in a different timezone.

**`jsonb` for `params`, `result`, `metrics`, `extra`.** These are read as a whole and their
shape is defined by Pydantic models that version faster than a migration cycle. Anything
queried or constrained — status, queue, model name, batch size — is a real column.

**`ON DELETE CASCADE` from users through assets to jobs and runs**, so deleting a user is
one statement and cannot leave orphans. `benchmark_runs.owner_id` is `SET NULL` instead,
because the measurement should survive the account.

**No soft deletes except on assets.** `AssetStatus.DELETED` exists because the blob may
outlive the row's usefulness and a reconciliation job needs to know the row was
deliberately retired. Everything else is deleted for real.

## Migrations

```bash
make migrate                                   # alembic upgrade head
make migrate-down                              # alembic downgrade -1
.venv/bin/alembic revision --autogenerate -m "add x"
```

Autogenerate is a starting point, not an answer: it does not detect renames (it emits a
drop plus an add, which discards data), it misses server-side default changes, and it will
happily generate a `DROP` for anything created outside Alembic. Every generated migration
gets read before it is committed.
