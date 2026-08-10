# ADR-007: Idempotency keys and a job/run split instead of exactly-once delivery

Status: Accepted

## Context

Two independent sources of duplication have to be handled, and they are usually conflated.

**Duplicate submissions.** A user double-clicks. A mobile client retries after a timeout
whose response was actually delivered. A load balancer replays a request. Each of these
produces a second `POST /v1/assets/{id}/process` for work that is already in flight. Doing
the work twice costs GPU seconds and produces two result sets, one of which is garbage that
nobody deletes.

**Duplicate deliveries.** A Redis broker with `task_acks_late=True` is at-least-once (see
ADR-002). A task body *will* occasionally run twice for the same message: after a
visibility-timeout expiry, after a worker is SIGKILLed between finishing and acking, after
a broker failover. This is not a bug to fix; it is the delivery semantic.

There is also a third, quieter problem. When a job fails and is retried, the interesting
questions are operational: how often does this happen, which worker, which device, what
batch size, did the retry succeed at a smaller batch? If a retry mutates the job row, all of
that is destroyed on each attempt and the only record is in logs.

## Decision

**Idempotency keys, unique per `(owner_id, idempotency_key)`.** A client may supply one. If
it does not, the API derives it from a SHA-256 over the asset id, the asset's content hash,
and the exact request body with `idempotency_key` excluded. Submitting the same bytes with
the same options is treated as the same request, and the existing job is returned with
`200` instead of `202`. A client that genuinely wants a second identical run passes its own
key.

Deriving the key from the *content hash* rather than only the asset id is what makes this
correct across re-uploads. It also forces a related decision: re-uploading content to an
asset that is already `ready` is rejected with `409`, because swapping the bytes underneath
would make a completed job's result inconsistent with the input its key was derived from.

**Tasks are idempotent, and idempotency is cheap.** `JobRunner` returns the stored result
manifest immediately when the job is already `succeeded`, so a replayed message is a single
database read. Output storage keys are deterministic
(`results/{owner}/{job}/{name}.{ext}`), so a replay that does re-run overwrites its own
objects rather than orphaning a second set.

**The database row, not the Celery result backend, is the source of truth for job state.**
Celery results expire after a day; a job row does not. Nothing in the API reads from the
result backend.

**A job and a run are different things.** `inference_jobs` is the user-visible unit of
work — one row, stable id, stable idempotency key. `inference_runs` is one *attempt*, with
its own worker hostname, device, device name, model, precision, batch size, `oom_retry`
flag, error classification, duration, frames processed, and peak RSS/VRAM. A retry appends
a run; it does not mutate the job.

**Retry policy is explicit, not `autoretry_for=(Exception,)`.** Failures are classified:
non-retryable (a corrupt upload or an unknown model does not improve on attempt three, and
retrying triples the log noise for an identical outcome), retryable (exponential backoff
with jitter, capped at three retries), and out-of-memory. OOM is retried like a transient
failure but with a **halved batch size**, persisted onto the job's params so it survives a
worker restart mid-retry and shows up in `GET /v1/jobs/{id}`. Re-running an identical OOM is
a guaranteed second OOM; shrinking the working set is the only thing that can change the
outcome.

## Alternatives considered

**Rely on exactly-once delivery.** It does not exist over a network. Systems that claim it
are doing effectively-once via deduplication, which is what this ADR describes, or they are
wrong.

**Deduplicate on the broker with a Redis `SETNX` lock per job id.** Simpler-looking, and it
does stop concurrent duplicates. Rejected as the primary mechanism: a lock has a TTL, and
choosing the TTL means choosing between "expires while the job is still running, so the
duplicate proceeds anyway" and "outlives a crashed worker, so the job is stuck". The
database row already has the state; a lock is a second, weaker copy of it that can disagree.

**Make the client responsible for idempotency.** Rejected: most clients will not, and the
cost of them not doing so is paid in GPU time by the server.

**One `inference_jobs` table with `attempts`, `last_error` and no run history.** This was
the first shape. Rejected once the first OOM retry happened and there was no way to answer
"what batch size did it succeed at" without grepping worker logs across replicas. A run
table makes it `SELECT batch_size FROM inference_runs WHERE oom_retry AND status='succeeded'`.

**Idempotency key derived from the asset id alone.** Rejected: it makes re-uploading
different bytes to the same asset return a stale result, which is a correctness bug rather
than an optimisation.

## Consequences

Good:

- A double-clicked button costs one database query, not a second inference.
- A redelivered message is harmless and observable — `metrics.idempotent_hits` counts it.
- Retry behaviour is answerable from SQL: how often each failure class occurs, on which
  worker, at which batch size. Batch-size tuning becomes an empirical exercise rather than a
  guess.
- A worker killed mid-job loses nothing: the message returns to the broker and the job row
  is still `running` with a run row explaining what happened.

Bad, and accepted:

- Derived idempotency keys are a hash of the request body, so a semantically identical
  request that differs in key ordering or an explicit-versus-default field produces a
  different key and a second job. `model_dump(mode="json")` normalises defaults, which
  covers the common case but not all of it.
- `(owner_id, idempotency_key)` is unique forever. A user who legitimately wants the same
  output again a month later gets the old job unless they pass their own key. There is no
  expiry, and adding one would mean choosing a window with no good justification.
- Two tables for one concept means every read of "what happened to my job" is a join, and
  `GET /v1/jobs/{id}` returns a nested structure rather than a flat row.
- Idempotency is load-bearing rather than best-effort, so it has to be tested rather than
  assumed. It is.
