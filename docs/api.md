# HTTP API

Base URL `http://127.0.0.1:8000`. Interactive documentation is generated from the code at
`/docs` (Swagger UI), `/redoc`, and `/openapi.json` — that is the authoritative reference
for request and response schemas. This page covers the parts that a schema cannot express:
the conventions, the error contract, and why the asynchronous endpoints behave the way they
do.

## Conventions

**Authentication.** Every `/v1` endpoint except `/v1/auth/register` and `/v1/auth/login`
requires `Authorization: Bearer <token>`. Tokens are HS256 JWTs from
`POST /v1/auth/login`, valid for one hour by default. `is_active` is checked on every
request, so deactivating a user takes effect immediately rather than at token expiry.

**Errors always have one shape.**

```json
{
  "error": {
    "code": "model_unavailable",
    "message": "model 'u2net' has no weights available on this worker",
    "request_id": "01JBQ9Z0K4M2X8T1N7W3",
    "details": {"model": "u2net"}
  }
}
```

`code` is stable and safe to branch on. `message` is for humans and may change. `details`
is optional and its shape depends on `code`. `request_id` matches the `X-Request-ID`
response header and appears in every server log line for that request, which is what makes
a bug report actionable.

**Request ids.** Send `X-Request-ID` and it is echoed and used; omit it and one is
generated. It propagates into the Celery task, so a job's worker logs carry the id of the
request that created it.

**Rate limiting.** A token bucket, 120 requests per minute with a burst of 30 by default,
keyed on the authenticated user id. Responses carry `X-RateLimit-Remaining`; a 429 carries
`Retry-After`. `/health*` and `/metrics` are not limited — the limiter is a router
dependency rather than middleware, so they are exempt by construction rather than by a path
list someone has to remember to maintain.

**Pagination** is `limit` and `offset` on list endpoints. There is no cursor pagination;
see Limitations.

## Endpoints

### Health

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, for humans and tools that assume this path |
| `GET` | `/health/live` | Liveness. Touches nothing external. A failure means *restart me*. |
| `GET` | `/health/ready` | Readiness. Checks database, Redis, model registry, storage, ffmpeg. |

`/health/ready` returns 503 when a **required** check fails (database, Redis). Storage and
ffmpeg are reported but not fatal: a missing ffmpeg only breaks video jobs, and refusing all
traffic for that would take the working image path down with it. Wiring dependency checks
into *liveness* is a classic outage amplifier — Postgres hiccups, every replica's liveness
probe fails, the orchestrator restarts the whole fleet, and cold caches turn a 30-second
blip into a ten-minute outage.

### Auth

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/auth/register` | Create an account |
| `POST` | `/v1/auth/login` | Exchange credentials for a token |
| `GET` | `/v1/auth/me` | The authenticated principal |

```bash
curl -sX POST localhost:8000/v1/auth/register \
  -H 'content-type: application/json' \
  -d '{"email":"me@example.com","password":"correct horse battery staple"}'

TOKEN=$(curl -sX POST localhost:8000/v1/auth/login \
  -H 'content-type: application/json' \
  -d '{"email":"me@example.com","password":"correct horse battery staple"}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
```

Login returns the same `401 invalid_credentials` for an unknown email and a wrong password.
Distinguishing them turns the endpoint into an account-existence oracle.

### Assets

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/assets` | Upload in one multipart request |
| `POST` | `/v1/assets/upload-url` | Reserve an asset and get somewhere to `PUT` the bytes |
| `PUT` | `/v1/assets/{id}/content` | Upload the bytes for a reserved asset |
| `POST` | `/v1/assets/{id}/complete` | Mark a presigned upload complete (S3 backend) |
| `GET` | `/v1/assets` | List the caller's assets (`limit`, `offset`, filters) |
| `GET` | `/v1/assets/{id}` | Metadata |
| `GET` | `/v1/assets/{id}/content` | Download the original bytes |
| `DELETE` | `/v1/assets/{id}` | Delete |

Two upload paths exist because they solve different problems. The multipart path is one
request and is what a CLI or a small file wants. The presigned path exists so a large video
can go **directly to object storage** without passing through the API — otherwise every API
replica needs bandwidth and memory proportional to upload volume, and a 256 MB upload
occupies a worker thread for its whole duration.

Uploads are validated in this order, and the order matters:

1. Non-empty.
2. **Content type sniffed from magic bytes.** The client's `Content-Type` and filename
   extension are cross-checked against the sniffed type and a mismatch is rejected. Neither
   is ever trusted as the answer.
3. Size against the per-kind limit, checked on the **real byte count**. The
   `Content-Length` check in middleware is a cheap first gate, not the real check —
   `Content-Length` is a claim.
4. For images, the header is parsed and the **pixel count** checked before any pixels are
   allocated. A 20 KB PNG can declare 50,000 × 50,000 pixels; decoding it first is a
   decompression bomb.
5. Only then is the image fully decoded.

Re-uploading content to an asset that is already `ready` is a `409 asset_already_uploaded`:
the content hash is part of the idempotency key of any job created from it, so swapping the
bytes would make a completed job's result inconsistent with its input.

### Processing

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/assets/{id}/process` | Queue a segmentation job |
| `GET` | `/v1/assets/{id}/result` | The latest successful result for an asset |

```bash
ASSET=$(curl -s -X POST localhost:8000/v1/assets \
  -H "authorization: Bearer $TOKEN" -F file=@photo.jpg \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl -s -X POST localhost:8000/v1/assets/$ASSET/process \
  -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"model":"cutoutnet","outputs":["transparent_png","mask_png"]}'
```

Returns **202** with a job id, or **200** with the existing job when the request is a
duplicate. Output kinds: `transparent_png`, `transparent_webp`, `mask_png`,
`color_composite`, `background_composite`, `blurred_background`. Outputs are requested
rather than always produced, because encoding a 4000 px PNG is not free and a caller who
only wants a mask should not pay for a transparent PNG as well.

**Idempotency.** Pass `idempotency_key` to control it. Without one, the key is derived from
the asset id, its content hash and the exact parameter set, so a double-click or a client
retry returns the original job rather than doing the work twice. A client that genuinely
wants a second identical run passes its own key. See
[ADR-007](decisions/ADR-007-idempotency.md).

### Jobs

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/jobs` | List the caller's jobs |
| `GET` | `/v1/jobs/{id}` | Status, progress and every attempt |
| `GET` | `/v1/jobs/{id}/result` | The output manifest |
| `GET` | `/v1/jobs/{id}/outputs/{kind}` | Download one output |
| `POST` | `/v1/jobs/{id}/cancel` | Cancel |

Poll `GET /v1/jobs/{id}` for `status` and `progress`. The detail response includes the
`inference_runs` history, so a job that was retried shows which worker, which device and
which batch size each attempt used — including whether the batch size was reduced after an
out-of-memory error.

Cancellation is cooperative. A `queued` job is revoked before it starts; a `running` job is
signalled and stops at its next progress checkpoint. There is no way to interrupt a forward
pass mid-flight, and pretending otherwise would be a lie in the API surface.

### Catalogue

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | Registered models with per-model availability |
| `GET` | `/v1/models/catalogue` | Raw registry specs, including licence and source |
| `GET` | `/v1/benchmarks` | Recorded benchmark runs |
| `GET` | `/v1/benchmarks/{run_id}` | One run, with full provenance |

`GET /v1/models` reports `weights_available` and `runtime_available` per model, computed
per call rather than cached, so a checkpoint that appears in `models/` — a finished training
run, a mounted volume — shows up without a restart. This is what lets a client discover that
a model cannot run *before* submitting a job to it, rather than finding out from a failed
job twenty seconds later.

### Operational

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/` | Service identity and doc links |

`/metrics` is unauthenticated, because a scraper is usually not a user. It is therefore
intended to be reachable only from inside the deployment network — see
[docs/security.md](security.md).

## Error codes

| Status | Code | Meaning |
|---|---|---|
| 400 | `bad_request` | Malformed or contradictory parameters |
| 400 | `unknown_model` | Not in the registry; the message lists what is |
| 400 | `model_unavailable` | Registered, but its weights or runtime are missing here |
| 401 | `unauthenticated` | No credentials presented |
| 401 | `invalid_token` | Expired, wrong signature, wrong issuer |
| 401 | `invalid_credentials` | Login failed. Deliberately the same for unknown email and wrong password. |
| 401 | `inactive_user` | Valid token, deactivated account |
| 403 | `forbidden` | Authenticated but not permitted |
| 404 | `not_found` | Absent **or not owned by the caller** — the two are indistinguishable on purpose |
| 404 | `result_not_available` | No successful job for this asset yet |
| 404 | `benchmark_not_found` | |
| 409 | `asset_already_uploaded` | Content already present |
| 409 | `asset_not_ready` | Referenced before its bytes arrived |
| 409 | `upload_incomplete` | Presigned upload never completed |
| 409 | `email_taken` | |
| 409 | `job_not_complete` | Result requested before the job finished |
| 409 | `job_already_terminal` | Cancel on a finished job |
| 413 | `payload_too_large` | Over the per-kind byte limit |
| 415 | `unsupported_type` | Sniffed type not in the allow-list |
| 422 | `content_type_mismatch` | Declared type disagrees with the bytes |
| 422 | `kind_mismatch` | Video uploaded where an image was expected, or the reverse |
| 422 | `corrupt_image` | Valid signature, undecodable content |
| 422 | `image_too_large` | Over the pixel limit; rejected before decoding |
| 422 | `empty_upload` | Zero bytes |
| 422 | `validation_error` | Pydantic rejected the body; `details` carries the field errors |
| 429 | `rate_limited` | `Retry-After` says when |
| 500 | `internal_error` | The message is generic; `request_id` is the way in |

Returning `404` rather than `403` for an object owned by someone else is deliberate: a `403`
confirms the id exists, which is an enumeration oracle.

## Limitations

Stated because the OpenAPI document cannot state them:

- **No webhooks or server-sent events.** Job completion is discovered by polling. The
  frontend polls with backoff. Webhooks are the obvious next step and are not implemented.
- **Offset pagination only.** Fine at this scale, and wrong at large offsets, where the
  database still walks the skipped rows and a concurrent insert can shift the window.
- **No API keys.** Only user JWTs, so machine-to-machine use means storing a password and
  logging in.
- **No refresh tokens.** A one-hour access token and then log in again.
- **No per-endpoint rate limits.** One bucket covers everything, so a burst of cheap `GET`
  requests consumes the same budget as expensive job submissions.
- **`/metrics` is unauthenticated** and must be network-restricted.
