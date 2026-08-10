# Security

This is a portfolio project, not an audited product. What follows is an honest account of
the controls that exist, the reasoning behind them, and — in the last section — what is
deliberately missing. The gaps are listed because a security page that only lists strengths
is worse than no security page.

## Threat model

Assumed adversary: an authenticated user of the API, or an unauthenticated party who can
reach it over the network. Assets they want are other users' uploads and outputs, the
ability to run code on a worker, and the ability to exhaust the service.

Explicitly out of scope: a compromised host, a malicious operator, and side channels in the
model itself.

## Authentication

**Passwords.** bcrypt with a per-hash salt at cost 12 (~250 ms per hash — high enough to
matter, low enough for a login).

Two bcrypt details are handled explicitly because both are easy to get wrong:

- **bcrypt truncates at 72 bytes.** A 200-character passphrase is only as strong as its
  first 72 bytes, and bcrypt 5.x raises rather than truncating. Passwords are pre-hashed
  with SHA-256 and base64-encoded to a fixed 44-byte string first, so length is bounded, no
  entropy is discarded, and there is no NUL-byte truncation issue.
- **passlib is not used.** passlib 1.7.x reads `bcrypt.__about__`, removed in bcrypt 4.1,
  and the compatibility shim is more code than calling bcrypt directly.

Verification is constant-time via `bcrypt.checkpw` and returns `False` rather than raising
on a malformed stored hash.

**Tokens.** HS256 JWTs with `sub`, `exp`, `iss` and a `typ` claim. Verification:

- **An explicit single-algorithm allow-list.** This is the defence against the classic
  `alg: none` and RS256→HS256 confusion attacks. Not passing `algorithms` to `jwt.decode`
  is the bug that makes those attacks work.
- `exp`, `sub` and `iss` are all *required* and validated. A token without an expiry is
  rejected rather than treated as non-expiring.
- `typ` must be `access`, so a token minted for another purpose cannot be replayed here.
- `is_active` is re-checked from the database on every request, so deactivating an account
  takes effect immediately rather than at token expiry.

**Login is not an account-existence oracle.** Unknown email and wrong password both return
`401 invalid_credentials` with the same body.

**The development signing key is refused in production.** `Settings` raises at
construction when `CUTOUTML_ENVIRONMENT=prod` and `jwt_secret` is still the default. A
default secret is not a weak secret — it is a *published* one, so anyone with a clone can
mint a token for any user id. The check lives on `Settings` rather than in the API so the
worker and the CLI refuse the same configuration, and it is a crash rather than a warning
because a warning in a log nobody reads is how this reaches production.

## Authorisation

Every query is scoped by `owner_id` at the database level. There is no "fetch then check"
step that can be forgotten, because the `WHERE` clause is the check.

**A resource owned by someone else returns `404`, not `403`.** A `403` confirms that the id
exists, which is an enumeration oracle. The two cases are deliberately indistinguishable.

The storage key layout provides a second, independent scope (`{kind}/{user_id}/…`), but it
is defence in depth and never the primary check.

## Upload handling

Uploads are the largest attack surface here, so validation is ordered deliberately —
cheapest and most decisive checks first — and nothing the client says is trusted:

1. **Reject empty.**
2. **Sniff the content type from magic bytes.** The declared `Content-Type` and the filename
   extension are then cross-checked against the sniffed type, and a mismatch is rejected
   (`content_type_mismatch`). Neither client-supplied value is ever used as the answer.
   Only an allow-list of image and video MIME types is accepted.
3. **Size against the real byte count.** The `Content-Length` check in middleware is a cheap
   first gate, not the real check: `Content-Length` is a claim. Images and videos have
   separate limits (32 MiB / 256 MiB by default).
4. **Pixel count from the header, before decoding.** A 20 KB PNG can declare 50,000 ×
   50,000 pixels. Decoding it first is a decompression bomb; the limit
   (`CUTOUTML_MAX_IMAGE_PIXELS`, 64 MP) is checked against declared dimensions while the
   file is still 20 KB.
5. **Then decode**, with a distinct `corrupt_image` error for a valid signature with
   undecodable content.

Videos are probed with `ffprobe` and rejected if they exceed `CUTOUTML_MAX_VIDEO_FRAMES`
(18,000 — ten minutes at 30 fps).

## Storage

Covered in full in [ADR-006](decisions/ADR-006-storage-layout.md). The security-relevant
parts:

- **Storage keys are server-generated and random** (128 bits). The client's filename never
  appears in a path, so path traversal is not mitigated — it is structurally impossible,
  because the untrusted input is never used to construct the path. The filename is kept in
  the database for display only.
- **Caller-supplied prefixes are rejected, not repaired.** `sanitize_prefix()` raises on any
  `..` segment. Normalising a traversal attempt away destroys the signal that someone tried.
- **The local backend contains escapes anyway.** Every key is resolved — symlinks first —
  and checked against the storage root, so a key that somehow bypassed generation still
  cannot write outside it.
- **Writes are atomic** (`os.replace` from a temp file in the destination directory). Without
  that, a crash mid-write leaves a truncated object that looks valid, and an API polling for
  a result a worker is writing can read a partial file.
- **Presigned URLs expire** (900 s by default) and are scoped to a single key and method.

## Rate limiting and denial of service

A **token bucket**, not a fixed window. A fixed window lets a client spend its whole quota
in the last millisecond of one window and again in the first millisecond of the next,
producing twice the intended peak.

The Redis implementation is a **Lua script**, so the read-modify-write is atomic. Doing
`GET`, compute, `SET` from Python is a race in which two concurrent requests read the same
token count and both succeed — a limit of 1 admits 2. Under real concurrency that is not an
edge case. Idle buckets expire, so the key space does not grow without bound.

**When Redis is unavailable the limiter degrades to a per-process in-memory bucket and logs
it.** This is a deliberate availability-over-strictness choice, and it has a consequence
worth stating plainly: **with N API replicas, the effective global limit during a Redis
outage becomes N × the configured value.** That is documented rather than hidden. It is the
right trade for this system and the wrong one for a system where the limit is a billing
control.

Other resource controls: body-size limits before routing; a hard frame ceiling per video
job; Celery hard and soft time limits (3600 s / 3480 s) so a pathological job is killed with
two minutes of warning to record why; `worker_prefetch_multiplier=1` so one worker cannot
hoard the queue.

## Injection and subprocess handling

- **SQL.** SQLAlchemy Core/ORM with bound parameters throughout. No string-built SQL
  anywhere.
- **Subprocess.** ffmpeg and ffprobe are invoked as **argument lists, never with
  `shell=True`**, so a filename cannot become a shell command. Binaries are resolved with
  `shutil.which` against the configured name.
- **Paths.** Both wrappers are context managers and kill the child process on exit,
  including on exception. An orphaned ffmpeg holding a pipe is the classic way a video
  worker leaks until the OOM killer arrives.
- **Deserialisation.** Celery is configured for JSON only; pickle is not enabled. Model
  checkpoints are loaded with `torch.load(..., weights_only=True)`, because `torch.load`
  without it is arbitrary code execution from a `.pt` file — which matters as soon as
  anyone downloads a checkpoint from the internet.

## Web and browser concerns

- **CORS is an explicit origin allow-list**, not `*`. The API uses bearer tokens that a
  browser stores, and a wildcard origin with credentials is what turns XSS on any site into
  account takeover here. Allowed methods, headers and exposed headers are all enumerated.
- **CSRF is not applicable** to the token flow: authentication is an `Authorization` header,
  not a cookie, so a cross-site form post carries no credentials. This stops being true the
  moment anyone adds cookie auth.
- **Downloads carry the sniffed content type** and are served from storage rather than from a
  path derived from user input.
- The Next.js frontend proxies `/api/*` to the API by default, so in normal operation the
  browser makes same-origin requests and CORS is not exercised at all.

## Secrets

- `.env` is gitignored; `.env.example` contains names and non-secret defaults only.
- No credential is committed. The compose stack's Postgres and MinIO credentials are
  development values for throwaway containers and are visible on purpose.
- Production S3 credentials are expected to come from the ambient credential chain
  (instance role, IRSA), which is why `CUTOUTML_S3_ACCESS_KEY_ID` defaults to unset.
- Errors returned to clients never include a stack trace or a connection string; `500`
  responses carry a generic message and a `request_id` that correlates to the server log.

## What is deliberately not implemented

Listed because these are the honest gaps, and because knowing which is which is the
difference between a limitation and a surprise:

- **No audit log.** Who deleted what, and when, is not recorded beyond application logs.
- **No API keys, no refresh tokens, no MFA, no token revocation list.** A leaked token is
  valid until it expires (one hour). Deactivating the *user* takes effect immediately, which
  is the only revocation mechanism.
- **No malware scanning** of uploads. Bytes are treated as pixels, never executed, but they
  are stored and served back.
- **`/metrics` is unauthenticated.** A scraper is not a user. It must be network-restricted;
  it leaks request volumes, job counts and model names.
- **No per-endpoint rate limits.** One bucket for everything, so cheap `GET`s consume the
  same budget as expensive job submissions.
- **No tenant isolation beyond the database scope and the key prefix.** All tenants share
  one bucket, one database and one worker fleet. A worker processes jobs from multiple
  users in one process, so a hypothetical memory-disclosure bug in a decoder crosses tenant
  boundaries.
- **No signed URLs on the local backend.** Presigning there has no cryptographic meaning, so
  development and production differ in whether a download can bypass the API.
- **No dependency scanning or SBOM** in CI.
- **No account lockout or login throttling** beyond the global rate limit, so password
  guessing is bounded at 120 attempts per minute per IP-derived bucket for unauthenticated
  requests.

## Reporting

This is a personal project with no security contact and no disclosure process. If you find
something, open an issue.
