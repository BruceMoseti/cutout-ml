# ADR-006: Server-generated random storage keys behind a narrow interface

Status: Accepted

## Context

The system stores user-uploaded images and videos and the outputs derived from them. Two
questions have to be answered before any of that code is written: what interface the
application sees, and what the keys look like.

The interface question is forced by deployment. Local development, CI and a single-node
install want a directory. Anything horizontally scaled needs object storage, because two
API replicas and three workers cannot share a filesystem without adding a network
filesystem to the architecture. Both have to work, and neither's semantics may leak into
the application.

The key question is forced by security. The obvious layout — put the file where the user's
filename says — is the source of a whole family of bugs:

- **Path traversal.** A filename of `../../etc/passwd`, or one with a NUL byte, or a
  Windows device name, or a Unicode normalisation that decomposes into `..` after some
  layer helpfully normalises it.
- **Enumeration.** Sequential ids or user-derived keys let anyone who can read one key
  guess others. On S3 with a misconfigured bucket policy, that is the whole dataset.
- **Collision and overwrite.** Two users uploading `photo.jpg` collide. One of them wins.
- **Content-type confusion.** A file named `.png` containing HTML, served back with a
  guessed content type, is stored XSS.

## Decision

**A narrow abstract `Storage` interface** — `put`, `put_stream`, `get`, `open`, `delete`,
`exists`, `stat`, `list`, `copy`, `presign_upload`, `presign_download` — with two
implementations (`LocalStorage`, `S3Storage`) chosen by configuration. Nothing larger,
because every method has to be implementable on both a filesystem and a bucket without one
medium's semantics leaking into the other's.

**Keys are server-generated and random.** The client's filename never appears in a path:

```
{kind}/{user_id}/{YYYY}/{MM}/{DD}/{32 hex chars}.{ext}
uploads/2e33d778-6abe-4497-837f-9c7a7ee923b2/2026/08/10/352ae7f84f49a7a4feafe5234560f8a2.png
```

Each component is load-bearing:

- **128 bits of randomness** (`secrets.token_hex(16)`) makes enumeration not a threat model
  worth discussing, and makes collision impossible in practice.
- **The user-id partition** makes per-tenant deletion a single prefix delete, and gives a
  cheap secondary check that a key belongs to the caller. It is defence in depth only:
  authorisation is enforced in the database, always.
- **The date partition** keeps S3 prefix listings bounded and makes lifecycle rules
  ("expire uploads older than 30 days") expressible as a prefix match rather than as a
  scan.
- **The extension is re-derived from the sniffed content type**, not from the client's
  claim, and is validated against `^[a-z0-9]{1,8}$`.

The original filename is stored in the `assets` table for display and never used to build
a path.

**Caller-supplied prefixes are rejected, not repaired.** `sanitize_prefix()` raises on any
`..` segment rather than normalising it away. Silently rewriting a traversal attempt makes
the logs useless — you lose the signal that someone tried.

**Output keys are deterministic**: `results/{owner}/{job}/{name}.{ext}`. Uploads must be
unguessable; outputs must be *idempotent*, because an at-least-once broker will sometimes
run a job twice and the second run has to overwrite its own objects rather than orphan a
second set (see ADR-007).

**Local storage still behaves like a real backend.** Every key is resolved and checked
against the root — symlinks resolved first — so a key that somehow bypassed key generation
still cannot escape. Writes go to a temporary file in the destination directory and are
`os.replace`d into place, because a crash mid-write otherwise leaves a truncated object
that looks valid, and a concurrent reader (an API polling for a result a worker is writing)
can observe a partial file.

## Alternatives considered

**Store blobs in Postgres as `bytea` or large objects.** Transactional with the metadata,
one backup to take, no second service. Rejected: a 256 MB video in a row makes every
backup, every replica and every `pg_dump` carry it, WAL amplification is severe, and
streaming it out means holding it in memory or fighting the large-object API.

**Filesystem only, with NFS/EFS for scale-out.** Rejected: it moves the problem into
infrastructure, and network filesystems have failure modes (stale handles, partial writes
under contention, lock semantics) that are worse than the S3 API.

**S3 only, with MinIO for local development.** Genuinely tempting — one code path, and the
compose stack runs MinIO anyway. Rejected because it makes `pytest` require a running
container, and unit tests that need a service get skipped, which means they rot. The local
backend keeps the test suite hermetic and the S3 backend is covered with a stubbed boto3
client.

**Content-addressed keys (`sha256` of the bytes).** Free deduplication, and idempotent by
construction. Rejected as the primary key layout: it makes deletion unsafe when two users
uploaded identical bytes (whose object is it?), it leaks equality — being able to check
whether a key exists is being able to check whether another user uploaded a specific
file — and it breaks the per-tenant prefix delete. The content hash *is* stored on the
asset row, where it powers deduplication decisions and idempotency keys without being the
path.

**Keep the user's filename, sanitised.** Rejected: sanitisation is a blocklist, and
blocklists for filenames have a long history of losing. Not using the input at all has no
such history.

## Consequences

Good:

- Path traversal is not mitigated, it is structurally impossible: the input is never used
  to build the path.
- Local development and tests need no object-storage service; production needs no shared
  filesystem.
- Per-tenant deletion and lifecycle expiry are both prefix operations.
- A duplicate job execution overwrites its own outputs.

Bad, and accepted:

- Keys are opaque. Nobody can find "that PNG I uploaded on Tuesday" by browsing the
  bucket; they have to query the database. That is the intended trade, and it means the
  database is authoritative — an object without a row is garbage, and a row without an
  object is a broken asset. Reconciling the two needs a job that does not yet exist.
- Presigned URLs are meaningless for the local backend: there is no local HTTP server to
  sign for, so it returns an application-relative URL and the API performs the normal
  database check. The two backends therefore differ in whether a download can bypass the
  API, which is a real behavioural difference between development and production.
- Deduplication is not implemented, only enabled: `content_sha256` is recorded but
  identical uploads still store two objects.
- The `copy()` fallback on the base class reads and rewrites the whole object. `S3Storage`
  overrides it with a server-side copy; anything else would move a 256 MB video through the
  application.
