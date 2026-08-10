"""Storage interface and key generation.

The interface is deliberately small - ``put``, ``get``, ``open``, ``delete``,
``exists``, ``stat``, ``list``, plus presigned URL generation - because that is all
the application needs and every method has to be implementable on both a
filesystem and an S3 bucket without one leaking into the other's semantics.

Key generation is a security control, not a convenience
------------------------------------------------------
Storage keys are **server-generated and random**; the client's filename never
appears in a path. This closes three separate holes at once:

* **Path traversal.** A filename of ``../../etc/passwd`` cannot escape the storage
  root if it is never used to build the path.
* **Enumeration.** Sequential or user-derived keys let one user guess another's
  objects. 128 bits of randomness does not.
* **Overwrite / collision.** Two users uploading ``photo.jpg`` never collide.

The original filename is kept in the database for display purposes only. The
extension is re-derived from the *sniffed* content type, not from the client's
claim.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime as dt
import re
import secrets
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import IO, Any

KEY_RANDOM_BYTES = 16
"""128 bits: enough that enumeration is not a threat model worth discussing."""

_SAFE_PREFIX = re.compile(r"[^a-zA-Z0-9._/-]")
_SAFE_EXT = re.compile(r"^[a-z0-9]{1,8}$")


class StorageError(RuntimeError):
    """Base class for storage failures."""


class ObjectNotFoundError(StorageError):
    """Raised when a key does not exist."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"object not found: {key}")


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectMetadata:
    """What every backend can report about a stored object."""

    key: str
    size: int
    content_type: str | None
    last_modified: dt.datetime | None
    etag: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "size": self.size,
            "content_type": self.content_type,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "etag": self.etag,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class PresignedUpload:
    """A time-limited direct-to-storage upload instruction."""

    key: str
    url: str
    method: str
    headers: dict[str, str]
    expires_at: dt.datetime
    max_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "url": self.url,
            "method": self.method,
            "headers": self.headers,
            "expires_at": self.expires_at.isoformat(),
            "max_bytes": self.max_bytes,
        }


def sanitize_prefix(prefix: str) -> str:
    """Reduce a caller-supplied prefix to a safe, relative POSIX path fragment.

    Rejects absolute paths and any ``..`` segment outright rather than trying to
    normalise them: silently rewriting a traversal attempt makes the logs useless.
    """
    cleaned = _SAFE_PREFIX.sub("_", prefix.strip("/"))
    parts = [p for p in PurePosixPath(cleaned).parts if p not in {"", "."}]
    if any(p == ".." for p in parts):
        raise ValueError(f"prefix must not contain '..': {prefix!r}")
    return "/".join(parts)


def build_storage_key(
    *,
    user_id: str,
    kind: str = "uploads",
    extension: str | None = None,
    now: dt.datetime | None = None,
) -> str:
    """Generate a random, tenant-scoped storage key.

    Shape: ``{kind}/{user_id}/{YYYY}/{MM}/{DD}/{32 hex chars}.{ext}``

    The date partition is not decoration: it keeps S3 prefix listings bounded and
    makes lifecycle rules ("expire uploads older than 30 days") expressible as a
    prefix match. The user id partition means a per-tenant deletion is one prefix
    delete. Authorisation is still enforced in the database - the key layout is
    defence in depth, never the primary check.
    """
    stamp = now or dt.datetime.now(dt.UTC)
    safe_kind = sanitize_prefix(kind) or "uploads"
    safe_user = sanitize_prefix(str(user_id)) or "anonymous"
    token = secrets.token_hex(KEY_RANDOM_BYTES)

    ext = ""
    if extension:
        candidate = extension.lower().lstrip(".")
        if not _SAFE_EXT.match(candidate):
            raise ValueError(f"unsafe storage extension: {extension!r}")
        ext = f".{candidate}"

    return f"{safe_kind}/{safe_user}/{stamp:%Y/%m/%d}/{token}{ext}"


class Storage(abc.ABC):
    """Backend-agnostic blob storage."""

    #: Human-readable backend name, recorded on assets so a migration between
    #: backends can be detected rather than guessed.
    backend: str = "abstract"

    @abc.abstractmethod
    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> ObjectMetadata:
        """Store ``data`` at ``key``, overwriting any existing object."""

    @abc.abstractmethod
    def put_stream(
        self, key: str, stream: IO[bytes], *, content_type: str | None = None
    ) -> ObjectMetadata:
        """Store from a file-like object without buffering it all in memory."""

    @abc.abstractmethod
    def get(self, key: str) -> bytes:
        """Read an object, raising :class:`ObjectNotFoundError` if absent."""

    @abc.abstractmethod
    def open(self, key: str) -> IO[bytes]:
        """Open an object for streaming reads."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Delete an object. Idempotent - deleting a missing key is not an error."""

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Whether ``key`` exists."""

    @abc.abstractmethod
    def stat(self, key: str) -> ObjectMetadata:
        """Metadata for ``key``."""

    @abc.abstractmethod
    def list(self, prefix: str = "", *, limit: int = 1000) -> Iterator[ObjectMetadata]:
        """Iterate objects under ``prefix``."""

    @abc.abstractmethod
    def presign_upload(
        self,
        key: str,
        *,
        content_type: str | None = None,
        max_bytes: int | None = None,
        expires_in: int | None = None,
    ) -> PresignedUpload:
        """A time-limited URL the client can PUT directly to."""

    @abc.abstractmethod
    def presign_download(self, key: str, *, expires_in: int | None = None) -> str:
        """A time-limited URL for reading an object."""

    def copy(self, src: str, dst: str) -> ObjectMetadata:
        """Copy within the same backend. Generic fallback via read-then-write."""
        data = self.get(src)
        meta = self.stat(src)
        return self.put(dst, data, content_type=meta.content_type)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(backend={self.backend!r})"
