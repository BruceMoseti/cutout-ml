"""Filesystem storage for development, tests and single-node deployments."""

from __future__ import annotations

import datetime as dt
import hashlib
import mimetypes
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from cutoutml.core.logging import get_logger
from cutoutml.storage.base import (
    ObjectMetadata,
    ObjectNotFoundError,
    PresignedUpload,
    Storage,
    StorageError,
)

log = get_logger(__name__)


class LocalStorage(Storage):
    """Objects as files under a root directory.

    Two details make this a real backend rather than a toy:

    * **Escape containment.** Every key is resolved and checked against the root, so
      even a key that got past :func:`~cutoutml.storage.base.build_storage_key`
      cannot write outside it. Symlinks are resolved before the check.
    * **Atomic writes.** Content is written to a temp file in the same directory and
      ``os.replace``-d into place. Without this, a crash mid-write leaves a
      truncated object that looks valid, and a concurrent reader can observe a
      partial file - both of which are real problems for a worker writing results
      that an API is polling.

    Presigning has no cryptographic meaning here: there is no local HTTP server to
    sign for, so it returns an application-relative URL and the API performs the
    normal database authorisation check. ``S3Storage`` is what production uses.
    """

    backend = "local"

    def __init__(self, root: Path | str, *, url_prefix: str = "/assets/local") -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.url_prefix = url_prefix.rstrip("/")

    # ------------------------------------------------------------------- paths

    def _path(self, key: str) -> Path:
        if not key or key.startswith("/"):
            raise ValueError(f"storage key must be a non-empty relative path: {key!r}")
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"storage key escapes the storage root: {key!r}")
        return candidate

    def _relkey(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # -------------------------------------------------------------------- write

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> ObjectMetadata:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, data)
        log.debug("local_storage_put", key=key, bytes=len(data))
        return self._stat_path(path, content_type)

    def put_stream(
        self, key: str, stream: IO[bytes], *, content_type: str | None = None
    ) -> ObjectMetadata:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                shutil.copyfileobj(stream, fh, length=1024 * 1024)
                fh.flush()
                os.fsync(fh.fileno())
            Path(tmp_name).replace(path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return self._stat_path(path, content_type)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            Path(tmp_name).replace(path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    # --------------------------------------------------------------------- read

    def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise ObjectNotFoundError(key) from None
        except IsADirectoryError:
            raise ObjectNotFoundError(key) from None

    def open(self, key: str) -> IO[bytes]:
        path = self._path(key)
        try:
            return path.open("rb")
        except FileNotFoundError:
            raise ObjectNotFoundError(key) from None

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).is_file()
        except ValueError:
            return False

    def stat(self, key: str) -> ObjectMetadata:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(key)
        return self._stat_path(path, None)

    def _stat_path(self, path: Path, content_type: str | None) -> ObjectMetadata:
        st = path.stat()
        guessed = content_type or mimetypes.guess_type(path.name)[0]
        return ObjectMetadata(
            key=self._relkey(path),
            size=st.st_size,
            content_type=guessed,
            last_modified=dt.datetime.fromtimestamp(st.st_mtime, tz=dt.UTC),
            etag=self._etag(path),
        )

    @staticmethod
    def _etag(path: Path) -> str:
        """MD5 of the content, matching S3's single-part ETag semantics.

        Used only for cache validation and integrity checks, never for security -
        hence MD5 rather than a cryptographic hash.
        """
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # ------------------------------------------------------------------- delete

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink(missing_ok=True)
        except ValueError:
            raise
        except OSError as exc:
            raise StorageError(f"failed to delete {key}: {exc}") from exc

    def list(self, prefix: str = "", *, limit: int = 1000) -> Iterator[ObjectMetadata]:
        base = self._path(prefix) if prefix else self.root
        search_root = base if base.is_dir() else base.parent
        count = 0
        for path in sorted(search_root.rglob("*")):
            if not path.is_file() or path.name.startswith(".upload-"):
                continue
            rel = self._relkey(path)
            if prefix and not rel.startswith(prefix):
                continue
            yield self._stat_path(path, None)
            count += 1
            if count >= limit:
                return

    # ---------------------------------------------------------------- presigning

    def presign_upload(
        self,
        key: str,
        *,
        content_type: str | None = None,
        max_bytes: int | None = None,
        expires_in: int | None = None,
    ) -> PresignedUpload:
        """Return an application-relative upload URL.

        Not a signed URL: the local backend has no independent HTTP endpoint, so the
        client PUTs back through the API, which authorises the request normally. The
        response shape matches S3's so client code does not branch.
        """
        expiry = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=expires_in or 900)
        headers = {"Content-Type": content_type} if content_type else {}
        return PresignedUpload(
            key=key,
            url=f"{self.url_prefix}/{key}",
            method="PUT",
            headers=headers,
            expires_at=expiry,
            max_bytes=max_bytes or 0,
        )

    def presign_download(
        self,
        key: str,
        *,
        expires_in: int | None = None,  # noqa: ARG002 - part of the Storage contract
    ) -> str:
        """A URL for reading the object.

        ``expires_in`` is accepted and ignored: a filesystem path has no expiry to
        encode. The interface keeps the parameter so callers do not branch on backend,
        and the API applies its own authorisation on the route this points at.
        """
        if not self.exists(key):
            raise ObjectNotFoundError(key)
        return f"{self.url_prefix}/{key}"
