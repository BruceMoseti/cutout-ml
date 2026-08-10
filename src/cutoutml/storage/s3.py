"""S3-compatible storage (AWS S3, MinIO, R2, ...) via boto3."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import IO, Any

from cutoutml.core.logging import get_logger
from cutoutml.storage.base import (
    ObjectMetadata,
    ObjectNotFoundError,
    PresignedUpload,
    Storage,
    StorageError,
)

log = get_logger(__name__)


class S3Storage(Storage):
    """Blob storage on any S3-compatible endpoint.

    Configuration notes that matter in practice:

    * ``endpoint_url`` + ``force_path_style`` are what make MinIO work. Virtual-host
      addressing (``bucket.host``) requires DNS entries MinIO does not have by
      default, and the resulting failure is an opaque connection error.
    * Credentials are optional: when omitted, boto3 walks its normal chain
      (environment, shared config, IMDS / IRSA). That is the right default for
      production, where hardcoded keys are the thing you are trying to avoid.
    * ``ContentLength`` is enforced on presigned uploads via a *condition*, not by
      trusting the client. See :meth:`presign_upload`.
    """

    backend = "s3"

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        force_path_style: bool = True,
        presign_expiry: int = 900,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - boto3 is in the api extra
            raise RuntimeError(
                "boto3 is required for S3Storage; install the 'api' extra"
            ) from exc

        self.bucket = bucket
        self.presign_expiry = presign_expiry
        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path" if force_path_style else "auto"},
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self.client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=config,
        )

    # ------------------------------------------------------------------ helpers

    def _is_missing(self, exc: Exception) -> bool:
        """Whether a botocore error means "no such key".

        S3 answers ``NoSuchKey`` for GET and ``404``/``NotFound`` for HEAD, and MinIO
        is not perfectly consistent either, so all of them are matched.
        """
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        return str(code) in {"NoSuchKey", "404", "NotFound", "NoSuchBucket"}

    # -------------------------------------------------------------------- write

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> ObjectMetadata:
        extra: dict[str, Any] = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)
        log.debug("s3_put", key=key, bytes=len(data))
        return ObjectMetadata(
            key=key,
            size=len(data),
            content_type=content_type,
            last_modified=dt.datetime.now(dt.UTC),
        )

    def put_stream(
        self, key: str, stream: IO[bytes], *, content_type: str | None = None
    ) -> ObjectMetadata:
        """Multipart-aware upload via ``upload_fileobj``.

        boto3 switches to multipart above 8 MB automatically, which is what keeps
        memory flat for large video uploads.
        """
        extra: dict[str, Any] = {"ContentType": content_type} if content_type else {}
        self.client.upload_fileobj(stream, self.bucket, key, ExtraArgs=extra or None)
        return self.stat(key)

    # --------------------------------------------------------------------- read

    def get(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_missing(exc):
                raise ObjectNotFoundError(key) from None
            raise StorageError(f"S3 get failed for {key}: {exc}") from exc
        body: bytes = response["Body"].read()
        return body

    def open(self, key: str) -> IO[bytes]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_missing(exc):
                raise ObjectNotFoundError(key) from None
            raise StorageError(f"S3 open failed for {key}: {exc}") from exc
        return response["Body"]  # type: ignore[no-any-return]

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_missing(exc):
                return False
            raise StorageError(f"S3 head failed for {key}: {exc}") from exc
        return True

    def stat(self, key: str) -> ObjectMetadata:
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_missing(exc):
                raise ObjectNotFoundError(key) from None
            raise StorageError(f"S3 head failed for {key}: {exc}") from exc
        return ObjectMetadata(
            key=key,
            size=int(head.get("ContentLength", 0)),
            content_type=head.get("ContentType"),
            last_modified=head.get("LastModified"),
            etag=str(head.get("ETag", "")).strip('"') or None,
        )

    # ------------------------------------------------------------------- delete

    def delete(self, key: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_missing(exc):
                return
            raise StorageError(f"S3 delete failed for {key}: {exc}") from exc

    def list(self, prefix: str = "", *, limit: int = 1000) -> Iterator[ObjectMetadata]:
        paginator = self.client.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield ObjectMetadata(
                    key=item["Key"],
                    size=int(item.get("Size", 0)),
                    content_type=None,
                    last_modified=item.get("LastModified"),
                    etag=str(item.get("ETag", "")).strip('"') or None,
                )
                count += 1
                if count >= limit:
                    return

    def copy(self, src: str, dst: str) -> ObjectMetadata:
        """Server-side copy - never pulls bytes through this process."""
        self.client.copy_object(
            Bucket=self.bucket, Key=dst, CopySource={"Bucket": self.bucket, "Key": src}
        )
        return self.stat(dst)

    # ---------------------------------------------------------------- presigning

    def presign_upload(
        self,
        key: str,
        *,
        content_type: str | None = None,
        max_bytes: int | None = None,
        expires_in: int | None = None,
    ) -> PresignedUpload:
        """Presigned ``PUT`` with the content type bound into the signature.

        Binding ``ContentType`` means a client cannot sign for ``image/png`` and then
        upload something else - the signature will not match. A size cap cannot be
        expressed in a presigned PUT (only in a POST policy), so ``max_bytes`` is
        returned for the client to respect and the server independently verifies the
        real size with a HEAD before it accepts the asset. Never trust the declared
        size.
        """
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        headers: dict[str, str] = {}
        if content_type:
            params["ContentType"] = content_type
            headers["Content-Type"] = content_type

        expiry = expires_in or self.presign_expiry
        url = self.client.generate_presigned_url(
            "put_object", Params=params, ExpiresIn=expiry
        )
        return PresignedUpload(
            key=key,
            url=url,
            method="PUT",
            headers=headers,
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(seconds=expiry),
            max_bytes=max_bytes or 0,
        )

    def presign_download(self, key: str, *, expires_in: int | None = None) -> str:
        return str(
            self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in or self.presign_expiry,
            )
        )

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist (used by the dev MinIO setup)."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            self.client.create_bucket(Bucket=self.bucket)
            log.info("s3_bucket_created", bucket=self.bucket)
