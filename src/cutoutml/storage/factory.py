"""Storage selection from configuration."""

from __future__ import annotations

import functools

from cutoutml.core.config import Settings, get_settings
from cutoutml.storage.base import Storage
from cutoutml.storage.local import LocalStorage
from cutoutml.storage.s3 import S3Storage


def build_storage(settings: Settings) -> Storage:
    """Construct the storage backend named by ``settings.storage_backend``.

    Unknown names raise rather than defaulting to the local backend. The config
    ``Literal`` already rejects a typo from the environment, so the only way to get
    here is by adding a backend to that ``Literal`` and forgetting this function -
    in which case a silent fallback would write a distributed deployment's results
    to one node's disk, where they would appear to vanish.
    """
    if settings.storage_backend == "local":
        return LocalStorage(settings.storage_root)
    if settings.storage_backend == "s3":
        return S3Storage(
            settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            force_path_style=settings.s3_force_path_style,
            presign_expiry=settings.presign_expiry_seconds,
        )
    raise ValueError(f"unknown storage backend: {settings.storage_backend!r}")


@functools.lru_cache(maxsize=1)
def get_storage() -> Storage:
    """Process-wide storage singleton.

    Cached because ``S3Storage`` builds a boto3 client, which is expensive
    (credential resolution, TLS context) and thread-safe to share.
    """
    return build_storage(get_settings())


def reset_storage_cache() -> None:
    """Drop the cached backend. Needed by tests that repoint ``storage_root``."""
    get_storage.cache_clear()
