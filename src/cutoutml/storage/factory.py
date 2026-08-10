"""Storage selection from configuration."""

from __future__ import annotations

import functools

from cutoutml.core.config import Settings, get_settings
from cutoutml.storage.base import Storage
from cutoutml.storage.local import LocalStorage
from cutoutml.storage.s3 import S3Storage


def build_storage(settings: Settings) -> Storage:
    """Construct the storage backend named by ``settings.storage_backend``."""
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
    return LocalStorage(settings.storage_root)


@functools.lru_cache(maxsize=1)
def get_storage() -> Storage:
    """Process-wide storage singleton.

    Cached because ``S3Storage`` builds a boto3 client, which is expensive
    (credential resolution, TLS context) and thread-safe to share. Tests call
    ``get_storage.cache_clear()``.
    """
    return build_storage(get_settings())
