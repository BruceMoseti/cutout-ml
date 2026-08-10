"""Object storage abstraction: local filesystem for dev/tests, S3 for production."""

from cutoutml.storage.base import (
    ObjectMetadata,
    PresignedUpload,
    Storage,
    StorageError,
    ObjectNotFoundError,
    build_storage_key,
)
from cutoutml.storage.factory import get_storage
from cutoutml.storage.local import LocalStorage
from cutoutml.storage.s3 import S3Storage

__all__ = [
    "LocalStorage",
    "ObjectMetadata",
    "ObjectNotFoundError",
    "PresignedUpload",
    "S3Storage",
    "Storage",
    "StorageError",
    "build_storage_key",
    "get_storage",
]
