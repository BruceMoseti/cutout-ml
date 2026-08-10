"""Storage backends and key generation.

The local backend is tested for real against a temp directory. The S3 backend is
tested with a stubbed boto3 client rather than a live MinIO: what is worth
asserting there is the *call shape* boto3 receives (path-style addressing,
ContentType bound into the presigned signature, NoSuchKey mapped to
ObjectNotFoundError), and a stub asserts that precisely. Round-tripping bytes
through MinIO would test MinIO.

Key generation gets the most attention because it is a security control - see the
module docstring of :mod:`cutoutml.storage.base`.
"""

from __future__ import annotations

import datetime as dt
import io
import os
from pathlib import Path
from typing import Any

import pytest

from cutoutml.storage.base import (
    ObjectMetadata,
    ObjectNotFoundError,
    Storage,
    StorageError,
    build_storage_key,
    sanitize_prefix,
)
from cutoutml.storage.factory import build_storage, get_storage, reset_storage_cache
from cutoutml.storage.local import LocalStorage

# ------------------------------------------------------------- key generation


def test_storage_keys_are_random_so_they_cannot_be_enumerated():
    keys = {build_storage_key(user_id="u1", extension="png") for _ in range(200)}
    assert len(keys) == 200


def test_storage_keys_are_partitioned_by_kind_user_and_date():
    key = build_storage_key(
        user_id="user-42",
        kind="outputs",
        extension="png",
        now=dt.datetime(2026, 3, 7, tzinfo=dt.UTC),
    )
    prefix, _, name = key.rpartition("/")
    assert prefix == "outputs/user-42/2026/03/07"
    assert name.endswith(".png")
    # 128 bits of randomness, hex-encoded.
    assert len(name.removesuffix(".png")) == 32


def test_the_client_filename_is_not_an_input_to_key_generation():
    """This is the structural reason ../../etc/passwd is a non-issue here rather
    than something to sanitise: there is nowhere to put it. Only the server-chosen
    kind, the user id and a sniffed extension reach the key."""
    import inspect

    assert set(inspect.signature(build_storage_key).parameters) == {
        "user_id",
        "kind",
        "extension",
        "now",
    }


def test_key_generation_rejects_traversal_in_the_server_supplied_parts():
    for kwargs in ({"kind": "../uploads"}, {"user_id": "../../root"}):
        with pytest.raises(ValueError, match=r"must not contain"):
            build_storage_key(**{"user_id": "u1", **kwargs})  # type: ignore[arg-type]


def test_key_generation_neutralises_odd_but_non_traversing_characters():
    key = build_storage_key(user_id="user 42;drop", kind="up loads", extension="png")
    assert key.startswith("up_loads/user_42_drop/")


def test_key_generation_rejects_an_unsafe_extension():
    with pytest.raises(ValueError, match="unsafe storage extension"):
        build_storage_key(user_id="u1", extension="../../sh")
    with pytest.raises(ValueError, match="unsafe storage extension"):
        build_storage_key(user_id="u1", extension="php5.suspiciouslylong")


def test_key_generation_normalises_the_extension():
    assert build_storage_key(user_id="u1", extension=".PNG").endswith(".png")
    assert "." not in build_storage_key(user_id="u1").rpartition("/")[2]


def test_empty_kind_and_user_fall_back_to_safe_defaults():
    key = build_storage_key(user_id="", kind="")
    assert key.startswith("uploads/anonymous/")


def test_sanitize_prefix_replaces_unsafe_characters_and_strips_slashes():
    assert sanitize_prefix("/a b/c;d/") == "a_b/c_d"
    assert sanitize_prefix("ok/path") == "ok/path"
    assert sanitize_prefix("") == ""


def test_sanitize_prefix_rejects_rather_than_rewrites_traversal():
    """Silently normalising an attack makes the logs useless."""
    with pytest.raises(ValueError, match=r"must not contain"):
        sanitize_prefix("a/../b")


# --------------------------------------------------------------- local backend


def test_local_storage_round_trips_bytes(local_storage: LocalStorage):
    meta = local_storage.put("a/b/c.png", b"payload", content_type="image/png")
    assert local_storage.get("a/b/c.png") == b"payload"
    assert meta.key == "a/b/c.png"
    assert meta.size == 7
    assert meta.content_type == "image/png"
    assert meta.etag


def test_local_storage_creates_intermediate_directories(local_storage: LocalStorage):
    local_storage.put("deep/nested/path/file.bin", b"x")
    assert (local_storage.root / "deep/nested/path/file.bin").is_file()


def test_local_storage_guesses_the_content_type_from_the_extension(local_storage: LocalStorage):
    local_storage.put("x.png", b"x")
    assert local_storage.stat("x.png").content_type == "image/png"


def test_local_storage_overwrites_an_existing_key(local_storage: LocalStorage):
    local_storage.put("k", b"first")
    local_storage.put("k", b"second")
    assert local_storage.get("k") == b"second"


def test_local_storage_writes_are_atomic_and_leave_no_partials(local_storage: LocalStorage):
    """A crash mid-write must not leave a truncated object that looks valid, and a
    concurrent reader must never observe one - the API polls files the worker writes."""
    local_storage.put("atomic/one.bin", b"0123456789")
    leftovers = [p.name for p in (local_storage.root / "atomic").iterdir()]
    assert leftovers == ["one.bin"]


def test_a_failed_stream_upload_cleans_up_its_temp_file(local_storage: LocalStorage):
    class Exploding(io.RawIOBase):
        def readinto(self, _: Any) -> int:
            raise OSError("connection reset")

        def readable(self) -> bool:
            return True

    with pytest.raises(OSError, match="connection reset"):
        local_storage.put_stream("broken/upload.bin", Exploding())  # type: ignore[arg-type]

    assert local_storage.exists("broken/upload.bin") is False
    directory = local_storage.root / "broken"
    assert list(directory.iterdir()) == []


def test_local_storage_put_stream_does_not_buffer_the_whole_body(local_storage: LocalStorage):
    payload = os.urandom(3 * 1024 * 1024)
    meta = local_storage.put_stream("big.bin", io.BytesIO(payload), content_type="video/mp4")
    assert meta.size == len(payload)
    assert local_storage.get("big.bin") == payload


def test_local_storage_open_streams(local_storage: LocalStorage):
    local_storage.put("s.bin", b"streamed")
    with local_storage.open("s.bin") as fh:
        assert fh.read() == b"streamed"


def test_local_storage_missing_keys_raise_object_not_found(local_storage: LocalStorage):
    with pytest.raises(ObjectNotFoundError):
        local_storage.get("nope")
    with pytest.raises(ObjectNotFoundError):
        local_storage.open("nope")
    with pytest.raises(ObjectNotFoundError):
        local_storage.stat("nope")
    assert local_storage.exists("nope") is False


def test_a_directory_is_not_an_object(local_storage: LocalStorage):
    local_storage.put("dir/inner.bin", b"x")
    assert local_storage.exists("dir") is False
    with pytest.raises(ObjectNotFoundError):
        local_storage.get("dir")


def test_local_storage_delete_is_idempotent(local_storage: LocalStorage):
    local_storage.put("gone.bin", b"x")
    local_storage.delete("gone.bin")
    local_storage.delete("gone.bin")
    assert local_storage.exists("gone.bin") is False


def test_local_storage_refuses_keys_that_escape_the_root(local_storage: LocalStorage):
    for bad in ("../escape.bin", "a/../../escape.bin", "/absolute.bin", ""):
        with pytest.raises(ValueError, match="storage key"):
            local_storage.put(bad, b"x")


def test_escaping_keys_report_absent_rather_than_leaking_existence(
    local_storage: LocalStorage, tmp_path: Path
):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    assert local_storage.exists(f"../{outside.name}") is False


def test_local_storage_resolves_symlinks_before_the_containment_check(
    local_storage: LocalStorage, tmp_path: Path
):
    """A symlink inside the root pointing out of it is the interesting escape."""
    secret = tmp_path / "secret.txt"
    secret.write_text("classified")
    (local_storage.root / "link").symlink_to(tmp_path)

    with pytest.raises(ValueError, match="escapes the storage root"):
        local_storage.get("link/secret.txt")


def test_local_storage_list_filters_by_prefix_and_honours_the_limit(local_storage: LocalStorage):
    for i in range(5):
        local_storage.put(f"uploads/u1/{i}.bin", b"x")
    local_storage.put("outputs/u1/other.bin", b"x")

    keys = [m.key for m in local_storage.list("uploads/")]
    assert len(keys) == 5
    assert all(k.startswith("uploads/") for k in keys)
    assert len(list(local_storage.list("uploads/", limit=2))) == 2


def test_local_storage_list_of_everything_includes_all_prefixes(local_storage: LocalStorage):
    local_storage.put("a/1.bin", b"x")
    local_storage.put("b/2.bin", b"x")
    assert {m.key for m in local_storage.list()} == {"a/1.bin", "b/2.bin"}


def test_local_storage_list_skips_in_flight_upload_temp_files(local_storage: LocalStorage):
    """Otherwise a concurrent upload shows up in an asset listing."""
    local_storage.put("real.bin", b"x")
    (local_storage.root / ".upload-abc123").write_bytes(b"partial")
    assert [m.key for m in local_storage.list()] == ["real.bin"]


def test_local_storage_copy_duplicates_content_and_type(local_storage: LocalStorage):
    local_storage.put("src.png", b"pixels", content_type="image/png")
    meta = local_storage.copy("src.png", "dst.png")
    assert local_storage.get("dst.png") == b"pixels"
    assert meta.content_type == "image/png"


def test_local_presign_download_requires_the_object_to_exist(local_storage: LocalStorage):
    local_storage.put("there.png", b"x")
    assert local_storage.presign_download("there.png").endswith("/there.png")
    with pytest.raises(ObjectNotFoundError):
        local_storage.presign_download("absent.png")


def test_local_presign_upload_matches_the_s3_response_shape(local_storage: LocalStorage):
    """Client code must not have to branch on the backend."""
    presigned = local_storage.presign_upload("k.png", content_type="image/png", max_bytes=1024)
    assert presigned.method == "PUT"
    assert presigned.headers == {"Content-Type": "image/png"}
    assert presigned.key == "k.png"
    assert presigned.max_bytes == 1024
    assert presigned.expires_at > dt.datetime.now(dt.UTC)
    assert set(presigned.as_dict()) == {
        "key",
        "url",
        "method",
        "headers",
        "expires_at",
        "max_bytes",
    }


def test_local_storage_reports_its_backend_name():
    """Recorded on assets so a backend migration is detectable rather than guessed."""
    assert LocalStorage.backend == "local"


# ------------------------------------------------------------------ S3 backend


class _StubS3:
    """Minimal in-memory stand-in for a boto3 S3 client."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str | None]] = {}
        self.presign_calls: list[tuple[str, dict[str, Any], int]] = []

    @staticmethod
    def _missing(code: str) -> Exception:
        exc = Exception(code)
        exc.response = {"Error": {"Code": code}}  # type: ignore[attr-defined]
        return exc

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **extra: Any) -> dict[str, Any]:
        self.objects[Key] = (Body, extra.get("ContentType"))
        return {}

    def upload_fileobj(self, stream: Any, bucket: str, key: str, ExtraArgs: Any = None) -> None:
        self.objects[key] = (stream.read(), (ExtraArgs or {}).get("ContentType"))

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise self._missing("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise self._missing("404")
        body, content_type = self.objects[Key]
        return {"ContentLength": len(body), "ContentType": content_type, "ETag": '"abc"'}

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.objects.pop(Key, None)
        return {}

    def copy_object(self, *, Bucket: str, Key: str, CopySource: dict[str, str]) -> dict[str, Any]:
        self.objects[Key] = self.objects[CopySource["Key"]]
        return {}

    def generate_presigned_url(
        self, operation: str, *, Params: dict[str, Any], ExpiresIn: int
    ) -> str:
        self.presign_calls.append((operation, Params, ExpiresIn))
        return f"https://s3.test/{Params['Key']}?op={operation}&exp={ExpiresIn}"

    def get_paginator(self, _: str) -> Any:
        objects = self.objects

        class _Paginator:
            def paginate(self, *, Bucket: str, Prefix: str) -> Any:
                yield {
                    "Contents": [
                        {"Key": k, "Size": len(v[0]), "ETag": '"e"'}
                        for k, v in sorted(objects.items())
                        if k.startswith(Prefix)
                    ]
                }

        return _Paginator()


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> Any:
    """An S3Storage whose boto3 client is the stub above."""
    pytest.importorskip("boto3", reason="boto3 ships in the api extra")
    from cutoutml.storage import s3 as s3_module

    stub = _StubS3()
    captured: dict[str, Any] = {}

    def fake_client(service: str, **kwargs: Any) -> _StubS3:
        captured["service"] = service
        captured.update(kwargs)
        return stub

    import boto3

    monkeypatch.setattr(boto3, "client", fake_client)
    storage = s3_module.S3Storage("test-bucket", endpoint_url="http://minio:9000")
    return storage, stub, captured


def test_s3_uses_path_style_addressing_for_minio_compatibility(s3):
    """Virtual-host addressing needs DNS entries MinIO does not have by default,
    and the resulting failure is an opaque connection error."""
    _, _, captured = s3
    config = captured["config"]
    assert config.s3["addressing_style"] == "path"
    assert config.signature_version == "s3v4"
    assert captured["endpoint_url"] == "http://minio:9000"


def test_s3_round_trips_bytes_and_reports_metadata(s3):
    storage, _, _ = s3
    storage.put("k.png", b"pixels", content_type="image/png")
    assert storage.get("k.png") == b"pixels"
    meta = storage.stat("k.png")
    assert meta.size == 6
    assert meta.content_type == "image/png"
    assert meta.etag == "abc"  # quotes stripped


def test_s3_maps_missing_keys_to_object_not_found(s3):
    """S3 says NoSuchKey for GET and 404/NotFound for HEAD, and MinIO is not
    perfectly consistent either; all of them must surface as one error."""
    storage, _, _ = s3
    with pytest.raises(ObjectNotFoundError):
        storage.get("absent")
    with pytest.raises(ObjectNotFoundError):
        storage.stat("absent")
    with pytest.raises(ObjectNotFoundError):
        storage.open("absent")
    assert storage.exists("absent") is False


def test_s3_wraps_unexpected_errors_as_storage_errors(s3, monkeypatch):
    storage, stub, _ = s3

    def boom(**_: Any) -> None:
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(stub, "get_object", boom)
    with pytest.raises(StorageError, match="S3 get failed"):
        storage.get("k")


def test_s3_delete_is_idempotent(s3):
    storage, _, _ = s3
    storage.put("k", b"x")
    storage.delete("k")
    storage.delete("k")
    assert storage.exists("k") is False


def test_s3_list_filters_by_prefix_and_honours_the_limit(s3):
    storage, _, _ = s3
    for i in range(4):
        storage.put(f"uploads/{i}", b"x")
    storage.put("outputs/0", b"x")

    assert len(list(storage.list("uploads/"))) == 4
    assert len(list(storage.list("uploads/", limit=2))) == 2


def test_s3_copy_is_server_side(s3):
    """Never pull bytes through this process to copy an object."""
    storage, stub, _ = s3
    storage.put("src", b"x")
    called: list[str] = []
    original = stub.get_object
    stub.get_object = lambda **kw: (called.append(kw["Key"]), original(**kw))[1]  # type: ignore[assignment]

    storage.copy("src", "dst")

    assert storage.get("dst") == b"x"
    assert "src" not in called


def test_s3_presign_upload_binds_the_content_type_into_the_signature(s3):
    """Otherwise a client can sign for image/png and upload anything."""
    storage, stub, _ = s3
    presigned = storage.presign_upload("k.png", content_type="image/png", max_bytes=2048)

    operation, params, expires = stub.presign_calls[-1]
    assert operation == "put_object"
    assert params["ContentType"] == "image/png"
    assert expires == storage.presign_expiry
    assert presigned.headers == {"Content-Type": "image/png"}
    # A size cap cannot be expressed in a presigned PUT, so it is advisory and the
    # server re-verifies the real size with a HEAD.
    assert presigned.max_bytes == 2048


def test_s3_presign_download_uses_the_requested_expiry(s3):
    storage, stub, _ = s3
    storage.presign_download("k.png", expires_in=60)
    operation, params, expires = stub.presign_calls[-1]
    assert operation == "get_object"
    assert params["Key"] == "k.png"
    assert expires == 60


def test_s3_stream_upload_delegates_to_boto3_multipart(s3):
    """upload_fileobj is what keeps memory flat for large video uploads."""
    storage, _, _ = s3
    storage.put_stream("v.mp4", io.BytesIO(b"video-bytes"), content_type="video/mp4")
    assert storage.get("v.mp4") == b"video-bytes"
    assert storage.stat("v.mp4").content_type == "video/mp4"


# ---------------------------------------------------------------------- factory


def test_the_factory_builds_the_configured_backend(tmp_path: Path, settings):
    storage = build_storage(settings)
    assert isinstance(storage, LocalStorage)
    assert storage.backend == "local"


def test_the_factory_rejects_an_unknown_backend(settings):
    with pytest.raises(ValueError, match="unknown storage backend"):
        build_storage(settings.model_copy(update={"storage_backend": "gopher"}))


def test_get_storage_is_cached_and_resettable(settings):
    """The API resolves storage per request, so it has to be cheap; tests repoint
    the root, so the cache has to be clearable."""
    assert get_storage() is get_storage()
    first = get_storage()
    reset_storage_cache()
    assert get_storage() is not first


# ------------------------------------------------------------- interface parity


def test_both_backends_implement_the_whole_interface():
    """The interface is small on purpose: every method must be implementable on a
    filesystem and on a bucket without one leaking the other's semantics."""
    from cutoutml.storage.s3 import S3Storage

    required = {
        name
        for name, value in vars(Storage).items()
        if getattr(value, "__isabstractmethod__", False)
    }
    assert required
    for backend in (LocalStorage, S3Storage):
        assert not required - set(dir(backend))
        assert not getattr(backend, "__abstractmethods__", frozenset())


def test_object_metadata_is_json_serialisable():
    meta = ObjectMetadata(
        key="k",
        size=3,
        content_type="image/png",
        last_modified=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
        etag="e",
    )
    assert meta.as_dict()["last_modified"] == "2026-01-02T00:00:00+00:00"
    assert ObjectMetadata("k", 0, None, None).as_dict()["last_modified"] is None
