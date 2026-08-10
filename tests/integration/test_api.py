"""API routers, exercised through a real test client against a real Postgres.

Why not mock the database: almost everything worth asserting about these routers is
in the SQL. Ownership is enforced by a ``WHERE owner_id = ...`` inside a dependency,
idempotency is enforced by a unique index, and soft-deletes are enforced by a status
predicate. A mocked session would let all three pass while broken.

The app is constructed per test through :func:`create_app` with the environment
repointed at a disposable database and a temp storage root, and every process-wide
cache (settings, engine, sessionmaker, storage) is cleared around it. Skipping that
is how one test silently receives another's configuration.

Celery runs in eager mode, so ``POST /assets/{id}/process`` executes the real task
body in-process. That makes these end-to-end tests: upload bytes, run the model,
read a transparent PNG back out.
"""

from __future__ import annotations

import io
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.integration

fastapi_client = pytest.importorskip("fastapi.testclient", reason="fastapi is required")
TestClient = fastapi_client.TestClient


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def api_settings(
    api_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Any]:
    """Point every process-wide singleton at throwaway resources.

    The rate limit is raised far above what any single test issues; the limiter has its
    own test that lowers it deliberately. Without this, unauthenticated calls from the
    whole module share one IP bucket and start 429-ing each other.
    """
    from cutoutml.core.config import get_settings
    from cutoutml.db.session import reset_caches
    from cutoutml.storage.factory import reset_storage_cache

    monkeypatch.setenv("CUTOUTML_ENVIRONMENT", "test")
    monkeypatch.setenv("CUTOUTML_DATABASE_URL", api_database)
    monkeypatch.setenv("CUTOUTML_STORAGE_BACKEND", "local")
    monkeypatch.setenv("CUTOUTML_STORAGE_ROOT", str(tmp_path / "objects"))
    monkeypatch.setenv("CUTOUTML_JWT_SECRET", "integration-test-signing-key")
    monkeypatch.setenv("CUTOUTML_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("CUTOUTML_DEVICE", "cpu")
    monkeypatch.setenv("CUTOUTML_TORCH_NUM_THREADS", "1")
    monkeypatch.setenv("CUTOUTML_RATE_LIMIT_PER_MINUTE", "10000")
    monkeypatch.setenv("CUTOUTML_RATE_LIMIT_BURST", "10000")
    monkeypatch.setenv("CUTOUTML_CELERY_TASK_ALWAYS_EAGER", "1")

    def _reset() -> None:
        get_settings.cache_clear()
        reset_caches()
        reset_storage_cache()

    _reset()
    try:
        yield get_settings()
    finally:
        _reset()


@pytest.fixture
def client(api_settings: Any) -> Iterator[Any]:
    """A test client with the lifespan actually run.

    ``with TestClient(...)`` rather than a bare constructor: the Redis client and the
    rate limiter are created in the lifespan, and without it every rate-limited route
    fails on a missing ``app.state.rate_limiter``.
    """
    from services.api.app.main import create_app
    from services.inference.app import celery_app as celery_module

    # The Celery app is built at import time, so a later environment change does not
    # reach it. Eager mode is set on the live object instead.
    celery_module.celery.conf.task_always_eager = True
    celery_module.celery.conf.task_eager_propagates = True

    with TestClient(create_app(api_settings)) as test_client:
        yield test_client


def register(client: Any, email: str | None = None, password: str = "correct horse battery") -> str:
    """Register a fresh account and return its bearer token.

    ``example.com`` rather than the more obvious ``.test``: ``email_validator`` rejects
    the RFC 6761 special-use TLDs outright, so ``user@example.test`` never reaches the
    handler under test - it fails schema validation with a 422 instead.
    """
    address = email or f"user-{uuid.uuid4().hex[:10]}@example.com"
    response = client.post(
        "/v1/auth/register",
        json={"email": address, "password": password, "display_name": "Test User"},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["access_token"])


@pytest.fixture
def token(client: Any) -> str:
    return register(client)


@pytest.fixture
def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def make_png(width: int = 96, height: int = 64) -> bytes:
    """A high-contrast blob on a dark field, so a real model produces a real mask."""
    from cutoutml.core.imaging import encode_image

    image = np.full((height, width, 3), 18, dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    radius = min(width, height) * 0.3
    blob = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= radius**2
    image[blob] = (245, 225, 70)
    return encode_image(image, "png")


def upload_image(client: Any, auth: dict[str, str], data: bytes | None = None) -> dict[str, Any]:
    """Upload an image through the single-request multipart route."""
    response = client.post(
        "/v1/assets",
        headers=auth,
        files={"file": ("blob.png", data or make_png(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


# ------------------------------------------------------------------ open routes


def test_root_advertises_the_docs_and_health_paths(client: Any):
    body = client.get("/").json()
    assert body["name"] == "CutoutML API"
    assert body["docs"] == "/docs"


def test_liveness_touches_nothing_external(client: Any):
    """Wiring dependency checks into liveness is an outage amplifier: one database
    hiccup restarts the whole fleet and the cold caches turn it into an outage."""
    for path in ("/health", "/health/live"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_readiness_reports_each_dependency_individually(client: Any):
    response = client.get("/health/ready")
    checks = {c["name"]: c for c in response.json()["checks"]}

    assert set(checks) == {"database", "redis", "model_registry", "storage", "ffmpeg"}
    assert checks["database"]["ok"] is True
    assert checks["model_registry"]["ok"] is True
    assert all(c["duration_ms"] >= 0 for c in checks.values())
    # 503 only when a *required* check fails, so the pod is pulled from the load
    # balancer without being killed.
    assert response.status_code == (200 if checks["redis"]["ok"] else 503)


def test_readiness_does_not_fail_the_pod_for_a_missing_ffmpeg(client: Any):
    """A missing ffmpeg only breaks video jobs; refusing all traffic would take the
    working image path down with it."""
    body = client.get("/health/ready").json()
    ffmpeg = next(c for c in body["checks"] if c["name"] == "ffmpeg")
    if not ffmpeg["ok"]:
        assert body["status"] == "ready"


def test_metrics_are_exposed_in_prometheus_format(client: Any):
    """The advertised content type has to match the body.

    OpenMetrics requires an ``# EOF`` terminator that the Prometheus text serialiser does
    not emit, so claiming ``application/openmetrics-text`` for a text-format body is a
    scrape a strict parser rejects.
    """
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "openmetrics" not in response.headers["content-type"]
    assert "# EOF" not in response.text
    assert "cutoutml_http_requests_total" in response.text


def test_the_request_histogram_labels_the_route_template_not_the_concrete_path(
    client: Any, auth: dict[str, str]
):
    """One time series per asset UUID is how a metrics endpoint grows without bound and
    eventually takes the scraper down with it."""
    asset = upload_image(client, auth)
    client.get(f"/v1/assets/{asset['id']}", headers=auth)

    body = client.get("/metrics").text
    assert 'route="/v1/assets/{asset_id}"' in body
    assert asset["id"] not in body


def test_openapi_documents_the_error_envelope(client: Any):
    """Clients generate from this; documenting FastAPI's default {"detail": ...} would
    be documenting a shape the API never returns."""
    schema = client.get("/openapi.json").json()
    assert "ErrorResponse" in schema["components"]["schemas"]
    responses = schema["paths"]["/v1/jobs"]["get"]["responses"]
    assert "401" in responses
    assert "ErrorResponse" in str(responses["401"])


# ------------------------------------------------------------------------- auth


def test_register_returns_a_usable_token(client: Any):
    token = register(client)
    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"].endswith("@example.com")


def test_register_never_echoes_the_password_hash(client: Any):
    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    register(client, email)
    body = client.get(
        "/v1/auth/me",
        headers={
            "Authorization": f"Bearer {register(client, f'other-{uuid.uuid4().hex[:8]}@example.org')}"
        },
    ).json()
    assert "password" not in str(body).lower()


def test_email_is_normalised_to_lower_case(client: Any):
    email = f"Mixed-{uuid.uuid4().hex[:8]}@Example.COM"
    register(client, email)
    response = client.post(
        "/v1/auth/login", json={"email": email.lower(), "password": "correct horse battery"}
    )
    assert response.status_code == 200


def test_a_duplicate_email_is_a_conflict_from_the_unique_index(client: Any):
    """Checking with a SELECT first and inserting second is a race two concurrent
    signups lose, so the index is the authority."""
    email = f"dupe-{uuid.uuid4().hex[:8]}@example.com"
    register(client, email)
    response = client.post(
        "/v1/auth/register", json={"email": email, "password": "another password entirely"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "email_taken"


def test_login_does_not_distinguish_a_bad_password_from_an_unknown_account(client: Any):
    """Distinguishing them turns the endpoint into an account-enumeration oracle."""
    email = f"known-{uuid.uuid4().hex[:8]}@example.com"
    register(client, email)

    wrong_password = client.post(
        "/v1/auth/login", json={"email": email, "password": "not the password"}
    )
    unknown_user = client.post(
        "/v1/auth/login", json={"email": "nobody@example.com", "password": "not the password"}
    )

    assert wrong_password.status_code == unknown_user.status_code == 401

    # request_id is per-request by construction, so it is excluded from the comparison;
    # everything a caller could use to tell the two cases apart must be identical.
    def comparable(response: Any) -> dict[str, Any]:
        return {k: v for k, v in response.json()["error"].items() if k != "request_id"}

    assert comparable(wrong_password) == comparable(unknown_user)


def test_short_passwords_are_rejected_by_validation(client: Any):
    response = client.post("/v1/auth/register", json={"email": "x@example.com", "password": "abc"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_a_validation_error_never_echoes_the_submitted_password(client: Any):
    """Pydantic's ctx can contain the offending value, which for a login body is the
    password, so ctx is dropped from the envelope."""
    secret = "hunter2-would-be-a-leak"
    response = client.post("/v1/auth/login", json={"email": "not-an-email", "password": secret})
    assert response.status_code == 422
    assert secret not in response.text


def test_protected_routes_require_a_bearer_token(client: Any):
    response = client.get("/v1/auth/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"
    assert response.headers["www-authenticate"] == "Bearer"


def test_a_garbage_token_is_rejected_as_invalid(client: Any):
    response = client.get("/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_token"


def test_a_token_signed_with_another_key_is_rejected(client: Any, api_settings: Any):
    """The signature is the whole control; accepting an unverified token would let
    anyone mint one for any user id."""
    from services.api.app.security import create_access_token

    forged = create_access_token(
        uuid.uuid4(), settings=api_settings.model_copy(update={"jwt_secret": "a different key"})
    )
    response = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_a_token_for_a_deleted_user_stops_working_immediately(client: Any, api_settings: Any):
    """The user row is loaded per request rather than trusted from the claims, so a
    disabled account cannot keep working for the remaining token TTL."""
    from sqlalchemy import delete

    from cutoutml.db.models import User
    from cutoutml.db.session import session_scope

    token = register(client)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/v1/auth/me", headers=headers).status_code == 200

    with session_scope() as session:
        user_id = client.get("/v1/auth/me", headers=headers).json()["id"]
        session.execute(delete(User).where(User.id == uuid.UUID(user_id)))

    response = client.get("/v1/auth/me", headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "inactive_user"


# --------------------------------------------------------------- error envelope


def test_every_error_uses_one_envelope_with_a_request_id(client: Any, auth: dict[str, str]):
    response = client.get(f"/v1/assets/{uuid.uuid4()}", headers=auth)
    assert response.status_code == 404
    error = response.json()["error"]
    assert set(error) >= {"code", "message", "request_id"}
    assert error["code"] == "asset_not_found"
    assert error["request_id"]


def test_the_request_id_in_the_body_matches_the_response_header(client: Any, auth: dict[str, str]):
    """This is the whole point of the field: a user quotes it and it correlates the
    response with the server log and with the worker that ran the job."""
    response = client.get(f"/v1/jobs/{uuid.uuid4()}", headers=auth)
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


def test_a_client_supplied_request_id_is_propagated(client: Any):
    response = client.get("/health", headers={"X-Request-ID": "caller-chosen-id"})
    assert response.headers["X-Request-ID"] == "caller-chosen-id"


def test_a_malformed_uuid_in_the_path_is_a_validation_error(client: Any, auth: dict[str, str]):
    response = client.get("/v1/assets/not-a-uuid", headers=auth)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["details"]["fields"]


def test_an_unknown_path_still_returns_the_envelope(client: Any):
    """Starlette's own 404 body is {"detail": ...}; a client must not have to branch."""
    response = client.get("/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_a_wrong_method_returns_the_envelope(client: Any):
    response = client.delete("/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


# ----------------------------------------------------------------- rate limiting


def test_the_rate_limiter_returns_429_with_retry_after(
    api_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Built with its own tiny limit rather than by hammering the shared one, so the
    test is deterministic and takes milliseconds."""
    from services.api.app.main import create_app

    from cutoutml.core.config import get_settings
    from cutoutml.db.session import reset_caches
    from cutoutml.storage.factory import reset_storage_cache

    monkeypatch.setenv("CUTOUTML_ENVIRONMENT", "test")
    monkeypatch.setenv("CUTOUTML_DATABASE_URL", api_database)
    monkeypatch.setenv("CUTOUTML_STORAGE_ROOT", str(tmp_path / "objects"))
    monkeypatch.setenv("CUTOUTML_JWT_SECRET", "integration-test-signing-key")
    monkeypatch.setenv("CUTOUTML_RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setenv("CUTOUTML_RATE_LIMIT_BURST", "2")
    get_settings.cache_clear()
    reset_caches()
    reset_storage_cache()

    try:
        with TestClient(create_app(get_settings())) as limited:
            codes = [limited.get("/v1/models").status_code for _ in range(12)]
            assert 429 in codes
            blocked = next(
                r for r in (limited.get("/v1/models") for _ in range(3)) if r.status_code == 429
            )
            assert blocked.json()["error"]["code"] == "rate_limited"
            assert float(blocked.headers["Retry-After"]) >= 0
            assert blocked.json()["error"]["details"]["retry_after_seconds"] >= 0
    finally:
        get_settings.cache_clear()
        reset_caches()
        reset_storage_cache()


def test_health_and_metrics_are_not_rate_limited(client: Any):
    """Excluded by construction - the limiter is a router dependency, not middleware -
    so a monitoring system cannot lock itself out."""
    assert all(client.get("/health/live").status_code == 200 for _ in range(30))
    assert client.get("/metrics").status_code == 200


# --------------------------------------------------------------------- catalog


def test_models_reports_availability_so_a_client_can_grey_out_a_selector(
    client: Any, auth: dict[str, str]
):
    body = client.get("/v1/models", headers=auth).json()
    by_name = {m["name"]: m for m in body["items"]}

    assert body["default_model"] == "cutoutnet"
    assert by_name["cutoutnet"]["weights_available"] is True
    assert by_name["classical"]["weights_available"] is True
    assert by_name["cutoutnet"]["license"].startswith("MIT")
    assert "input_size" in by_name["cutoutnet"]


def test_models_discloses_the_licence_caveat_on_the_reimplementations(
    client: Any, auth: dict[str, str]
):
    """A user picking birefnet needs to know the official weights are elsewhere and
    that some third-party fine-tunes are non-commercial."""
    by_name = {m["name"]: m for m in client.get("/v1/models", headers=auth).json()["items"]}
    assert "non-commercial" in by_name["birefnet"]["license"]
    assert "reimplement" in by_name["birefnet"]["description"].lower()


def test_the_raw_catalogue_matches_the_registry(client: Any, auth: dict[str, str]):
    from cutoutml.models.registry import list_model_names

    entries = client.get("/v1/models/catalogue", headers=auth).json()
    assert {e["name"] for e in entries} == set(list_model_names())


def test_benchmarks_reads_committed_json_rather_than_measuring_on_request(
    client: Any, auth: dict[str, str], api_settings: Any
):
    """A measurement taken while serving traffic is a measurement of a contended
    machine, and it would carry no provenance."""
    body = client.get("/v1/benchmarks", headers=auth).json()
    assert body["results_dir"].endswith("benchmarks/results")
    assert isinstance(body["items"], list)
    assert body["total_files"] == len(list(Path(body["results_dir"]).glob("*.json")))


def test_a_benchmark_run_id_cannot_traverse_out_of_the_results_directory(
    client: Any, auth: dict[str, str]
):
    """run_id is matched against the directory listing rather than joined onto a path."""
    response = client.get("/v1/benchmarks/..%2f..%2fetc%2fpasswd", headers=auth)
    assert response.status_code in {404, 422}
    assert "root:" not in response.text


# ---------------------------------------------------------------------- assets


def test_multipart_upload_stores_the_bytes_and_sniffs_the_dimensions(
    client: Any, auth: dict[str, str]
):
    asset = upload_image(client, auth)
    assert asset["kind"] == "image"
    assert asset["status"] == "ready"
    assert (asset["width"], asset["height"]) == (96, 64)
    assert asset["size_bytes"] == len(make_png())
    assert asset["content_type"] == "image/png"
    assert asset["storage_backend"] == "local"


def test_the_asset_response_does_not_leak_the_storage_layout(client: Any, auth: dict[str, str]):
    """A client addresses an asset by id and downloads through the API, so the key and the
    content hash are internal. Exposing the key invites clients to construct their own
    object-store URLs, which then constrains any future change to the key scheme."""
    asset = upload_image(client, auth)
    assert "storage_key" not in asset
    assert "content_sha256" not in asset
    assert "owner_id" not in asset


def test_the_content_hash_is_recorded_for_content_addressed_idempotency(
    client: Any, auth: dict[str, str]
):
    """Read from the row rather than the response because the hash is what the job
    idempotency key is derived from, not something a client is meant to see."""
    import hashlib

    from cutoutml.db.models import Asset
    from cutoutml.db.session import session_scope

    data = make_png()
    asset = upload_image(client, auth, data)
    with session_scope() as session:
        row = session.get(Asset, uuid.UUID(asset["id"]))
        assert row is not None
        assert row.content_sha256 == hashlib.sha256(data).hexdigest()


def test_the_stored_key_is_server_generated_and_ignores_the_client_filename(
    client: Any, auth: dict[str, str]
):
    """The filename never reaches a path, which is what makes ../../etc/passwd a
    non-issue rather than something to sanitise."""
    from cutoutml.db.models import Asset
    from cutoutml.db.session import session_scope

    response = client.post(
        "/v1/assets",
        headers=auth,
        files={"file": ("../../../etc/passwd.png", make_png(), "image/png")},
    )
    assert response.status_code == 201

    with session_scope() as session:
        row = session.get(Asset, uuid.UUID(response.json()["id"]))
        assert row is not None
        key = row.storage_key
        # Reduced to a basename on the way in, because this value is display-only and can
        # end up in a Content-Disposition header or an HTML page.
        assert row.original_filename == "passwd.png"

    assert ".." not in key
    assert "passwd" not in key
    assert key.startswith("uploads/")


def test_an_upload_whose_content_contradicts_its_declared_type_is_rejected(
    client: Any, auth: dict[str, str]
):
    """The sniffer is authoritative: a .png that is really a script must not be stored
    as an image and handed to a decoder."""
    response = client.post(
        "/v1/assets",
        headers=auth,
        files={"file": ("evil.png", b"#!/bin/sh\nrm -rf /\n", "image/png")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_type"


def test_an_empty_upload_is_rejected_with_its_own_code(client: Any, auth: dict[str, str]):
    """Distinct from unsupported_type so the client can say "the file is empty" rather
    than "we could not identify the format", which is a confusing thing to read."""
    response = client.post("/v1/assets", headers=auth, files={"file": ("empty.png", b"")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_upload"


def test_an_explicit_kind_that_contradicts_the_content_is_rejected(
    client: Any, auth: dict[str, str]
):
    response = client.post(
        "/v1/assets",
        headers=auth,
        data={"kind": "video"},
        files={"file": ("blob.png", make_png(), "image/png")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "kind_mismatch"


def test_an_invalid_kind_form_field_is_rejected(client: Any, auth: dict[str, str]):
    response = client.post(
        "/v1/assets",
        headers=auth,
        data={"kind": "hologram"},
        files={"file": ("blob.png", make_png(), "image/png")},
    )
    assert response.status_code == 400
    assert "image or video" in response.json()["error"]["message"]


def test_the_two_phase_upload_reserves_then_accepts_the_bytes(client: Any, auth: dict[str, str]):
    reserved = client.post(
        "/v1/assets/upload-url",
        headers=auth,
        json={"kind": "image", "filename": "photo.png", "content_type": "image/png"},
    )
    assert reserved.status_code == 201
    body = reserved.json()
    assert body["method"] == "PUT"
    assert body["max_bytes"] > 0

    # The local backend cannot sign for an endpoint it does not own, so the URL points
    # back at this API, which authorises normally.
    upload = client.put(body["upload_url"], headers=auth, content=make_png())
    assert upload.status_code == 200
    assert upload.json()["status"] == "ready"


def test_reserving_an_upload_above_the_declared_limit_is_refused_up_front(
    client: Any, auth: dict[str, str], api_settings: Any
):
    response = client.post(
        "/v1/assets/upload-url",
        headers=auth,
        json={
            "kind": "image",
            "filename": "huge.png",
            "content_type": "image/png",
            "size_bytes": api_settings.max_upload_bytes + 1,
        },
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_re_uploading_a_ready_asset_is_a_conflict(client: Any, auth: dict[str, str]):
    """The content hash is part of the idempotency key of any job created from it, so
    swapping the bytes would make a completed job inconsistent with its input."""
    asset = upload_image(client, auth)
    response = client.put(f"/v1/assets/{asset['id']}/content", headers=auth, content=make_png())
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "asset_already_uploaded"


def test_listing_assets_paginates_and_filters_by_kind(client: Any, auth: dict[str, str]):
    for _ in range(3):
        upload_image(client, auth)

    listed = client.get("/v1/assets", headers=auth, params={"limit": 2}).json()
    assert listed["total"] == 3
    assert len(listed["items"]) == 2
    assert listed["limit"] == 2

    assert client.get("/v1/assets", headers=auth, params={"kind": "video"}).json()["total"] == 0
    assert client.get("/v1/assets", headers=auth, params={"limit": 0}).status_code == 422


def test_asset_content_round_trips(client: Any, auth: dict[str, str]):
    data = make_png(48, 32)
    asset = upload_image(client, auth, data)
    response = client.get(f"/v1/assets/{asset['id']}/content", headers=auth)
    assert response.status_code == 200
    assert response.content == data
    assert response.headers["cache-control"] == "private, max-age=300"


def test_a_reserved_asset_has_no_content_to_download_yet(client: Any, auth: dict[str, str]):
    reserved = client.post(
        "/v1/assets/upload-url",
        headers=auth,
        json={"kind": "image", "filename": "x.png", "content_type": "image/png"},
    ).json()
    response = client.get(f"/v1/assets/{reserved['asset_id']}/content", headers=auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "asset_not_ready"


def test_delete_soft_deletes_the_row_and_hard_deletes_the_object(
    client: Any, auth: dict[str, str], api_settings: Any
):
    """The row survives so jobs referencing it still resolve and the audit trail
    stands; the bytes go immediately, because that is what a user means by delete."""
    from cutoutml.db.models import Asset, AssetStatus
    from cutoutml.db.session import session_scope
    from cutoutml.storage.factory import get_storage

    asset = upload_image(client, auth)
    asset_id = uuid.UUID(asset["id"])
    with session_scope() as session:
        row = session.get(Asset, asset_id)
        assert row is not None
        key = row.storage_key
    assert get_storage().exists(key) is True

    assert client.delete(f"/v1/assets/{asset['id']}", headers=auth).status_code == 204

    assert get_storage().exists(key) is False
    with session_scope() as session:
        row = session.get(Asset, asset_id)
        assert row is not None, "the row is soft-deleted, not removed"
        assert row.status == AssetStatus.DELETED.value

    assert client.get(f"/v1/assets/{asset['id']}", headers=auth).status_code == 404
    assert client.get("/v1/assets", headers=auth).json()["total"] == 0


def test_deleting_twice_is_a_404_not_a_500(client: Any, auth: dict[str, str]):
    asset = upload_image(client, auth)
    client.delete(f"/v1/assets/{asset['id']}", headers=auth)
    assert client.delete(f"/v1/assets/{asset['id']}", headers=auth).status_code == 404


# ------------------------------------------------------------------- ownership


def test_one_tenant_cannot_read_another_tenants_asset(client: Any, auth: dict[str, str]):
    """Enforced in the WHERE clause of the dependency that resolves the id, not by an
    ``if`` in the handler that a future edit could drop."""
    asset = upload_image(client, auth)
    intruder = {"Authorization": f"Bearer {register(client)}"}

    for path in ("", "/content", "/result"):
        response = client.get(f"/v1/assets/{asset['id']}{path}", headers=intruder)
        assert response.status_code == 404

    assert client.delete(f"/v1/assets/{asset['id']}", headers=intruder).status_code == 404


def test_a_foreign_asset_reports_404_rather_than_403(client: Any, auth: dict[str, str]):
    """403 for "exists but not yours" and 404 for "does not exist" is an existence
    oracle that lets one tenant enumerate another's ids."""
    asset = upload_image(client, auth)
    intruder = {"Authorization": f"Bearer {register(client)}"}

    foreign = client.get(f"/v1/assets/{asset['id']}", headers=intruder)
    absent = client.get(f"/v1/assets/{uuid.uuid4()}", headers=intruder)
    assert foreign.status_code == absent.status_code == 404
    assert foreign.json()["error"]["code"] == absent.json()["error"]["code"]


def test_another_tenants_asset_cannot_be_used_as_a_background_image(
    client: Any, auth: dict[str, str]
):
    """Otherwise a request could name a foreign asset as its "background" and receive
    those pixels back in the composited output."""
    victim_background = upload_image(client, auth)

    attacker = {"Authorization": f"Bearer {register(client)}"}
    own_asset = upload_image(client, attacker)

    response = client.post(
        f"/v1/assets/{own_asset['id']}/process",
        headers=attacker,
        json={
            "model": "trivial-center",
            "image": {
                "background": "image",
                "background_asset_id": victim_background["id"],
                "outputs": ["mask_png"],
            },
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "asset_not_found"


# ------------------------------------------------------------- process and jobs


def test_processing_an_image_end_to_end_returns_a_transparent_png(
    client: Any, auth: dict[str, str]
):
    """The whole point of the product, asserted end to end: upload bytes, run the
    model through Celery, download a PNG whose alpha channel is real."""
    from PIL import Image

    asset = upload_image(client, auth)
    accepted = client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "cutoutnet", "image": {"outputs": ["transparent_png", "mask_png"]}},
    )
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["id"]

    job = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert job["status"] == "succeeded", job
    assert job["model_name"] == "cutoutnet"
    assert job["runs"], "a job must record at least one attempt"

    result = client.get(f"/v1/jobs/{job_id}/result", headers=auth).json()
    kinds = {o["kind"] for o in result["outputs"]}
    assert {"transparent_png", "mask_png"} <= kinds
    assert result["metrics"]["timings_ms"]["inference"] > 0

    download = client.get(f"/v1/jobs/{job_id}/outputs/transparent_png", headers=auth)
    assert download.status_code == 200
    with Image.open(io.BytesIO(download.content)) as png:
        assert png.mode == "RGBA"
        assert png.size == (96, 64)
        alpha = np.array(png)[:, :, 3]
    assert alpha.min() < 250, "a fully opaque alpha plane means the cutout did nothing"


def test_a_resubmission_returns_the_original_job_rather_than_duplicating_work(
    client: Any, auth: dict[str, str]
):
    """Submitting the same bytes with the same options twice is a double-click or a
    client retry, not a request for a second identical output."""
    asset = upload_image(client, auth)
    body = {"model": "trivial-center", "image": {"outputs": ["mask_png"]}}

    first = client.post(f"/v1/assets/{asset['id']}/process", headers=auth, json=body)
    second = client.post(f"/v1/assets/{asset['id']}/process", headers=auth, json=body)

    assert first.status_code == 202
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]


def test_an_explicit_idempotency_key_forces_a_distinct_job(client: Any, auth: dict[str, str]):
    asset = upload_image(client, auth)
    body: dict[str, Any] = {"model": "trivial-center", "image": {"outputs": ["mask_png"]}}

    first = client.post(f"/v1/assets/{asset['id']}/process", headers=auth, json=body)
    second = client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={**body, "idempotency_key": "deliberately-different"},
    )
    assert second.status_code == 202
    assert first.json()["id"] != second.json()["id"]


def test_different_options_are_different_jobs(client: Any, auth: dict[str, str]):
    """The key covers the parameter set, so changing the requested outputs must not
    return the previous job's result."""
    asset = upload_image(client, auth)
    first = client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "trivial-center", "image": {"outputs": ["mask_png"]}},
    )
    second = client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "trivial-center", "image": {"outputs": ["transparent_png"]}},
    )
    assert second.status_code == 202
    assert first.json()["id"] != second.json()["id"]


def test_the_idempotency_key_is_scoped_to_the_owner(client: Any, auth: dict[str, str]):
    """Two users uploading identical bytes must not collide on one job."""
    data = make_png()
    mine = upload_image(client, auth, data)
    other_auth = {"Authorization": f"Bearer {register(client)}"}
    theirs = upload_image(client, other_auth, data)

    body = {"model": "trivial-center", "image": {"outputs": ["mask_png"]}}
    first = client.post(f"/v1/assets/{mine['id']}/process", headers=auth, json=body)
    second = client.post(f"/v1/assets/{theirs['id']}/process", headers=other_auth, json=body)

    assert first.status_code == second.status_code == 202
    assert first.json()["id"] != second.json()["id"]


def test_processing_an_unknown_model_is_a_400_naming_the_alternatives(
    client: Any, auth: dict[str, str]
):
    asset = upload_image(client, auth)
    response = client.post(
        f"/v1/assets/{asset['id']}/process", headers=auth, json={"model": "segment-anything"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_model"
    assert "cutoutnet" in response.json()["error"]["message"]


def test_processing_a_gpu_only_model_is_refused_on_this_hardware(client: Any, auth: dict[str, str]):
    """There is no GPU here, so the TensorRT spec must be refused at submission rather
    than accepted and failed asynchronously."""
    asset = upload_image(client, auth)
    response = client.post(
        f"/v1/assets/{asset['id']}/process", headers=auth, json={"model": "tensorrt"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_unavailable"


def test_processing_an_asset_with_no_content_is_a_conflict(client: Any, auth: dict[str, str]):
    reserved = client.post(
        "/v1/assets/upload-url",
        headers=auth,
        json={"kind": "image", "filename": "x.png", "content_type": "image/png"},
    ).json()
    response = client.post(f"/v1/assets/{reserved['asset_id']}/process", headers=auth, json={})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "asset_not_ready"


def test_an_image_job_is_routed_to_a_cpu_queue_on_this_machine(client: Any, auth: dict[str, str]):
    asset = upload_image(client, auth)
    job = client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "trivial-center", "device": "cpu"},
    ).json()
    assert job["queue"] == "cpu"


def test_listing_jobs_filters_by_status_and_rejects_an_unknown_one(
    client: Any, auth: dict[str, str]
):
    asset = upload_image(client, auth)
    client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "trivial-center", "image": {"outputs": ["mask_png"]}},
    )

    assert len(client.get("/v1/jobs", headers=auth).json()) == 1
    assert len(client.get("/v1/jobs", headers=auth, params={"status": "succeeded"}).json()) == 1
    assert client.get("/v1/jobs", headers=auth, params={"status": "pending"}).json() == []

    bad = client.get("/v1/jobs", headers=auth, params={"status": "wat"})
    assert bad.status_code == 400
    assert "unknown status" in bad.json()["error"]["message"]


def test_a_job_records_every_attempt_not_just_the_last(client: Any, auth: dict[str, str]):
    """The attempts are what show whether a success came first try or after an OOM
    retry at a smaller batch size, which a single mutable row per job cannot express."""
    asset = upload_image(client, auth)
    job_id = client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "trivial-center", "image": {"outputs": ["mask_png"]}},
    ).json()["id"]

    runs = client.get(f"/v1/jobs/{job_id}", headers=auth).json()["runs"]
    assert len(runs) >= 1
    assert runs[0]["attempt"] == 1
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["device"] == "cpu"


def insert_queued_job(client: Any, auth: dict[str, str]) -> str:
    """Write a job row that is genuinely still queued, and return its id.

    Celery runs eagerly here, so a job dispatched through the API has already finished by
    the time the 202 comes back. Cancellation and the not-yet-complete branch of the
    result endpoint are only reachable against a non-terminal row, so it is written
    directly rather than dispatched.
    """
    from cutoutml.db.models import InferenceJob
    from cutoutml.db.session import session_scope

    asset = upload_image(client, auth)
    owner_id = uuid.UUID(client.get("/v1/auth/me", headers=auth).json()["id"])
    job_id = uuid.uuid4()
    with session_scope() as session:
        session.add(
            InferenceJob(
                id=job_id,
                owner_id=owner_id,
                asset_id=uuid.UUID(asset["id"]),
                status="queued",
                kind="image",
                model_name="trivial-center",
                precision="fp32",
                queue="cpu",
                idempotency_key=uuid.uuid4().hex,
                params={"device": "cpu", "image": {"outputs": ["mask_png"]}},
            )
        )
    return str(job_id)


def test_results_are_only_served_for_succeeded_jobs(client: Any, auth: dict[str, str]):
    job_id = insert_queued_job(client, auth)

    response = client.get(f"/v1/jobs/{job_id}/result", headers=auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_complete"


def test_downloading_an_output_kind_that_was_not_requested_is_a_404(
    client: Any, auth: dict[str, str]
):
    asset = upload_image(client, auth)
    job_id = client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "trivial-center", "image": {"outputs": ["mask_png"]}},
    ).json()["id"]

    assert client.get(f"/v1/jobs/{job_id}/outputs/mask_png", headers=auth).status_code == 200
    response = client.get(f"/v1/jobs/{job_id}/outputs/transparent_webp", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "output_not_found"


def test_the_asset_result_shortcut_returns_the_newest_succeeded_job(
    client: Any, auth: dict[str, str]
):
    asset = upload_image(client, auth)
    assert client.get(f"/v1/assets/{asset['id']}/result", headers=auth).status_code == 404

    client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "trivial-center", "image": {"outputs": ["mask_png"]}},
    )
    response = client.get(f"/v1/assets/{asset['id']}/result", headers=auth)
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"


def test_cancelling_a_terminal_job_is_a_conflict(client: Any, auth: dict[str, str]):
    job_id = insert_queued_job(client, auth)

    first = client.post(f"/v1/jobs/{job_id}/cancel", headers=auth)
    assert first.status_code == 200
    assert first.json()["status"] == "cancelled"

    second = client.post(f"/v1/jobs/{job_id}/cancel", headers=auth)
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "job_already_terminal"


def test_one_tenant_cannot_read_or_cancel_another_tenants_job(client: Any, auth: dict[str, str]):
    asset = upload_image(client, auth)
    job_id = client.post(
        f"/v1/assets/{asset['id']}/process",
        headers=auth,
        json={"model": "trivial-center", "image": {"outputs": ["mask_png"]}},
    ).json()["id"]
    intruder = {"Authorization": f"Bearer {register(client)}"}

    assert client.get(f"/v1/jobs/{job_id}", headers=intruder).status_code == 404
    assert client.get(f"/v1/jobs/{job_id}/result", headers=intruder).status_code == 404
    assert client.get(f"/v1/jobs/{job_id}/outputs/mask_png", headers=intruder).status_code == 404
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=intruder).status_code == 404
    assert client.get("/v1/jobs", headers=intruder).json() == []
