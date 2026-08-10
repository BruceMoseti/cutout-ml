#!/usr/bin/env python
"""End-to-end smoke test against a running API and worker.

Walks the path a real caller takes -- register, log in, upload an image, queue a job, poll
it, read the result back and download the cutout -- and prints the timings. It talks HTTP
only. If this passes, the API, Redis, the Celery worker, Postgres and the object store are
genuinely wired to each other, which is the one thing no unit test can establish: every
test in `tests/` either runs the pipeline in-process or mocks the queue.

It asks for the `classical` model on purpose. Learned checkpoints are trained in-repo and
gitignored, so a container image has none of them, and a smoke test that needs weights would
be testing whether someone remembered to mount a volume. GrabCut needs nothing but OpenCV,
which means a red frame on a white background is enough to prove the whole path moved bytes.

Usage:
    python scripts/smoke.py [--api http://127.0.0.1:8000] [--model classical]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import uuid

import httpx

TIMEOUT = httpx.Timeout(120.0, connect=10.0)
#: Terminal job states, so a failure stops the poll instead of waiting out the deadline.
DONE = {"succeeded", "failed", "cancelled"}


def _fail(step: str, response: httpx.Response) -> None:
    print(f"\n  FAILED at {step}: HTTP {response.status_code}")
    print(f"  {response.text[:1200]}")
    sys.exit(1)


def _step(label: str, started: float) -> None:
    print(f"  {label:<44} {(time.monotonic() - started) * 1000:7.0f} ms")


def _subject_png() -> bytes:
    """A red square on white: unambiguous foreground, so any segmenter finds something.

    Built here rather than committed because a fixture file is one more thing that can be
    absent in the environment this is meant to be diagnosing.
    """
    from PIL import Image

    image = Image.new("RGB", (256, 256), (255, 255, 255))
    for x in range(64, 192):
        for y in range(64, 192):
            image.putpixel((x, y), (220, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="classical")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for the job")
    args = parser.parse_args()

    api = args.api.rstrip("/")
    email = f"smoke-{uuid.uuid4().hex[:12]}@example.com"
    password = "correct horse battery staple"

    print(f"\nSmoke test against {api}\n")
    with httpx.Client(base_url=api, timeout=TIMEOUT) as client:
        t0 = time.monotonic()
        response = client.get("/health/ready")
        if response.status_code != 200:
            _fail("readiness", response)
        checks = [check["name"] for check in response.json().get("checks", [])]
        _step(f"ready ({', '.join(sorted(checks))})", t0)

        t0 = time.monotonic()
        response = client.post("/v1/auth/register", json={"email": email, "password": password})
        if response.status_code not in (200, 201):
            _fail("register", response)
        _step("registered", t0)

        t0 = time.monotonic()
        response = client.post("/v1/auth/login", json={"email": email, "password": password})
        if response.status_code != 200:
            _fail("login", response)
        client.headers["authorization"] = f"Bearer {response.json()['access_token']}"
        _step("logged in", t0)

        t0 = time.monotonic()
        response = client.post(
            "/v1/assets",
            files={"file": ("smoke.png", _subject_png(), "image/png")},
        )
        if response.status_code not in (200, 201):
            _fail("upload", response)
        asset_id = response.json()["id"]
        _step(f"uploaded asset {asset_id[:8]}", t0)

        t0 = time.monotonic()
        response = client.post(
            f"/v1/assets/{asset_id}/process",
            json={"model": args.model, "outputs": ["transparent_png", "mask_png"]},
        )
        if response.status_code not in (200, 202):
            _fail("process", response)
        job_id = response.json()["id"]
        _step(f"queued job {job_id[:8]} on {args.model}", t0)

        t0 = time.monotonic()
        deadline = t0 + args.timeout
        status = "pending"
        while time.monotonic() < deadline:
            response = client.get(f"/v1/jobs/{job_id}")
            if response.status_code != 200:
                _fail("poll", response)
            status = response.json()["status"]
            if status in DONE:
                break
            time.sleep(1.0)
        if status != "succeeded":
            print(f"\n  FAILED: job finished as {status!r} after {time.monotonic() - t0:.0f}s")
            detail = client.get(f"/v1/jobs/{job_id}")
            print(f"  {detail.text[:1200]}")
            return 1
        _step("worker finished the job", t0)

        t0 = time.monotonic()
        response = client.get(f"/v1/assets/{asset_id}/result")
        if response.status_code != 200:
            _fail("result", response)
        result = response.json()
        outputs = result.get("outputs") or []
        if not outputs:
            print(f"\n  FAILED: job succeeded but produced no outputs\n  {response.text[:600]}")
            return 1
        _step(f"result lists {len(outputs)} output(s)", t0)

        # Downloading is the part that proves the object store round-tripped rather than the
        # database merely holding a row that claims it did.
        t0 = time.monotonic()
        for output in outputs:
            url = output["url"]
            downloaded = client.get(url if url.startswith("http") else f"{api}{url}")
            if downloaded.status_code != 200 or not downloaded.content:
                _fail(f"download {output.get('kind')}", downloaded)
            if not downloaded.content.startswith(b"\x89PNG"):
                print(f"\n  FAILED: {output.get('kind')} is not a PNG")
                return 1
        _step("downloaded every output", t0)

    print("\n  OK — API, queue, worker, database and storage are wired together.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
