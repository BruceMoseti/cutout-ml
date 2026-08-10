"""Fetch pretrained weights for models that ship without them.

This script is written to be *useful when the network allows it and honest when it does
not*. Nothing here is a stub: it performs a real streamed download with checksum
verification, atomic replacement and a clear diagnosis when the host is unreachable.

Reachability is a genuine constraint, not a hypothetical. In the environment this
repository was built in, ``huggingface.co`` is blocked at the network layer, which is
where the official U^2-Net and BiRefNet checkpoints are mirrored. So:

* ``u2net`` / ``u2net-lite`` have real mirror URLs configured; the download works
  wherever those hosts are reachable, and the SHA-256 of the canonical files is checked
  when known.
* ``birefnet`` has **no** URL, because the official BiRefNet weights target a Swin
  backbone whose tensor shapes do not match this repository's compact
  reimplementation. Downloading them would produce a checkpoint that cannot load, so
  the script says so instead of pretending.
* ``cutoutnet`` is trained in-repo; the script points at ``make train`` rather than a URL.

Licensing is printed before every download. U^2-Net is Apache-2.0. BiRefNet's code is
MIT, but **some third-party fine-tuned BiRefNet checkpoints are released under
non-commercial terms** - which the person downloading them, not this script, is
responsible for checking. See ``docs/models.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import socket
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from cutoutml.core.config import get_settings
from cutoutml.core.logging import configure_logging, get_logger

log = get_logger(__name__)

USER_AGENT = "cutoutml-download-weights/0.1"
CHUNK_BYTES = 1024 * 256


@dataclasses.dataclass(frozen=True, slots=True)
class WeightSource:
    """Where a checkpoint comes from and what it is licensed under."""

    model: str
    filename: str
    subdir: str
    license: str
    homepage: str
    urls: tuple[str, ...] = ()
    sha256: str | None = None
    note: str = ""

    def target(self) -> Path:
        return get_settings().model_weights_dir / self.subdir / self.filename


SOURCES: dict[str, WeightSource] = {
    "u2net": WeightSource(
        model="u2net",
        filename="u2net.pth",
        subdir="u2net",
        license="Apache-2.0",
        homepage="https://github.com/xuebinqin/U-2-Net",
        # The authors distribute via Google Drive; these are the commonly used
        # redistribution mirrors. Both are unreachable from some CI networks, which is
        # handled rather than assumed away.
        urls=(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
            "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2net.pth",
        ),
        note=(
            "The first mirror is an ONNX graph, not a .pth: use it with the OnnxAdapter "
            "rather than U2NetAdapter. The .pth mirror is on HuggingFace and is blocked "
            "in some environments."
        ),
    ),
    "u2net-lite": WeightSource(
        model="u2net-lite",
        filename="u2netp.pth",
        subdir="u2net",
        license="Apache-2.0",
        homepage="https://github.com/xuebinqin/U-2-Net",
        urls=("https://huggingface.co/tomjackson2023/rembg/resolve/main/u2netp.pth",),
    ),
    "birefnet": WeightSource(
        model="birefnet",
        filename="birefnet-compact.pt",
        subdir="birefnet",
        license=(
            "Official BiRefNet code: MIT. Official weights: see the model card. "
            "SOME third-party fine-tuned BiRefNet weights are NON-COMMERCIAL - verify "
            "before use."
        ),
        homepage="https://github.com/ZhengPeng7/BiRefNet",
        urls=(),
        note=(
            "No download is offered. The official checkpoints are built on a Swin "
            "backbone and are NOT shape-compatible with this repository's compact "
            "reimplementation, so loading them would fail. Train this architecture "
            "yourself, or use the official BiRefNet repository directly for its weights."
        ),
    ),
    "cutoutnet": WeightSource(
        model="cutoutnet",
        filename="cutoutnet-small.pt",
        subdir="cutoutnet",
        license="MIT (this repository)",
        homepage="https://github.com/your-org/cutout-ml",
        urls=(),
        note="Trained in-repo. Run `make train` (a few minutes on 8 CPU cores).",
    ),
}


class DownloadUnavailableError(RuntimeError):
    """No configured URL was reachable. Message names every host that was tried."""


def sha256_file(path: Path, *, chunk: int = CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def host_reachable(url: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    """TCP-level reachability check, so a blocked host is diagnosed as blocked.

    Done before the HTTP request because a DNS or connection failure produces a much
    clearer message than a mid-download ``URLError``, and because it distinguishes "the
    network blocks this host" from "the file moved".
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return (False, f"malformed URL: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return (True, "reachable")
    except OSError as exc:
        return (False, f"{host}:{port} unreachable ({type(exc).__name__}: {exc})")


def download(
    url: str, destination: Path, *, expected_sha256: str | None = None, timeout: float = 60.0
) -> Path:
    """Stream a URL to ``destination`` atomically, verifying the checksum.

    Written to a temp file in the same directory and ``os.replace``-d, so an interrupted
    download never leaves a truncated file that a later run would treat as valid.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    tmp = Path(tempfile.mkstemp(dir=destination.parent, prefix=".download-")[1])
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with tmp.open("wb") as fh:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
            if total and written != total:
                raise DownloadUnavailableError(
                    f"truncated download from {url}: got {written} of {total} bytes"
                )

        if expected_sha256:
            actual = sha256_file(tmp)
            if actual != expected_sha256:
                raise DownloadUnavailableError(
                    f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
                )

        tmp.replace(destination)
        log.info("weights_downloaded", url=url, path=str(destination), bytes=written)
        return destination
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def fetch(model: str, *, force: bool = False, timeout: float = 60.0) -> Path:
    """Fetch weights for ``model``, trying each mirror in order."""
    try:
        source = SOURCES[model]
    except KeyError:
        raise SystemExit(f"unknown model {model!r}; known: {', '.join(sorted(SOURCES))}") from None

    target = source.target()
    print(f"model:    {source.model}")
    print(f"target:   {target}")
    print(f"license:  {source.license}")
    print(f"homepage: {source.homepage}")
    if source.note:
        print(f"note:     {source.note}")

    if target.is_file() and not force:
        print(f"already present ({target.stat().st_size / 1e6:.1f} MB); use --force to re-download")
        return target

    if not source.urls:
        raise DownloadUnavailableError(
            f"no download URL is configured for {model!r}. {source.note}"
        )

    failures: list[str] = []
    for url in source.urls:
        ok, detail = host_reachable(url, timeout=min(timeout, 10.0))
        if not ok:
            failures.append(detail)
            log.warning("weights_host_unreachable", url=url, detail=detail)
            continue
        try:
            return download(url, target, expected_sha256=source.sha256, timeout=timeout)
        except (urllib.error.URLError, OSError, DownloadUnavailableError) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            log.warning("weights_download_failed", url=url, error=str(exc))

    raise DownloadUnavailableError(
        f"could not fetch weights for {model!r}. Tried {len(source.urls)} mirror(s):\n  "
        + "\n  ".join(failures)
        + "\n\nOptions: (1) download manually from "
        + source.homepage
        + f" and place the file at {target}; (2) use --random-init for latency-only "
        "benchmarking (accuracy will be meaningless); (3) use the `cutoutnet` model, "
        "whose weights are produced in-repo by `make train`."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download pretrained segmentation weights")
    p.add_argument("--model", default="u2net", help=f"one of: {', '.join(sorted(SOURCES))}")
    p.add_argument("--all", action="store_true", help="attempt every configured model")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--list", action="store_true", help="list sources and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(fmt="console")

    if args.list:
        for name, source in sorted(SOURCES.items()):
            state = "present" if source.target().is_file() else "missing"
            print(f"{name:14s} {state:8s} {source.license}")
            if source.note:
                print(f"{'':14s}          {source.note}")
        return 0

    targets = sorted(SOURCES) if args.all else [args.model]
    failed = 0
    for name in targets:
        print(f"\n=== {name}")
        try:
            fetch(name, force=args.force, timeout=args.timeout)
        except DownloadUnavailableError as exc:
            print(f"UNAVAILABLE: {exc}")
            failed += 1
    return 1 if failed and not args.all else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
