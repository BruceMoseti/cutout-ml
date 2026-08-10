"""Upload validation.

Everything a client tells us about an upload is untrusted: the filename, the extension
and the ``Content-Type`` header are all attacker-controlled. Validation therefore works
in this order:

1. **Size**, checked against the real byte count, not ``Content-Length``.
2. **Magic bytes**, sniffed from the payload itself. This is the authoritative check.
3. **Declared type**, compared against the sniffed type and rejected on *conflict* only
   (browsers send ``application/octet-stream`` constantly, so a missing or vague
   declaration is tolerated while a contradictory one is not).
4. **Decodability**, for images: PIL must actually parse it, and the pixel count must be
   under the decompression-bomb limit. A 20 KB PNG can declare 50000x50000 pixels and
   allocate 7 GB on decode.

The extension used for the storage key is derived from the **sniffed** type, never from
the filename, so ``evil.php.png`` cannot become ``evil.php`` on disk.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Literal

from cutoutml.core.logging import get_logger

log = get_logger(__name__)

AssetKindLiteral = Literal["image", "video"]


@dataclasses.dataclass(frozen=True, slots=True)
class DetectedType:
    """Result of content sniffing."""

    mime: str
    extension: str
    kind: AssetKindLiteral


class UploadValidationError(ValueError):
    """Upload rejected. ``code`` becomes the API error code (HTTP 400/413/415)."""

    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


# Magic-byte signatures. Ordered longest-first where prefixes overlap.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
    (b"BM", "image/bmp", "bmp"),
    (b"II*\x00", "image/tiff", "tiff"),
    (b"MM\x00*", "image/tiff", "tiff"),
)

_ALLOWED_IMAGE_MIMES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff", "image/webp"}
)
_ALLOWED_VIDEO_MIMES = frozenset(
    {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/x-msvideo"}
)

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._ \-()]+")


def sniff(data: bytes) -> DetectedType | None:
    """Identify a payload from its magic bytes.

    RIFF and ISO-BMFF need two-stage checks: ``RIFF....WEBP`` versus ``RIFF....AVI``, and
    an ``ftyp`` box whose brand distinguishes MP4 from QuickTime. Matching only the outer
    container would let an AVI through as a WebP.
    """
    if len(data) < 12:
        return None

    for signature, mime, ext in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return DetectedType(mime, ext, "image")

    if data[:4] == b"RIFF":
        if data[8:12] == b"WEBP":
            return DetectedType("image/webp", "webp", "image")
        if data[8:11] == b"AVI":
            return DetectedType("video/x-msvideo", "avi", "video")

    # ISO base media file format: size(4) + 'ftyp' + brand(4)
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"qt  "}:
            return DetectedType("video/quicktime", "mov", "video")
        return DetectedType("video/mp4", "mp4", "video")

    # Matroska/WebM share the EBML header; the DocType distinguishes them and lives a
    # little further in, so the first KB is scanned.
    if data[:4] == b"\x1a\x45\xdf\xa3":
        head = data[:1024]
        if b"webm" in head:
            return DetectedType("video/webm", "webm", "video")
        return DetectedType("video/x-matroska", "mkv", "video")

    return None


def safe_filename(filename: str | None, *, max_length: int = 200) -> str | None:
    """Sanitise a filename for *display only*.

    Never used to build a path - storage keys are random (see
    :func:`cutoutml.storage.base.build_storage_key`). This exists so that a filename
    rendered back into an HTML page or a log line cannot carry control characters or
    path separators.
    """
    if not filename:
        return None
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _FILENAME_SAFE.sub("_", base).strip(" .")
    return cleaned[:max_length] or None


def validate_upload(
    data: bytes,
    *,
    declared_content_type: str | None = None,
    filename: str | None = None,
    max_image_bytes: int,
    max_video_bytes: int,
    max_image_pixels: int,
    expected_kind: AssetKindLiteral | None = None,
) -> tuple[DetectedType, dict[str, int | None]]:
    """Validate an upload, returning its detected type and probed properties.

    Raises :class:`UploadValidationError` with a specific ``code`` for each failure mode,
    so the API can return an actionable message instead of a generic 400.
    """
    if not data:
        raise UploadValidationError("uploaded file is empty", code="empty_upload")

    detected = sniff(data)
    if detected is None:
        raise UploadValidationError(
            "could not determine file type from its contents; supported: PNG, JPEG, "
            "GIF, BMP, TIFF, WebP images and MP4, MOV, WebM, MKV, AVI video",
            code="unsupported_type",
            status_code=415,
        )

    allowed = _ALLOWED_IMAGE_MIMES if detected.kind == "image" else _ALLOWED_VIDEO_MIMES
    if detected.mime not in allowed:
        raise UploadValidationError(
            f"file type {detected.mime} is not allowed", code="unsupported_type", status_code=415
        )

    if expected_kind is not None and detected.kind != expected_kind:
        raise UploadValidationError(
            f"expected {expected_kind} but the upload is {detected.kind} ({detected.mime})",
            code="kind_mismatch",
            status_code=415,
        )

    limit = max_image_bytes if detected.kind == "image" else max_video_bytes
    if len(data) > limit:
        raise UploadValidationError(
            f"file is {len(data)} bytes which exceeds the {detected.kind} limit of {limit}",
            code="payload_too_large",
            status_code=413,
        )

    if declared_content_type:
        declared = declared_content_type.split(";", 1)[0].strip().lower()
        vague = {"application/octet-stream", "binary/octet-stream", ""}
        if declared not in vague and declared != detected.mime:
            # Log it: a systematic mismatch is either a broken client or a probe.
            log.warning("upload_content_type_mismatch", declared=declared, detected=detected.mime)
            raise UploadValidationError(
                f"declared Content-Type {declared} does not match the actual file "
                f"contents ({detected.mime})",
                code="content_type_mismatch",
                status_code=415,
            )

    _warn_on_extension_mismatch(filename, detected)

    properties: dict[str, int | None] = {"width": None, "height": None}
    if detected.kind == "image":
        properties = _probe_image(data, max_image_pixels)

    return detected, properties


def _warn_on_extension_mismatch(filename: str | None, detected: DetectedType) -> None:
    """Log, but never reject, a filename whose extension contradicts the content.

    Deliberately not an error. Legitimate mismatches are common - ``.jpeg`` versus
    ``.jpg``, a ``.png`` that a phone actually encoded as HEIC-turned-JPEG, a download
    saved with no extension - and the sniffed type is already authoritative, so rejecting
    would only break real users. It is worth logging because a *systematic* mismatch,
    especially towards an executable extension, is the signature of someone probing for a
    path where the client-supplied name is trusted.
    """
    if not filename or "." not in filename:
        return
    claimed = filename.rsplit(".", 1)[-1].lower()
    aliases = {"jpeg": "jpg", "tif": "tiff", "htm": "html"}
    claimed = aliases.get(claimed, claimed)
    if claimed != detected.extension:
        log.info(
            "upload_extension_mismatch",
            claimed_extension=claimed,
            detected_extension=detected.extension,
            detected_mime=detected.mime,
        )


def _probe_image(data: bytes, max_pixels: int) -> dict[str, int | None]:
    """Read image dimensions without decoding pixels, and enforce the pixel cap.

    ``Image.open`` only reads the header, so this is cheap and - crucially - happens
    *before* any full decode. Checking dimensions after decoding would be pointless: the
    allocation has already happened.
    """
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            img.verify()
    except UnidentifiedImageError as exc:
        raise UploadValidationError(
            "file has an image signature but could not be decoded; it is probably "
            "truncated or corrupt",
            code="corrupt_image",
        ) from exc
    except Exception as exc:
        raise UploadValidationError(
            f"image could not be read: {type(exc).__name__}", code="corrupt_image"
        ) from exc

    if width <= 0 or height <= 0:
        raise UploadValidationError("image has zero area", code="corrupt_image")
    if width * height > max_pixels:
        raise UploadValidationError(
            f"image is {width}x{height} = {width * height} pixels, over the limit of "
            f"{max_pixels}. Large images are rejected before decoding to prevent "
            "decompression-bomb memory exhaustion.",
            code="image_too_large",
            status_code=413,
        )
    return {"width": width, "height": height}


def content_hash(data: bytes) -> str:
    """SHA-256 of the payload, used for deduplication and idempotency keys."""
    return hashlib.sha256(data).hexdigest()
