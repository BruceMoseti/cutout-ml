"""ffmpeg process wrappers for streaming video I/O.

Why raw subprocess pipes instead of PyAV: this keeps the dependency surface to a
binary that is already present on every container that can process video, gives
exact control over the encoder flags that matter for alpha (``-pix_fmt yuva420p``,
``-auto-alt-ref 0``), and streams frame-by-frame so peak memory is
``O(batch_size)`` rather than ``O(frame_count)``. A 60-second 4K clip is ~1.5 GB of
raw RGB; holding it would be a bug, not an inefficiency.

Both wrappers are context managers and kill the child process on exit, including on
exception. An orphaned ffmpeg holding a pipe is the classic way a video worker leaks
until OOM.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from cutoutml.core.logging import get_logger

log = get_logger(__name__)


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe failed. Carries the tail of stderr, which is where the reason is."""

    def __init__(self, message: str, *, stderr: str = "", returncode: int | None = None) -> None:
        self.stderr = stderr
        self.returncode = returncode
        detail = f" (exit {returncode})" if returncode is not None else ""
        tail = f"\nffmpeg stderr tail:\n{stderr[-2000:]}" if stderr else ""
        super().__init__(f"{message}{detail}{tail}")


class UnsupportedCodecError(FFmpegError):
    """The input cannot be decoded. Non-retryable: retrying will fail identically."""


@dataclasses.dataclass(frozen=True, slots=True)
class VideoInfo:
    """Probed properties of a video file."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    pix_fmt: str
    has_audio: bool
    rotation: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _binary(name: str, configured: str) -> str:
    resolved = shutil.which(configured) or shutil.which(name)
    if resolved is None:
        raise FFmpegError(
            f"{name} not found on PATH. Install ffmpeg (the container images do) or "
            f"set CUTOUTML_{name.upper()}_BINARY."
        )
    return resolved


def probe(path: Path | str, *, ffprobe: str = "ffprobe") -> VideoInfo:
    """Probe a video with ffprobe.

    ``nb_frames`` is missing or wrong in many containers (notably WebM and
    fragmented MP4), so it is only trusted when present and otherwise derived from
    duration x fps. Getting this wrong makes progress reporting nonsense.
    """
    binary = _binary("ffprobe", ffprobe)
    cmd = [
        binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise UnsupportedCodecError(
            f"ffprobe could not read {path}", stderr=proc.stderr, returncode=proc.returncode
        )

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise UnsupportedCodecError(f"no video stream found in {path}")

    fps = _parse_fraction(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    declared = video.get("nb_frames")
    if declared and str(declared).isdigit() and int(declared) > 0:
        frame_count = int(declared)
    elif fps > 0 and duration > 0:
        frame_count = round(duration * fps)
    else:
        frame_count = 0

    return VideoInfo(
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps=fps if fps > 0 else 25.0,
        frame_count=frame_count,
        duration_seconds=duration,
        codec=str(video.get("codec_name", "unknown")),
        pix_fmt=str(video.get("pix_fmt", "unknown")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        rotation=_parse_rotation(video),
    )


def _parse_fraction(value: str) -> float:
    try:
        num, _, den = value.partition("/")
        d = float(den) if den else 1.0
        return float(num) / d if d else 0.0
    except (TypeError, ValueError):
        return 0.0


def _parse_rotation(stream: dict[str, Any]) -> int:
    """Rotation from side data or the legacy tag, normalised to 0/90/180/270.

    Phone video is routinely stored landscape with a rotation flag. ffmpeg applies it
    during decode by default, so this is recorded for metadata rather than acted on -
    but silently ignoring its existence is how portrait video ends up sideways.
    """
    for entry in stream.get("side_data_list", []) or []:
        if "rotation" in entry:
            with contextlib.suppress(TypeError, ValueError):
                return int(entry["rotation"]) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        with contextlib.suppress(TypeError, ValueError):
            return int(tag) % 360
    return 0


class FrameReader:
    """Stream decoded ``rgb24`` frames out of ffmpeg.

    ffmpeg writes tightly packed rows with no padding for ``rgb24``, so each frame is
    exactly ``width * height * 3`` bytes and can be read with a fixed-size
    ``readinto``. Short reads at the end of the stream are normal and terminate
    iteration.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        width: int,
        height: int,
        ffmpeg: str = "ffmpeg",
        scale: tuple[int, int] | None = None,
        start_seconds: float | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.out_width, self.out_height = scale or (width, height)
        self.frame_bytes = self.out_width * self.out_height * 3
        self.max_frames = max_frames
        self._binary = _binary("ffmpeg", ffmpeg)
        self._start = start_seconds
        self._scale = scale
        self.process: subprocess.Popen[bytes] | None = None

    def _command(self) -> list[str]:
        cmd = [self._binary, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self._start:
            # Input seeking (before -i) is keyframe-accurate and fast; output
            # seeking would decode and discard everything before the start point.
            cmd += ["-ss", f"{self._start:.3f}"]
        cmd += ["-i", str(self.path)]
        if self.max_frames:
            cmd += ["-frames:v", str(self.max_frames)]
        if self._scale:
            cmd += ["-vf", f"scale={self.out_width}:{self.out_height}:flags=bicubic"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-sn", "-"]
        return cmd

    def __enter__(self) -> FrameReader:
        cmd = self._command()
        log.debug("ffmpeg_decode_start", cmd=" ".join(cmd))
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_bytes
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def frames(self) -> Iterator[bytes]:
        """Yield raw frame buffers. Caller wraps them in NumPy views."""
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("FrameReader must be used as a context manager")
        stdout = self.process.stdout
        while True:
            buf = stdout.read(self.frame_bytes)
            if not buf:
                break
            if len(buf) < self.frame_bytes:
                # Truncated trailing frame: the stream ended mid-frame.
                break
            yield buf
        self._check_exit()

    def _check_exit(self) -> None:
        assert self.process is not None
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read() or b""
        code = self.process.wait()
        if code not in (0, None):
            raise UnsupportedCodecError(
                f"decoding {self.path} failed",
                stderr=stderr.decode("utf-8", "replace"),
                returncode=code,
            )

    def close(self) -> None:
        if self.process is None:
            return
        proc, self.process = self.process, None
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
                proc.wait(timeout=5)


class FrameWriter:
    """Stream frames into an ffmpeg encoder.

    ``pix_fmt`` is the *input* pixel format (``rgb24`` or ``rgba``); the encoder
    arguments decide the output. See :func:`encoder_args` for the per-container
    choices and why they are what they are.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        width: int,
        height: int,
        fps: float,
        pix_fmt: str = "rgb24",
        encoder_arguments: Sequence[str] | None = None,
        ffmpeg: str = "ffmpeg",
        audio_source: Path | str | None = None,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps if fps > 0 else 25.0
        self.pix_fmt = pix_fmt
        self.encoder_arguments = list(encoder_arguments or [])
        self.audio_source = Path(audio_source) if audio_source else None
        self._binary = _binary("ffmpeg", ffmpeg)
        self.process: subprocess.Popen[bytes] | None = None
        self.frames_written = 0

    def _command(self) -> list[str]:
        cmd = [
            self._binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.pix_fmt,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
        ]
        if self.audio_source is not None:
            # Copy the original audio rather than re-encoding it: it is untouched by
            # segmentation, and re-encoding would only lose quality.
            cmd += ["-i", str(self.audio_source), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]
        cmd += self.encoder_arguments
        cmd += [str(self.path)]
        return cmd

    def __enter__(self) -> FrameWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._command()
        log.debug("ffmpeg_encode_start", cmd=" ".join(cmd))
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.close(failed=exc_type is not None)

    def write(self, frame: bytes) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("FrameWriter must be used as a context manager")
        try:
            self.process.stdin.write(frame)
        except BrokenPipeError:
            # ffmpeg died; its stderr explains why and is far more useful than the
            # BrokenPipeError itself.
            stderr = b""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read() or b""
            raise FFmpegError(
                "ffmpeg encoder exited while frames were still being written",
                stderr=stderr.decode("utf-8", "replace"),
                returncode=self.process.poll(),
            ) from None
        self.frames_written += 1

    def close(self, *, failed: bool = False) -> None:
        if self.process is None:
            return
        proc, self.process = self.process, None
        if proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
        stderr = b""
        if proc.stderr is not None:
            with contextlib.suppress(Exception):
                stderr = proc.stderr.read() or b""
        code = proc.wait()
        if code != 0 and not failed:
            raise FFmpegError(
                f"encoding {self.path} failed",
                stderr=stderr.decode("utf-8", "replace"),
                returncode=code,
            )


def encoder_args(
    container: str,
    *,
    crf: int = 23,
    preset: str = "medium",
    alpha: bool = False,
    bitrate: str | None = None,
) -> list[str]:
    """Encoder flags per container.

    The important asymmetry, and the reason ``docs/decisions/ADR-003-video-output.md``
    exists:

    * **MP4/H.264 has no usable alpha channel.** The spec allows auxiliary alpha
      pictures; no browser or NLE decodes them. So MP4 output is always a
      *composite* over a chosen background, never transparent.
    * **WebM/VP9 can** carry alpha (``yuva420p``) and Chrome/Firefox play it, which
      makes it the obvious choice for transparent video on the web - but whether a
      given ffmpeg/libvpx build actually *writes* the alpha plane varies, and when it
      does not it produces a fully opaque file with no warning. ``-auto-alt-ref 0``
      is a hard requirement (alternate reference frames are incompatible with the
      alpha plane), and even with it the result must be verified rather than assumed.
      :func:`alpha_roundtrip_works` does that verification; ``docs/benchmarks.md``
      records what this machine's build actually does.
    * **ProRes 4444 / QuickTime RLE** are the lossless-ish alpha formats editors
      actually want, and their alpha support is far more reliable across builds.
      Both are enormous, which is why they are opt-in.

    ``-pix_fmt yuv420p`` on H.264 is not optional in practice: without it ffmpeg may
    pick ``yuv444p`` from an RGB input, which Safari and most hardware decoders
    refuse to play.
    """
    fmt = container.lower().lstrip(".")

    if fmt in {"mp4", "h264", "m4v"}:
        if alpha:
            raise ValueError(
                "MP4/H.264 cannot carry a usable alpha channel; request 'webm' for "
                "transparency or supply a background for compositing"
            )
        args = [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        args += ["-b:v", bitrate] if bitrate else ["-crf", str(crf)]
        return args

    if fmt in {"webm", "vp9"}:
        args = [
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p" if alpha else "yuv420p",
            "-row-mt",
            "1",
            "-deadline",
            "good",
            "-cpu-used",
            "2",
        ]
        if alpha:
            args += ["-auto-alt-ref", "0"]
        args += ["-b:v", bitrate] if bitrate else ["-crf", str(crf), "-b:v", "0"]
        return args

    if fmt in {"mov", "prores"}:
        # profile 4 = ProRes 4444, the only ProRes profile with an alpha channel.
        return ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]

    if fmt == "qtrle":
        return ["-c:v", "qtrle", "-pix_fmt", "argb"]

    raise ValueError(f"unsupported container {container!r}; expected mp4, webm, mov or qtrle")


ALPHA_CONTAINERS: tuple[str, ...] = ("webm", "mov", "qtrle")
"""Containers whose codecs are *specified* to carry alpha. Not a capability claim."""

_CONTAINER_EXTENSIONS: dict[str, str] = {
    "mp4": "mp4",
    "h264": "mp4",
    "m4v": "m4v",
    "webm": "webm",
    "vp9": "webm",
    "mov": "mov",
    "prores": "mov",
    # qtrle is a *codec*, and its container is QuickTime. ffmpeg picks the muxer
    # from the extension, so writing "out.qtrle" fails to select one at all.
    "qtrle": "mov",
}


def container_extension(container: str) -> str:
    """File extension for a container/codec name, since the two are not the same."""
    fmt = container.lower().lstrip(".")
    try:
        return _CONTAINER_EXTENSIONS[fmt]
    except KeyError:
        raise ValueError(
            f"unsupported container {container!r}; expected one of "
            f"{', '.join(sorted(_CONTAINER_EXTENSIONS))}"
        ) from None


def container_supports_alpha(container: str) -> bool:
    """Whether a container/codec pair is *specified* to carry transparency.

    This is a statement about the format, not about the local ffmpeg build. Use
    :func:`alpha_roundtrip_works` before promising a caller a transparent result.
    """
    return container.lower().lstrip(".") in {"webm", "vp9", "mov", "prores", "qtrle"}


ALPHA_DECODER_ARGS: dict[str, tuple[str, ...]] = {
    "webm": ("-c:v", "libvpx-vp9"),
    "vp9": ("-c:v", "libvpx-vp9"),
    "vp8": ("-c:v", "libvpx"),
}
"""Containers whose alpha plane is invisible to ffmpeg's *default* decoder.

For VP9-in-WebM the alpha plane lives in Matroska ``BlockAdditional`` side data rather
than in the video frame, and ffmpeg's native ``vp9`` decoder ignores it: decoding
without ``-c:v libvpx-vp9`` returns alpha = 255 for every pixel of a genuinely
transparent file. ProRes and QuickTime RLE keep alpha in the pixel format and need no
override, which is why they are absent here.
"""


def alpha_decoder_args(container: str) -> tuple[str, ...]:
    """Decoder flags required to *see* ``container``'s alpha plane, if any.

    These go before ``-i`` on the ffmpeg command line, since they select the decoder
    for the input rather than describing the output.
    """
    return ALPHA_DECODER_ARGS.get(container.lower().lstrip("."), ())


def _decode_alpha_range(
    path: Path, size: tuple[int, int], ffmpeg: str, *, decoder: Sequence[str] = ()
) -> tuple[int, int] | None:
    """Decode ``path`` to RGBA and return ``(min_alpha, max_alpha)``, or ``None``.

    ``decoder`` must be passed for WebM: see :data:`ALPHA_DECODER_ARGS`. Without it the
    caller measures the *decoder's* alpha support rather than the file's, and concludes
    that a perfectly good transparent WebM is opaque.
    """
    width, height = size
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            *decoder,
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    stride = width * height * 4
    if proc.returncode != 0 or len(proc.stdout) < stride:
        return None
    alpha = proc.stdout[3:stride:4]
    return (min(alpha), max(alpha))


def alpha_roundtrip_works(container: str, ffmpeg: str = "ffmpeg") -> bool:
    """Encode a two-frame half-transparent clip and check the alpha survives.

    This exists because a codec advertising ``yuva420p`` in ``ffmpeg -h encoder=...``
    does **not** mean the muxer will write the alpha plane, and a build that drops it
    still exits 0. Promising a caller transparency on the strength of the codec table
    risks handing them a "transparent" video with no transparency in it, so the
    capability is measured instead: encode a clip whose left half is fully transparent
    and whose right half is opaque, decode it back, and require both extremes to
    survive. A uniform plane would not distinguish real alpha from a codec that writes
    a constant.

    The decode step forces an alpha-aware decoder via :func:`alpha_decoder_args`.
    Omitting that is the trap this function is most likely to fall into: ffmpeg's
    native ``vp9`` decoder silently ignores WebM's alpha side data, so the probe would
    report "alpha dropped" for a file that is in fact fully transparent.

    The probe is a real encode/decode of a 32x32x2-frame clip - a few milliseconds -
    and is cached per process, so readiness checks and job validation can call it
    freely. Arguments are normalised before the cache lookup so that positional and
    defaulted calls share one entry.
    """
    return _alpha_roundtrip_works(container.lower().lstrip("."), ffmpeg)


@functools.lru_cache(maxsize=16)
def _alpha_roundtrip_works(container: str, ffmpeg: str) -> bool:
    try:
        binary = _binary("ffmpeg", ffmpeg)
    except FFmpegError:
        return False

    size = (32, 32)
    try:
        args = encoder_args(container, crf=30, alpha=True)
        extension = container_extension(container)
    except ValueError:
        return False

    with tempfile.TemporaryDirectory(prefix="cutoutml-alpha-probe-") as tmp:
        out = Path(tmp) / f"probe.{extension}"
        # Left half fully transparent, right half opaque: a uniform alpha could be
        # mistaken for a codec that writes a constant plane.
        frame = bytearray()
        for _ in range(size[1]):
            frame += bytes([200, 40, 40, 0]) * (size[0] // 2)
            frame += bytes([200, 40, 40, 255]) * (size[0] // 2)
        cmd = [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{size[0]}x{size[1]}",
            "-r",
            "10",
            "-i",
            "-",
            *args,
            str(out),
        ]
        proc = subprocess.run(cmd, input=bytes(frame) * 2, capture_output=True, check=False)
        if proc.returncode != 0 or not out.is_file():
            log.warning(
                "alpha_probe_encode_failed",
                container=container,
                stderr=proc.stderr.decode("utf-8", "replace")[-400:],
            )
            return False
        observed = _decode_alpha_range(out, size, binary, decoder=alpha_decoder_args(container))

    if observed is None:
        return False
    low, high = observed
    works = low < 32 and high > 224
    log.info("alpha_probe", container=container, alpha_min=low, alpha_max=high, works=works)
    return works


def working_alpha_containers(ffmpeg: str = "ffmpeg") -> list[str]:
    """Which alpha-capable containers this ffmpeg build actually delivers.

    Reported by the readiness endpoint so an operator learns that transparent video
    is unavailable at deploy time rather than from a user's complaint.
    """
    return [c for c in ALPHA_CONTAINERS if alpha_roundtrip_works(c, ffmpeg)]


def has_alpha(path: Path | str, *, ffprobe: str = "ffprobe") -> bool:
    """Whether a file genuinely carries per-pixel transparency.

    This is trickier than it should be, and getting it wrong is how "transparent
    video" ships opaque:

    * For **VP9 in WebM** the alpha plane is *not* part of the video frame. It is
      stored per-block in Matroska ``BlockAdditional`` side data, and the track
      carries an ``AlphaMode`` flag. Consequently ``ffprobe`` reports
      ``pix_fmt=yuv420p`` even for a file that is fully transparent - the alpha is
      real but invisible to a pixel-format check. The reliable signal is the
      ``alpha_mode`` stream tag.
    * ffmpeg's **native ``vp9`` decoder ignores that side data entirely**, so
      decoding without ``-c:v libvpx-vp9`` yields alpha = 255 everywhere. Anyone
      verifying their own output with a plain ``ffmpeg -i ... -pix_fmt rgba`` will
      conclude, incorrectly, that alpha was dropped.
    * ProRes 4444 and QuickTime RLE *do* put alpha in the pixel format, so for those
      the ``pix_fmt`` check is the correct one.
    """
    binary = _binary("ffprobe", ffprobe)
    proc = subprocess.run(
        [
            binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-print_format",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    streams = json.loads(proc.stdout or "{}").get("streams", [])
    if not streams:
        return False
    stream = streams[0]
    if str((stream.get("tags") or {}).get("alpha_mode", "0")) == "1":
        return True
    return str(stream.get("pix_fmt", "")) in ALPHA_PIX_FMTS


ALPHA_PIX_FMTS = frozenset(
    {
        "yuva420p",
        "yuva420p10le",
        "yuva422p",
        "yuva422p10le",
        "yuva444p",
        "yuva444p10le",
        "yuva444p12le",
        "yuva444p16le",
        "rgba",
        "argb",
        "bgra",
        "abgr",
        "rgba64le",
        "rgba64be",
        "ya8",
        "ya16le",
    }
)
"""Pixel formats that carry alpha in-frame (ProRes 4444, QTRLE, PNG sequences)."""


def available_encoders(ffmpeg: str = "ffmpeg") -> set[str]:
    """Encoder names this ffmpeg build supports.

    Used at readiness-check time: a container image without ``libvpx-vp9`` cannot
    produce transparent video, and finding that out when the first job fails is worse
    than finding out at startup.
    """
    try:
        binary = _binary("ffmpeg", ffmpeg)
    except FFmpegError:
        return set()
    proc = subprocess.run(
        [binary, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    )
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0][:1] in {"V", "A", "S"} and len(parts[0]) == 6:
            names.add(parts[1])
    return names
