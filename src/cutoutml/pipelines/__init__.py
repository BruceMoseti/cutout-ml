"""End-to-end image and video pipelines."""

from cutoutml.pipelines.ffmpeg import (
    FFmpegError,
    FrameReader,
    FrameWriter,
    UnsupportedCodecError,
    VideoInfo,
    container_supports_alpha,
    encoder_args,
    probe,
)
from cutoutml.pipelines.image import (
    DEFAULT_OUTPUTS,
    ImagePipeline,
    ImageRequest,
    ImageResult,
    OutputKind,
)
from cutoutml.pipelines.video import (
    TemporalSmoother,
    VideoPipeline,
    VideoProgress,
    VideoRequest,
    VideoResult,
    estimate_flicker,
    make_test_video,
)

__all__ = [
    "DEFAULT_OUTPUTS",
    "FFmpegError",
    "FrameReader",
    "FrameWriter",
    "ImagePipeline",
    "ImageRequest",
    "ImageResult",
    "OutputKind",
    "TemporalSmoother",
    "UnsupportedCodecError",
    "VideoInfo",
    "VideoPipeline",
    "VideoProgress",
    "VideoRequest",
    "VideoResult",
    "container_supports_alpha",
    "encoder_args",
    "estimate_flicker",
    "make_test_video",
    "probe",
]
