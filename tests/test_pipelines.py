"""Image and video pipelines.

The video tests split into two groups on purpose:

* **Batching and smoothing logic**, tested against a synthetic frame iterator with
  no ffmpeg involved. This is where the memory bound lives (``_batched`` must
  never hold more than ``batch_size`` frames) and it is cheap to test exhaustively.
* **End-to-end through ffmpeg**, which encodes a real clip, segments it and probes
  the output. Slower, and skipped rather than mocked when ffmpeg is absent - a
  mocked encoder would not catch the failure these tests exist for, which is
  ffmpeg silently dropping the alpha plane.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import decode_rgba
from cutoutml.core.refine import RefineConfig
from cutoutml.pipelines.ffmpeg import (
    ALPHA_CONTAINERS,
    FFmpegError,
    alpha_decoder_args,
    alpha_roundtrip_works,
    container_extension,
    container_supports_alpha,
    encoder_args,
    has_alpha,
    probe,
    working_alpha_containers,
)
from cutoutml.pipelines.image import ImagePipeline, ImageRequest
from cutoutml.pipelines.video import (
    TemporalSmoother,
    VideoPipeline,
    VideoProgress,
    VideoRequest,
    _batched,
    archive_frames,
    estimate_flicker,
    make_test_video,
)

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


@pytest.fixture(scope="module")
def pipeline(trivial_model) -> ImagePipeline:
    return ImagePipeline(trivial_model)


@pytest.fixture(scope="module")
def video_pipeline(trivial_model) -> VideoPipeline:
    return VideoPipeline(trivial_model)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real 24-frame MP4, encoded once for the whole module."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    directory = tmp_path_factory.mktemp("clip")
    return make_test_video(directory / "source.mp4", frames=24, width=96, height=64, fps=12.0)


# =========================================================== image pipeline


def test_image_pipeline_loads_an_unloaded_model(trivial_model):
    """Callers should not have to remember to load; the pipeline owns the lifecycle."""
    from cutoutml.models.registry import get_model

    model = get_model("trivial-center", load=False, device="cpu")
    assert model.is_loaded is False
    ImagePipeline(model)
    assert model.is_loaded is True


def test_process_bytes_returns_the_requested_outputs_only(pipeline, make_png):
    """Encoding a 4000px PNG is not free; a caller who wants a mask should not pay
    for a transparent PNG as well."""
    result = pipeline.process_bytes(make_png(64, 48), ImageRequest(outputs=("mask_png",)))
    assert set(result.outputs) == {"mask_png"}


def test_process_bytes_produces_a_transparent_png_at_the_original_size(pipeline, make_png):
    result = pipeline.process_bytes(make_png(70, 50))

    assert result.width == 70
    assert result.height == 50
    rgba = decode_rgba(result.outputs["transparent_png"])
    assert rgba.shape == (50, 70, 4)
    # The alpha plane is genuinely varying, not a constant 255.
    assert rgba[:, :, 3].min() < 128 < rgba[:, :, 3].max()


def test_process_bytes_records_the_content_hash_for_idempotency(pipeline, make_png):
    import hashlib

    data = make_png(32, 32)
    result = pipeline.process_bytes(data)
    assert result.content_sha256 == hashlib.sha256(data).hexdigest()


def test_timings_cover_every_stage_and_decode_is_only_set_for_bytes(pipeline, make_png):
    """The stage breakdown in docs/benchmarks.md is built from these keys."""
    from_bytes = pipeline.process_bytes(make_png(32, 32))
    assert set(from_bytes.timings_ms) == {
        "decode",
        "preprocess",
        "inference",
        "postprocess",
        "refine",
        "encode",
    }
    from_array = pipeline.process_array(np.zeros((32, 32, 3), dtype=np.uint8))
    assert "decode" not in from_array.timings_ms
    assert all(v >= 0.0 for v in from_array.timings_ms.values())


def test_alpha_coverage_reports_the_thresholded_foreground_fraction(pipeline):
    result = pipeline.process_array(np.zeros((64, 64, 3), dtype=np.uint8))
    assert 0.0 < result.alpha_coverage < 1.0


def test_batching_gives_the_same_outputs_as_processing_one_at_a_time(pipeline):
    rng = np.random.default_rng(11)
    images = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(4)]

    batched = pipeline.process_batch(images)
    singles = [pipeline.process_array(img) for img in images]

    assert len(batched) == 4
    for got, want in zip(batched, singles, strict=True):
        assert got.outputs["mask_png"] == want.outputs["mask_png"]


def test_a_heterogeneous_batch_is_fine_because_letterboxing_normalises_it(pipeline):
    """In a real workload the batch is always heterogeneous."""
    images = [
        np.zeros((40, 100, 3), dtype=np.uint8),
        np.zeros((100, 40, 3), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.uint8),
    ]
    results = pipeline.process_batch(images)
    assert [(r.height, r.width) for r in results] == [(40, 100), (100, 40), (64, 64)]


def test_an_empty_batch_is_not_an_error(pipeline):
    assert pipeline.process_batch([]) == []


def test_per_image_timings_are_the_batch_cost_divided_by_the_batch_size(pipeline):
    """Otherwise a batched row in the benchmark table would look N times slower."""
    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
    results = pipeline.process_batch(images)
    inference = {round(r.timings_ms["inference"], 6) for r in results}
    assert len(inference) == 1


def test_every_output_kind_encodes(pipeline):
    request = ImageRequest(
        outputs=(
            "transparent_png",
            "transparent_webp",
            "mask_png",
            "color_composite",
            "background_composite",
            "blurred_background",
        ),
        background_image=np.full((64, 64, 3), 90, dtype=np.uint8),
    )
    outputs = pipeline.process_array(np.zeros((48, 48, 3), dtype=np.uint8), request).outputs
    assert len(outputs) == 6
    assert all(len(v) > 0 for v in outputs.values())
    assert outputs["transparent_png"][:8] == b"\x89PNG\r\n\x1a\n"


def test_background_composite_without_a_background_image_is_rejected(pipeline):
    request = ImageRequest(outputs=("background_composite",))
    with pytest.raises(ValueError, match=r"requires request\.background_image"):
        pipeline.process_array(np.zeros((32, 32, 3), dtype=np.uint8), request)


def test_a_pixel_budget_protects_against_a_decompression_bomb(pipeline, make_png):
    with pytest.raises(ValueError, match="pixel"):
        pipeline.process_bytes(make_png(64, 64), ImageRequest(max_pixels=16))


def test_refinement_runs_at_full_resolution_not_on_the_low_res_mask(pipeline):
    """Refining before upsampling reintroduces exactly the stair-stepping the
    guided filter exists to remove, so the refine config must visibly matter at
    a resolution well above the model's 320px input."""
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[128:384, 128:384] = 240

    off = pipeline.alpha_only(image, RefineConfig.off())
    quality = pipeline.alpha_only(image, RefineConfig.quality())

    assert off.shape == quality.shape == (512, 512)
    assert not np.array_equal(off, quality)


def test_alpha_batch_matches_alpha_only(pipeline):
    rng = np.random.default_rng(12)
    images = [rng.integers(0, 255, (40, 40, 3), dtype=np.uint8) for _ in range(3)]
    batch = pipeline.alpha_batch(images)
    for got, image in zip(batch, images, strict=True):
        np.testing.assert_allclose(got, pipeline.alpha_only(image), atol=1e-5)


def test_image_result_summary_is_json_safe(pipeline, make_png):
    summary = pipeline.process_bytes(make_png(32, 32)).summary()
    assert isinstance(summary["outputs"]["mask_png"], int)
    assert isinstance(summary["timings_ms"]["inference"], float)


def test_image_request_summary_does_not_leak_the_background_pixels():
    request = ImageRequest(background_image=np.zeros((4, 4, 3), dtype=np.uint8))
    payload = request.as_dict()
    assert payload["has_background_image"] is True
    assert "background_image" not in payload


# ======================================================== video: batching


def test_batched_never_yields_more_than_the_batch_size():
    """This is the memory bound: peak RSS is batch_size x W x H x 3, so a bug here
    is how a 4K clip turns into 1.5 GB of resident RGB."""
    frames = iter([np.zeros((2, 2), dtype=np.uint8) for _ in range(10)])
    sizes = [len(b) for b in _batched(frames, 4)]
    assert sizes == [4, 4, 2]
    assert max(sizes) <= 4


def test_batched_yields_the_short_final_batch():
    frames = iter([np.zeros((1, 1), dtype=np.uint8) for _ in range(5)])
    assert [len(b) for b in _batched(frames, 2)] == [2, 2, 1]


def test_batched_of_an_empty_iterator_yields_nothing():
    assert list(_batched(iter([]), 4)) == []


def test_batched_treats_a_non_positive_size_as_one():
    frames = iter([np.zeros((1, 1), dtype=np.uint8) for _ in range(3)])
    assert [len(b) for b in _batched(frames, 0)] == [1, 1, 1]


def test_batched_is_lazy_so_frames_are_not_all_decoded_up_front():
    """If it consumed the iterator eagerly the streaming discipline would be lost."""
    consumed = 0

    def source():
        nonlocal consumed
        for _ in range(100):
            consumed += 1
            yield np.zeros((1, 1), dtype=np.uint8)

    batches = _batched(source(), 4)
    next(batches)
    assert consumed == 4


def test_batched_preserves_frame_order():
    frames = iter([np.full((1, 1), i, dtype=np.uint8) for i in range(7)])
    flattened = [int(f[0, 0]) for batch in _batched(frames, 3) for f in batch]
    assert flattened == list(range(7))


# ================================================ video: temporal smoothing


def test_smoother_none_is_a_pass_through():
    smoother = TemporalSmoother("none")
    frame = np.full((4, 4), 0.7, dtype=np.float32)
    assert np.array_equal(smoother(frame), frame)


def test_smoother_ema_keeps_exactly_one_frame_of_state():
    smoother = TemporalSmoother("ema", ema_weight=0.5)
    smoother(np.zeros((2, 2), dtype=np.float32))
    out = smoother(np.ones((2, 2), dtype=np.float32))
    assert np.allclose(out, 0.5)
    assert smoother._buffer == []


def test_smoother_median_window_is_forced_odd_so_a_middle_exists():
    assert TemporalSmoother("median", window=4).window == 5
    assert TemporalSmoother("median", window=3).window == 3
    assert TemporalSmoother("median", window=0).window == 1


def test_smoother_median_buffer_stays_bounded():
    """The other half of the memory bound: median holds `window` frames, not all."""
    smoother = TemporalSmoother("median", window=3)
    for _ in range(20):
        smoother(np.zeros((4, 4), dtype=np.float32))
    assert len(smoother._buffer) == 3


def test_smoother_median_removes_a_single_frame_dropout():
    smoother = TemporalSmoother("median", window=3)
    good = np.ones((4, 4), dtype=np.float32)
    smoother(good)
    smoother(good)
    out = smoother(np.zeros((4, 4), dtype=np.float32))
    assert np.array_equal(out, good)


def test_smoother_reset_clears_both_kinds_of_state():
    ema = TemporalSmoother("ema")
    ema(np.ones((2, 2), dtype=np.float32))
    ema.reset()
    assert ema._previous is None

    median = TemporalSmoother("median")
    median(np.ones((2, 2), dtype=np.float32))
    median.reset()
    assert median._buffer == []


def test_estimate_flicker_is_zero_for_a_frozen_sequence():
    """Reported next to IoU precisely because a frozen mask scores a perfect 0 here
    while being useless."""
    frame = np.zeros((8, 8), dtype=np.float32)
    assert estimate_flicker([frame, frame.copy()]) == 0.0
    assert estimate_flicker([frame]) == 0.0
    assert estimate_flicker([]) == 0.0


def test_estimate_flicker_grows_with_frame_to_frame_change():
    a = np.zeros((8, 8), dtype=np.float32)
    b = np.full((8, 8), 0.2, dtype=np.float32)
    c = np.full((8, 8), 0.9, dtype=np.float32)
    assert estimate_flicker([a, b]) < estimate_flicker([a, c])


# ======================================================= video: ffmpeg args


def test_mp4_refuses_alpha_rather_than_producing_an_opaque_file():
    """The spec allows auxiliary alpha pictures; no browser or NLE decodes them, so
    an "alpha" MP4 is a silently opaque file."""
    with pytest.raises(ValueError, match="cannot carry a usable alpha channel"):
        encoder_args("mp4", alpha=True)


def test_h264_pins_yuv420p_because_safari_refuses_yuv444p():
    args = encoder_args("mp4")
    assert "libx264" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert "+faststart" in args


def test_vp9_alpha_requires_disabling_alternate_reference_frames():
    """-auto-alt-ref 1 is incompatible with the alpha plane and the failure is
    silent: an opaque file with exit code 0."""
    args = encoder_args("webm", alpha=True)
    assert args[args.index("-pix_fmt") + 1] == "yuva420p"
    assert args[args.index("-auto-alt-ref") + 1] == "0"


def test_vp9_without_alpha_stays_on_yuv420p():
    assert encoder_args("webm")[encoder_args("webm").index("-pix_fmt") + 1] == "yuv420p"
    assert "-auto-alt-ref" not in encoder_args("webm")


def test_a_bitrate_replaces_crf_rather_than_joining_it():
    args = encoder_args("mp4", bitrate="2M")
    assert "-crf" not in args
    assert args[args.index("-b:v") + 1] == "2M"


def test_prores_uses_profile_4444_the_only_one_with_alpha():
    args = encoder_args("mov")
    assert args[args.index("-profile:v") + 1] == "4"
    assert "yuva444p10le" in args


def test_an_unsupported_container_is_rejected():
    with pytest.raises(ValueError, match="unsupported container"):
        encoder_args("avi")


def test_container_extension_maps_codec_names_to_real_muxers():
    """qtrle is a codec whose container is QuickTime; writing out.qtrle selects no
    muxer at all."""
    assert container_extension("qtrle") == "mov"
    assert container_extension("prores") == "mov"
    assert container_extension("vp9") == "webm"
    assert container_extension(".mp4") == "mp4"
    with pytest.raises(ValueError, match="unsupported container"):
        container_extension("mkv")


def test_container_supports_alpha_is_a_claim_about_the_format_only():
    assert container_supports_alpha("webm") is True
    assert container_supports_alpha("mov") is True
    assert container_supports_alpha("mp4") is False


def test_webm_needs_an_explicit_libvpx_decoder_to_see_its_alpha_plane():
    """ffmpeg's native vp9 decoder ignores Matroska BlockAdditional alpha entirely,
    so a genuinely transparent file decodes as alpha=255 everywhere."""
    assert alpha_decoder_args("webm") == ("-c:v", "libvpx-vp9")
    # ProRes and QTRLE carry alpha in the pixel format and need no override.
    assert alpha_decoder_args("mov") == ()
    assert alpha_decoder_args("qtrle") == ()


def test_alpha_roundtrip_probe_rejects_containers_that_cannot_carry_alpha():
    assert alpha_roundtrip_works("mp4") is False
    assert alpha_roundtrip_works("avi") is False


@ffmpeg_required
def test_alpha_roundtrip_probe_agrees_with_working_alpha_containers():
    """Recorded as a test because docs/benchmarks.md publishes what this build does."""
    usable = working_alpha_containers()
    assert set(usable) <= set(ALPHA_CONTAINERS)
    for container in ALPHA_CONTAINERS:
        assert alpha_roundtrip_works(container) is (container in usable)


# ================================================= video: end to end


@ffmpeg_required
def test_probe_reads_the_dimensions_and_frame_count(clip: Path):
    info = probe(clip)
    assert (info.width, info.height) == (96, 64)
    assert info.frame_count == 24
    assert info.fps == pytest.approx(12.0, abs=0.1)
    assert info.has_audio is False
    assert info.codec == "h264"


@ffmpeg_required
def test_probe_rejects_a_file_that_is_not_video(tmp_path: Path):
    junk = tmp_path / "not-video.mp4"
    junk.write_bytes(b"definitely not an mp4")
    with pytest.raises(FFmpegError):
        probe(junk)


@ffmpeg_required
def test_composite_mode_writes_a_playable_mp4(video_pipeline, clip: Path, tmp_path: Path):
    out = tmp_path / "composite.mp4"
    result = video_pipeline.process(
        clip, out, VideoRequest(mode="composite", batch_size=4, smoothing="ema", crf=30)
    )

    assert result.frames_processed == 24
    assert out.is_file()
    assert result.output_bytes > 0
    assert result.has_alpha is False
    assert probe(out).frame_count == 24


@ffmpeg_required
def test_mask_mode_writes_a_grayscale_video(video_pipeline, clip: Path, tmp_path: Path):
    out = tmp_path / "mask.mp4"
    result = video_pipeline.process(clip, out, VideoRequest(mode="mask", max_frames=6, crf=30))
    assert result.frames_processed == 6
    assert out.is_file()


@ffmpeg_required
def test_frames_mode_writes_rgba_pngs_and_zips_them(video_pipeline, clip: Path, tmp_path: Path):
    """The only output option a video editor will always accept, and the fallback
    when this ffmpeg build cannot carry alpha in any container."""
    out = tmp_path / "frames"
    result = video_pipeline.process(
        clip, out, VideoRequest(mode="frames", max_frames=5, batch_size=2)
    )

    assert len(result.frame_paths) == 5
    assert result.has_alpha is True
    rgba = decode_rgba(result.frame_paths[0].read_bytes())
    assert rgba.shape[2] == 4
    assert rgba[:, :, 3].min() < 255

    assert result.archive_path is not None
    assert result.deliverable == result.archive_path
    with zipfile.ZipFile(result.archive_path) as zf:
        # Flat archive: the server's temp path must not be reproduced inside it.
        assert [n for n in zf.namelist() if "/" in n] == []
        assert len(zf.namelist()) == 5


@ffmpeg_required
def test_frames_mode_can_skip_the_archive(video_pipeline, clip: Path, tmp_path: Path):
    result = video_pipeline.process(
        clip,
        tmp_path / "loose",
        VideoRequest(mode="frames", max_frames=3, archive_frames=False),
    )
    assert result.archive_path is None
    assert result.deliverable == result.output_path


@ffmpeg_required
@pytest.mark.skipif(
    not alpha_roundtrip_works("webm"),
    reason="this ffmpeg build drops the alpha plane in WebM",
)
def test_transparent_mode_produces_a_file_that_really_carries_alpha(
    video_pipeline, clip: Path, tmp_path: Path
):
    out = tmp_path / "transparent.webm"
    result = video_pipeline.process(
        clip, out, VideoRequest(mode="transparent", container="webm", max_frames=4, crf=40)
    )
    assert result.has_alpha is True
    assert has_alpha(out) is True


@ffmpeg_required
def test_transparent_mode_is_refused_for_mp4_before_any_work_happens(
    video_pipeline, clip: Path, tmp_path: Path
):
    """Rejected up front rather than surfacing as a confusing ffmpeg stderr dump
    several seconds in - or worse, as a silently opaque "transparent" video."""
    with pytest.raises(ValueError, match="cannot carry alpha"):
        video_pipeline.process(
            clip, tmp_path / "bad.mp4", VideoRequest(mode="transparent", container="mp4")
        )


@ffmpeg_required
def test_max_frames_processes_only_a_prefix(video_pipeline, clip: Path, tmp_path: Path):
    result = video_pipeline.process(
        clip, tmp_path / "short.mp4", VideoRequest(max_frames=7, crf=30)
    )
    assert result.frames_processed == 7


@ffmpeg_required
def test_a_frame_limit_is_enforced_before_decoding(video_pipeline, clip: Path, tmp_path: Path):
    out = tmp_path / "limited.mp4"
    with pytest.raises(ValueError, match="above the configured limit"):
        video_pipeline.process(clip, out, VideoRequest(frame_limit=5))
    assert not out.exists()


@ffmpeg_required
def test_scale_to_changes_the_output_resolution(video_pipeline, clip: Path, tmp_path: Path):
    out = tmp_path / "scaled.mp4"
    video_pipeline.process(clip, out, VideoRequest(scale_to=(48, 32), max_frames=4, crf=30))
    assert (probe(out).width, probe(out).height) == (48, 32)


@ffmpeg_required
def test_batch_size_does_not_change_the_frame_count(video_pipeline, clip: Path, tmp_path: Path):
    counts = {
        size: video_pipeline.process(
            clip, tmp_path / f"b{size}.mp4", VideoRequest(batch_size=size, max_frames=9, crf=35)
        ).frames_processed
        for size in (1, 2, 4)
    }
    assert set(counts.values()) == {9}


@ffmpeg_required
def test_progress_callbacks_are_monotonic_and_end_at_complete(
    video_pipeline, clip: Path, tmp_path: Path
):
    events: list[VideoProgress] = []
    video_pipeline.process(
        clip,
        tmp_path / "progress.mp4",
        VideoRequest(max_frames=12, batch_size=2, progress_interval=2, crf=35),
        on_progress=events.append,
    )

    assert events
    assert events[-1].stage == "complete"
    assert events[-1].fraction == 1.0
    assert [e.frames_done for e in events] == sorted(e.frames_done for e in events)
    assert all(0.0 <= e.fraction <= 1.0 for e in events)
    assert set(events[0].as_dict()) == {
        "stage",
        "frames_done",
        "frames_total",
        "fraction",
        "seconds_elapsed",
        "fps",
        "message",
    }


def test_progress_fraction_is_zero_when_the_total_is_unknown():
    """Probing a fragmented MP4 or WebM often yields no frame count."""
    assert (
        VideoProgress(
            stage="inference", frames_done=5, frames_total=0, fps=1.0, seconds_elapsed=1.0
        ).fraction
        == 0.0
    )


@ffmpeg_required
def test_measure_flicker_reports_both_sides_of_the_smoothing_trade_off(
    video_pipeline, clip: Path, tmp_path: Path
):
    """Smoothing is measured, not assumed: the flicker figures are what justify the
    default EMA weight in docs/benchmarks.md."""
    result = video_pipeline.process(
        clip,
        tmp_path / "flicker.mp4",
        VideoRequest(max_frames=12, smoothing="ema", measure_flicker=True, crf=35),
    )
    assert result.flicker_raw is not None
    assert result.flicker_smoothed is not None
    assert result.flicker_smoothed <= result.flicker_raw + 1e-6


@ffmpeg_required
def test_a_missing_source_file_fails_before_ffmpeg_is_spawned(video_pipeline, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        video_pipeline.process(tmp_path / "absent.mp4", tmp_path / "out.mp4")


@ffmpeg_required
def test_compare_smoothing_isolates_the_smoother_from_inference_variation(
    video_pipeline, clip: Path
):
    """It decodes once and reuses the cached alphas, so the comparison cannot be
    confounded by frame-to-frame differences in the model output."""
    study = video_pipeline.compare_smoothing(clip, max_frames=8, batch_size=4)

    assert set(study) == {"none", "ema", "median", "_baseline"}
    assert study["none"]["flicker"] == pytest.approx(study["_baseline"]["flicker"])
    assert study["ema"]["flicker"] <= study["_baseline"]["flicker"] + 1e-6
    assert all(study[m]["frames"] == 8 for m in ("none", "ema", "median"))


@ffmpeg_required
def test_video_result_summary_is_json_safe(video_pipeline, clip: Path, tmp_path: Path):
    result = video_pipeline.process(
        clip, tmp_path / "summary.mp4", VideoRequest(max_frames=3, crf=35)
    )
    summary = result.summary()
    assert isinstance(summary["source"], dict)
    assert summary["frames_processed"] == 3
    assert isinstance(summary["output_path"], str)


def test_video_request_summary_does_not_leak_the_background_pixels():
    payload = VideoRequest(background_image=np.zeros((4, 4, 3), dtype=np.uint8)).as_dict()
    assert payload["has_background_image"] is True
    assert "background_image" not in payload


def test_archive_frames_writes_a_flat_stored_zip(tmp_path: Path):
    """ZIP_STORED because PNG is already deflate-compressed: recompressing costs
    CPU proportional to the whole output and typically saves under 1%."""
    frames = tmp_path / "nested" / "deeper"
    frames.mkdir(parents=True)
    paths = []
    for i in range(3):
        p = frames / f"frame_{i:06d}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 512)
        paths.append(p)

    archive = archive_frames(paths, tmp_path / "out.zip")

    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["frame_000000.png", "frame_000001.png", "frame_000002.png"]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
