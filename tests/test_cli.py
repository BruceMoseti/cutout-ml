"""The ``cutoutml`` command line interface.

The CLI is the documented way to use everything that does not need the API, so it is
also the first thing a new reader runs. Two properties are worth pinning:

* **Exit codes and messages.** A wrong path or an unexportable model has to produce a
  status code and one readable line, not a traceback. Nothing else in the suite covers
  that, because every other caller of these functions is a test that passes valid input.
* **The container precedence in ``video``.** Explicit flag, then the filename extension,
  then the mode. The middle step exists so ``--mode transparent -o out.webm`` is not
  rejected for asking an MP4 to carry alpha, and it is easy to drop in a refactor because
  the resulting bug looks like a user error.

These are unit tests: the only model used is ``trivial-center``, which needs no weights,
and nothing here shells out or opens a socket. ``doctor`` is deliberately pointed at a
missing ffmpeg and an unreachable database so it exercises its degraded paths.
"""

from __future__ import annotations

import importlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutoutml.cli import (
    GLOBAL_VALUE_FLAGS,
    _resolve_container,
    build_parser,
    cmd_doctor,
    cmd_export_onnx,
    cmd_models,
    cmd_segment,
    cmd_video,
    main,
)


@pytest.fixture
def cli_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Settings that keep the CLI off every external dependency.

    The database URL points at a closed local port rather than being unset, so
    ``check_database`` fails immediately with a refused connection instead of either
    hanging or reaching a developer's real Postgres.
    """
    from cutoutml.core.config import get_settings

    monkeypatch.setenv("CUTOUTML_ENVIRONMENT", "test")
    monkeypatch.setenv("CUTOUTML_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("CUTOUTML_JWT_SECRET", "cli-test-secret-not-a-real-one")
    monkeypatch.setenv("CUTOUTML_DATABASE_URL", "postgresql+psycopg://x:x@127.0.0.1:1/nope")
    monkeypatch.setenv("CUTOUTML_DEVICE", "cpu")
    monkeypatch.setenv("CUTOUTML_TORCH_NUM_THREADS", "1")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """A small PNG on disk, since every command takes a path rather than bytes."""
    from cutoutml.core.imaging import encode_image

    image = np.full((48, 64, 3), 20, dtype=np.uint8)
    image[12:36, 16:48] = (240, 210, 60)
    path = tmp_path / "photo.png"
    path.write_bytes(encode_image(image, "png"))
    return path


def training_module() -> Any:
    """The ``cutoutml.training.train`` *module*.

    ``import cutoutml.training.train as m`` binds the ``train`` *function* instead: the
    package re-exports it under that name, and the re-export overwrites the submodule
    attribute. ``import_module`` reads from ``sys.modules``, which is unaffected.
    """
    return importlib.import_module("cutoutml.training.train")


def record_forwarded_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace the trainer's entry point with a recorder and return the recording.

    The trainer itself is not run: what is under test is which tokens reach it, and a
    real epoch costs seconds of CPU for no extra assurance about the argv.
    """
    recorded: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        recorded.append(list(argv))
        return 0

    monkeypatch.setattr(training_module(), "main", fake_main)
    return recorded


def namespace(command: str, *argv: str) -> Any:
    """Parse real argv, so defaults come from the parser rather than a hand-built stub.

    A hand-built Namespace is how a test keeps passing after a flag's default changes.
    """
    return build_parser().parse_args([command, *argv])


# =============================================================== argument parsing


def test_a_bare_invocation_is_an_error_rather_than_a_silent_no_op():
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([])
    assert exit_info.value.code == 2


def test_an_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["enhance"])


def test_every_advertised_subcommand_parses():
    """The module docstring lists these as the public surface; a rename that leaves the
    docstring behind is exactly the sort of thing nobody notices."""
    parser = build_parser()
    assert parser.parse_args(["models"]).command == "models"
    assert parser.parse_args(["segment", "a.png"]).command == "segment"
    assert parser.parse_args(["video", "a.mp4", "-o", "b.mp4"]).command == "video"
    assert parser.parse_args(["export-onnx", "cutoutnet", "-o", "m.onnx"]).command == "export-onnx"
    assert parser.parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["train"]).command == "train"
    assert parser.parse_args(["benchmark"]).command == "benchmark"


def test_video_requires_an_output_because_there_is_no_sensible_default():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["video", "clip.mp4"])


def test_an_invalid_output_kind_is_rejected_at_parse_time():
    """Better here than after the model has loaded and the image has been segmented."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["segment", "a.png", "--outputs", "transparent_gif"])


def test_the_background_colour_takes_exactly_three_components():
    assert namespace("segment", "a.png", "--background-color", "1", "2", "3").background_color == [
        1,
        2,
        3,
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["segment", "a.png", "--background-color", "1", "2"])


# ========================================================== container precedence


def test_an_explicit_container_flag_wins_over_the_extension():
    args = namespace("video", "in.mp4", "-o", "out.webm", "--container", "mov")
    assert _resolve_container(args) == "mov"


def test_the_extension_is_used_when_no_container_is_given():
    """The step that keeps ``--mode transparent -o out.webm`` from being rejected for
    asking an MP4 to carry alpha - a flag default contradicting a filename the user
    typed."""
    args = namespace("video", "in.mp4", "-o", "out.webm", "--mode", "transparent")
    assert _resolve_container(args) == "webm"


def test_an_extensionless_destination_falls_back_to_the_mode():
    """Transparency is only deliverable by an alpha-capable container, so the mode is
    what decides when the filename says nothing."""
    assert _resolve_container(
        namespace("video", "in.mp4", "-o", "out", "--mode", "transparent")
    ) == ("webm")
    assert _resolve_container(namespace("video", "in.mp4", "-o", "out", "--mode", "composite")) == (
        "mp4"
    )


def test_the_mode_fallback_does_not_override_a_contradictory_extension():
    """``--mode composite -o out.webm`` is a perfectly reasonable request: composite
    output in a VP9 file. The mode must not quietly rewrite it to mp4."""
    args = namespace("video", "in.mp4", "-o", "out.webm", "--mode", "composite")
    assert _resolve_container(args) == "webm"


# ===================================================================== models


def test_models_json_lists_the_registry_with_a_usable_weights_flag(
    cli_settings: None, capsys: pytest.CaptureFixture[str]
):
    assert cmd_models(namespace("models", "--json")) == 0

    rows = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row for row in rows}

    from cutoutml.models.registry import list_model_names

    assert set(by_name) == set(list_model_names())
    assert by_name["trivial-center"]["weights"] == "yes", "a model needing no weights is usable"
    assert set(by_name["trivial-center"]) == {
        "name",
        "architecture",
        "runtime",
        "input",
        "weights",
        "tags",
    }
    assert by_name["trivial-center"]["input"] == "320x320"


def test_models_reports_missing_weights_rather_than_omitting_the_model(
    cli_settings: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
):
    """Listing only usable models would leave a reader wondering why the registry in the
    docs has entries their machine does not."""
    from cutoutml.core.config import get_settings

    monkeypatch.setenv("CUTOUTML_MODEL_WEIGHTS_DIR", str(tmp_path / "no-weights-here"))
    get_settings.cache_clear()

    cmd_models(namespace("models", "--json"))
    by_name = {row["name"]: row for row in json.loads(capsys.readouterr().out)}

    assert by_name["u2net"]["weights"] == "MISSING"
    assert by_name["trivial-center"]["weights"] == "yes"


def test_models_prints_an_aligned_table_by_default(
    cli_settings: None, capsys: pytest.CaptureFixture[str]
):
    assert cmd_models(namespace("models")) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["name", "architecture", "runtime", "input", "weights", "tags"]
    assert set(lines[1]) == {"-"}
    assert all(len(line) == len(lines[1]) for line in lines[2:] if line.strip())


# ==================================================================== segment


def test_segment_writes_one_file_per_requested_output(
    cli_settings: None, photo: Path, tmp_path: Path
):
    outdir = tmp_path / "out"
    args = namespace(
        "segment",
        str(photo),
        "-o",
        str(outdir),
        "-m",
        "trivial-center",
        "--outputs",
        "transparent_png",
        "mask_png",
    )
    assert cmd_segment(args) == 0

    assert (outdir / "photo.transparent_png.png").is_file()
    assert (outdir / "photo.mask_png.png").is_file()
    assert sorted(p.name for p in outdir.iterdir()) == [
        "photo.mask_png.png",
        "photo.transparent_png.png",
    ], "nothing beyond the requested kinds is written"


def test_segment_names_webp_outputs_with_a_webp_extension(
    cli_settings: None, photo: Path, tmp_path: Path
):
    """The extension is derived from the output kind; a .png holding WebP bytes is the
    kind of thing that only surfaces when a browser refuses the file."""
    outdir = tmp_path / "out"
    args = namespace(
        "segment",
        str(photo),
        "-o",
        str(outdir),
        "-m",
        "trivial-center",
        "--outputs",
        "transparent_webp",
    )
    assert cmd_segment(args) == 0
    assert (outdir / "photo.transparent_webp.webp").is_file()


def test_segment_creates_the_output_directory(cli_settings: None, photo: Path, tmp_path: Path):
    outdir = tmp_path / "deep" / "nested" / "out"
    args = namespace("segment", str(photo), "-o", str(outdir), "-m", "trivial-center")
    assert cmd_segment(args) == 0
    assert outdir.is_dir()


def test_segment_prints_the_result_summary_as_json(
    cli_settings: None, photo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Printed so the CLI can be piped into jq, which means it has to be valid JSON and
    not a repr."""
    args = namespace(
        "segment",
        str(photo),
        "-o",
        str(tmp_path / "out"),
        "-m",
        "trivial-center",
        "--outputs",
        "mask_png",
    )
    cmd_segment(args)

    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{") :])
    assert "timings_ms" in summary
    assert summary["model"]


def test_segment_reports_a_missing_input_with_a_status_code_not_a_traceback(
    cli_settings: None, tmp_path: Path
):
    args = namespace("segment", str(tmp_path / "absent.png"), "-o", str(tmp_path / "out"))
    assert cmd_segment(args) == 2


def test_segment_treats_a_directory_as_a_missing_input(cli_settings: None, tmp_path: Path):
    """``is_file`` rather than ``exists``, or a directory reaches the decoder."""
    assert cmd_segment(namespace("segment", str(tmp_path), "-o", str(tmp_path / "out"))) == 2


def test_segment_composites_onto_a_background_image(
    cli_settings: None, photo: Path, tmp_path: Path
):
    outdir = tmp_path / "out"
    args = namespace(
        "segment",
        str(photo),
        "-o",
        str(outdir),
        "-m",
        "trivial-center",
        "--background",
        str(photo),
        "--outputs",
        "background_composite",
    )
    assert cmd_segment(args) == 0
    assert (outdir / "photo.background_composite.png").is_file()


# ====================================================================== video


def test_video_reports_a_missing_input_with_a_status_code(cli_settings: None, tmp_path: Path):
    args = namespace("video", str(tmp_path / "absent.mp4"), "-o", str(tmp_path / "out.mp4"))
    assert cmd_video(args) == 2


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)
def test_video_turns_an_impossible_request_into_a_message_and_a_status_code(
    cli_settings: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """MP4 cannot carry alpha. The point of catching ValueError is that the user sees the
    reason on one line instead of a traceback ending in the encoder.

    Needs a real clip because the pipeline probes the input before it validates the
    container, so a placeholder file fails as an unreadable video first and never reaches
    the branch under test.
    """
    from cutoutml.pipelines.video import make_test_video

    source = make_test_video(tmp_path / "clip.mp4", frames=4, width=64, height=48)

    args = namespace(
        "video",
        str(source),
        "-o",
        str(tmp_path / "out.mp4"),
        "--mode",
        "transparent",
        "--container",
        "mp4",
        "-m",
        "trivial-center",
        "--quiet",
    )
    assert cmd_video(args) == 2

    captured = capsys.readouterr()
    assert "alpha" in captured.err.lower()
    assert "Traceback" not in captured.err


# ================================================================ export-onnx


def test_export_onnx_refuses_a_model_that_is_not_a_torch_adapter(
    cli_settings: None, tmp_path: Path
):
    """The classical baselines have no graph to trace. Better a status code than an
    exception from deep inside the exporter."""
    args = namespace("export-onnx", "classical", "-o", str(tmp_path / "classical.onnx"))
    assert cmd_export_onnx(args) == 2
    assert not (tmp_path / "classical.onnx").exists()


# ===================================================================== doctor


def test_doctor_json_reports_the_environment_without_a_database_or_ffmpeg(
    cli_settings: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The documented reason doctor exists is to answer "no GPU, no ffmpeg, no
    onnxruntime" without a conversation, so it has to survive all three being true."""
    from cutoutml.core.config import get_settings

    monkeypatch.setenv("CUTOUTML_FFMPEG_BINARY", "ffmpeg-that-is-not-installed")
    monkeypatch.setenv("CUTOUTML_FFPROBE_BINARY", "ffprobe-that-is-not-installed")
    get_settings.cache_clear()

    assert cmd_doctor(namespace("doctor", "--json")) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["ffmpeg"] == "NOT FOUND"
    assert report["ffprobe"] == "NOT FOUND"
    assert report["database"].startswith("UNAVAILABLE:")
    assert report["models_registered"] > 0
    assert report["torch"]
    assert report["cpu"]


def test_doctor_reports_the_device_it_would_actually_resolve_to(
    cli_settings: None, capsys: pytest.CaptureFixture[str]
):
    """Not a claim about what is installed but about what ``auto`` picks, which is the
    question behind "why is this slow"."""
    cmd_doctor(namespace("doctor", "--json"))
    report = json.loads(capsys.readouterr().out)

    assert report["device_resolved"] in {"cpu", "cuda", "mps"}
    assert report["cuda_available"] is False or report["device_resolved"] != "cpu"


def test_doctor_prints_aligned_key_values_by_default(
    cli_settings: None, capsys: pytest.CaptureFixture[str]
):
    """Looked up by key rather than by scanning every line, because a driver error is
    multi-line and its continuation lines legitimately carry no key at all."""
    assert cmd_doctor(namespace("doctor")) == 0

    lines = capsys.readouterr().out.splitlines()
    value_columns = set()
    for key in ("torch", "cuda_available", "ffmpeg", "database", "default_model"):
        line = next(candidate for candidate in lines if candidate.startswith(f"{key} "))
        value = line[len(key) :].lstrip()
        value_columns.add(len(line) - len(value))
        assert value

    assert len(value_columns) == 1, "the values line up in one column"


# ======================================================================= main


def test_main_configures_logging_and_returns_the_command_status(
    cli_settings: None, capsys: pytest.CaptureFixture[str]
):
    assert main(["--log-format", "json", "models", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)


def test_main_propagates_a_non_zero_status(cli_settings: None, tmp_path: Path):
    assert main(["segment", str(tmp_path / "absent.png")]) == 2


def test_a_leading_flag_reaches_the_delegate_instead_of_being_rejected(
    cli_settings: None, monkeypatch: pytest.MonkeyPatch
):
    """The regression this guards against is total, not cosmetic: under
    ``nargs=REMAINDER`` argparse refuses to capture an option in first position, so
    ``cutoutml train --epochs 2`` exited 2 with "unrecognized arguments: --epochs" - one
    of the invocations the module docstring advertises."""
    recorded = record_forwarded_argv(monkeypatch)

    assert main(["train", "--arch", "cutoutnet-tiny", "--epochs", "2"]) == 0
    assert recorded == [["--arch", "cutoutnet-tiny", "--epochs", "2"]]


def test_global_flags_before_a_delegating_subcommand_are_not_forwarded(
    cli_settings: None, monkeypatch: pytest.MonkeyPatch
):
    """``--log-format`` belongs to this CLI; passing it on would make the trainer reject
    a flag the user was told to use."""
    recorded = record_forwarded_argv(monkeypatch)

    assert main(["--log-format", "json", "train", "--epochs", "1"]) == 0
    assert recorded == [["--epochs", "1"]]


def test_a_flag_value_that_reads_like_a_subcommand_is_not_mistaken_for_one(
    cli_settings: None, monkeypatch: pytest.MonkeyPatch
):
    """The argv is sliced at the resolved subcommand rather than at the first token that
    happens to spell one, so a value like this cannot move the split point."""
    recorded = record_forwarded_argv(monkeypatch)

    assert main(["--log-level", "train", "train", "--epochs", "1"]) == 0
    assert recorded == [["--epochs", "1"]]


def test_the_delegate_status_code_is_propagated(
    cli_settings: None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(training_module(), "main", lambda _argv: 3)
    assert main(["train"]) == 3


def test_a_typo_in_a_normal_subcommand_is_still_rejected(cli_settings: None):
    """Identifying the subcommand with ``parse_known_args`` must not turn the strict parse
    into a permissive one for everything else."""
    with pytest.raises(SystemExit) as exit_info:
        main(["models", "--jsonn"])
    assert exit_info.value.code == 2


def test_the_hardcoded_global_flag_list_matches_the_parser():
    """``subcommand_index`` has to know which global options consume a following token,
    and argparse exposes no public accessor for that. Reaching into the parser's private
    actions is acceptable here and not in the CLI itself: adding a third global flag with
    a value would otherwise silently split the argv one token early."""
    parser = build_parser()
    with_values = {
        option
        for action in parser._actions
        for option in action.option_strings
        if action.nargs != 0 and option.startswith("--") and option != "--help"
    }
    assert with_values == set(GLOBAL_VALUE_FLAGS)


def test_help_is_forwarded_to_the_delegate_rather_than_summarised(cli_settings: None):
    """``cutoutml train --help`` should print the trainer's real flags. Keeping a copy of
    them in this parser is how the two drift apart."""
    with pytest.raises(SystemExit) as exit_info:
        main(["train", "--help"])
    assert exit_info.value.code == 0
