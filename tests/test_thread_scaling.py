"""Tests for intra-op thread control and how it is reported.

Thread count is the variable that most easily invalidates a CPU benchmark, and it fails
*silently*: nothing raises when one row runs on one thread and the row it is compared
against runs on eight. These tests pin the three places that would let that happen -
the count reaching the runtime, the count being read back from the runtime rather than
from the request, and the sweep rows staying out of tables they would corrupt.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from cutoutml.benchmarks.harness import BenchmarkCase, BenchmarkConfig, BenchmarkHarness
from cutoutml.benchmarks.render_report import (
    environment_block,
    main_table,
    readme_table,
    runtime_comparison_table,
    thread_scaling_block,
)


@pytest.fixture
def restore_torch_threads():
    """The harness sets a process-wide value; leaking it would skew later tests."""
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


# --------------------------------------------------------------- reaching the runtime


def test_the_harness_pins_torch_threads_from_its_config(restore_torch_threads):
    BenchmarkHarness(BenchmarkConfig(threads=2, accuracy_samples=1))
    assert torch.get_num_threads() == 2


def test_a_thread_count_of_zero_leaves_the_runtime_default_alone(restore_torch_threads):
    default = torch.get_num_threads()
    BenchmarkHarness(BenchmarkConfig(threads=0, accuracy_samples=1))
    assert torch.get_num_threads() == default


def test_a_case_overrides_the_run_wide_count(restore_torch_threads):
    harness = BenchmarkHarness(BenchmarkConfig(threads=1, accuracy_samples=1))
    assert harness._case_threads(BenchmarkCase(model="trivial-ones")) == 1
    assert harness._case_threads(BenchmarkCase(model="trivial-ones", threads=4)) == 4


def test_an_onnx_case_is_handed_the_thread_count_torch_set_num_threads_cannot_reach(
    restore_torch_threads, monkeypatch
):
    """ONNX Runtime sizes its pool at session creation and ignores PyTorch's setting.

    Without this the runtime-comparison table would compare 1 thread against 8 and
    attribute the difference to the runtime.
    """
    captured: dict[str, Any] = {}

    def fake_get_model(name: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise RuntimeError("stop here; only the kwargs matter")

    monkeypatch.setattr("cutoutml.benchmarks.harness.get_model", fake_get_model)
    harness = BenchmarkHarness(BenchmarkConfig(threads=3, accuracy_samples=1))

    with pytest.raises(RuntimeError):
        harness._load_model(BenchmarkCase(model="cutoutnet-onnx"))
    assert captured["intra_op_threads"] == 3

    captured.clear()
    with pytest.raises(RuntimeError):
        harness._load_model(BenchmarkCase(model="cutoutnet"))
    assert "intra_op_threads" not in captured, "a torch adapter would reject the kwarg"


def test_an_explicit_case_option_beats_the_run_wide_count(restore_torch_threads, monkeypatch):
    captured: dict[str, Any] = {}

    def fake_get_model(name: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr("cutoutml.benchmarks.harness.get_model", fake_get_model)
    harness = BenchmarkHarness(BenchmarkConfig(threads=3, accuracy_samples=1))
    case = BenchmarkCase(model="cutoutnet-onnx", options={"intra_op_threads": 7})
    with pytest.raises(RuntimeError):
        harness._load_model(case)
    assert captured["intra_op_threads"] == 7


# -------------------------------------------------------- reading it back accurately


def test_onnx_resolves_a_zero_request_to_a_real_core_count():
    """Reporting the request would record 0 threads for the commonest configuration."""
    from cutoutml.models.onnx_adapter import OnnxAdapter

    assert OnnxAdapter(onnx_path="unused.onnx", intra_op_threads=0).effective_intra_op_threads >= 1
    assert OnnxAdapter(onnx_path="unused.onnx", intra_op_threads=5).effective_intra_op_threads == 5


def test_a_skipped_accuracy_is_recorded_as_absent_with_a_reason(restore_torch_threads):
    """Not as a zero, and not as the random-weights excuse, which would be a lie."""
    harness = BenchmarkHarness(
        BenchmarkConfig(threads=1, accuracy_samples=2, warmup=0, repetitions=1)
    )
    result = harness.run_case(
        BenchmarkCase(model="trivial-ones", device="cpu", measure_accuracy=False)
    )
    assert result.status == "ok"
    assert result.accuracy is None
    assert result.accuracy_valid is False
    assert "not remeasured" in result.notes
    assert "random" not in result.notes


def test_a_measured_case_records_the_thread_count_it_ran_at(restore_torch_threads):
    harness = BenchmarkHarness(
        BenchmarkConfig(threads=1, accuracy_samples=2, warmup=0, repetitions=1)
    )
    result = harness.run_case(BenchmarkCase(model="trivial-ones", device="cpu", threads=2))
    assert result.status == "ok"
    assert result.latency is not None
    assert result.latency.threads == 2
    assert result.as_dict()["latency"]["threads"] == 2


# ------------------------------------------------------------------- reporting it


def _case(
    name: str,
    *,
    runtime: str,
    threads: int,
    p50: float,
    sweep: bool,
    model: str = "m",
) -> dict[str, Any]:
    return {
        "case": {
            "name": name,
            "model": model,
            "precision": "fp32",
            "batch_size": 1,
            "random_init": False,
            "threads": threads if sweep else None,
        },
        "status": "ok",
        "runtime": runtime,
        "model_metadata": {"runtime": runtime, "name": model},
        "latency": {
            "threads": threads,
            "p50_ms": p50,
            "p95_ms": p50 * 1.2,
            "stddev_ms": 1.0,
            "per_image_p50_ms": p50,
            "batch_size": 1,
            "throughput_images_per_second": 1000.0 / p50,
            "peak_rss_bytes": 1024 * 1024,
        },
        "accuracy": {"iou": 0.5, "mae": 0.1, "f_beta": 0.5, "boundary_f1": 0.5},
        "accuracy_valid": True,
        "model_size_bytes": 1024 * 1024,
        "latency_trustworthy": True,
        "compile": {"attempted": False, "succeeded": False},
        "stage_timings_ms": {"inference": p50},
        "notes": "",
    }


def _report(cases: list[dict[str, Any]], *, threads: int = 1) -> dict[str, Any]:
    return {
        "run_id": "20260101T000000Z-test",
        "created_at": "2026-01-01T00:00:00Z",
        "duration_seconds": 1.0,
        "cases": cases,
        "summary": {},
        "environment": {
            "hardware": "test CPU, 8 vCPU, 32 GB RAM, no GPU (CPU-only)",
            "gpu": "none",
            "cpu_count_logical": 8,
            "os_description": "Linux test",
            "python_version": "3.12.0",
            "torch_threads": threads or 8,
            "git": {"short_commit": "abc123", "branch": "main", "dirty": False},
            "library_versions": {"torch": "2.13.0"},
        },
        "config": {"warmup": 3, "repetitions": 20, "accuracy_samples": 64, "threads": threads},
        "dataset": {},
    }


SWEEP = [
    _case("eager-t1", runtime="pytorch-eager", threads=1, p50=46.7, sweep=True),
    _case("eager-t8", runtime="pytorch-eager", threads=8, p50=2854.0, sweep=True),
    _case("onnx-t1", runtime="onnxruntime:CPU", threads=1, p50=20.0, sweep=True),
    _case("onnx-t8", runtime="onnxruntime:CPU", threads=8, p50=10.0, sweep=True),
]


def test_the_main_table_shows_the_thread_count_for_every_row():
    table = main_table(
        _report([_case("plain", runtime="pytorch-eager", threads=4, p50=50.0, sweep=False)])
    )
    assert "| Threads |" in table
    header, *_ = table.splitlines()
    row = next(line for line in table.splitlines() if line.startswith("| m |"))
    assert row.split("|")[5].strip() == "4"
    assert header.split("|")[5].strip() == "Threads"


def test_sweep_rows_are_held_out_of_the_headline_tables():
    """They duplicate a model already present and would read as separate models."""
    plain = _case("plain", runtime="pytorch-eager", threads=1, p50=50.0, sweep=False)
    report = _report([plain, *SWEEP])
    assert main_table(report).count("\n| m |") == 1
    assert readme_table(report).count("\n| **m**") == 1


def test_a_sweep_row_cannot_reach_the_runtime_comparison_table():
    """Mixed thread counts there would attribute a thread effect to a runtime."""
    report = _report(
        [
            _case("eager", runtime="pytorch-eager", threads=1, p50=46.7, sweep=False),
            _case("onnx", runtime="onnxruntime:CPU", threads=1, p50=20.0, sweep=False),
            *SWEEP,
        ]
    )
    table = runtime_comparison_table(report)
    assert "2.33x" in table, "the two matched rows should still be compared"
    assert "t8" not in table


def test_the_scaling_block_reports_the_curve_and_the_inversion():
    block = thread_scaling_block(_report(SWEEP))
    assert "| pytorch-eager | 1 |" in block
    assert "| pytorch-eager | 8 |" in block
    # 46.7 / 2854 is a slowdown, and it must be printed as one rather than dropped.
    assert "0.02x" in block
    assert "2.00x" in block, "ONNX does speed up, and that has to show too"
    assert "61x between its own extremes" in block, "the inversion is quantified per runtime"
    assert "`eager-t8`" in block, "the worst row is named, not just quantified"
    assert "more threads made it slower" in block
    assert "threads bought what they should have" in block, "ONNX scaled, and that is stated"


def test_the_scaling_block_explains_the_single_thread_default():
    block = thread_scaling_block(_report(SWEEP))
    assert "single-threaded by default" in block
    assert "barrier" in block


def test_a_run_without_a_sweep_says_so_rather_than_printing_an_empty_table():
    block = thread_scaling_block(
        _report([_case("p", runtime="pytorch-eager", threads=1, p50=1.0, sweep=False)])
    )
    assert "no thread-scaling sweep" in block


def test_the_environment_block_states_the_thread_count_and_that_it_was_pinned():
    assert "1 per runtime, pinned by the harness" in environment_block(_report([], threads=1))
    assert "one per core" in environment_block(_report([], threads=0))


def test_an_older_report_without_a_thread_setting_is_not_described_as_pinned():
    """Results committed before the setting existed must not gain a false claim."""
    report = _report([])
    del report["config"]["threads"]
    assert "not pinned" in environment_block(report)


def test_the_readme_warns_that_single_threaded_latency_is_pessimistic():
    table = readme_table(
        _report([_case("p", runtime="pytorch-eager", threads=1, p50=1.0, sweep=False)])
    )
    assert "single-threaded" in table
    assert "dedicated machine would beat them" in table
