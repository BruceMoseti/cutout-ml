"""Tests for contention measurement and the way it is reported.

The reporting tests build minimal report dicts rather than running the harness: what matters
is that a contended row cannot reach a table without its marker, and that is a property of
the renderer.
"""

from __future__ import annotations

from typing import Any

from cutoutml.benchmarks.contention import LoadSnapshot, sample
from cutoutml.benchmarks.render_report import (
    contention_block,
    main_table,
    readme_table,
)


def _snapshot(external: float, *, threshold: float = 0.5) -> LoadSnapshot:
    return LoadSnapshot(
        logical_cpus=8,
        load_average_1m=external,
        load_average_5m=external,
        load_average_15m=external,
        total_busy_cores=external,
        own_busy_cores=0.0,
        external_busy_cores=external,
        quiet_threshold_cores=threshold,
        sample_seconds=1.0,
    )


def _report(*loads: LoadSnapshot) -> dict[str, Any]:
    cases = []
    for index, load in enumerate(loads):
        cases.append(
            {
                "case": {
                    "name": f"case-{index}",
                    "model": f"model-{index}",
                    "precision": "fp32",
                    "batch_size": 1,
                    "random_init": False,
                },
                "status": "ok",
                "runtime": "pytorch-eager",
                "model_metadata": {"runtime": "pytorch", "name": f"model-{index}"},
                "latency": {
                    "per_image_p50_ms": 10.0,
                    "p95_ms": 12.0,
                    "batch_size": 1,
                    "throughput_images_per_second": 100.0,
                    "peak_rss_bytes": 1024 * 1024,
                },
                "accuracy": {"iou": 0.5, "mae": 0.1, "f_beta": 0.5, "boundary_f1": 0.5},
                "accuracy_valid": True,
                "model_size_bytes": 1024 * 1024,
                "load": load.as_dict(),
                "latency_trustworthy": load.quiet,
                "compile": {"attempted": False, "succeeded": False},
                "notes": "",
            }
        )
    return {
        "run_id": "20260101T000000Z-test",
        "created_at": "2026-01-01T00:00:00Z",
        "duration_seconds": 1.0,
        "cases": cases,
        "summary": {
            "cases_measured_under_contention": sum(1 for load in loads if not load.quiet),
            "peak_external_busy_cores": max(
                (load.external_busy_cores for load in loads), default=0
            ),
        },
        "environment": {
            "hardware": "test CPU, 8 vCPU, 32 GB RAM, no GPU (CPU-only)",
            "gpu": "none",
            "cpu_count_logical": 8,
            "os_description": "Linux test",
            "python_version": "3.12.0",
            "torch_threads": 8,
            "git": {"short_commit": "abc123", "branch": "main", "dirty": False},
            "library_versions": {"torch": "2.13.0"},
        },
        "config": {"warmup": 3, "repetitions": 20, "accuracy_samples": 64},
        "dataset": {},
    }


# ------------------------------------------------------------------ the measurement


def test_a_machine_below_the_threshold_is_quiet_and_above_it_is_not():
    assert _snapshot(0.2).quiet
    assert _snapshot(0.5).quiet, "the threshold itself counts as quiet"
    assert not _snapshot(0.6).quiet
    assert not _snapshot(7.9).quiet


def test_the_summary_names_the_state_and_the_evidence():
    assert "quiet" in _snapshot(0.1).summary
    contended = _snapshot(7.2).summary
    assert "CONTENDED" in contended
    assert "7.2 of 8" in contended


def test_sampling_a_real_machine_produces_self_consistent_numbers():
    """External demand is a derived quantity, so it must not go negative or exceed capacity."""
    snapshot = sample(interval=0.1)
    assert snapshot.logical_cpus >= 1
    assert snapshot.external_busy_cores >= 0.0
    assert snapshot.own_busy_cores >= 0.0
    assert snapshot.total_busy_cores <= snapshot.logical_cpus + 0.01
    assert snapshot.as_dict()["quiet"] in (True, False)


def test_the_snapshot_serialises_the_derived_fields_too():
    """`quiet` is a property, so dataclasses.asdict() alone would silently drop it."""
    data = _snapshot(0.1).as_dict()
    assert data["quiet"] is True
    assert "summary" in data
    assert data["external_busy_cores"] == 0.1


# -------------------------------------------------------------------- the reporting


def test_a_quiet_run_claims_the_hardware_and_shows_no_table():
    block = contention_block(_report(_snapshot(0.1), _snapshot(0.2)))
    assert "quiet machine" in block
    assert "| Case |" not in block


def test_a_contended_run_reports_the_count_the_peak_and_a_per_case_table():
    block = contention_block(_report(_snapshot(0.1), _snapshot(7.5)))
    assert "1 of 2 timed cases were measured under contention" in block
    assert "7.5 of 8 cores" in block
    assert "| `case-1` | 7.5 / 8 |" in block
    assert "upper bounds" in block


def test_the_contention_block_states_that_accuracy_is_not_affected():
    """The distinction is the whole point: a contended run still publishes accuracy."""
    block = contention_block(_report(_snapshot(7.5)))
    assert "Accuracy is unaffected" in block


def test_a_run_without_load_evidence_says_so_rather_than_claiming_quiet():
    report = _report(_snapshot(0.1))
    for case in report["cases"]:
        case.pop("load")
    assert "no load evidence" in contention_block(report)


def test_a_contended_latency_cell_is_marked_in_the_detailed_table():
    table = main_table(_report(_snapshot(0.1), _snapshot(7.5)))
    lines = [line for line in table.splitlines() if line.startswith("| model-")]
    assert "†" not in lines[0], "a quiet row must not be marked"
    assert "†" in lines[1], "a contended row must be marked"
    assert "†" in table.splitlines()[-1], "the marker must be explained"


def test_a_quiet_table_carries_no_marker_and_no_footnote():
    table = main_table(_report(_snapshot(0.1)))
    assert "†" not in table


def test_the_readme_table_marks_contended_rows_and_explains_the_marker():
    table = readme_table(_report(_snapshot(0.1), _snapshot(7.5)))
    assert "†" in table
    assert "measured under CPU contention" in table
    assert "accuracy columns" in table


def test_the_readme_table_omits_the_contention_caveat_when_the_machine_was_quiet():
    table = readme_table(_report(_snapshot(0.1)))
    assert "†" not in table
    assert "contention" not in table.lower()
