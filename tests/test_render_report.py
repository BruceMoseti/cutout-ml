"""The benchmark renderer.

The renderer is what turns measurements into published claims, so the properties worth
testing are the ones whose failure would put a false statement in the README rather than
a formatting glitch: marking a timing as contended when it was not, printing a legend for
a mark that appears nowhere, explaining a discrepancy by a cause the data contradicts, and
mixing thread-sweep rows into tables where they would be read as duplicates.

One test renders the committed report end to end. That is deliberately coupled to the real
schema: fixtures in this file describe what the renderer is *told*, and only a real file
catches the case where the harness stops telling it that.
"""

from __future__ import annotations

from typing import Any

import pytest

from cutoutml.benchmarks.harness import latest_report
from cutoutml.benchmarks.render_report import (
    METHODOLOGY,
    _contention_mark,
    _sweep_consistency_notes,
    _was_contended,
    main_table,
    readme_table,
    render_benchmarks_doc,
    update_readme,
)


def case(
    *,
    name: str = "cutoutnet-fp32",
    model: str = "cutoutnet",
    p50: float = 30.0,
    threads: int | None = None,
    trustworthy: bool | None = True,
    status: str = "ok",
    batch_size: int = 1,
    external_cores: float = 0.0,
) -> dict[str, Any]:
    """One case as the harness serialises it, reduced to the fields the renderer reads."""
    return {
        "status": status,
        "case": {
            "name": name,
            "model": model,
            "precision": "fp32",
            "batch_size": batch_size,
            "random_init": False,
            "threads": threads,
        },
        "latency": {
            "p50_ms": p50,
            "p95_ms": p50 * 1.02,
            "per_image_p50_ms": p50 / batch_size,
            "batch_size": batch_size,
            "threads": threads or 1,
            "stddev_ms": 0.1,
            "throughput_images_per_second": 1000.0 * batch_size / p50,
            "peak_rss_bytes": 350 * 1024 * 1024,
            "cold_start_seconds": 0.04,
        },
        "accuracy": {"iou": 0.85, "mae": 0.057, "f_beta": 0.91, "boundary_f1": 0.79},
        "accuracy_valid": True,
        "latency_trustworthy": trustworthy,
        "load": {
            "quiet": trustworthy is not False,
            "external_busy_cores": external_cores,
            "logical_cpus": 8,
            "load_average_1m": 0.4,
        },
        "model_metadata": {"runtime": "pytorch", "name": model},
        "runtime": "pytorch-eager",
        "compile": {"attempted": False, "succeeded": False},
        "model_size_bytes": 4 * 1024 * 1024,
        "stage_timings_ms": {"preprocess": 0.8, "inference": p50, "postprocess": 0.1},
    }


def report_of(*cases: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "20260101T000000Z-test",
        "created_at": "2026-01-01T00:00:00Z",
        "duration_seconds": 1.0,
        "cases": list(cases),
        "summary": {"cases_measured_under_contention": 0},
        "config": {"warmup": 3, "repetitions": 20, "accuracy_samples": 64, "threads": 1},
        "dataset": {
            "dataset_id": "synthetic-v1.0.0-seed1",
            "splits": [{"name": "test", "count": 8}],
        },
        "environment": {
            "hardware": "test CPU, 8 vCPU",
            "gpu": "none",
            "os_description": "Linux",
            "python_version": "3.12.3",
            "torch_threads": 1,
            "cpu_count_logical": 8,
            "library_versions": {"torch": "2.13.0+cpu"},
            "git": {"short_commit": "abc123", "branch": "main", "dirty": False},
        },
    }


# ------------------------------------------------------- contention bookkeeping


def test_a_case_that_produced_no_timing_is_not_reported_as_contended():
    """A skipped case records ``null``, and ``not None`` is true - which is how an
    unmeasured row ends up wearing a "the CPU was busy" mark it never earned."""
    assert _was_contended(case(status="skipped", trustworthy=None)) is False
    assert _contention_mark(case(status="skipped", trustworthy=None)) == ""


def test_only_an_explicit_false_counts_as_contended():
    assert _was_contended(case(trustworthy=False)) is True
    assert _was_contended(case(trustworthy=True)) is False
    assert _was_contended({}) is False


def test_a_contended_row_is_marked_and_a_quiet_one_is_not():
    assert _contention_mark(case(trustworthy=False)).strip() == "†"
    assert _contention_mark(case(trustworthy=True)) == ""


def test_the_contention_legend_is_printed_only_when_something_wears_the_mark():
    quiet = main_table(report_of(case(), case(status="skipped", trustworthy=None)))
    assert "†" not in quiet, "a legend for a mark that appears nowhere is a false footnote"

    contended = main_table(report_of(case(trustworthy=False)))
    assert "†" in contended


# ------------------------------------------------- the repeatability comparison


def _notes_for(sweep_trustworthy: bool, main_trustworthy: bool) -> str:
    """A sweep row and its main-table twin, 1.5x apart, with the given load verdicts."""
    sweep = case(name="threadscale-eager-t1", p50=20.0, threads=1, trustworthy=sweep_trustworthy)
    main = case(name="cutoutnet-fp32", p50=30.0, threads=None, trustworthy=main_trustworthy)
    return "\n".join(_sweep_consistency_notes(report_of(sweep, main), [sweep]))


def test_a_gap_between_two_quiet_rows_is_not_blamed_on_the_machine():
    """The harness sampled the load before both timing loops and both were idle. Calling
    that contention would contradict the run's own evidence."""
    notes = _notes_for(sweep_trustworthy=True, main_trustworthy=True)

    assert "1.5x apart" in notes
    assert "Both rows sampled an idle machine" in notes
    assert "order_effect.py" in notes, "the claim should cite the experiment that measured it"
    assert "not the same machine" not in notes


def test_a_gap_involving_a_contended_row_is_attributed_to_the_load():
    notes = _notes_for(sweep_trustworthy=True, main_trustworthy=False)

    assert "another workload" in notes
    assert "order_effect.py" not in notes


def test_rows_that_agree_produce_no_note_at_all():
    sweep = case(name="threadscale-eager-t1", p50=30.2, threads=1)
    main = case(name="cutoutnet-fp32", p50=30.0, threads=None)

    assert _sweep_consistency_notes(report_of(sweep, main), [sweep]) == []


def test_a_sweep_row_with_no_twin_is_not_compared_against_an_unrelated_model():
    sweep = case(name="threadscale-eager-t1", model="cutoutnet", p50=20.0, threads=1)
    other = case(name="u2netp", model="u2netp", p50=280.0, threads=None)

    assert _sweep_consistency_notes(report_of(sweep, other), [sweep]) == []


# ------------------------------------------------------------- table membership


def test_thread_sweep_rows_stay_out_of_the_headline_and_readme_tables():
    """They duplicate a model that already appears there at a different thread count, so
    including them would read as the same configuration measured twice."""
    report = report_of(
        case(name="cutoutnet-fp32", p50=30.0),
        case(name="threadscale-eager-t8", p50=7.4, threads=8),
    )

    assert "threadscale-eager-t8" not in main_table(report)
    assert "threadscale-eager-t8" not in readme_table(report)


def test_the_readme_table_reports_batch_one_only():
    """Mixing batch sizes into a single latency column compares responsiveness against
    throughput without saying so."""
    report = report_of(case(name="b1", p50=30.0), case(name="b8", p50=240.0, batch_size=8))
    table = readme_table(report)

    assert "30.0 ms" in table
    assert "| 8 |" not in table


# -------------------------------------------------------------- whole-doc render


def test_the_readme_region_is_replaced_between_its_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- BENCHMARKS:BEGIN -->\nstale\n<!-- BENCHMARKS:END -->\nafter\n")

    assert update_readme(readme, "fresh") is True
    text = readme.read_text()

    assert "stale" not in text
    assert "fresh" in text
    assert text.startswith("before")
    assert text.rstrip().endswith("after")


def test_a_readme_without_markers_is_left_alone_rather_than_appended_to(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")

    assert update_readme(readme, "fresh") is False
    assert readme.read_text() == "no markers here\n"


def test_the_methodology_states_the_ordering_caveat():
    """It is the only documented limit on comparing two rows of the same table, so it has
    to survive edits to the surrounding prose."""
    assert "order_effect.py" in METHODOLOGY
    assert "Position in the run" in METHODOLOGY


def test_the_committed_report_still_renders():
    """Guards against harness/renderer schema drift: the fixtures above would keep passing
    if the harness stopped emitting a field the renderer reads."""
    report = latest_report()
    if report is None:
        pytest.skip("no committed benchmark results to render")

    document = render_benchmarks_doc(report, METHODOLOGY)

    assert "# Benchmarks" in document
    assert report["run_id"] in document
    # Every case in the file must appear somewhere, whether it succeeded or was skipped.
    for entry in report["cases"]:
        assert entry["case"]["name"] in document or entry["case"]["model"] in document
