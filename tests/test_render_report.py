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

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from cutoutml.benchmarks.harness import latest_report
from cutoutml.benchmarks.render_report import (
    SmokeReportError,
    _contention_mark,
    _sweep_consistency_notes,
    _was_contended,
    checkpoint_table,
    contention_block,
    main_table,
    methodology_block,
    readme_table,
    render,
    render_benchmarks_doc,
    update_readme,
)
from cutoutml.core.config import REPO_ROOT, get_settings


def load_run_module() -> Any:
    """Import ``benchmarks/run.py``, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location("benchmarks_run", REPO_ROOT / "benchmarks/run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _ordering_report() -> dict[str, Any]:
    """A run whose sweep re-measures a main-table row and disagrees with it by 1.5x."""
    return report_of(
        case(name="threadscale-eager-t1", p50=20.0, threads=1),
        case(name="cutoutnet-fp32", p50=30.0, threads=None),
    )


def test_the_methodology_states_the_ordering_caveat():
    """It is the only documented limit on comparing two rows of the same table, so it has
    to survive edits to the surrounding prose."""
    block = methodology_block(_ordering_report())

    assert "order_effect.py" in block
    assert "Position in the run" in block


def test_the_ordering_caveat_quantifies_itself_from_the_run_it_describes():
    """Written down once, the spread would be quoted for runs that never measured it: it
    has already moved between the archived runs as the machine got quieter."""
    assert "1.5x apart" in methodology_block(_ordering_report())

    tighter = report_of(
        case(name="threadscale-eager-t1", p50=20.0, threads=1),
        case(name="cutoutnet-fp32", p50=24.0, threads=None),
    )
    block = methodology_block(tighter)

    assert "1.2x apart" in block
    assert "1.5x" not in block


def test_the_methodology_separates_how_latency_and_accuracy_are_measured():
    """The published claim was once that accuracy follows from the weights and the eval
    set, which says nothing about a baseline that has no weights. The rule that actually
    holds - timed as production runs it, scored from a fixed RNG state - has to be on the
    page, because it is the reason the classical rows are reproducible at all."""
    block = " ".join(methodology_block(report_of(case(name="classical-grabcut"))).split())

    assert "Latency and accuracy are measured under different rules" in block
    assert "resets that RNG immediately before every accuracy pass" in block
    assert "Nothing is reset inside the timed section" in block


def test_a_run_that_duplicates_no_configuration_makes_no_spread_claim():
    block = methodology_block(report_of(case(name="cutoutnet-fp32", p50=30.0)))

    assert "measures no configuration twice" in " ".join(block.split())
    assert "x apart" not in block


def test_the_shared_machine_limitation_is_listed_only_when_a_row_was_contended():
    """A caveat on a run that has nothing to caveat invites a reader to discount rows that
    carry no qualification, which is its own kind of dishonesty."""
    quiet = report_of(case())
    assert "The machine was shared" not in methodology_block(quiet)

    busy = report_of(case(trustworthy=False, external_cores=7.5))
    busy["summary"] = {"cases_measured_under_contention": 1}
    assert "The machine was shared" in methodology_block(busy)


# --------------------------------------------------- not rounding in your favour


def test_a_small_but_nonzero_external_load_is_not_rounded_away():
    """One decimal place printed a measured 0.04 busy cores as "0.0", which reads as "no
    external demand at all" - a stronger claim than the data makes, in the flattering
    direction, attached to every timing in the document."""
    report = report_of(case(external_cores=0.04))
    block = contention_block(report)

    assert "0.04 of 8 cores" in block
    assert "0.0 of 8 cores" not in block


def test_a_load_large_enough_to_show_keeps_one_decimal():
    assert "1.9 of 8 cores" in contention_block(report_of(case(external_cores=1.904)))
    assert "0.0 of 8 cores" in contention_block(report_of(case(external_cores=0.0)))


def test_the_calibration_floor_is_read_from_the_run_not_written_down():
    """`trivial-ones` predicts foreground everywhere, so its IoU is the eval set's mean
    foreground coverage. It was hardcoded as "~35% / 0.35 IoU" against a measured 0.3590,
    understating the floor every other row has to clear."""
    trivial = case(name="trivial-ones", model="trivial-ones", p50=0.1)
    trivial["accuracy"]["iou"] = 0.3590
    block = methodology_block(report_of(trivial))

    assert "35.9% of the frame" in block
    assert "0.3590 IoU" in block

    # A different eval set must move the sentence, not leave the old number behind.
    other = case(name="trivial-ones", model="trivial-ones", p50=0.1)
    other["accuracy"]["iou"] = 0.4812
    moved = methodology_block(report_of(other))
    assert "48.1% of the frame" in moved and "0.4812 IoU" in moved
    assert "35.9%" not in moved


def test_a_run_without_a_calibration_row_makes_no_specific_floor_claim():
    block = methodology_block(report_of(case()))
    assert "content-blind rows" in block
    assert "0.3590 IoU" not in block


# ------------------------------------------------------------ smoke-run guard


def test_a_smoke_run_is_refused_rather_than_published(tmp_path):
    """`--quick` is three repetitions over eight samples. Rendering one would replace every
    published figure with it *and* satisfy the CI check that re-renders and diffs, so the
    guard against hand-typed numbers would certify a smoke test."""
    report = report_of(case())
    report["config"] = {"repetitions": 3, "accuracy_samples": 8, "threads": 1, "smoke": True}
    path = tmp_path / "20260101T000000Z-test.json"
    path.write_text(json.dumps(report))

    with pytest.raises(SmokeReportError) as caught:
        render(path, docs_path=tmp_path / "benchmarks.md", readme_path=tmp_path / "README.md")

    assert "smoke run" in str(caught.value)
    # Refusing means writing nothing, not writing something and then complaining.
    assert not (tmp_path / "benchmarks.md").exists()


def test_a_report_without_the_smoke_flag_is_published_as_before(tmp_path):
    """The committed reports predate the flag, so its absence must mean 'publishable'."""
    report = report_of(case())
    assert "smoke" not in report["config"]
    path = tmp_path / "20260101T000000Z-test.json"
    path.write_text(json.dumps(report))

    docs, _ = render(path, docs_path=tmp_path / "benchmarks.md", readme_path=tmp_path / "none.md")

    assert docs.is_file()


def test_quick_keeps_its_report_out_of_the_archive_and_off_the_docs():
    """The two halves of the fix, asserted against the argument parsing rather than by
    running the suite: a smoke run is redirected out of the directory the renderer reads,
    and it does not re-render."""
    run = load_run_module()

    args = run.build_parser().parse_args(["--quick"])
    assert args.output_dir is None and not args.no_render

    resolved = run.resolve_quick_run(args)

    assert resolved.no_render is True
    assert resolved.output_dir == run.QUICK_RESULTS_DIR
    assert run.QUICK_RESULTS_DIR.parent == get_settings().benchmark_results_dir
    # An explicit destination is still honoured; the default is what changes.
    explicit = run.resolve_quick_run(
        run.build_parser().parse_args(["--quick", "--output-dir", "x"])
    )
    assert explicit.output_dir == Path("x")


# --------------------------------------------------------- checkpoint provenance


def with_weights(
    entry: dict[str, Any], *, filename: str, digest: str, source: str | None = None
) -> dict[str, Any]:
    """Attach the weights fields the harness records for a case that loaded a checkpoint."""
    entry["model_metadata"] = {
        **entry["model_metadata"],
        "weights_path": f"/repo/models/{filename}",
        "weights_sha256": digest,
        "weights_source_sha256": source,
    }
    return entry


def test_a_run_of_only_locally_trained_checkpoints_claims_no_conversion():
    """The column has to be absent, not empty. A "Converted from" heading over a table where
    nothing was converted invites the reader to look for a provenance chain that does not
    exist."""
    table = checkpoint_table(
        report_of(with_weights(case(), filename="cutoutnet-small.pt", digest="a" * 64))
    )

    assert "Converted from" not in table
    assert "`aaaaaaaaaaaaaaaa...`" in table


def test_a_converted_checkpoint_publishes_the_digest_that_survives_re_conversion():
    """The point of the column. `weights_sha256` names one conversion of the source artefact
    and moves when the conversion is re-run, so on its own it cannot distinguish new weights
    from a new afternoon - which is how a correct benchmark row came to look stale."""
    table = checkpoint_table(
        report_of(
            with_weights(
                case(name="u2net-pretrained", model="u2net"),
                filename="u2net.pt",
                digest="b" * 64,
                source="c" * 64,
            )
        )
    )

    assert "Converted from" in table
    assert "`cccccccccccccccc...`" in table
    # And the source is published *beside* the file digest rather than replacing it: the
    # file digest is still what a reader compares against the bytes they have.
    assert "`bbbbbbbbbbbbbbbb...`" in table


def test_a_trained_checkpoint_in_a_converted_runs_table_says_so_rather_than_blank():
    """A blank cell reads as "we did not record this". "trained in-repo" is the actual fact
    and is a different claim."""
    table = checkpoint_table(
        report_of(
            with_weights(
                case(name="u2net-pretrained", model="u2net"),
                filename="u2net.pt",
                digest="b" * 64,
                source="c" * 64,
            ),
            with_weights(case(), filename="cutoutnet-small.pt", digest="a" * 64),
        )
    )

    cutoutnet_row = next(line for line in table.splitlines() if "cutoutnet-small.pt" in line)
    assert "trained in-repo" in cutoutnet_row
    assert cutoutnet_row.count("|") == table.splitlines()[0].count("|")


def test_the_committed_report_still_renders():
    """Guards against harness/renderer schema drift: the fixtures above would keep passing
    if the harness stopped emitting a field the renderer reads."""
    report = latest_report()
    if report is None:
        pytest.skip("no committed benchmark results to render")

    document = render_benchmarks_doc(report)

    assert "# Benchmarks" in document
    assert report["run_id"] in document
    # Every case in the file must appear somewhere, whether it succeeded or was skipped.
    for entry in report["cases"]:
        assert entry["case"]["name"] in document or entry["case"]["model"] in document
