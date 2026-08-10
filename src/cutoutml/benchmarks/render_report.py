"""Render committed benchmark JSON into Markdown.

The README and ``docs/benchmarks.md`` tables are **generated**, never hand-written.
That is the mechanism that makes "every number is real" enforceable rather than
aspirational: there is no way to type a number into the README that did not come out
of a measurement, because the region between the marker comments is overwritten on
every render, and CI can re-render and diff.

Rows with random weights print ``n/a (random weights)`` in the accuracy columns. Rows
that were skipped (missing weights) are listed separately with the reason, rather than
silently omitted - a table that hides its own gaps is misleading.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Any

from cutoutml.core.config import REPO_ROOT
from cutoutml.core.logging import get_logger

log = get_logger(__name__)

README_BEGIN = "<!-- BENCHMARKS:BEGIN -->"
README_END = "<!-- BENCHMARKS:END -->"


def _fmt(value: float | int | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int) and digits == 0:
        return f"{value:,}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def _mib(value: int | None) -> str:
    if not value:
        return "n/a"
    return f"{value / (1024 * 1024):.1f} MiB"


#: Appended to every timing cell measured while another workload had the CPU. A marker on
#: the cell rather than a note under the table, because a reader scanning for a latency
#: figure will not read the note.
CONTENDED_MARK = " †"


def _was_contended(case: dict[str, Any]) -> bool:
    """Whether this case's *timings* were taken while something else used the CPU.

    Tested against ``False`` rather than for falsiness: a case that produced no timing at
    all - a skipped one, whose weights were missing - records ``null`` here, and ``not
    None`` is true, which would report an unmeasured case as a contended one and print the
    contention legend under a table that has nothing to mark.
    """
    return case.get("latency_trustworthy") is False


def _contention_mark(case: dict[str, Any]) -> str:
    return CONTENDED_MARK if _was_contended(case) else ""


def _is_thread_sweep(case: dict[str, Any]) -> bool:
    """Whether the case overrode the run-wide thread count.

    Sweep rows are held out of the main, README and runtime-comparison tables. They
    duplicate a model that already appears there, and mixing thread counts into a
    runtime comparison would silently turn it into a thread comparison.
    """
    return case["case"].get("threads") is not None


def _display_name(case: dict[str, Any], metadata: dict[str, Any] | None) -> str:
    name = case["case"]["model"]
    parts = [name]
    if metadata and metadata.get("runtime", "").startswith("onnxruntime"):
        provider = metadata["runtime"].split(":", 1)[-1].replace("ExecutionProvider", "")
        parts.append(f"ONNX/{provider}")
    if (case.get("compile") or {}).get("succeeded"):
        parts.append("compiled")
    if case["case"]["random_init"]:
        parts.append("random-init")
    return " ".join(parts)


def _runtime_label(case: dict[str, Any], metadata: dict[str, Any] | None) -> str:
    """What actually executed the row.

    The harness's own label is preferred over the adapter's ``metadata.runtime`` because
    only the former distinguishes eager from compiled - and, critically, distinguishes a
    successful compile from a failed one that silently fell back to eager. Reading the
    adapter's value here would print "pytorch" for both and quietly invite a reader to
    attribute an eager measurement to Inductor.
    """
    return str(case.get("runtime") or (metadata or {}).get("runtime") or "?")


def main_table(report: dict[str, Any]) -> str:
    """The headline table: accuracy + latency for every successful case."""
    rows: list[str] = [
        "| Model | Runtime | Precision | Batch | Threads | IoU | MAE | F-beta | Boundary F1 | "
        "p50 ms/img | p95 ms/img | img/s | Peak RSS | Model size |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if case["status"] != "ok" or _is_thread_sweep(case):
            continue
        meta = case.get("model_metadata") or {}
        lat = case.get("latency") or {}
        acc = case.get("accuracy")
        valid = case.get("accuracy_valid", False)

        if valid and acc:
            iou = _fmt(acc.get("iou"), 4)
            mae = _fmt(acc.get("mae"), 4)
            fbeta = _fmt(acc.get("f_beta"), 4)
            bf1 = _fmt(acc.get("boundary_f1"), 4)
        else:
            iou = mae = fbeta = bf1 = "n/a *"

        rows.append(
            "| {model} | {runtime} | {precision} | {batch} | {threads} | {iou} | {mae} | "
            "{fbeta} | {bf1} | {p50} | {p95} | {ips} | {rss} | {size} |".format(
                model=_display_name(case, meta),
                runtime=_runtime_label(case, meta),
                precision=case["case"]["precision"],
                batch=case["case"]["batch_size"],
                threads=lat.get("threads", "?"),
                iou=iou,
                mae=mae,
                fbeta=fbeta,
                bf1=bf1,
                p50=_fmt(lat.get("per_image_p50_ms"), 2) + _contention_mark(case),
                p95=_fmt(lat.get("p95_ms", 0) / max(1, lat.get("batch_size", 1)), 2),
                ips=_fmt(lat.get("throughput_images_per_second"), 1),
                rss=_mib(lat.get("peak_rss_bytes")),
                size=_mib(case.get("model_size_bytes")),
            )
        )
    rows.append("")
    rows.append(
        "`n/a *` = accuracy not measurable for this row: the network ran with **random "
        "weights** so that latency could still be benchmarked without a loadable "
        "checkpoint. Latency in those rows is real; accuracy is meaningless."
    )
    if any(_was_contended(c) for c in report["cases"]):
        rows.append("")
        rows.append(
            "`†` = measured while another workload held the CPU, so the figure is an upper "
            "bound rather than this model's cost. Accuracy columns are unaffected: they are "
            "deterministic in the weights and the eval set. See "
            "[Machine contention](#machine-contention)."
        )
    return "\n".join(rows)


def contention_block(report: dict[str, Any]) -> str:
    """Per-case external CPU demand, and what it does and does not invalidate."""
    cases = [c for c in report["cases"] if c.get("load")]
    if not cases:
        return "_This run predates contention measurement; its timings carry no load evidence._"

    contended = [c for c in cases if not c["load"]["quiet"]]
    peak = max(c["load"]["external_busy_cores"] for c in cases)
    cores = cases[0]["load"]["logical_cpus"]

    if not contended:
        return (
            f"Every case was measured on a quiet machine: external demand never exceeded "
            f"{peak:.1f} of {cores} cores. The latency figures are this hardware's."
        )

    rows = [
        f"**{len(contended)} of {len(cases)} timed cases were measured under contention.** "
        f"External demand peaked at {peak:.1f} of {cores} cores ({peak / cores:.0%} of the "
        "machine) while these timings were taken.",
        "",
        "The latency, throughput and peak-memory columns for those rows are therefore upper "
        "bounds on this hardware's cost, not measurements of it. They are published with the "
        "evidence attached rather than omitted, and marked `†` wherever they appear. Nothing "
        "here is corrected or extrapolated: a scaled-down number would be a guess.",
        "",
        "Accuracy is unaffected and is not qualified. IoU, MAE, F-measure and boundary F1 are "
        "deterministic functions of the weights and the eval set, and come out bit-identical "
        "whatever else the scheduler was doing.",
        "",
        "| Case | External cores busy | Load avg (1m) | Latency trustworthy |",
        "|---|---|---|---|",
    ]
    for case in cases:
        load = case["load"]
        rows.append(
            "| `{name}` | {external:.1f} / {cores} | {avg} | {ok} |".format(
                name=case["case"]["name"],
                external=load["external_busy_cores"],
                cores=load["logical_cpus"],
                avg=_fmt(load.get("load_average_1m"), 1),
                ok="yes" if load["quiet"] else "**no**",
            )
        )
    return "\n".join(rows)


def _duplicate_pairs(report: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every sweep row paired with the main-table row measuring the same configuration.

    Pairs are what make the sweep a free repeatability check, so both the note under the
    sweep and the methodology's ordering caveat are computed from them rather than from
    two independent walks of the report that could disagree.
    """

    def key(case: dict[str, Any]) -> tuple[str, int, int, str]:
        spec, lat = case["case"], case["latency"]
        # The runtime label belongs in the key: `cutoutnet` at batch 1 appears both eager
        # and Inductor-compiled, and comparing an eager sweep row against the compiled
        # one would report a compile speedup as measurement noise.
        return (
            spec["model"],
            spec["batch_size"],
            lat.get("threads", 0),
            _runtime_label(case, case.get("model_metadata")),
        )

    baseline: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    sweep: list[dict[str, Any]] = []
    for case in report["cases"]:
        if case["status"] != "ok" or not case.get("latency") or not case["latency"].get("p50_ms"):
            continue
        if _is_thread_sweep(case):
            sweep.append(case)
        else:
            baseline[key(case)] = case

    pairs = []
    for case in sweep:
        other = baseline.get(key(case))
        if other is not None and other["latency"].get("p50_ms"):
            pairs.append((case, other))
    return pairs


def _widest_duplicate_spread(
    report: dict[str, Any],
) -> tuple[float, dict[str, Any], dict[str, Any]] | None:
    """The duplicated configuration whose two measurements disagree most, if any.

    Returned as data rather than prose because two documents quote it - the note under
    the sweep and the methodology's ordering caveat - and a hand-written number in either
    would go stale the next time the suite runs.
    """
    worst: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for case, other in _duplicate_pairs(report):
        a, b = case["latency"]["p50_ms"], other["latency"]["p50_ms"]
        ratio = max(a, b) / min(a, b)
        if worst is None or ratio > worst[0]:
            worst = (ratio, case, other)
    return worst


def _sweep_consistency_notes(
    report: dict[str, Any], sweep: list[dict[str, Any]], *, tolerance: float = 1.25
) -> list[str]:
    """Cross-check each sweep row against the main-table row it duplicates.

    The sweep measures configurations the main table already contains, which makes it a
    free repeatability check - and on a contended machine the two disagree. Publishing
    the disagreement is the point: a reader who spots the same model at two different
    latencies would otherwise reasonably conclude one of them is wrong, when in fact
    both are real and the spread *is* the finding.
    """
    swept = {id(c) for c in sweep}
    notes: list[str] = []
    for case, other in _duplicate_pairs(report):
        if id(case) not in swept:
            continue
        lat, spec = case["latency"], case["case"]
        other_p50 = other["latency"]["p50_ms"]
        ratio = max(other_p50, lat["p50_ms"]) / min(other_p50, lat["p50_ms"])
        if ratio < tolerance:
            continue

        preamble = (
            f"- **Repeatability**: `{spec['model']}` at {lat['threads']} thread(s) measured "
            f"{_fmt(lat['p50_ms'], 1)} ms here and {_fmt(other_p50, 1)} ms in the table "
            f"above - {ratio:.1f}x apart for the same configuration."
        )
        # Which explanation is true is a question about the recorded load, not a house
        # style. Attributing a quiet-machine gap to contention would be inventing a cause
        # that this run's own per-case load samples contradict.
        if _was_contended(case) or _was_contended(other):
            notes.append(
                f"{preamble} At least one of the two was measured while another workload "
                "held the CPU, which is what the `†` marks record: neither figure is wrong, "
                "and the machine was not the same machine at the two moments."
            )
        else:
            notes.append(
                f"{preamble} Both rows sampled an idle machine, so contention does not "
                "account for it. What differs is where each sat in the run: latency here "
                "depends measurably on what ran earlier in the same process. "
                "`benchmarks/order_effect.py` isolates that effect - timing this "
                "configuration after a larger model, on a quiet machine, reproduces the "
                "faster figure with a standard deviation under 0.2 ms - and its result is "
                "archived under `benchmarks/results/experiments/`. Compare rows within "
                "this sweep, which ran back to back, rather than against the table above."
            )
    if notes:
        notes.insert(0, "")
    return notes


def thread_scaling_block(report: dict[str, Any]) -> str:
    """Intra-op thread scaling, and why this suite is single-threaded by default.

    Printing the curve is the only way a reader can tell that the one-thread default is a
    measured decision rather than a shrug. Which way the curve goes is a property of the
    machine, not of this repository: on a box with more runnable threads than cores
    PyTorch's scaling goes *backwards*, and on an idle one it does not. Both outcomes have
    been recorded here, so the explanatory prose below is selected by what the run being
    rendered actually measured rather than by which of the two was seen first.
    """
    sweep = [
        c
        for c in report["cases"]
        if c["status"] == "ok" and _is_thread_sweep(c) and c.get("latency")
    ]
    if not sweep:
        return "_This run contains no thread-scaling sweep; re-run without `--no-threads`._"

    groups: dict[str, list[dict[str, Any]]] = {}
    for case in sweep:
        groups.setdefault(_runtime_label(case, case.get("model_metadata")), []).append(case)

    rows = [
        "| Runtime | Threads | p50 ms | p95 ms | stddev ms | img/s | Speedup vs 1 thread |",
        "|---|---|---|---|---|---|---|",
    ]
    for runtime, cases in sorted(groups.items()):
        ordered = sorted(cases, key=lambda c: c["latency"]["threads"])
        single = next(
            (c["latency"]["p50_ms"] for c in ordered if c["latency"]["threads"] == 1), None
        )
        for case in ordered:
            lat = case["latency"]
            speedup = f"{single / lat['p50_ms']:.2f}x" if single and lat["p50_ms"] else "n/a"
            rows.append(
                "| {rt} | {t} | {p50} | {p95} | {sd} | {ips} | {su} |".format(
                    rt=runtime,
                    t=lat["threads"],
                    p50=_fmt(lat.get("p50_ms"), 1),
                    p95=_fmt(lat.get("p95_ms"), 1),
                    sd=_fmt(lat.get("stddev_ms"), 1),
                    ips=_fmt(lat.get("throughput_images_per_second"), 1),
                    su=speedup,
                )
            )

    rows += [
        "",
        "Within each runtime the weights, the batch size and the image are identical; the "
        "only variable is how many intra-op threads the runtime was given. Compare down a "
        "runtime's rows, not across runtimes - the two runtimes execute different code.",
        "",
    ]
    any_inverted = False
    for runtime, cases in sorted(groups.items()):
        ordered = sorted(cases, key=lambda c: c["latency"]["p50_ms"])
        best, worst = ordered[0], ordered[-1]
        if worst["latency"]["p50_ms"] <= best["latency"]["p50_ms"]:
            continue
        ratio = worst["latency"]["p50_ms"] / best["latency"]["p50_ms"]
        inverted = worst["latency"]["threads"] > best["latency"]["threads"]
        any_inverted = any_inverted or inverted
        verdict = (
            "more threads made it slower" if inverted else "threads bought what they should have"
        )
        rows += [
            f"- **{runtime}**: {ratio:.0f}x between its own extremes - "
            f"{_fmt(best['latency']['p50_ms'], 1)} ms at {best['latency']['threads']} "
            f"thread(s) against {_fmt(worst['latency']['p50_ms'], 1)} ms at "
            f"{worst['latency']['threads']} (`{worst['case']['name']}`). "
            f"That is, {verdict}.",
        ]
    rows += _sweep_consistency_notes(report, sweep)
    rows += ["", _barrier_explanation(any_inverted), ""]
    rows += [
        "Two consequences shape the rest of this document:",
        "",
        "1. **The suite runs single-threaded by default** (`--threads 1`). "
        + _single_thread_default_rationale(any_inverted),
        "2. **A runtime comparison must fix the thread count.** ONNX Runtime resolves a "
        "request of 0 to one thread per core while PyTorch has its own default, so an "
        "uncontrolled 'PyTorch vs ONNX' row pair can differ by eight threads before it "
        "differs by a runtime. The harness now passes one count to both.",
    ]
    return "\n".join(rows)


def _barrier_explanation(observed_here: bool) -> str:
    """The barrier-wait mechanism, stated as observed or as not observed.

    Printed unconditionally it would read as an account of the table above it, which on an
    idle machine that table contradicts. The mechanism is still worth documenting when it
    is absent, because it is why the suite pins a thread count at all - but it has to be
    labelled as a thing this run did not see.
    """
    mechanism = (
        "A U-Net forward pass is roughly a hundred parallel regions, each ending in a "
        "barrier, and a barrier cannot retire until every worker thread has been scheduled "
        "onto a core. Ask for eight threads on a machine whose cores are already committed "
        "and every one of those barriers waits on a descheduled thread, so the cost becomes "
        "a function of the scheduler rather than of the model. ONNX Runtime resists this "
        "better than PyTorch because it fuses the graph into far fewer parallel regions and "
        "controls its own spin-then-yield policy at each one."
    )
    if observed_here:
        return f"Where a runtime gets *slower* with more threads, the extra time is not arithmetic but waiting. {mechanism}"
    return (
        "No runtime regressed with more threads in this run, so nothing here needs "
        f"explaining away - but the mechanism that makes wide runs unsafe to publish from a "
        f"shared machine is worth stating, because it is why the harness pins a count. "
        f"{mechanism} Earlier runs of this same suite, taken while a neighbouring job held "
        "all eight cores, show exactly that regression; they are kept in "
        "`benchmarks/results/` for the comparison."
    )


def _single_thread_default_rationale(inverted_here: bool) -> str:
    """Why one thread, given what this run measured about wider ones."""
    if inverted_here:
        return (
            "One thread has no barriers to lose, which makes it the only CPU latency "
            "figure on a shared machine that means the same thing twice. It also "
            "understates what dedicated hardware would do, and that is the correct "
            "direction for a published number to be wrong in."
        )
    return (
        "Threads did pay off in this run, so the default is not a claim that they cannot: "
        "it is that a one-thread figure is the only one that does not silently encode the "
        "core count of whichever machine took it, and the only one whose reproducibility "
        "does not depend on that machine staying idle. It understates what dedicated "
        "hardware would do, and that is the correct direction for a published number to "
        "be wrong in."
    )


def stage_table(report: dict[str, Any]) -> str:
    """Per-stage breakdown, which is where surprises usually live."""
    rows = [
        "| Model | Preprocess ms | Inference ms | Postprocess ms | Refine ms | Cold start s |",
        "|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if case["status"] != "ok" or not case.get("stage_timings_ms") or _is_thread_sweep(case):
            continue
        stages = case["stage_timings_ms"]
        lat = case.get("latency") or {}
        rows.append(
            "| {m} | {pre} | {inf} | {post} | {ref} | {cold} |".format(
                m=_display_name(case, case.get("model_metadata")),
                pre=_fmt(stages.get("preprocess"), 2),
                inf=_fmt(stages.get("inference"), 2),
                post=_fmt(stages.get("postprocess"), 2),
                ref=_fmt(stages.get("refine"), 2),
                cold=_fmt(lat.get("cold_start_seconds"), 3),
            )
        )
    return "\n".join(rows)


def accuracy_detail_table(report: dict[str, Any]) -> str:
    """All accuracy metrics, for rows where accuracy is valid."""
    rows = [
        "| Model | IoU | Dice | MAE | F-beta | max F-beta | S-measure | Boundary F1 | BER | Precision | Recall |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if case["status"] != "ok" or not case.get("accuracy_valid") or not case.get("accuracy"):
            continue
        if _is_thread_sweep(case):
            continue
        a = case["accuracy"]
        rows.append(
            "| {m} | {iou} | {dice} | {mae} | {fb} | {fbm} | {sm} | {bf1} | {ber} | {p} | {r} |".format(
                m=_display_name(case, case.get("model_metadata")),
                iou=_fmt(a.get("iou"), 4),
                dice=_fmt(a.get("dice"), 4),
                mae=_fmt(a.get("mae"), 4),
                fb=_fmt(a.get("f_beta"), 4),
                fbm=_fmt(a.get("f_beta_max"), 4),
                sm=_fmt(a.get("s_measure"), 4),
                bf1=_fmt(a.get("boundary_f1"), 4),
                ber=_fmt(a.get("ber"), 4),
                p=_fmt(a.get("precision"), 4),
                r=_fmt(a.get("recall"), 4),
            )
        )
    return "\n".join(rows)


def runtime_comparison_table(report: dict[str, Any]) -> str:
    """Eager vs ``torch.compile`` vs ONNX Runtime, grouped so the delta is attributable.

    Grouped by (model, batch size) because a runtime comparison across different batch
    sizes is not a comparison. Speedup is expressed against the eager row in the same
    group, and a group without one prints no speedup rather than a ratio against whatever
    else happened to be present.
    """
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for case in report["cases"]:
        if case["status"] != "ok" or not case.get("latency") or _is_thread_sweep(case):
            continue
        spec = case["case"]
        # Random-weight rows are excluded: their latency is real, but they exist to price
        # an architecture, and mixing them in invites a comparison across weights.
        if spec["random_init"]:
            continue
        groups.setdefault((spec["model"].removesuffix("-onnx"), spec["batch_size"]), []).append(
            case
        )

    comparable = {k: v for k, v in groups.items() if len(v) > 1}
    if not comparable:
        return (
            "_This run contains no model measured under more than one runtime at the same "
            "batch size, so there is nothing to compare._"
        )

    rows = [
        "| Model | Batch | Runtime | Compiled | Codegen s | p50 ms/img | img/s | vs eager |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (model, batch), cases in sorted(comparable.items()):
        baseline = next(
            (
                c["latency"]["per_image_p50_ms"]
                for c in cases
                if c.get("runtime") == "pytorch-eager"
            ),
            None,
        )
        for case in sorted(cases, key=lambda c: c["latency"]["per_image_p50_ms"]):
            compile_outcome = case.get("compile") or {}
            per_image = case["latency"]["per_image_p50_ms"]
            speedup = (
                f"{baseline / per_image:.2f}x" if baseline and per_image else "n/a (no eager row)"
            )
            rows.append(
                "| {model} | {batch} | {rt} | {compiled} | {warm} | {p50} | {ips} | {speedup} |".format(
                    model=model,
                    batch=batch,
                    rt=_runtime_label(case, case.get("model_metadata")),
                    compiled=(
                        "yes"
                        if compile_outcome.get("succeeded")
                        else ("FAILED" if compile_outcome.get("attempted") else "-")
                    ),
                    warm=_fmt(compile_outcome.get("warm_seconds"), 1),
                    p50=_fmt(per_image, 2),
                    ips=_fmt(case["latency"]["throughput_images_per_second"], 1),
                    speedup=speedup,
                )
            )

    failures = [
        c
        for c in report["cases"]
        if (c.get("compile") or {}).get("attempted") and not c["compile"].get("succeeded")
    ]
    if failures:
        rows.append("")
        rows.append("Compilation failures (the row above fell back to eager execution):")
        rows.append("")
        for case in failures:
            rows.append(f"- `{case['case']['name']}`: {case['compile'].get('error')}")
    return "\n".join(rows)


def checkpoint_table(report: dict[str, Any]) -> str:
    """Which exact weights produced the accuracy figures above.

    A checkpoint *path* is not provenance: the file behind it is overwritten by the next
    training run. The digest is what lets a reader confirm that a published IoU came from
    the weights currently in the repository.
    """
    seen: dict[str, tuple[str, str]] = {}
    for case in report["cases"]:
        meta = case.get("model_metadata") or {}
        digest = meta.get("weights_sha256")
        if not digest or not case.get("accuracy_valid"):
            continue
        seen[str(meta.get("weights_path"))] = (str(meta.get("name")), str(digest))
    if not seen:
        return "_No row in this run loaded a checkpoint from disk._"
    rows = ["| Model | Weights | SHA-256 |", "|---|---|---|"]
    for path, (name, digest) in sorted(seen.items(), key=lambda kv: kv[1][0]):
        rows.append(f"| {name} | `{Path(path).name}` | `{digest[:16]}...` |")
    return "\n".join(rows)


def skipped_table(report: dict[str, Any]) -> str:
    """Cases that could not run, with the reason."""
    entries = [c for c in report["cases"] if c["status"] != "ok"]
    if not entries:
        return "_No cases were skipped or failed in this run._"
    rows = ["| Case | Status | Reason |", "|---|---|---|"]
    for case in entries:
        reason = (case.get("error") or case.get("notes") or "").replace("\n", " ")
        rows.append(f"| `{case['case']['name']}` | {case['status']} | {reason[:220]} |")
    return "\n".join(rows)


def _threads_description(report: dict[str, Any]) -> str:
    """The thread count in force, and a pointer to why it is what it is."""
    cfg = report.get("config", {})
    env = report["environment"]
    requested = cfg.get("threads")
    if requested is None:  # a run recorded before the setting existed
        return f"{env.get('torch_threads', '?')} (PyTorch default; not pinned by the harness)"
    if requested == 0:
        return f"each runtime's default, one per core ({env.get('torch_threads', '?')} for PyTorch)"
    suffix = " - see [Thread scaling](#thread-scaling)" if requested == 1 else ""
    return f"{requested} per runtime, pinned by the harness{suffix}"


def environment_block(report: dict[str, Any]) -> str:
    """The honesty section: exactly what hardware produced these numbers."""
    env = report["environment"]
    git = env.get("git", {})
    libs = env.get("library_versions", {})
    lib_text = ", ".join(f"{k} {v}" for k, v in sorted(libs.items()) if v != "not-installed")
    dirty = (
        " (**working tree dirty** - numbers are not attributable to this commit)"
        if git.get("dirty")
        else ""
    )
    return "\n".join(
        [
            f"- **Hardware**: {env['hardware']}",
            f"- **GPU**: {env['gpu']}"
            + (
                "  <-- all numbers below are CPU-only; no GPU was available on this machine"
                if env["gpu"] == "none"
                else ""
            ),
            f"- **OS / Python**: {env['os_description']} / Python {env['python_version']}",
            f"- **Intra-op threads**: {_threads_description(report)}",
            f"- **Git commit**: `{git.get('short_commit') or 'unknown'}`"
            f" on `{git.get('branch') or 'unknown'}`{dirty}",
            f"- **Libraries**: {lib_text}",
            f"- **Run id**: `{report['run_id']}` ({report['created_at']}, "
            f"{report['duration_seconds']} s wall clock)",
        ]
    )


def dataset_block(report: dict[str, Any]) -> str:
    dataset = report.get("dataset", {})
    lines = [
        f"- **Dataset id**: `{dataset.get('dataset_id', 'unknown')}`",
        f"- **Generator**: `{dataset.get('generator', 'unknown')}` "
        f"v{dataset.get('generator_version', '?')}",
        f"- **Master seed**: `{dataset.get('master_seed', '?')}`",
        f"- **Resolution**: {dataset.get('resolution', '?')}",
    ]
    if dataset.get("fingerprint"):
        lines.append(
            f"- **Content fingerprint**: `{dataset['fingerprint'][:32]}...` "
            f"(first {dataset.get('fingerprint_samples')} samples)"
        )
    splits = dataset.get("splits") or []
    if splits:
        lines.append("- **Splits**: " + ", ".join(f"{s['name']}={s['count']}" for s in splits))
    cfg = report.get("config", {})
    lines.append(
        f"- **Harness**: {cfg.get('warmup')} warmup + {cfg.get('repetitions')} timed "
        f"repetitions per case, {cfg.get('accuracy_samples')} accuracy samples"
    )
    return "\n".join(lines)


def _thread_rungs(report: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Per runtime, its one-thread rung and its widest rung, where both were measured."""
    sweep = [
        c
        for c in report["cases"]
        if c["status"] == "ok" and _is_thread_sweep(c) and c.get("latency")
    ]
    by_runtime: dict[str, list[dict[str, Any]]] = {}
    for case in sweep:
        by_runtime.setdefault(_runtime_label(case, case.get("model_metadata")), []).append(case)

    rungs = []
    for runtime, cases in sorted(by_runtime.items()):
        single = next((c for c in cases if c["latency"]["threads"] == 1), None)
        widest = max(cases, key=lambda c: c["latency"]["threads"])
        if single is None or widest is single or not single["latency"]["p50_ms"]:
            continue
        rungs.append((runtime, single, widest))
    return rungs


def _thread_inversion_evidence(report: dict[str, Any]) -> str:
    """The worst measured thread inversion, phrased as a clause, or nothing.

    Read out of the report rather than written down, because the README's rule is that
    every number in it traces to a committed JSON artifact. A run with no sweep gets no
    numbers rather than last run's.
    """
    worst: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for _runtime, single, widest in _thread_rungs(report):
        ratio = widest["latency"]["p50_ms"] / single["latency"]["p50_ms"]
        if ratio > 1 and (worst is None or ratio > worst[0]):
            worst = (ratio, single, widest)

    if worst is None:
        return ""
    _, single, widest = worst
    return (
        f" - the same weights measured {_fmt(single['latency']['p50_ms'], 1)} ms on "
        f"{single['latency']['threads']} thread and "
        f"{_fmt(widest['latency']['p50_ms'], 1)} ms on {widest['latency']['threads']}"
    )


def _single_thread_caveat(report: dict[str, Any]) -> str:
    """The README's single-threaded caveat, justified by what the run's sweep measured.

    The caveat itself never changes - a one-thread figure is a per-core cost either way -
    but the *reason* one thread is nevertheless what gets published does. Asserting that
    wide runs drown in barrier waits is only honest when the sweep shows them doing so; on
    an idle machine it shows the opposite, and the reason to pin one thread becomes
    comparability rather than collapse.
    """
    lead = (
        "**Latency is single-threaded**, so these are per-core costs and a dedicated "
        "machine would beat them"
    )
    tail = (
        "The measured curve, and the reasoning, are in "
        "[docs/benchmarks.md](docs/benchmarks.md#thread-scaling)."
    )
    inversion = _thread_inversion_evidence(report)
    if inversion:
        return (
            f"{lead}. That is deliberate: this box runs other tenants, and multi-threaded "
            f"timings on it are dominated by barrier waits rather than by the model"
            f"{inversion}. {tail}"
        )

    rungs = _thread_rungs(report)
    if not rungs:
        return (
            f"{lead}. One thread is what gets published because it is the only figure that "
            f"does not silently encode the core count of the machine that took it. {tail}"
        )

    # The weakest-scaling runtime, so the "would beat them" claim is supported by the
    # smallest measured margin rather than the most flattering one.
    runtime, single, widest = min(
        rungs, key=lambda r: r[1]["latency"]["p50_ms"] / r[2]["latency"]["p50_ms"]
    )
    ratio = single["latency"]["p50_ms"] / widest["latency"]["p50_ms"]
    scope = f" the weaker-scaling of the {len(rungs)} runtimes swept" if len(rungs) > 1 else ""
    return (
        f"{lead} - by at least the {ratio:.1f}x this run's own sweep measured, "
        f"{_fmt(single['latency']['p50_ms'], 1)} ms on {single['latency']['threads']} thread "
        f"against {_fmt(widest['latency']['p50_ms'], 1)} ms on "
        f"{widest['latency']['threads']} for `{runtime}`,{scope}. One thread is published "
        f"anyway, because it is the only figure that does not silently encode the core count "
        f"of the machine that took it, and because a multi-threaded figure on a shared box "
        f"tracks the other tenants rather than the model. {tail}"
    )


def readme_table(report: dict[str, Any]) -> str:
    """Compact table for the README, plus the honesty line and a link to the JSON."""
    rows = [
        "| Model | Runtime | IoU | MAE | p50 latency | Throughput | Size |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if case["status"] != "ok":
            continue
        meta = case.get("model_metadata") or {}
        lat = case.get("latency") or {}
        acc = case.get("accuracy")
        valid = case.get("accuracy_valid", False)
        if case["case"]["batch_size"] != 1 or _is_thread_sweep(case):
            continue
        rows.append(
            "| **{m}** | {rt} | {iou} | {mae} | {p50} ms{mark} | {ips} img/s | {size} |".format(
                m=_display_name(case, meta),
                rt=_runtime_label(case, meta),
                iou=_fmt(acc.get("iou"), 4) if valid and acc else "n/a *",
                mae=_fmt(acc.get("mae"), 4) if valid and acc else "n/a *",
                p50=_fmt(lat.get("per_image_p50_ms"), 1),
                mark=_contention_mark(case),
                ips=_fmt(lat.get("throughput_images_per_second"), 1),
                size=_mib(case.get("model_size_bytes")),
            )
        )

    env = report["environment"]
    summary = report.get("summary", {})
    threads = (report.get("config") or {}).get("threads")
    rows += [
        "",
        f"**Benchmark environment**: {env['hardware']}. "
        f"GPU: **{env['gpu']}**. "
        f"Every number above was measured by `benchmarks/run.py` on this machine - "
        f"none are copied from a paper or estimated.",
        "",
    ]
    if threads == 1:
        rows += [_single_thread_caveat(report), ""]
    rows += [
        "`n/a *` = the network ran with **random weights** (no checkpoint exists that this "
        "architecture can load), so its latency is real but accuracy is not measurable.",
    ]
    contended = int(summary.get("cases_measured_under_contention") or 0)
    if contended:
        peak = summary.get("peak_external_busy_cores")
        rows += [
            "",
            f"`†` = **measured under CPU contention.** A concurrent workload was using up to "
            f"{_fmt(peak, 1)} of this machine's {env['cpu_count_logical']} cores while "
            f"{contended} of the timed cases ran, so those latency and throughput figures are "
            "upper bounds rather than this hardware's cost. They are published with the "
            "per-case load evidence rather than quietly cleaned up. **The accuracy columns "
            "are unaffected** - they are deterministic in the weights and the eval set.",
        ]
    rows += [
        "",
        f"Source data: [`benchmarks/results/{report['run_id']}.json`]"
        f"(benchmarks/results/{report['run_id']}.json) - "
        f"regenerate with `make bench`. Full methodology and the per-case load table: "
        "[docs/benchmarks.md](docs/benchmarks.md).",
    ]
    return "\n".join(rows)


def render_benchmarks_doc(report: dict[str, Any], methodology: str | None = None) -> str:
    """Assemble the full ``docs/benchmarks.md``."""
    if methodology is None:
        methodology = methodology_block(report)
    return "\n".join(
        [
            "<!-- GENERATED FILE - do not edit by hand.",
            "     Produced by `python -m cutoutml.benchmarks.render_report`",
            f"     from benchmarks/results/{report['run_id']}.json. -->",
            "",
            "# Benchmarks",
            "",
            "## Environment",
            "",
            environment_block(report),
            "",
            "## Dataset",
            "",
            dataset_block(report),
            "",
            "## Results",
            "",
            main_table(report),
            "",
            "## Machine contention",
            "",
            contention_block(report),
            "",
            "## Thread scaling",
            "",
            thread_scaling_block(report),
            "",
            "## Runtime comparison",
            "",
            "The same weights at the same batch size under PyTorch eager, "
            "`torch.compile` (Inductor) and ONNX Runtime, so the difference between the",
            "rows is attributable to the runtime and nothing else. `Codegen s` is the",
            "one-off tracing and compilation cost, which the timed loop excludes.",
            "",
            runtime_comparison_table(report),
            "",
            "## Checkpoint provenance",
            "",
            checkpoint_table(report),
            "",
            "## Per-stage timing breakdown",
            "",
            "Where the wall clock actually goes for one image. Useful because the model",
            "is frequently not the bottleneck - preprocessing and alpha refinement are",
            "resolution-dependent while inference is fixed at the letterboxed size.",
            "",
            stage_table(report),
            "",
            "## Full accuracy metrics",
            "",
            accuracy_detail_table(report),
            "",
            "## Skipped / failed cases",
            "",
            skipped_table(report),
            "",
            methodology,
            "",
        ]
    )


#: Static methodology prose. The two placeholders are the parts that describe *this* run
#: rather than the harness, and are filled by ``methodology_block``; everything else is
#: true of any run and is therefore safe to write down.
METHODOLOGY_TEMPLATE = """## Methodology

### Why single-run timings are misleading

A number like "37 ms" from one `time.perf_counter()` pair around one forward pass is
close to useless, for four reasons that all apply on the machine these numbers came
from:

1. **The first call is not representative.** PyTorch and oneDNN choose convolution
   algorithms lazily and cache them; onnxruntime builds an execution plan; CUDA creates
   a context and autotunes. The first inference is routinely 2-50x the steady-state
   cost. The harness runs warmup iterations and *discards* them, reporting the first
   iteration separately as `first_inference_ms` and model load as
   `cold_start_seconds`.

2. **CUDA is asynchronous.** `model(x)` returns before the GPU has finished. Timing it
   without `torch.cuda.synchronize()` measures the launch overhead - often producing
   "0.4 ms" for work that takes 20 ms. The harness synchronises before starting and
   before stopping the clock. (On the CPU-only machine used here this is a no-op, but
   the code path is the same one a GPU run would take.)

3. **The distribution has a long right tail.** Frequency scaling, other tenants on a
   shared cloud VM, page faults and GC produce outliers. A mean absorbs them; a p99
   exposes them. The harness reports p50/p95/p99/mean/stddev/min/max, and the stddev is
   the number to look at first: if it is large relative to p50, the machine was not
   quiet and no other figure in the row should be trusted.

   Leaving that inference to the reader is not good enough, though - a wide stddev is
   equally consistent with "this model has variable cost" and "someone else had the
   CPU". So the harness also *measures* how busy the machine was, per case, and marks
   the rows where the answer makes their timings meaningless. See
   [Machine contention](#machine-contention) for this run's numbers.

4. **Batch size changes the meaning of "latency".** At batch 1 you measure
   *responsiveness*; at batch 8 you measure *throughput*, and per-image latency
   improves while the latency any individual request experiences gets worse. Both are
   reported, and per-image figures are always explicitly per-image.

5. **A CPU latency figure without a thread count is not a measurement.** The same
   weights on the same machine can differ by more than an order of magnitude depending
   on how many intra-op threads the runtime was given, and which direction they move in
   is a property of the machine rather than of the model - see
   [Thread scaling](#thread-scaling) for the curve this run measured and
   `benchmarks/results/README.md` for an archived run of the same sweep in which more
   threads were dramatically worse. Every row records the thread count the runtime
   actually ran with, taken from the runtime rather than from the request, because ONNX
   Runtime silently resolves a request of 0 to one thread per core.

{ordering_caveat}
### What is measured

- **Latency**: wall clock around `model.predict(tensor)` only - preprocessing and
  encoding are excluded here and reported separately in the stage breakdown, because
  they scale with the source image size rather than the model.
- **Throughput**: `batch_size / mean_latency`. For video, frames/s equals images/s
  because frames go through the identical path.
- **Peak RSS**: process resident set size after the run, from `psutil`. It includes the
  interpreter and loaded libraries (~250 MB for PyTorch), so compare *differences*
  between rows, not absolute values.
- **Intra-op threads**: the width the runtime actually ran at, read back from the
  runtime. Pinned to the same value for every runtime in a comparison, because
  otherwise the comparison is partly a thread-count comparison.
- **Machine contention**: busy cores attributable to processes outside this process
  tree, sampled immediately before each timing loop. Measured as *external* demand
  rather than as a raw load average so that the harness's own consumption does not count
  against it, and in cores rather than as a load-average figure so the threshold means
  the same thing on a 4-core and a 64-core machine. A case is treated as quiet below
  half a busy core.
- **Peak VRAM**: `torch.cuda.max_memory_allocated`, or `null` off-GPU.
- **Cold start**: wall clock of `model.load()` - weight loading, device transfer and
  graph/session construction. This is what a scale-from-zero request pays.
- **Model size**: on-disk checkpoint/graph size, or the in-memory parameter size when
  weights are random.

### Runtimes compared, and how a failure is reported

Three runtimes execute the *same* trained weights:

- **PyTorch eager** - the reference. Convolutions already go through oneDNN, which is
  why the compiled speedup below is smaller than a GPU reader might expect.
- **`torch.compile` (Inductor)** - traces the graph and generates C++. Two things make
  this easy to report dishonestly, so both are handled explicitly: the first call costs
  seconds to tens of seconds (recorded separately as `Codegen s` and excluded from the
  timed loop), and the compile can *fail* at runtime on a machine without a C++
  toolchain. A failure falls back to eager and is printed as `FAILED` with the exception,
  never as a compiled row - which is why the Runtime column comes from the harness rather
  than from the model adapter.
- **ONNX Runtime (CPU execution provider)** - a genuinely different implementation of
  the same graph. The export is asserted to compute the same function to within 2e-3 in
  `tests/test_registry.py`, so a runtime row cannot silently be a different model.

**TensorRT is implemented but unmeasured.** The adapter exists and is type-checked, but
building an engine requires a CUDA GPU, and no row is published for it. That is a gap,
not a result.

### Accuracy metrics

`IoU`, `Dice`, `MAE`, `F-beta` (beta^2 = 0.3, the salient-object-detection
convention), `max/mean F-beta` swept over 255 thresholds, `S-measure`, `Boundary F1`
within a 3 px tolerance, `BER`, precision and recall. Definitions and the reasoning
for each are in `src/cutoutml/core/metrics.py`.

Two metrics deserve attention because they disagree usefully:

- **IoU vs Boundary F1.** A star-shaped mask with thin spikes can score high IoU (most
  of the area is right) while Boundary F1 collapses (the spikes are wrong). Boundary F1
  is what correlates with "does this cutout look good".
- **IoU vs MAE.** IoU thresholds at 0.5 and is blind to how confident the model is.
  MAE uses the continuous alpha, so a model that produces correct-but-mushy soft edges
  is punished by MAE and not by IoU. For matting, MAE is the more honest number.

### Calibration references

The table includes deliberately content-blind rows (`trivial-ones`, `trivial-center`).
They exist because IoU is only interpretable relative to what predicting *nothing*
achieves: on a set where the foreground covers ~35% of the frame, "predict everything"
already scores 0.35 IoU. Any row that does not clearly beat those has learned nothing.
`classical` (GrabCut from a centred rectangle) is the strongest non-learned baseline
and is the number a learned model has to beat to be worth its weights.

### Honest limitations

- **The eval set is synthetic.** See
  [`docs/decisions/ADR-004-synthetic-dataset.md`](decisions/ADR-004-synthetic-dataset.md).
  Absolute numbers here are **not comparable to published DUTS/DIS5K results**. The
  same harness runs unchanged on real data via `cutoutml.datasets.real` - pass
  `--dataset-root /path/to/DUTS`.
- **No GPU was available.** Every measurement is CPU-only. The fp16/TensorRT code paths
  are implemented and type-checked but unmeasured here; rows for them are absent rather
  than estimated.
{contention_limitation}- **Latency here is single-threaded and therefore pessimistic.** These are per-core
  costs, not the best this hardware can do. A dedicated machine given one thread per
  core would be faster - by how much is a question this environment cannot answer, so
  no multi-threaded headline figure is published. [Thread scaling](#thread-scaling)
  shows what was measured instead.
- **Random-weight rows measure architecture cost, not quality.** Only BiRefNet is in that
  position: its official checkpoints target a Swin backbone whose shapes do not match this
  repository's reimplementation, so no download would help and its row shows real latency
  with `n/a` accuracy. U^2-Net's published weights *are* loaded here - see
  [docs/models.md](models.md) for the route.
- **The pretrained models are evaluated out of domain.** U^2-Net was trained on DUTS,
  a real-photograph saliency dataset, and is scored here against a synthetic eval set. It
  is expected to place below a small model trained in-repo on that eval set's own
  distribution, and it does. That ordering is a statement about the eval set, not about
  the models: read it as evidence that these synthetic numbers do not transfer to
  photographs, in either direction.
"""


def _wrap_item(number: int | None, text: str) -> str:
    """Wrap one methodology list item to the width the static prose around it uses.

    Generated prose sits in the middle of a hand-written numbered list, so it has to wrap
    like its neighbours; an unwrapped paragraph renders identically and reads as machine
    output in the diff, which is where anyone auditing these claims will see it.
    """
    prefix = f"{number}. " if number is not None else "   "
    return textwrap.fill(
        text, width=88, initial_indent=prefix, subsequent_indent="   ", break_on_hyphens=False
    )


def _ordering_caveat(report: dict[str, Any]) -> str:
    """Methodology item 6: how far apart the run's own duplicated rows landed.

    The size of the effect is a property of the run being rendered, so it is read out of
    the report. Writing it down once would make it a number nobody re-checks: the spread
    has already moved between the runs in ``benchmarks/results/`` as the machine got
    quieter, and a stale figure here would be quoted as this run's.
    """
    worst = _widest_duplicate_spread(report)
    if worst is None:
        return (
            _wrap_item(
                6,
                "**Position in the run changes the number, and it is not noise.** A case's "
                "measured latency depends on what ran earlier in the same process; "
                "`benchmarks/order_effect.py` measures the effect directly and its output "
                "is archived in `benchmarks/results/experiments/`. This run measures no "
                "configuration twice, so it carries no internal check of its own on the "
                "size of the effect - **compare rows that ran near each other**.",
            )
            + "\n"
        )

    ratio, sweep_case, main_case = worst
    model = sweep_case["case"]["model"]
    threads = sweep_case["latency"]["threads"]
    # Rounded up, so the printed bound is one the data satisfies rather than one it misses
    # by a rounding step. Expressed relative to each row's own median because the absolute
    # figure is meaningless without knowing which of the two rows it belongs to.
    noise = math.ceil(
        100
        * max(
            (c["latency"].get("stddev_ms") or 0.0) / c["latency"]["p50_ms"]
            for c in (sweep_case, main_case)
        )
    )
    if _was_contended(sweep_case) or _was_contended(main_case):
        cause = (
            "Part of that gap is contention, which the `†` marks record; the rest is "
            "position. Latency here depends on what ran earlier in the same process, and"
        )
    else:
        cause = (
            "That is not contention - the harness sampled the load before both timing "
            "loops and both were idle - and it is not warmup, which is discarded from "
            "both. It is where each sat in the run: latency here depends on what ran "
            "earlier in the same process, and"
        )

    first = _wrap_item(
        6,
        f"**Position in the run changes the number, and it is not noise.** This suite "
        f"measures `{model}` eager at batch 1 on {threads} thread twice - once in the main "
        f"table, once as the matching rung of the thread sweep - and the two land about "
        f"{ratio:.1f}x apart, each with a standard deviation inside {noise}% of its own "
        f"median. {cause} `benchmarks/order_effect.py` times the identical configuration "
        f"after each of several preludes and finds that one particular earlier model "
        f"reproduces the faster figure while others, including a much more expensive one, "
        f"change nothing.",
    )
    second = _wrap_item(
        None,
        "The mechanism is not established here, and two plausible ones were tested and "
        "rejected: pre-faulting up to 1 GiB of heap before the timed loop changes nothing, "
        "and running a compiled case first changes nothing. What follows for a reader is "
        "concrete regardless of the cause - **compare rows that ran near each other**, "
        "which is why the thread sweep is a self-contained block rather than figures "
        "scattered through the main table, and treat a "
        f"{ratio:.1f}x agreement between two distant rows as the floor on this harness's "
        "cross-row precision. The experiment's own output, with the per-arm load samples "
        "that rule out contention, is archived in `benchmarks/results/experiments/`.",
    )
    return f"{first}\n\n{second}\n"


def _contention_limitation(report: dict[str, Any]) -> str:
    """The shared-machine limitation, or nothing when no row was measured under load.

    Listing it unconditionally invites a reader to discount timings that carry no
    qualification, which is its own kind of dishonesty - the run either has contended rows
    or it does not, and the summary says which.
    """
    if not int((report.get("summary") or {}).get("cases_measured_under_contention") or 0):
        return ""
    return (
        "- **The machine was shared.** See [Machine contention](#machine-contention) for "
        "exactly\n  which rows this affects and by how much. Timings on a contended row are "
        "upper bounds;\n  accuracy is unaffected.\n"
    )


def methodology_block(report: dict[str, Any]) -> str:
    """The methodology section, with its run-specific figures filled from the report."""
    return METHODOLOGY_TEMPLATE.format(
        ordering_caveat=_ordering_caveat(report),
        contention_limitation=_contention_limitation(report),
    )


def render(
    report_path: Path | str | None = None,
    *,
    docs_path: Path | str | None = None,
    readme_path: Path | str | None = None,
) -> tuple[Path, Path | None]:
    """Render a report into ``docs/benchmarks.md`` and the README table."""
    from cutoutml.benchmarks.harness import latest_report, load_report

    report = load_report(report_path) if report_path else latest_report()
    if report is None:
        raise FileNotFoundError("no benchmark results found; run `python benchmarks/run.py` first")

    docs = Path(docs_path) if docs_path else REPO_ROOT / "docs" / "benchmarks.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(render_benchmarks_doc(report))
    log.info("benchmarks_doc_rendered", path=str(docs), run_id=report["run_id"])

    readme = Path(readme_path) if readme_path else REPO_ROOT / "README.md"
    updated: Path | None = None
    if readme.is_file():
        if update_readme(readme, readme_table(report)):
            updated = readme
            log.info("readme_table_updated", path=str(readme))
        else:
            log.warning(
                "readme_markers_missing",
                path=str(readme),
                hint=f"add {README_BEGIN} / {README_END} around the benchmark table",
            )
    return (docs, updated)


def update_readme(path: Path, table: str) -> bool:
    """Replace the region between the benchmark markers. Returns False if absent."""
    text = path.read_text()
    start = text.find(README_BEGIN)
    end = text.find(README_END)
    if start == -1 or end == -1 or end < start:
        return False
    new_text = text[: start + len(README_BEGIN)] + "\n" + table + "\n" + text[end:]
    path.write_text(new_text)
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render benchmark JSON into Markdown")
    p.add_argument("report", nargs="?", type=Path, help="path to a results JSON (default: latest)")
    p.add_argument("--docs", type=Path, default=None)
    p.add_argument("--readme", type=Path, default=None)
    p.add_argument("--print-table", action="store_true", help="print the README table only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_table:
        from cutoutml.benchmarks.harness import latest_report, load_report

        report = load_report(args.report) if args.report else latest_report()
        if report is None:
            print("no benchmark results found")
            return 1
        print(readme_table(report))
        return 0

    docs, readme = render(args.report, docs_path=args.docs, readme_path=args.readme)
    print(f"wrote {docs}")
    if readme:
        print(f"updated {readme}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "METHODOLOGY_TEMPLATE",
    "checkpoint_table",
    "contention_block",
    "main",
    "methodology_block",
    "readme_table",
    "render",
    "render_benchmarks_doc",
    "runtime_comparison_table",
    "thread_scaling_block",
    "update_readme",
]
