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


def _display_name(case: dict[str, Any], metadata: dict[str, Any] | None) -> str:
    name = case["case"]["model"]
    parts = [name]
    if metadata and metadata.get("runtime", "").startswith("onnxruntime"):
        provider = metadata["runtime"].split(":", 1)[-1].replace("ExecutionProvider", "")
        parts.append(f"ONNX/{provider}")
    if case["case"]["random_init"]:
        parts.append("random-init")
    return " ".join(parts)


def main_table(report: dict[str, Any]) -> str:
    """The headline table: accuracy + latency for every successful case."""
    rows: list[str] = [
        "| Model | Runtime | Precision | Batch | IoU | MAE | F-beta | Boundary F1 | "
        "p50 ms/img | p95 ms/img | img/s | Peak RSS | Model size |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if case["status"] != "ok":
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
            "| {model} | {runtime} | {precision} | {batch} | {iou} | {mae} | {fbeta} | {bf1} | "
            "{p50} | {p95} | {ips} | {rss} | {size} |".format(
                model=_display_name(case, meta),
                runtime=meta.get("runtime", "?"),
                precision=case["case"]["precision"],
                batch=case["case"]["batch_size"],
                iou=iou,
                mae=mae,
                fbeta=fbeta,
                bf1=bf1,
                p50=_fmt(lat.get("per_image_p50_ms"), 2),
                p95=_fmt(lat.get("p95_ms", 0) / max(1, lat.get("batch_size", 1)), 2),
                ips=_fmt(lat.get("throughput_images_per_second"), 1),
                rss=_mib(lat.get("peak_rss_bytes")),
                size=_mib(case.get("model_size_bytes")),
            )
        )
    rows.append("")
    rows.append(
        "`n/a *` = accuracy not measurable for this row: the network ran with **random "
        "weights** so that latency could still be benchmarked without downloadable "
        "checkpoints. Latency in those rows is real; accuracy is meaningless."
    )
    return "\n".join(rows)


def stage_table(report: dict[str, Any]) -> str:
    """Per-stage breakdown, which is where surprises usually live."""
    rows = [
        "| Model | Preprocess ms | Inference ms | Postprocess ms | Refine ms | Cold start s |",
        "|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if case["status"] != "ok" or not case.get("stage_timings_ms"):
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
            f"- **PyTorch threads**: {env['torch_threads']}",
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
        if case["case"]["batch_size"] != 1:
            continue
        rows.append(
            "| **{m}** | {rt} | {iou} | {mae} | {p50} ms | {ips} img/s | {size} |".format(
                m=_display_name(case, meta),
                rt=meta.get("runtime", "?"),
                iou=_fmt(acc.get("iou"), 4) if valid and acc else "n/a *",
                mae=_fmt(acc.get("mae"), 4) if valid and acc else "n/a *",
                p50=_fmt(lat.get("per_image_p50_ms"), 1),
                ips=_fmt(lat.get("throughput_images_per_second"), 1),
                size=_mib(case.get("model_size_bytes")),
            )
        )

    env = report["environment"]
    rows += [
        "",
        f"**Benchmark environment**: {env['hardware']}. "
        f"GPU: **{env['gpu']}**. "
        f"Every number above was measured by `benchmarks/run.py` on this machine - "
        f"none are copied from a paper or estimated.",
        "",
        "`n/a *` = the network ran with **random weights** (pretrained checkpoints are "
        "not downloadable in this environment), so its latency is real but accuracy is "
        "not measurable. See [docs/benchmarks.md](docs/benchmarks.md).",
        "",
        f"Source data: [`benchmarks/results/{report['run_id']}.json`]"
        f"(benchmarks/results/{report['run_id']}.json) - "
        f"regenerate with `make bench`.",
    ]
    return "\n".join(rows)


def render_benchmarks_doc(report: dict[str, Any], methodology: str) -> str:
    """Assemble the full ``docs/benchmarks.md``."""
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


METHODOLOGY = """## Methodology

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

4. **Batch size changes the meaning of "latency".** At batch 1 you measure
   *responsiveness*; at batch 8 you measure *throughput*, and per-image latency
   improves while the latency any individual request experiences gets worse. Both are
   reported, and per-image figures are always explicitly per-image.

### What is measured

- **Latency**: wall clock around `model.predict(tensor)` only - preprocessing and
  encoding are excluded here and reported separately in the stage breakdown, because
  they scale with the source image size rather than the model.
- **Throughput**: `batch_size / mean_latency`. For video, frames/s equals images/s
  because frames go through the identical path.
- **Peak RSS**: process resident set size after the run, from `psutil`. It includes the
  interpreter and loaded libraries (~250 MB for PyTorch), so compare *differences*
  between rows, not absolute values.
- **Peak VRAM**: `torch.cuda.max_memory_allocated`, or `null` off-GPU.
- **Cold start**: wall clock of `model.load()` - weight loading, device transfer and
  graph/session construction. This is what a scale-from-zero request pays.
- **Model size**: on-disk checkpoint/graph size, or the in-memory parameter size when
  weights are random.

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
- **Random-weight rows measure architecture cost, not quality.** U^2-Net and BiRefNet
  need pretrained checkpoints that could not be downloaded, so their rows show real
  latency with `n/a` accuracy.
"""


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
    docs.write_text(render_benchmarks_doc(report, METHODOLOGY))
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
    "METHODOLOGY",
    "main",
    "readme_table",
    "render",
    "render_benchmarks_doc",
    "update_readme",
]
