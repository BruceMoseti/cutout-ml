"""Reading and writing benchmark report files.

Producing a report needs torch, a dataset and a loaded model; reading one back needs
:mod:`json`. These three functions are the second kind, and they are separated from
:mod:`cutoutml.benchmarks.harness` because two callers only ever read: ``GET
/v1/benchmarks``, which must not import a deep-learning framework to serve a JSON file,
and the documentation renderer, which turns a recorded run into the tables in
``docs/benchmarks.md``.

A report is the unit of provenance in this repo - it carries the commit, the hardware,
the library versions and the dataset fingerprint that produced every number - so the
file layout is deliberately dull: one JSON document per run, named by its ``run_id``,
sorted keys, newline-terminated, so that a diff between two runs is readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cutoutml.core.config import get_settings
from cutoutml.core.logging import get_logger

log = get_logger(__name__)


def save_report(report: dict[str, Any], directory: Path | str | None = None) -> Path:
    """Write a report to ``benchmarks/results/<timestamp>-<id>.json``."""
    target_dir = Path(directory) if directory else get_settings().benchmark_results_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log.info("benchmark_report_saved", path=str(path))
    return path


def load_report(path: Path | str) -> dict[str, Any]:
    """Read a saved report."""
    return json.loads(Path(path).read_text())  # type: ignore[no-any-return]


def latest_report(directory: Path | str | None = None) -> dict[str, Any] | None:
    """Most recent report in ``directory`` by filename (timestamp-prefixed)."""
    target = Path(directory) if directory else get_settings().benchmark_results_dir
    files = sorted(target.glob("*.json"))
    if not files:
        return None
    return load_report(files[-1])
