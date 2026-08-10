"""Benchmark harness, environment capture and Markdown rendering.

Lazy re-exports (PEP 562): reading a recorded report is a ``json.loads`` and the API does
it on ``GET /v1/benchmarks``, so importing this package must not drag in the harness -
and with it torch, a dataset and every model adapter. ``load_report`` and its siblings
resolve to :mod:`cutoutml.benchmarks.results`, which is framework-free.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import for type checkers only; see the module docstring
    from cutoutml.benchmarks.environment import Environment, capture, git_state, hardware_label
    from cutoutml.benchmarks.harness import (
        BenchmarkCase,
        BenchmarkConfig,
        BenchmarkHarness,
        CaseResult,
        LatencyStats,
    )
    from cutoutml.benchmarks.render_report import readme_table, render, render_benchmarks_doc
    from cutoutml.benchmarks.results import latest_report, load_report, save_report

_EXPORTS: dict[str, str] = {
    "BenchmarkCase": "cutoutml.benchmarks.harness",
    "BenchmarkConfig": "cutoutml.benchmarks.harness",
    "BenchmarkHarness": "cutoutml.benchmarks.harness",
    "CaseResult": "cutoutml.benchmarks.harness",
    "Environment": "cutoutml.benchmarks.environment",
    "LatencyStats": "cutoutml.benchmarks.harness",
    "capture": "cutoutml.benchmarks.environment",
    "git_state": "cutoutml.benchmarks.environment",
    "hardware_label": "cutoutml.benchmarks.environment",
    "latest_report": "cutoutml.benchmarks.results",
    "load_report": "cutoutml.benchmarks.results",
    "readme_table": "cutoutml.benchmarks.render_report",
    "render": "cutoutml.benchmarks.render_report",
    "render_benchmarks_doc": "cutoutml.benchmarks.render_report",
    "save_report": "cutoutml.benchmarks.results",
}

__all__ = [
    "BenchmarkCase",
    "BenchmarkConfig",
    "BenchmarkHarness",
    "CaseResult",
    "Environment",
    "LatencyStats",
    "capture",
    "git_state",
    "hardware_label",
    "latest_report",
    "load_report",
    "readme_table",
    "render",
    "render_benchmarks_doc",
    "save_report",
]


def __getattr__(name: str) -> Any:
    try:
        module_path = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    # importlib rather than `from . import harness`: the latter reaches this same hook to
    # resolve the name and recurses until the stack runs out.
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
