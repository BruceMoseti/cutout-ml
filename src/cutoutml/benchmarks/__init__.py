"""Benchmark harness, environment capture and Markdown rendering."""

from cutoutml.benchmarks.environment import Environment, capture, git_state, hardware_label
from cutoutml.benchmarks.harness import (
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkHarness,
    CaseResult,
    LatencyStats,
    latest_report,
    load_report,
    save_report,
)
from cutoutml.benchmarks.render_report import readme_table, render, render_benchmarks_doc

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
