"""Measure how busy the machine is, so a latency number can be trusted or discounted.

Every latency figure in this repository was produced on a shared cloud VM. A benchmark
that ignores that publishes whatever the scheduler happened to give it: on an 8-vCPU box
carrying a load average of 20, a model's p50 can be several times its unloaded value, and
nothing in the timing distribution says so. The stddev widens, but a reader cannot
distinguish "this model has variable cost" from "this machine had other tenants".

So contention is measured rather than assumed away, and it is measured as *external* CPU
demand - total busy cores minus the cores this process tree is using - because the
harness's own consumption is the thing being measured and must not count against it.

What contention does and does not invalidate
--------------------------------------------
It invalidates latency, throughput and peak-memory figures, which are the only quantities
that depend on how much of the machine was available.

It does **not** touch accuracy. IoU, MAE, F-measure and boundary F1 are deterministic
functions of the weights and the eval set; they are bit-identical on a quiet machine and a
loaded one. That asymmetry is worth stating plainly, because it means a contended run
still produces publishable accuracy numbers - only the timing columns need a caveat.

The threshold is expressed in cores rather than as a load-average number so that it means
the same thing on a 4-vCPU and a 64-vCPU machine.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import psutil

#: External demand below this many busy cores is treated as a quiet machine. Half a core
#: allows for the usual background daemons without admitting a second real workload.
QUIET_EXTERNAL_CORES = 0.5

#: Sampling window. psutil needs a non-zero interval to compute a utilisation delta, and
#: shorter windows are dominated by whichever processes happen to be scheduled in them.
SAMPLE_SECONDS = 1.0


@dataclasses.dataclass(frozen=True, slots=True)
class LoadSnapshot:
    """CPU demand at one instant, split into this process tree's share and everyone else's."""

    logical_cpus: int
    load_average_1m: float | None
    load_average_5m: float | None
    load_average_15m: float | None
    total_busy_cores: float
    own_busy_cores: float
    external_busy_cores: float
    quiet_threshold_cores: float
    sample_seconds: float

    @property
    def quiet(self) -> bool:
        """Whether the machine was idle enough for a latency figure to mean something."""
        return self.external_busy_cores <= self.quiet_threshold_cores

    @property
    def summary(self) -> str:
        """One-line description for a log line or a docs footnote."""
        state = "quiet" if self.quiet else "CONTENDED"
        return (
            f"{state}: {self.external_busy_cores:.1f} of {self.logical_cpus} cores busy with "
            f"external work (load average {self.load_average_1m or float('nan'):.1f})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "quiet": self.quiet, "summary": self.summary}


def _own_busy_cores() -> float:
    """Cores consumed by this process and its children, in cores (1.0 == one full core).

    ``cpu_percent()`` without an interval returns the average since the process started,
    which is the right basis here: it is compared against a utilisation sample taken over
    the same machine, and a per-call interval would double the sampling cost for every
    child.
    """
    try:
        process = psutil.Process(os.getpid())
        members = [process, *process.children(recursive=True)]
    except psutil.Error:  # pragma: no cover - the process cannot fail to see itself
        return 0.0

    total = 0.0
    for member in members:
        try:
            total += member.cpu_percent() / 100.0
        except psutil.Error:
            # A child that exited between listing and sampling contributes nothing.
            continue
    return total


def sample(
    interval: float = SAMPLE_SECONDS, *, threshold: float = QUIET_EXTERNAL_CORES
) -> LoadSnapshot:
    """Measure CPU demand over ``interval`` seconds."""
    logical = psutil.cpu_count(logical=True) or 1
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):  # pragma: no cover - not available on every platform
        one = five = fifteen = None  # type: ignore[assignment]

    # Prime the per-process counters, then sample the machine over the same window.
    _own_busy_cores()
    total_percent = psutil.cpu_percent(interval=max(0.1, interval))
    own = _own_busy_cores()

    total_cores = total_percent / 100.0 * logical
    return LoadSnapshot(
        logical_cpus=logical,
        load_average_1m=one,
        load_average_5m=five,
        load_average_15m=fifteen,
        total_busy_cores=round(total_cores, 3),
        own_busy_cores=round(min(own, total_cores), 3),
        external_busy_cores=round(max(0.0, total_cores - own), 3),
        quiet_threshold_cores=threshold,
        sample_seconds=max(0.1, interval),
    )
