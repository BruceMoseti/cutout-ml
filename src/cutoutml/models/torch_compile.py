"""Optional ``torch.compile`` wrapping, with the outcome recorded rather than assumed.

``torch.compile`` is the cheapest possible inference optimisation to *try* - one call,
no model changes - and one of the easiest to report dishonestly. Three things go wrong
in practice:

* **It can fail at runtime.** Inductor generates C++ (and on CUDA, Triton) and shells
  out to a compiler. On a slim container with no toolchain the compile raises, and code
  that assumed success either crashes in production or silently swallows the error and
  reports "compiled" numbers that came from eager mode.
* **The first call is enormously expensive.** Tracing plus codegen costs seconds to
  tens of seconds. A benchmark that does not warm up past it measures the compiler.
* **It is not always faster.** On small CPU models the win is often nil or negative,
  because oneDNN already handles the convolutions and Inductor's gains come mostly
  from fusing the pointwise work around them.

So this module returns a :class:`CompileOutcome` describing exactly what happened, the
harness records it next to the timings, and a row can never claim to be compiled
unless the compile actually succeeded.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import torch

from cutoutml.core.logging import get_logger

log = get_logger(__name__)

CompileMode = str
"""One of ``default``, ``reduce-overhead`` or ``max-autotune`` in current PyTorch."""


@dataclasses.dataclass(frozen=True, slots=True)
class CompileOutcome:
    """What happened when compilation was attempted."""

    attempted: bool
    succeeded: bool
    mode: CompileMode | None
    backend: str | None
    #: Wall-clock cost of the first (tracing + codegen) forward pass.
    warm_seconds: float | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def runtime_label(self) -> str:
        """Label for the ``runtime`` column of a benchmark row."""
        if not self.attempted:
            return "pytorch-eager"
        if self.succeeded:
            return f"pytorch-compile:{self.backend}:{self.mode}"
        return "pytorch-eager (compile failed)"


def compile_module(
    module: torch.nn.Module,
    example: torch.Tensor,
    *,
    mode: CompileMode = "default",
    backend: str = "inductor",
    fullgraph: bool = False,
) -> tuple[torch.nn.Module, CompileOutcome]:
    """Compile ``module`` and force the codegen to happen now.

    ``torch.compile`` is lazy: it returns instantly and compiles on the first call with
    real shapes. Driving one forward pass here means the caller's warmup loop measures
    a compiled model rather than the compiler, and any failure surfaces at this call
    site where it can be reported, instead of inside a timing loop.

    Returns the original module unchanged when compilation fails, so a caller that
    ignores the outcome still gets a working model - it just is not compiled.
    """
    started = time.perf_counter()
    try:
        compiled = torch.compile(module, mode=mode, backend=backend, fullgraph=fullgraph)
        with torch.inference_mode():
            compiled(example)
        warm = time.perf_counter() - started
    except Exception as exc:
        # Deliberately broad: Inductor surfaces missing compilers, unsupported ops and
        # guard failures as several unrelated exception types, and none of them should
        # take down a benchmark run or an API worker.
        outcome = CompileOutcome(
            attempted=True,
            succeeded=False,
            mode=mode,
            backend=backend,
            warm_seconds=None,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
        log.warning("torch_compile_failed", mode=mode, backend=backend, error=outcome.error)
        return module, outcome

    outcome = CompileOutcome(
        attempted=True,
        succeeded=True,
        mode=mode,
        backend=backend,
        warm_seconds=round(warm, 3),
    )
    log.info("torch_compile_ok", mode=mode, backend=backend, warm_seconds=outcome.warm_seconds)
    return compiled, outcome


def not_attempted() -> CompileOutcome:
    """The outcome for an eager-mode case, so every row has the same shape."""
    return CompileOutcome(
        attempted=False, succeeded=False, mode=None, backend=None, warm_seconds=None
    )
