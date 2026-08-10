"""Capture the environment a benchmark ran in.

A latency number without the machine, the commit and the library versions attached is
not reproducible and therefore not evidence. Everything here goes into the committed
result JSON so a reader can tell whether a number is comparable to theirs.

The git state includes a ``dirty`` flag. A benchmark run against uncommitted changes is
not attributable to any commit, and recording that honestly is more useful than
recording a commit hash that does not describe the code that ran.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil

from cutoutml.core.config import REPO_ROOT
from cutoutml.core.devices import cuda_available

TRACKED_PACKAGES = (
    "torch", "numpy", "opencv-python-headless", "onnx", "onnxruntime",
    "pillow", "scipy", "fastapi", "sqlalchemy", "celery",
)


@dataclasses.dataclass(frozen=True, slots=True)
class GitState:
    """Repository state at benchmark time."""

    commit: str | None
    short_commit: str | None
    branch: str | None
    dirty: bool
    describe: str | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Environment:
    """Full environment snapshot."""

    hardware: str
    cpu_model: str
    cpu_count_logical: int
    cpu_count_physical: int | None
    cpu_max_mhz: float | None
    total_ram_bytes: int
    available_ram_bytes: int
    os_description: str
    python_version: str
    torch_threads: int
    gpu: str
    gpu_count: int
    cuda_version: str | None
    cudnn_version: str | None
    library_versions: dict[str, str]
    git: GitState
    env_flags: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["git"] = self.git.as_dict()
        return data

    @property
    def is_gpu(self) -> bool:
        return self.gpu != "none"


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, cwd=cwd, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def git_state(repo: Path | None = None) -> GitState:
    """Read git metadata, tolerating a non-repo or missing git binary."""
    root = repo or REPO_ROOT
    commit = _run(["git", "rev-parse", "HEAD"], root)
    status = _run(["git", "status", "--porcelain"], root)
    return GitState(
        commit=commit,
        short_commit=commit[:12] if commit else None,
        branch=_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        # `status` is None on failure and "" on a clean tree - distinct cases.
        dirty=bool(status) if status is not None else False,
        describe=_run(["git", "describe", "--tags", "--always", "--dirty"], root),
    )


def cpu_model() -> str:
    """CPU model string from procfs, falling back to platform info."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or platform.machine() or "unknown"
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def gpu_description() -> tuple[str, int, str | None, str | None]:
    """``(name, count, cuda_version, cudnn_version)``; ``("none", 0, ...)`` off-GPU.

    Deliberately reports the literal string ``"none"`` rather than omitting the field:
    a benchmark table that silently drops the GPU column on a CPU-only machine invites
    the reader to assume a GPU was used.
    """
    if not cuda_available():
        return ("none", 0, None, None)
    import torch

    count = torch.cuda.device_count()
    names = sorted({torch.cuda.get_device_name(i) for i in range(count)})
    cudnn = None
    try:
        version = torch.backends.cudnn.version()
        cudnn = str(version) if version else None
    except Exception:  # pragma: no cover  # noqa: BLE001 - provenance capture must never fail a run
        cudnn = None
    return (", ".join(names), count, torch.version.cuda, cudnn)


def library_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of the packages that can move a benchmark number."""
    out: dict[str, str] = {}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def relevant_env_flags() -> dict[str, str]:
    """Environment variables that materially change performance.

    Thread-count variables are the usual reason two runs on the same machine differ by
    2x, so they are recorded rather than assumed.
    """
    keys = (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "TORCH_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "ORT_TENSORRT_FP16_ENABLE",
        "CUTOUTML_DEVICE", "CUTOUTML_PRECISION",
    )
    return {k: os.environ[k] for k in keys if k in os.environ}


def hardware_label() -> str:
    """One-line, honest hardware description for the README and docs.

    Explicitly says "no GPU" when there is none. The whole point is that a reader
    should never have to guess whether a latency figure came from an A100 or an 8-core
    cloud VM.
    """
    cores = psutil.cpu_count(logical=True) or 0
    physical = psutil.cpu_count(logical=False)
    ram_gb = round(psutil.virtual_memory().total / (1024**3))
    gpu, gpu_count, _, _ = gpu_description()
    core_text = f"{cores} vCPU" + (f" ({physical} physical cores)" if physical else "")
    gpu_text = f"{gpu_count}x {gpu}" if gpu != "none" else "no GPU (CPU-only)"
    return f"{cpu_model()}, {core_text}, {ram_gb} GB RAM, {gpu_text}"


def capture() -> Environment:
    """Snapshot the current environment."""
    import torch

    gpu, gpu_count, cuda_version, cudnn = gpu_description()
    freq = psutil.cpu_freq()
    memory = psutil.virtual_memory()
    return Environment(
        hardware=hardware_label(),
        cpu_model=cpu_model(),
        cpu_count_logical=psutil.cpu_count(logical=True) or 0,
        cpu_count_physical=psutil.cpu_count(logical=False),
        cpu_max_mhz=float(freq.max) if freq and freq.max else None,
        total_ram_bytes=int(memory.total),
        available_ram_bytes=int(memory.available),
        os_description=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=platform.python_version(),
        torch_threads=torch.get_num_threads(),
        gpu=gpu,
        gpu_count=gpu_count,
        cuda_version=cuda_version,
        cudnn_version=cudnn,
        library_versions=library_versions(),
        git=git_state(),
        env_flags=relevant_env_flags(),
    )
