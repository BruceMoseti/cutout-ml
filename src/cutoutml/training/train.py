"""Training loop for every from-scratch architecture in this repository.

The loop itself knows nothing about networks: it takes a
:class:`~cutoutml.training.architectures.TrainableArch` and drives batches, a
schedule and a loss through it. That is what lets the same code, the same synthetic
data and the same metrics produce the CutoutNet capacity curve *and* the U^2-Net
comparison, so the rows of the benchmark table differ only in the architecture.

Constraints this loop is designed around: **8 CPU cores, no GPU, tens of minutes**.
That budget dictates every choice below.

* 256x256 inputs and a ~1.1M-parameter network, so a forward+backward step costs
  tens of milliseconds rather than seconds.
* The dataset is generated on the fly by worker processes. Generation costs ~19 ms
  per sample, which is comparable to a training step, so it is parallelised across
  DataLoader workers rather than pre-materialised to disk. A fixed seed per
  ``(split, index)`` keeps it reproducible regardless of worker count.
* **Cosine LR with warmup.** Warmup matters more than usual here: with random
  initialisation and BatchNorm, the first few hundred steps have wildly wrong
  running statistics, and a full-size LR during that window frequently diverges.
* **AMP-aware but honest about it.** ``GradScaler`` is only engaged for CUDA fp16;
  bf16 needs no scaling, and on CPU autocast is a no-op for speed in most builds.
  The code path exists and is correct for GPU users; the numbers in this repo were
  produced in fp32 on CPU.
* Metrics are appended to a JSON file per epoch - no TensorBoard, no wandb, nothing
  that needs a server or an account to read a training curve.

Determinism: seeds for Python/NumPy/torch, and ``use_deterministic_algorithms`` is
*not* forced because a few CPU kernels lack deterministic implementations and
raising there would be worse than the small nondeterminism. Runs are reproducible
to within floating-point reduction order.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cutoutml.core.config import REPO_ROOT
from cutoutml.core.devices import (
    Precision,
    describe_device,
    resolve_device,
    resolve_precision,
    to_memory_format,
)
from cutoutml.core.imaging import normalize
from cutoutml.core.logging import configure_logging, get_logger
from cutoutml.core.metrics import iou as iou_metric
from cutoutml.core.metrics import mae as mae_metric
from cutoutml.datasets.synthetic import SyntheticConfig, SyntheticSegmentationDataset
from cutoutml.training.architectures import ARCHITECTURES, Normalization, resolve_arch
from cutoutml.training.losses import LossWeights, SegmentationLoss

log = get_logger(__name__)

DEFAULT_ARCH = "cutoutnet-small"


@dataclasses.dataclass(slots=True)
class TrainConfig:
    """Everything that defines a training run. Serialised into the run JSON."""

    arch: str = DEFAULT_ARCH
    resolution: int = 256
    train_samples: int = 3072
    val_samples: int = 192
    epochs: int = 8
    batch_size: int = 16
    lr: float = 3e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 60
    seed: int = 1337
    num_workers: int = 6
    device: str = "auto"
    precision: Precision = "fp32"
    grad_clip: float = 1.0
    ema_decay: float = 0.0
    channels_last: bool = True
    loss: LossWeights = dataclasses.field(default_factory=LossWeights)
    output_dir: Path | None = None
    run_dir: Path = REPO_ROOT / "training" / "runs"
    dataset_seed: int = 20240817
    torch_threads: int = 0

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["loss"] = self.loss.as_dict()
        d["output_dir"] = str(self.output_dir) if self.output_dir else None
        d["run_dir"] = str(self.run_dir)
        return d

    def checkpoint_path(self) -> Path:
        """Where this run's best checkpoint is written.

        Defaults to the location the serving adapter looks in, so a finished run is
        immediately servable without a copy step.
        """
        arch = resolve_arch(self.arch)
        if self.output_dir is not None:
            return self.output_dir / Path(arch.checkpoint).name
        return REPO_ROOT / "models" / arch.checkpoint


class TensorDatasetWrapper(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Adapts the NumPy synthetic dataset to normalised torch tensors.

    The normalisation constants come from the architecture registry, which is also
    what the serving adapter declares. If the two ever disagree the model appears to
    train fine and then scores near chance at inference, so there is exactly one
    source for the constant and ``tests/unit/test_training_architectures.py``
    asserts the two agree for every registered architecture.
    """

    def __init__(self, base: SyntheticSegmentationDataset, normalization: Normalization) -> None:
        self.base = base
        self.mean, self.std = normalization

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, alpha = self.base[index]
        x = torch.from_numpy(normalize(image, self.mean, self.std))
        y = torch.from_numpy(np.ascontiguousarray(alpha))[None]
        return x, y


MIN_SHM_BYTES = 512 * 1024 * 1024
"""Below this much free ``/dev/shm``, multi-worker loading is not safe."""


def shm_bytes_available() -> int:
    """Free bytes on ``/dev/shm``, or 0 if it cannot be inspected."""
    try:
        stat = os.statvfs("/dev/shm")
    except OSError:
        return 0
    return int(stat.f_bavail * stat.f_frsize)


def safe_num_workers(requested: int) -> int:
    """Clamp DataLoader workers to what ``/dev/shm`` can actually support.

    Every tensor a worker hands to the main process goes through POSIX shared
    memory. Docker defaults ``/dev/shm`` to **64 MB**, and *both* of PyTorch's
    sharing strategies (``file_descriptor`` and ``file_system``) allocate there on
    Linux, so the usual "set the sharing strategy" advice does not help. Training
    then dies a few steps in with ``No space left on device``, which looks like a
    disk problem and is really an IPC one.

    Real fixes, in order of preference: run the container with ``--shm-size=1g``
    (docker-compose.yml does this), or remount
    (``mount -o remount,size=2G /dev/shm``). If neither has happened, falling back
    to in-process loading is slower but correct, and far better than crashing
    mid-run.
    """
    if requested <= 0:
        return 0
    available = shm_bytes_available()
    if available < MIN_SHM_BYTES:
        log.warning(
            "dataloader_workers_disabled",
            requested=requested,
            shm_available_mb=round(available / 1e6, 1),
            hint="run with --shm-size=1g or remount /dev/shm; loading in-process instead",
        )
        return 0
    return requested


def set_seeds(seed: int) -> None:
    """Seed every RNG that affects a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def cosine_lr(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> dict[str, float]:
    """Validation pass reporting the same IoU/MAE the benchmark harness uses.

    Sharing the metric implementation with the benchmark is deliberate: a training
    curve that improves under a bespoke metric while the reported benchmark metric
    stagnates is a trap worth engineering away.
    """
    model.eval()
    ious: list[float] = []
    maes: list[float] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        primary = logits[0] if isinstance(logits, (tuple, list)) else logits
        probs = torch.sigmoid(primary.float()).cpu().numpy()
        target = y.numpy()
        for i in range(probs.shape[0]):
            ious.append(iou_metric(probs[i, 0], target[i, 0]))
            maes.append(mae_metric(probs[i, 0], target[i, 0]))
    return {
        "val_iou": float(np.mean(ious)) if ious else 0.0,
        "val_mae": float(np.mean(maes)) if maes else 1.0,
        "val_samples": float(len(ious)),
    }


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run training and return the run record (also written to ``run_dir``)."""
    configure_logging(fmt="console")
    set_seeds(cfg.seed)
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)

    arch = resolve_arch(cfg.arch)
    device = resolve_device(cfg.device)
    precision = resolve_precision(cfg.precision, device)
    log.info(
        "train_start",
        arch=cfg.arch,
        device=str(device),
        precision=precision,
        threads=torch.get_num_threads(),
        resolution=cfg.resolution,
    )
    if not arch.cpu_feasible and device.type == "cpu":
        log.warning(
            "arch_not_cpu_feasible",
            arch=cfg.arch,
            note=(
                "this architecture is not sized for CPU training; the run will "
                "complete but is unlikely to reach a useful accuracy"
            ),
        )

    data_cfg = SyntheticConfig(resolution=(cfg.resolution, cfg.resolution))
    train_ds = TensorDatasetWrapper(
        SyntheticSegmentationDataset(
            count=cfg.train_samples, split="train", seed=cfg.dataset_seed, config=data_cfg
        ),
        arch.normalization,
    )
    val_ds = TensorDatasetWrapper(
        SyntheticSegmentationDataset(
            count=cfg.val_samples, split="val", seed=cfg.dataset_seed, config=data_cfg
        ),
        arch.normalization,
    )

    num_workers = safe_num_workers(cfg.num_workers)
    loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds, shuffle=True, **loader_kwargs
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=min(2, num_workers),
        drop_last=False,
    )

    model = arch.build().to(device)
    if cfg.channels_last:
        # 1.4x faster training steps on this 8-core CPU: oneDNN has NHWC kernels for
        # depthwise convolutions and a slower fallback for NCHW. Same layout is used
        # at inference (see TorchSegmentationModel.use_channels_last) so training and
        # serving exercise identical kernels.
        model = to_memory_format(model, torch.channels_last)
    param_count = sum(p.numel() for p in model.parameters())

    # No weight decay on BatchNorm parameters or biases: decaying a BN scale toward
    # zero actively fights the normalisation it is supposed to provide.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.999),
    )

    criterion = SegmentationLoss(cfg.loss, gradient_output_index=arch.gradient_output_index)
    use_scaler = precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs

    history: list[dict[str, Any]] = []
    best_iou = -1.0
    best_mae = 1.0
    best_path = cfg.checkpoint_path()
    best_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    started = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_started = time.perf_counter()
        running: dict[str, float] = {}
        seen = 0

        for x, y in train_loader:
            lr = cosine_lr(global_step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = precision != "fp32"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                enabled=autocast_enabled,
            ):
                outputs = model(x)
                loss, parts = criterion(outputs, y)

            if use_scaler:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value * x.shape[0]
            seen += x.shape[0]
            global_step += 1

        train_metrics = {f"train_{k}": v / max(1, seen) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_started

        record = {
            "epoch": epoch + 1,
            "lr": cosine_lr(global_step - 1, total_steps, cfg),
            "seconds": round(epoch_seconds, 3),
            "samples_per_second": round(seen / max(epoch_seconds, 1e-6), 2),
            **{k: round(v, 6) for k, v in train_metrics.items()},
            **{k: round(v, 6) for k, v in val_metrics.items()},
        }
        history.append(record)
        log.info("epoch_done", **record)

        if val_metrics["val_iou"] > best_iou:
            best_iou = val_metrics["val_iou"]
            best_mae = val_metrics["val_mae"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": cfg.arch,
                    "param_count": param_count,
                    "epoch": epoch + 1,
                    "val_iou": best_iou,
                    "val_mae": val_metrics["val_mae"],
                    "config": cfg.as_dict(),
                    "normalization": {
                        "mean": list(arch.normalization[0]),
                        "std": list(arch.normalization[1]),
                    },
                },
                best_path,
            )
            log.info("checkpoint_saved", path=str(best_path), val_iou=round(best_iou, 4))

    total_seconds = time.perf_counter() - started
    checkpoint_bytes = best_path.stat().st_size if best_path.is_file() else 0

    run = {
        "run_id": f"{cfg.arch}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "arch": cfg.arch,
        "serves_as": arch.serves_as,
        "config": cfg.as_dict(),
        "param_count": param_count,
        "device": str(device),
        "device_name": describe_device(device).name,
        "precision": precision,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "dataset": train_ds.base.manifest(
            {"train": cfg.train_samples, "val": cfg.val_samples}
        ).as_dict(),
        "normalization": {
            "mean": list(arch.normalization[0]),
            "std": list(arch.normalization[1]),
        },
        "total_seconds": round(total_seconds, 2),
        "best_val_iou": round(best_iou, 6),
        "best_val_mae": round(best_mae, 6),
        "checkpoint": str(best_path),
        "checkpoint_bytes": checkpoint_bytes,
        "history": history,
    }
    run_path = cfg.run_dir / f"{run['run_id']}.json"
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    # A stable filename so docs and the report generator can find the latest curve
    # for an architecture without globbing timestamps.
    (cfg.run_dir / f"{cfg.arch}-latest.json").write_text(json.dumps(run, indent=2) + "\n")
    log.info(
        "train_done",
        seconds=round(total_seconds, 1),
        best_val_iou=round(best_iou, 4),
        checkpoint=str(best_path),
        checkpoint_mb=round(checkpoint_bytes / 1e6, 2),
        run_json=str(run_path),
    )
    return run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a CutoutML architecture on the synthetic dataset"
    )
    p.add_argument("--arch", default=DEFAULT_ARCH, choices=sorted(ARCHITECTURES))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-samples", type=int, default=3072)
    p.add_argument("--val-samples", type=int, default=192)
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="input size; defaults to the architecture's design resolution",
    )
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--ssim-weight", type=float, default=0.0)
    p.add_argument("--edge-weight", type=float, default=0.5)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the checkpoint directory (default: the path the adapter loads)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arch = resolve_arch(args.arch)
    cfg = TrainConfig(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        resolution=args.resolution or arch.default_resolution,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        precision=args.precision,
        torch_threads=args.torch_threads,
        loss=LossWeights(ssim=args.ssim_weight, edge=args.edge_weight),
        output_dir=args.output_dir,
    )
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
