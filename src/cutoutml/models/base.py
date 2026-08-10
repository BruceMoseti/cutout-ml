"""The ``SegmentationModel`` contract every backend implements.

The point of this interface is that the API, the pipelines and the benchmark
harness are written against it exactly once, and adding a runtime (ONNX,
TensorRT) or an architecture (U^2-Net, BiRefNet, CutoutNet) never touches them.
See ``docs/decisions/ADR-001-model-registry.md``.

Lifecycle
---------
``__init__`` is cheap and must not touch the filesystem or allocate a device;
``load()`` does all expensive work and is idempotent. Everything after that is
``preprocess -> predict -> postprocess``, where ``predict`` operates on batched
tensors and knows nothing about files.

Shapes
------
* ``preprocess(images)`` takes a list of ``(H, W, 3)`` uint8 RGB arrays and
  returns ``(N, 3, h, w)`` float32 plus one :class:`LetterboxInfo` per image.
* ``predict(tensor)`` returns **logits** shaped ``(N, 1, h, w)``. Returning
  logits rather than probabilities keeps the sigmoid in one place and lets the
  training loop reuse the same forward path with a numerically stable loss.
* ``postprocess(logits, infos)`` returns one ``float32`` alpha map per image at
  the original resolution.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cutoutml.core.devices import (
    Precision,
    autocast_context,
    describe_device,
    resolve_device,
    resolve_precision,
    to_memory_format,
)
from cutoutml.core.imaging import LetterboxInfo, letterbox, normalize, unletterbox_mask
from cutoutml.core.logging import get_logger

# Re-exported: the declarative half of the contract lives in a torch-free module so that
# the API can read the catalogue without importing a framework. See
# :mod:`cutoutml.models.spec`.
from cutoutml.models.spec import (
    ModelMetadata,
    ModelSpec,
    WeightsUnavailableError,
    weights_digest,
)

log = get_logger(__name__)

__all__ = [
    "ModelMetadata",
    "ModelSpec",
    "SegmentationModel",
    "TorchSegmentationModel",
    "WeightsUnavailableError",
    "load_state_dict",
    "weights_digest",
]


class SegmentationModel(abc.ABC):
    """Abstract base for every segmentation backend."""

    #: Filled in by the registry so metadata can report where the model came from.
    spec: ModelSpec | None = None

    def __init__(
        self,
        *,
        name: str,
        input_size: tuple[int, int] = (320, 320),
        device: str | torch.device | None = "auto",
        precision: Precision = "fp32",
        weights_path: Path | str | None = None,
        random_init: bool = False,
    ) -> None:
        self.name = name
        self.input_size = input_size
        self.requested_device = device
        self.device = resolve_device(device)
        self.requested_precision: Precision = precision
        self.precision: Precision = resolve_precision(precision, self.device)
        self.weights_path = Path(weights_path) if weights_path else None
        self.random_init = random_init
        self._loaded = False
        self._load_seconds: float | None = None
        #: Whatever the checkpoint recorded about where it came from, empty for one that
        #: recorded nothing. See :func:`load_checkpoint` for why this is worth keeping.
        self._weights_provenance: dict[str, Any] = {}

    # ------------------------------------------------------------------ loading

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_seconds(self) -> float | None:
        """Wall-clock cost of :meth:`load`, reported as cold-start time."""
        return self._load_seconds

    def load(self) -> SegmentationModel:
        """Materialise the model. Idempotent; returns ``self`` for chaining."""
        if self._loaded:
            return self
        started = time.perf_counter()
        self._load()
        self._load_seconds = time.perf_counter() - started
        self._loaded = True
        log.info(
            "model_loaded",
            model=self.name,
            device=str(self.device),
            precision=self.precision,
            seconds=round(self._load_seconds, 4),
        )
        return self

    @abc.abstractmethod
    def _load(self) -> None:
        """Backend-specific loading. Called at most once by :meth:`load`."""

    def unload(self) -> None:
        """Release device memory. Safe to call on an unloaded model."""
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"model '{self.name}' used before load(); call load() first")

    # ------------------------------------------------------------- preprocessing

    def preprocess(
        self, images: Sequence[np.ndarray] | np.ndarray
    ) -> tuple[torch.Tensor, list[LetterboxInfo]]:
        """Letterbox + normalise a batch of RGB uint8 arrays into a tensor.

        Accepts a single ``(H, W, 3)`` array for convenience; the returned tensor
        always has a batch dimension.
        """
        batch = [images] if isinstance(images, np.ndarray) and images.ndim == 3 else list(images)
        arrays: list[np.ndarray] = []
        infos: list[LetterboxInfo] = []
        for img in batch:
            source = np.asarray(img)
            # The divisor is computed from the source pixels, not the letterboxed
            # canvas, so the constant padding cannot change it.
            divisor = self.intensity_divisor(source)
            padded, info = letterbox(source, self.input_size)
            arrays.append(normalize(padded, *self.normalization, scale_by=divisor))
            infos.append(info)
        tensor = torch.from_numpy(np.stack(arrays, axis=0))
        return tensor, infos

    @property
    def normalization(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """``(mean, std)`` used by :meth:`preprocess`. Override per architecture."""
        from cutoutml.core.imaging import IMAGENET_MEAN, IMAGENET_STD

        return IMAGENET_MEAN, IMAGENET_STD

    def intensity_divisor(self, image: np.ndarray) -> float | None:  # noqa: ARG002 - hook
        """Extra per-image divisor applied to ``[0, 1]`` pixels before mean/std.

        ``None`` - the default - means plain ImageNet normalisation. Override only to
        reproduce a reference pipeline that does something else; see
        :meth:`cutoutml.models.u2net.adapter.U2NetAdapter.intensity_divisor`.
        """
        return None

    # ---------------------------------------------------------------- prediction

    @abc.abstractmethod
    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the network. Input ``(N, 3, h, w)``, output logits ``(N, 1, h, w)``."""

    def postprocess(
        self,
        logits: torch.Tensor,
        infos: Sequence[LetterboxInfo],
    ) -> list[np.ndarray]:
        """Sigmoid + un-letterbox each logit map back to its original size."""
        return self.alpha_from_probabilities(torch.sigmoid(logits.detach().float()), infos)

    def alpha_from_probabilities(
        self,
        probs: torch.Tensor,
        infos: Sequence[LetterboxInfo],
    ) -> list[np.ndarray]:
        """Un-letterbox already-activated probabilities back to original resolution.

        Split out from :meth:`postprocess` for backends whose artefact bakes the
        sigmoid into the graph, which must not apply a second one - see
        :class:`cutoutml.models.onnx_adapter.OnnxAdapter`.
        """
        probs = probs.detach().float()
        if probs.ndim == 4:
            probs = probs[:, 0]
        arr = probs.cpu().numpy()
        if arr.shape[0] != len(infos):
            raise ValueError(f"got {arr.shape[0]} predictions for {len(infos)} letterbox infos")
        return [unletterbox_mask(arr[i], infos[i]) for i in range(len(infos))]

    # ----------------------------------------------------------------- one-shot

    def infer(self, images: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
        """Convenience ``preprocess -> predict -> postprocess`` for RGB arrays."""
        self._ensure_loaded()
        tensor, infos = self.preprocess(images)
        logits = self.predict(tensor)
        return self.postprocess(logits, infos)

    # ----------------------------------------------------------------- metadata

    @abc.abstractmethod
    def metadata(self) -> ModelMetadata:
        """Describe this instance for API responses and benchmark records."""

    def _base_metadata_kwargs(self) -> dict[str, Any]:
        """Shared metadata fields, so subclasses only fill in what differs."""
        info = describe_device(self.device)
        return {
            "name": self.name,
            "input_size": self.input_size,
            "precision": self.precision,
            "device": str(self.device),
            "device_name": info.name,
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "weights_sha256": weights_digest(self.weights_path),
            # The digest of what this checkpoint was converted from, when it was converted
            # from anything. This is the identity a reader can reproduce: converting the
            # same ONNX twice yields two files with two digests and identical weights, so
            # `weights_sha256` alone cannot tell "these are different weights" from "this
            # was converted on a different afternoon".
            "weights_source_sha256": self._weights_provenance.get("source_sha256"),
            "randomly_initialized": self.random_init,
            "accuracy_valid": not self.random_init,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, device={self.device}, "
            f"precision={self.precision}, loaded={self._loaded})"
        )


class TorchSegmentationModel(SegmentationModel):
    """Base class for adapters backed by an ``nn.Module``.

    Centralises the parts that are identical for every PyTorch architecture:
    device placement, ``eval()``, ``inference_mode()``, autocast, checkpoint
    loading and parameter counting.

    Note the deliberate combination in :meth:`predict`: ``model.eval()`` is set
    once at load time (it switches BatchNorm to running statistics and disables
    Dropout), while ``torch.inference_mode()`` wraps each call (it disables
    autograd bookkeeping and tensor version counters). They are orthogonal, and
    both are needed: ``eval()`` alone still builds a graph and still tracks
    versions, and ``inference_mode()`` alone would leave BatchNorm updating its
    running statistics from whatever batch a request happened to contain.
    """

    module: torch.nn.Module | None

    #: NHWC ("channels last") weight/activation layout. On CPU this measurably
    #: speeds up depthwise-separable convolutions, because oneDNN has NHWC kernels
    #: for them and falls back to a slower path for NCHW; on CUDA it is what the
    #: tensor-core kernels want. Measured 1.4-1.6x on cutoutnet-small training steps
    #: (batch 4, 256x256, 8 threads) over three alternating runs on this 8-core CPU -
    #: the spread is run-to-run variance on a shared machine, not a range across
    #: configurations, and it is a micro-benchmark rather than a committed artifact.
    #: Disabled for architectures that gain nothing (see U2NetAdapter).
    use_channels_last: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.module = None

    @abc.abstractmethod
    def build_module(self) -> torch.nn.Module:
        """Instantiate the architecture with random weights."""

    def transform_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Hook for adapters whose checkpoints use different parameter names.

        A hook rather than an overridden :meth:`_load`: the loader also resolves the
        device and the memory format, and an adapter that reimplemented it to add key
        remapping silently lost those steps - which is exactly what happened to
        :class:`~cutoutml.models.u2net.adapter.U2NetAdapter`.
        """
        return state

    def _load(self) -> None:
        module = self.build_module()
        if not self.random_init:
            path = self.resolve_weights_path()
            loaded, provenance = load_checkpoint(path, map_location="cpu")
            self._weights_provenance = provenance
            state = self.transform_state_dict(loaded)
            missing, unexpected = module.load_state_dict(state, strict=False)
            if missing or unexpected:
                log.warning(
                    "state_dict_mismatch",
                    model=self.name,
                    missing=len(missing),
                    unexpected=len(unexpected),
                    missing_sample=list(missing)[:5],
                    unexpected_sample=list(unexpected)[:5],
                )
            self.weights_path = path
        else:
            log.warning(
                "random_init_model",
                model=self.name,
                warning="weights are random; latency is valid, accuracy is NOT",
            )
        module.eval()
        module.to(self.device)
        if self.use_channels_last:
            module = to_memory_format(module, torch.channels_last)
        self.module = module

    def resolve_weights_path(self) -> Path:
        """Locate the checkpoint, raising :class:`WeightsUnavailableError` if absent."""
        if self.weights_path is not None and self.weights_path.is_file():
            return self.weights_path
        expected = self.weights_path or Path(self.default_weights_hint())
        raise WeightsUnavailableError(self.name, expected, self.weights_hint())

    def default_weights_hint(self) -> str:
        return f"models/{self.name}/{self.name}.pt"

    def weights_hint(self) -> str:
        return (
            "Run `python -m cutoutml.models.download_weights --model "
            f"{self.name}` to fetch them, or pass random_init=True for "
            "latency-only benchmarking (accuracy will be meaningless)."
        )

    def unload(self) -> None:
        self.module = None
        super().unload()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        assert self.module is not None
        tensor = tensor.to(self.device, non_blocking=True)
        if self.use_channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), autocast_context(self.device, self.precision):
            out = self.module(tensor)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return logits.float()

    def param_counts(self) -> tuple[int, int]:
        """``(total, trainable)`` parameter counts; ``(0, 0)`` before load."""
        if self.module is None:
            return (0, 0)
        total = sum(p.numel() for p in self.module.parameters())
        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        return (total, trainable)

    def metadata(self) -> ModelMetadata:
        total, trainable = self.param_counts()
        spec = self.spec
        return ModelMetadata(
            architecture=spec.architecture if spec else type(self).__name__,
            param_count=total,
            trainable_param_count=trainable,
            runtime="pytorch",
            license=spec.license if spec else "unknown",
            source=spec.source if spec else "",
            notes=(
                "RANDOM WEIGHTS - latency measurements are valid, accuracy is not."
                if self.random_init
                else ""
            ),
            **self._base_metadata_kwargs(),
        )

    def to_onnx(
        self,
        output_path: Path | str,
        *,
        opset: int = 17,
        dynamic_batch: bool = True,
    ) -> Path:
        """Export to ONNX with a fixed spatial size and optional dynamic batch.

        The spatial dimensions stay static on purpose: dynamic H/W forces
        onnxruntime to re-plan memory on every new shape and blocks most graph
        optimisations, and our pipelines always letterbox to a fixed size anyway.
        """
        self._ensure_loaded()
        assert self.module is not None
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        w, h = self.input_size
        # Export from a contiguous NCHW graph: onnxruntime chooses its own layout,
        # and exporting from channels_last bakes in transposes it would otherwise
        # optimise away.
        dummy = torch.zeros(1, 3, h, w, dtype=torch.float32, device=self.device).contiguous()
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
        wrapper = to_memory_format(_SingleOutputWrapper(self.module), torch.contiguous_format)
        wrapper.eval()
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(out),
                input_names=["input"],
                output_names=["logits"],
                opset_version=opset,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                dynamo=False,
            )
        log.info("onnx_exported", model=self.name, path=str(out), opset=opset)
        return out


class _SingleOutputWrapper(torch.nn.Module):
    """Collapse a multi-side-output network to its fused output for export.

    U^2-Net style networks return six side outputs plus a fusion; ONNX consumers
    only need the fusion, and exporting the rest bloats the graph.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def load_checkpoint(
    path: Path | str, map_location: str = "cpu"
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a checkpoint, unwrapping the common container shapes, with its provenance.

    ``weights_only=True`` is passed so a malicious checkpoint cannot execute
    arbitrary code during unpickling - relevant because weights may be
    user-supplied via the ``models/`` directory.

    The provenance block is returned rather than dropped because for a converted
    checkpoint it holds the only reproducible identity the file has. A ``.pt`` written
    by :mod:`cutoutml.models.u2net.from_onnx` is not bit-reproducible - ``torch.save``
    does not promise identical bytes for identical tensors - so its own SHA-256 names one
    local conversion. Re-run the conversion and the digest changes while the weights do
    not, which makes a published benchmark row look stale when it is not. The digest of
    the ONNX file it was converted from does not move, and that file is downloaded
    against a pinned digest.
    """
    ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
    provenance: dict[str, Any] = {}
    if isinstance(ckpt, dict):
        recorded = ckpt.get("provenance")
        if isinstance(recorded, dict):
            provenance = recorded
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            inner = ckpt.get(key)
            if isinstance(inner, dict):
                ckpt = inner
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint at {path} did not contain a state dict")
    return {k.removeprefix("module."): v for k, v in ckpt.items()}, provenance


def load_state_dict(path: Path | str, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    """The tensors alone, for callers with no use for the provenance block."""
    return load_checkpoint(path, map_location=map_location)[0]
