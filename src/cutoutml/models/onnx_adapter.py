"""ONNX Runtime adapter with prioritised execution-provider selection.

Why ONNX at all? Two reasons that matter in production:

1. **Deployment decoupling.** The serving container needs onnxruntime (~50 MB),
   not PyTorch + CUDA (~2.5 GB). That is the difference between a cold start of
   seconds and one of minutes.
2. **Graph-level optimisation.** onnxruntime fuses Conv+BN+ReLU, folds constants
   and picks better memory layouts than eager PyTorch. On CPU this is usually a
   real speedup; on GPU the gap narrows because cuDNN already does much of it.

Provider priority is TensorRT -> CUDA -> CPU, and the provider that onnxruntime
*actually* chose is logged and recorded in metadata. That distinction matters:
listing ``CUDAExecutionProvider`` in the priority list does not make it available,
and silently falling back to CPU while reporting "GPU" is exactly the kind of
mistake that produces bogus benchmark numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from cutoutml.core.imaging import LetterboxInfo
from cutoutml.core.logging import get_logger
from cutoutml.models.base import ModelMetadata, SegmentationModel, weights_digest

log = get_logger(__name__)

PROVIDER_PRIORITY: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


def available_providers() -> list[str]:
    """Execution providers this onnxruntime build offers, or ``[]`` if missing."""
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnxruntime is an optional extra
        return []
    return list(ort.get_available_providers())


def select_providers(requested: Sequence[str] | None = None, *, device: str = "auto") -> list[str]:
    """Intersect the priority list with what is installed, honouring ``device``.

    ``device="cpu"`` forces CPU-only even on a machine with CUDA, which the
    benchmark harness relies on to produce a comparable CPU row.
    """
    installed = available_providers()
    if not installed:
        return []
    candidates = list(requested) if requested else list(PROVIDER_PRIORITY)
    if device.startswith("cpu"):
        candidates = [p for p in candidates if p == "CPUExecutionProvider"]
    chosen = [p for p in candidates if p in installed]
    if "CPUExecutionProvider" in installed and "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")  # always keep a fallback
    return chosen


class OnnxAdapter(SegmentationModel):
    """Run an exported ``.onnx`` segmentation graph.

    The graph is expected to have a single float32 input ``(N, 3, h, w)`` and a single
    output shaped ``(N, 1, h, w)`` - which is what
    :meth:`cutoutml.models.base.TorchSegmentationModel.to_onnx` produces.

    Two properties of a third-party graph cannot be recovered from the file and must be
    declared by the registry entry, because guessing either one wrong is silent rather
    than loud:

    ``output_activation``
        Whether the graph emits logits or already-sigmoided probabilities. Published
        segmentation graphs commonly bake the sigmoid in; applying a second one squashes
        a mask towards 0.5 - alpha stays plausible-looking, edges go soft, and IoU drops
        by a few points with nothing raising an error. When this is ``"sigmoid"``,
        :meth:`predict` returns probabilities rather than the logits the base contract
        specifies, and :meth:`postprocess` compensates. That deviation is confined to
        this pair of methods.

    ``intensity_scaling``
        Preprocessing is not part of an ONNX artefact. A graph exported from U^2-Net
        needs that architecture's per-image max division; a graph exported from this
        repository does not.
    """

    def __init__(
        self,
        *,
        onnx_path: Path | str,
        providers: Sequence[str] | None = None,
        intra_op_threads: int = 0,
        normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        output_activation: str = "logits",
        intensity_scaling: str = "none",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", "onnx")
        super().__init__(**kwargs)
        if output_activation not in ("logits", "sigmoid"):
            raise ValueError(
                f"output_activation must be 'logits' or 'sigmoid', got {output_activation!r}"
            )
        if intensity_scaling not in ("none", "max"):
            raise ValueError(
                f"intensity_scaling must be 'none' or 'max', got {intensity_scaling!r}"
            )
        self.onnx_path = Path(onnx_path)
        self.requested_providers = list(providers) if providers else None
        self.intra_op_threads = intra_op_threads
        self._normalization = normalization
        self.output_activation = output_activation
        self.intensity_scaling = intensity_scaling
        self.session: Any = None
        self.active_providers: list[str] = []
        #: Batch size the graph is fixed to, or ``None`` when the axis is dynamic.
        self.static_batch: int | None = None
        self._input_name = "input"
        self._output_name = "logits"

    @property
    def normalization(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Must match whatever the exporting model used, or accuracy silently rots.

        The registry passes this through from the source adapter's spec; there is
        no way to recover it from the ONNX file itself, which is a genuine sharp
        edge of shipping ONNX artefacts.
        """
        if self._normalization is not None:
            return self._normalization
        return super().normalization

    def intensity_divisor(self, image: np.ndarray) -> float | None:
        if self.intensity_scaling != "max":
            return None
        peak = float(np.asarray(image).max())
        return peak / 255.0 if peak > 0 else None

    @property
    def effective_intra_op_threads(self) -> int:
        """The thread count the session really runs with.

        ONNX Runtime treats ``0`` as "one thread per physical core", so reporting the
        requested value would record 0 for the most common configuration and leave a
        benchmark row with no thread count at all.
        """
        if self.intra_op_threads > 0:
            return self.intra_op_threads
        return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1

    def _load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed; install the 'onnx' extra "
                "(pip install -e '.[onnx]') to use OnnxAdapter"
            ) from exc

        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at {self.onnx_path}. Export one with "
                "`python -m cutoutml.models.export_onnx --model <name>`."
            )

        providers = select_providers(self.requested_providers, device=str(self.device))
        if not providers:
            raise RuntimeError("onnxruntime reported no usable execution providers")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.intra_op_threads > 0:
            opts.intra_op_num_threads = self.intra_op_threads

        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=providers)
        # get_providers() is the ground truth: a requested provider that failed to
        # initialise will simply not appear here.
        self.active_providers = list(self.session.get_providers())

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        shape = inputs[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = (int(shape[3]), int(shape[2]))
        # A symbolic batch axis arrives as a string; an int means the graph was exported
        # without a dynamic batch and will reject any other size.
        self.static_batch = int(shape[0]) if shape and isinstance(shape[0], int) else None

        log.info(
            "onnx_session_created",
            path=str(self.onnx_path),
            requested=providers,
            active=self.active_providers,
            input_size=self.input_size,
            static_batch=self.static_batch,
        )
        if providers[0] != self.active_providers[0]:
            log.warning(
                "onnx_provider_downgraded",
                requested=providers[0],
                active=self.active_providers[0],
            )

    def unload(self) -> None:
        self.session = None
        self.active_providers = []
        self.static_batch = None
        super().unload()

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the graph. Returns logits, or probabilities for a sigmoid-baked graph."""
        self._ensure_loaded()
        assert self.session is not None
        batch = int(tensor.shape[0])
        if self.static_batch is not None and batch != self.static_batch:
            raise ValueError(
                f"{self.onnx_path.name} was exported with a fixed batch size of "
                f"{self.static_batch} and cannot run a batch of {batch}. Re-export with "
                "a dynamic batch axis (`to_onnx(..., dynamic_batch=True)`), or submit "
                f"work in groups of {self.static_batch}. Padding the batch and "
                "discarding the surplus would make a throughput number that looks "
                "better than the artefact actually is."
            )
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        (out,) = self.session.run([self._output_name], {self._input_name: arr})
        raw = np.asarray(out, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[:, None]
        return torch.from_numpy(raw)

    def postprocess(self, logits: torch.Tensor, infos: Sequence[LetterboxInfo]) -> list[np.ndarray]:
        """Skip the sigmoid when the graph already applied one."""
        if self.output_activation == "sigmoid":
            return self.alpha_from_probabilities(logits, infos)
        return super().postprocess(logits, infos)

    def metadata(self) -> ModelMetadata:
        size_bytes = self.onnx_path.stat().st_size if self.onnx_path.is_file() else 0
        active = self.active_providers[0] if self.active_providers else "unloaded"
        spec = self.spec
        return ModelMetadata(
            architecture=spec.architecture if spec else "onnx-graph",
            # ONNX initialisers are not exposed as "parameters"; estimate from the
            # fp32 file size so the benchmark table still has a comparable column.
            param_count=size_bytes // 4,
            trainable_param_count=0,
            runtime=f"onnxruntime:{active}",
            license=spec.license if spec else "depends on source model",
            source=spec.source if spec else str(self.onnx_path),
            notes=(
                f"Execution providers active: {', '.join(self.active_providers) or 'none'}. "
                "param_count is estimated from the fp32 graph size. "
                f"Graph output: {self.output_activation}; intensity scaling: "
                f"{self.intensity_scaling}; batch axis: "
                f"{'fixed at ' + str(self.static_batch) if self.static_batch else 'dynamic'}; "
                f"intra-op threads: {self.effective_intra_op_threads}."
            ),
            **{
                **self._base_metadata_kwargs(),
                # The graph *is* the weights here, so both fields have to point at it
                # rather than at the .pt this adapter never loads.
                "weights_path": str(self.onnx_path),
                "weights_sha256": weights_digest(self.onnx_path),
            },
        )
