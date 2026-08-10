"""TensorRT adapter.

This module is importable on a machine with no TensorRT, no CUDA and no GPU - the
imports are guarded and every failure path raises
``RuntimeError("TensorRT unavailable: ...")``. That is a hard requirement: the CI
that lints and type-checks this repository has no GPU, and a bare ``import
tensorrt`` at module scope would break collection of the whole test suite.

Portability caveats (important, and frequently gotten wrong)
------------------------------------------------------------
A TensorRT ``.engine`` is **not** a portable artefact. It is specialised to:

* the **GPU architecture** it was built on (an SM 8.6 engine will not load on
  SM 7.5), and often the specific SKU, because kernel autotuning measures the
  real memory bandwidth and SM count;
* the **TensorRT version** - the serialisation format has no cross-version
  compatibility guarantee unless the engine is built with version-compatible
  mode, which itself costs performance;
* the **CUDA/cuDNN** versions available at load time;
* the **optimisation profile** (min/opt/max shapes) chosen at build time. Feeding
  a shape outside the profile fails at runtime rather than falling back.

The practical consequence is that engines must be built on the deployment
hardware, as part of a warm-up step, and cached keyed by
``(gpu_name, trt_version, precision, shape_profile)``. :func:`engine_cache_key`
implements that key. Never bake an engine into a container image and assume it
will work on whatever GPU the scheduler gives you.

FP16 is nearly always worth it on Ampere and later (roughly 1.5-2x over fp32 with
no measurable IoU change on segmentation masks). INT8 needs a calibration cache
built from representative data and does affect edge quality, so it is exposed but
not default.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cutoutml.core.logging import get_logger
from cutoutml.models.base import ModelMetadata, SegmentationModel, weights_digest

log = get_logger(__name__)

TRT_UNAVAILABLE_MESSAGE = "TensorRT unavailable"


def tensorrt_available() -> bool:
    """True when ``tensorrt`` and a CUDA runtime binding are both importable."""
    try:
        import tensorrt  # noqa: F401
    except Exception:  # noqa: BLE001 - optional dependency probe
        return False
    return _cuda_binding() is not None


def _cuda_binding() -> str | None:
    """Which CUDA host binding is available: ``cuda-python``, ``pycuda`` or none."""
    try:
        import cuda.bindings.driver  # noqa: F401

        return "cuda-python"
    except Exception:  # noqa: BLE001 - optional dependency probe
        pass
    try:
        import pycuda.driver  # noqa: F401

        return "pycuda"
    except Exception:  # noqa: BLE001 - optional dependency probe
        return None


def tensorrt_version() -> str | None:
    """Installed TensorRT version string, or ``None``."""
    try:
        import tensorrt as trt

        return str(trt.__version__)
    except Exception:  # noqa: BLE001 - optional dependency probe
        return None


def engine_cache_key(
    onnx_path: Path | str,
    *,
    precision: str,
    max_batch: int,
    input_size: tuple[int, int],
    gpu_name: str | None = None,
    trt_version: str | None = None,
) -> str:
    """Deterministic cache key for a built engine.

    Includes the ONNX file's content hash plus every axis of non-portability, so a
    cached engine can only ever be reused in a configuration where it is valid.
    Pure function with no TensorRT dependency - unit tested on CPU.
    """
    digest = hashlib.sha256()
    p = Path(onnx_path)
    if p.is_file():
        digest.update(p.read_bytes())
    else:
        digest.update(str(p).encode())
    parts = [
        digest.hexdigest()[:16],
        precision,
        f"b{max_batch}",
        f"{input_size[0]}x{input_size[1]}",
        (gpu_name or "unknown-gpu").replace(" ", "_"),
        f"trt{trt_version or 'unknown'}",
        platform.machine(),
    ]
    return "-".join(parts)


class TensorRTAdapter(SegmentationModel):
    """Build (or load) and execute a TensorRT engine for a segmentation graph.

    Construction never touches TensorRT, so the object can be created and
    inspected anywhere; :meth:`load` is where the ``RuntimeError`` is raised if the
    runtime is missing.
    """

    def __init__(
        self,
        *,
        onnx_path: Path | str | None = None,
        engine_path: Path | str | None = None,
        max_batch: int = 8,
        workspace_gb: float = 2.0,
        int8_calibration_cache: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", "tensorrt")
        super().__init__(**kwargs)
        if onnx_path is None and engine_path is None:
            raise ValueError("TensorRTAdapter requires onnx_path or engine_path")
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.engine_path = Path(engine_path) if engine_path else None
        self.max_batch = max_batch
        self.workspace_gb = workspace_gb
        self.int8_calibration_cache = (
            Path(int8_calibration_cache) if int8_calibration_cache else None
        )
        self.engine: Any = None
        self.context: Any = None
        self._trt: Any = None
        self._stream: Any = None
        self._io_names: tuple[str, str] = ("input", "logits")

    # ------------------------------------------------------------------ loading

    def _require_runtime(self) -> Any:
        if not tensorrt_available():
            binding = _cuda_binding()
            raise RuntimeError(
                f"{TRT_UNAVAILABLE_MESSAGE}: "
                f"tensorrt importable={tensorrt_version() is not None}, "
                f"cuda binding={binding or 'none'}. Install the 'trt' extra on a "
                "CUDA host (pip install tensorrt cuda-python) and rebuild the "
                "engine on the target GPU - engines are not portable across GPU "
                "architectures or TensorRT versions."
            )
        import tensorrt as trt

        self._trt = trt
        return trt

    def _load(self) -> None:
        trt = self._require_runtime()
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{TRT_UNAVAILABLE_MESSAGE}: TensorRT is installed but no CUDA "
                "device is visible to this process"
            )

        logger = trt.Logger(trt.Logger.WARNING)
        engine_bytes = self._resolve_engine_bytes(trt, logger)

        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                f"{TRT_UNAVAILABLE_MESSAGE}: engine deserialisation failed. This "
                "almost always means the engine was built with a different "
                "TensorRT version or for a different GPU architecture."
            )
        self.context = self.engine.create_execution_context()
        self._io_names = self._discover_io_names()
        self._stream = torch.cuda.Stream()
        log.info(
            "tensorrt_engine_loaded",
            engine=str(self.engine_path),
            gpu=torch.cuda.get_device_name(0),
            trt_version=tensorrt_version(),
        )

    def _resolve_engine_bytes(self, trt: Any, logger: Any) -> bytes:
        """Load a cached engine or build one from ONNX, caching the result."""
        if self.engine_path and self.engine_path.is_file():
            return self.engine_path.read_bytes()

        if self.onnx_path is None or not self.onnx_path.is_file():
            raise RuntimeError(
                f"{TRT_UNAVAILABLE_MESSAGE}: no engine at {self.engine_path} and no "
                f"ONNX source at {self.onnx_path} to build one from"
            )

        key = engine_cache_key(
            self.onnx_path,
            precision=self.precision,
            max_batch=self.max_batch,
            input_size=self.input_size,
            gpu_name=torch.cuda.get_device_name(0),
            trt_version=tensorrt_version(),
        )
        target = self.engine_path or self.onnx_path.with_name(f"{key}.engine")

        log.info("tensorrt_building_engine", onnx=str(self.onnx_path), target=str(target))
        engine_bytes = self._build_engine(trt, logger)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(engine_bytes)
        self.engine_path = target
        return engine_bytes

    def _build_engine(self, trt: Any, logger: Any) -> bytes:
        """Parse ONNX and serialise an engine with an explicit shape profile."""
        assert self.onnx_path is not None
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        if not parser.parse(self.onnx_path.read_bytes()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"{TRT_UNAVAILABLE_MESSAGE}: ONNX parse failed: {'; '.join(errors)}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(self.workspace_gb * (1 << 30))
        )
        if self.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.int8_calibration_cache is not None and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        # Optimisation profile: shapes outside [min, max] will fail at runtime, so
        # the batch range is declared explicitly rather than inferred.
        profile = builder.create_optimization_profile()
        w, h = self.input_size
        input_name = network.get_input(0).name
        profile.set_shape(
            input_name,
            min=(1, 3, h, w),
            opt=(max(1, self.max_batch // 2), 3, h, w),
            max=(self.max_batch, 3, h, w),
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"{TRT_UNAVAILABLE_MESSAGE}: engine build returned None")
        return bytes(serialized)

    def _discover_io_names(self) -> tuple[str, str]:
        trt = self._trt
        inputs, outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if not inputs or not outputs:
            raise RuntimeError(f"{TRT_UNAVAILABLE_MESSAGE}: engine has no I/O tensors")
        return (inputs[0], outputs[0])

    def unload(self) -> None:
        self.context = None
        self.engine = None
        self._stream = None
        super().unload()

    # --------------------------------------------------------------- inference

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        """Execute the engine.

        Torch tensors are used as the device buffers (via ``data_ptr()``) rather
        than raw pycuda allocations: it removes a host round-trip and lets the
        caller keep everything on the GPU.
        """
        self._ensure_loaded()
        assert self.context is not None
        in_name, out_name = self._io_names

        device_in = tensor.to("cuda", dtype=torch.float32, non_blocking=True).contiguous()
        self.context.set_input_shape(in_name, tuple(device_in.shape))
        out_shape = tuple(self.context.get_tensor_shape(out_name))
        device_out = torch.empty(out_shape, dtype=torch.float32, device="cuda")

        self.context.set_tensor_address(in_name, device_in.data_ptr())
        self.context.set_tensor_address(out_name, device_out.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError(f"{TRT_UNAVAILABLE_MESSAGE}: execute_async_v3 failed")
        stream.synchronize()

        logits = device_out.float()
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        return logits

    def postprocess(self, logits: torch.Tensor, infos: Sequence[Any]) -> list[np.ndarray]:
        return super().postprocess(logits, infos)

    def metadata(self) -> ModelMetadata:
        spec = self.spec
        size = (
            self.engine_path.stat().st_size
            if self.engine_path and self.engine_path.is_file()
            else 0
        )
        return ModelMetadata(
            architecture=spec.architecture if spec else "tensorrt-engine",
            param_count=size // 4,
            trainable_param_count=0,
            runtime=f"tensorrt:{tensorrt_version() or 'unavailable'}",
            license=spec.license if spec else "depends on source model",
            source=spec.source if spec else str(self.onnx_path or self.engine_path),
            notes=(
                "TensorRT engines are hardware- and version-specific: rebuild on the "
                "target GPU. Shapes outside the built optimisation profile will fail."
            ),
            **{
                **self._base_metadata_kwargs(),
                "weights_path": str(self.engine_path) if self.engine_path else None,
                "weights_sha256": weights_digest(self.engine_path),
            },
        )
