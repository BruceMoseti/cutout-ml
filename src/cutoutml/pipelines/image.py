"""End-to-end image pipeline.

::

    bytes -> validate -> decode -> EXIF orient -> letterbox -> normalize
          -> model -> logits -> sigmoid -> un-letterbox to original size
          -> alpha refinement -> requested outputs

The pipeline owns *no* model-specific knowledge: it takes a
:class:`~cutoutml.models.base.SegmentationModel` and asks it to preprocess, predict
and postprocess. That is what lets the same code serve CutoutNet, an ONNX graph and
a GrabCut baseline.

Design notes
------------
* **Batching is real.** ``process_batch`` runs one forward pass for N images rather
  than N passes, which on CPU is worth roughly 1.3-1.6x and on GPU much more. The
  per-image path is a thin wrapper so there is only one code path to test.
* **Outputs are requested, not always produced.** Encoding a 4000 px PNG is not
  cheap; a caller who only wants a mask should not pay for a transparent PNG too.
* **Refinement happens at full resolution.** Refining the low-resolution mask and
  then upsampling reintroduces exactly the stair-stepping the guided filter is
  there to remove.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from cutoutml.core.imaging import (
    composite_blurred_background,
    composite_over_color,
    composite_over_image,
    decode_image,
    encode_image,
    encode_mask,
)
from cutoutml.core.logging import get_logger
from cutoutml.core.refine import RefineConfig, refine_alpha
from cutoutml.models.base import SegmentationModel

log = get_logger(__name__)

OutputKind = Literal[
    "transparent_png",
    "transparent_webp",
    "mask_png",
    "color_composite",
    "background_composite",
    "blurred_background",
]

DEFAULT_OUTPUTS: tuple[OutputKind, ...] = ("transparent_png", "mask_png")


@dataclasses.dataclass(slots=True)
class ImageRequest:
    """One image processing request."""

    outputs: tuple[OutputKind, ...] = DEFAULT_OUTPUTS
    background_color: tuple[int, int, int] = (255, 255, 255)
    background_image: np.ndarray | None = None
    blur_sigma: float = 12.0
    webp_quality: int = 90
    refine: RefineConfig = dataclasses.field(default_factory=RefineConfig)
    max_pixels: int | None = 64_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "outputs": list(self.outputs),
            "background_color": list(self.background_color),
            "has_background_image": self.background_image is not None,
            "blur_sigma": self.blur_sigma,
            "webp_quality": self.webp_quality,
            "refine": self.refine.as_dict(),
        }


@dataclasses.dataclass(slots=True)
class ImageResult:
    """Encoded outputs plus timing breakdown for one image."""

    width: int
    height: int
    alpha_coverage: float
    outputs: dict[str, bytes]
    timings_ms: dict[str, float]
    model_name: str
    content_sha256: str | None = None

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary (byte payloads replaced by sizes)."""
        return {
            "width": self.width,
            "height": self.height,
            "alpha_coverage": round(self.alpha_coverage, 5),
            "model": self.model_name,
            "outputs": {k: len(v) for k, v in self.outputs.items()},
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
            "content_sha256": self.content_sha256,
        }


class ImagePipeline:
    """Reusable image pipeline bound to one loaded model."""

    def __init__(self, model: SegmentationModel) -> None:
        if not model.is_loaded:
            model.load()
        self.model = model

    # -------------------------------------------------------------- entry points

    def process_bytes(
        self, data: bytes, request: ImageRequest | None = None
    ) -> ImageResult:
        """Decode encoded image bytes and process them."""
        req = request or ImageRequest()
        started = time.perf_counter()
        image = decode_image(data, apply_exif=True, max_pixels=req.max_pixels)
        decode_ms = (time.perf_counter() - started) * 1000.0

        result = self.process_array(image, req)
        result.timings_ms["decode"] = decode_ms
        result.content_sha256 = hashlib.sha256(data).hexdigest()
        return result

    def process_array(self, image: np.ndarray, request: ImageRequest | None = None) -> ImageResult:
        """Process a decoded RGB array."""
        return self.process_batch([image], request)[0]

    def process_batch(
        self, images: Sequence[np.ndarray], request: ImageRequest | None = None
    ) -> list[ImageResult]:
        """Process several images in a single forward pass.

        All images are letterboxed to the model's input size, so a heterogeneous
        batch is fine - which matters, because in a real workload it always is.
        """
        req = request or ImageRequest()
        if not images:
            return []

        t0 = time.perf_counter()
        tensor, infos = self.model.preprocess(images)
        t1 = time.perf_counter()
        logits = self.model.predict(tensor)
        t2 = time.perf_counter()
        alphas = self.model.postprocess(logits, infos)
        t3 = time.perf_counter()

        n = len(images)
        pre_ms = (t1 - t0) * 1000.0 / n
        infer_ms = (t2 - t1) * 1000.0 / n
        post_ms = (t3 - t2) * 1000.0 / n

        results: list[ImageResult] = []
        for image, alpha in zip(images, alphas, strict=True):
            t4 = time.perf_counter()
            refined = refine_alpha(alpha, image, req.refine)
            refine_ms = (time.perf_counter() - t4) * 1000.0

            t5 = time.perf_counter()
            outputs = self._encode_outputs(image, refined, req)
            encode_ms = (time.perf_counter() - t5) * 1000.0

            results.append(
                ImageResult(
                    width=image.shape[1],
                    height=image.shape[0],
                    alpha_coverage=float((refined > 0.5).mean()),
                    outputs=outputs,
                    timings_ms={
                        "preprocess": pre_ms,
                        "inference": infer_ms,
                        "postprocess": post_ms,
                        "refine": refine_ms,
                        "encode": encode_ms,
                    },
                    model_name=self.model.name,
                )
            )

        log.info(
            "image_batch_processed",
            model=self.model.name,
            batch=n,
            inference_ms_per_image=round(infer_ms, 2),
            outputs=list(req.outputs),
        )
        return results

    # -------------------------------------------------------------------- outputs

    def _encode_outputs(
        self, image: np.ndarray, alpha: np.ndarray, req: ImageRequest
    ) -> dict[str, bytes]:
        outputs: dict[str, bytes] = {}
        for kind in req.outputs:
            if kind == "transparent_png":
                outputs[kind] = encode_image(image, "png", alpha=alpha)
            elif kind == "transparent_webp":
                outputs[kind] = encode_image(
                    image, "webp", alpha=alpha, quality=req.webp_quality
                )
            elif kind == "mask_png":
                outputs[kind] = encode_mask(alpha)
            elif kind == "color_composite":
                composite = composite_over_color(image, alpha, req.background_color)
                outputs[kind] = encode_image(composite, "png")
            elif kind == "background_composite":
                if req.background_image is None:
                    raise ValueError(
                        "output 'background_composite' requires request.background_image"
                    )
                composite = composite_over_image(image, alpha, req.background_image)
                outputs[kind] = encode_image(composite, "png")
            elif kind == "blurred_background":
                composite = composite_blurred_background(
                    image, alpha, blur_sigma=req.blur_sigma
                )
                outputs[kind] = encode_image(composite, "png")
            else:  # pragma: no cover - Literal makes this unreachable via typing
                raise ValueError(f"unknown output kind: {kind!r}")
        return outputs

    def alpha_only(self, image: np.ndarray, refine: RefineConfig | None = None) -> np.ndarray:
        """Just the refined alpha map. Used by the video pipeline and benchmarks."""
        alpha = self.model.infer([image])[0]
        return refine_alpha(alpha, image, refine or RefineConfig())

    def alpha_batch(
        self, images: Sequence[np.ndarray], refine: RefineConfig | None = None
    ) -> list[np.ndarray]:
        """Refined alpha maps for a batch, in one forward pass."""
        cfg = refine or RefineConfig()
        alphas = self.model.infer(list(images))
        return [refine_alpha(a, img, cfg) for a, img in zip(alphas, images, strict=True)]
