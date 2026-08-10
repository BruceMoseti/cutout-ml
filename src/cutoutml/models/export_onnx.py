"""Export a registered PyTorch model to ONNX and verify numerical parity.

Exporting without verifying is how a "working" ONNX deployment silently serves
different masks than the PyTorch model it was validated as. This script always runs a
parity check and refuses to declare success if the maximum absolute difference exceeds
the tolerance.

Tolerance rationale: the export changes operator implementations (onnxruntime fuses
Conv+BN, may reassociate additions, and picks its own GEMM kernels), so bit-exactness is
not achievable and not the goal. What matters is that the *alpha map* is
indistinguishable. The default 1e-3 on **probabilities after sigmoid** corresponds to
less than one 8-bit alpha level, i.e. differences that cannot survive quantisation to a
PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cutoutml.core.config import get_settings
from cutoutml.core.logging import configure_logging, get_logger
from cutoutml.models.base import TorchSegmentationModel
from cutoutml.models.registry import get_model, resolve_spec

log = get_logger(__name__)

DEFAULT_LOGIT_TOLERANCE = 2e-3
DEFAULT_PROB_TOLERANCE = 1e-3


def export(
    model_name: str,
    output: Path | str | None = None,
    *,
    opset: int = 17,
    device: str = "cpu",
    dynamic_batch: bool = True,
    random_init: bool = False,
) -> Path:
    """Export ``model_name`` to ONNX, returning the written path."""
    model = get_model(model_name, device=device, random_init=random_init, load=True)
    if not isinstance(model, TorchSegmentationModel):
        raise TypeError(
            f"model {model_name!r} is not a PyTorch model and cannot be exported to ONNX"
        )

    spec = resolve_spec(model_name)
    if output is None:
        weights_dir = get_settings().model_weights_dir
        output = weights_dir / f"{model_name.split('-')[0]}" / f"{spec.name}-{model.variant}.onnx" \
            if hasattr(model, "variant") else weights_dir / f"{spec.name}.onnx"
    path = Path(output)
    model.to_onnx(path, opset=opset, dynamic_batch=dynamic_batch)
    log.info(
        "onnx_export_complete",
        model=model_name,
        path=str(path),
        bytes=path.stat().st_size,
        opset=opset,
    )
    return path


def verify_parity(
    model_name: str,
    onnx_path: Path | str,
    *,
    device: str = "cpu",
    batch_sizes: tuple[int, ...] = (1, 4),
    logit_tolerance: float = DEFAULT_LOGIT_TOLERANCE,
    prob_tolerance: float = DEFAULT_PROB_TOLERANCE,
    seed: int = 12345,
    random_init: bool = False,
) -> dict[str, Any]:
    """Compare PyTorch and onnxruntime outputs on identical random inputs.

    Several batch sizes are checked because a dynamic batch axis is exactly where an
    export goes wrong: a graph that folded the batch dimension into a constant works at
    batch 1 and produces garbage or an error at batch 4.
    """
    import onnxruntime as ort

    torch_model = get_model(model_name, device=device, random_init=random_init, load=True)
    if not isinstance(torch_model, TorchSegmentationModel):
        raise TypeError(f"model {model_name!r} is not a PyTorch model")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rng = np.random.default_rng(seed)
    w, h = torch_model.input_size
    report: dict[str, Any] = {
        "model": model_name,
        "onnx_path": str(onnx_path),
        "providers": session.get_providers(),
        "input_size": [w, h],
        "checks": [],
        "passed": True,
    }

    for batch in batch_sizes:
        array = rng.standard_normal((batch, 3, h, w), dtype=np.float32)
        tensor = torch.from_numpy(array)

        torch_logits = torch_model.predict(tensor).detach().cpu().numpy()
        (onnx_logits,) = session.run([output_name], {input_name: array})
        onnx_logits = np.asarray(onnx_logits, dtype=np.float32)
        if onnx_logits.ndim == 3:
            onnx_logits = onnx_logits[:, None]

        if torch_logits.shape != onnx_logits.shape:
            report["checks"].append(
                {
                    "batch": batch,
                    "passed": False,
                    "error": f"shape mismatch {torch_logits.shape} vs {onnx_logits.shape}",
                }
            )
            report["passed"] = False
            continue

        logit_diff = float(np.abs(torch_logits - onnx_logits).max())
        prob_diff = float(
            np.abs(_sigmoid(torch_logits) - _sigmoid(onnx_logits)).max()
        )
        # An 8-bit alpha step is 1/255 ~= 0.0039, so express the difference in those
        # units: that is the number a reader can reason about.
        alpha_levels = prob_diff * 255.0
        passed = logit_diff <= logit_tolerance and prob_diff <= prob_tolerance
        report["checks"].append(
            {
                "batch": batch,
                "passed": passed,
                "max_abs_logit_diff": logit_diff,
                "max_abs_prob_diff": prob_diff,
                "max_alpha_levels_of_255": alpha_levels,
                "logit_tolerance": logit_tolerance,
                "prob_tolerance": prob_tolerance,
            }
        )
        report["passed"] = report["passed"] and passed
        log.info(
            "onnx_parity_check",
            batch=batch,
            max_logit_diff=round(logit_diff, 8),
            max_prob_diff=round(prob_diff, 8),
            alpha_levels=round(alpha_levels, 4),
            passed=passed,
        )

    torch_model.unload()
    return report


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (``exp`` of a large positive overflows)."""
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a model to ONNX and verify parity")
    p.add_argument("--model", default="cutoutnet")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--static-batch", action="store_true", help="disable the dynamic batch axis")
    p.add_argument("--random-init", action="store_true", help="export untrained weights")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--report", type=Path, default=None, help="write the parity report as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(fmt="console")

    path = export(
        args.model,
        args.output,
        opset=args.opset,
        device=args.device,
        dynamic_batch=not args.static_batch,
        random_init=args.random_init,
    )
    print(f"exported {path} ({path.stat().st_size / 1e6:.2f} MB)")

    if args.skip_verify:
        return 0

    batch_sizes = (1,) if args.static_batch else (1, 4)
    report = verify_parity(
        args.model, path, device=args.device, batch_sizes=batch_sizes, random_init=args.random_init
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for check in report["checks"]:
        print(f"  batch {check['batch']}: {check}")
    if not report["passed"]:
        print("PARITY CHECK FAILED", file=__import__("sys").stderr)
        return 1
    print("parity OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
