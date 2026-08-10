"""Materialise a PyTorch U^2-Net checkpoint from a BatchNorm-folded ONNX graph.

Why this exists
---------------
The official U^2-Net weights are Apache-2.0, but the authors distribute them via Google
Drive and the ``.pth`` mirrors are on HuggingFace. Neither host is reachable from every
build machine - on the machine this repository was developed on, ``huggingface.co`` is
blocked at the network layer. What *is* redistributed over plain HTTPS is an **ONNX
export** of the same weights (see :mod:`cutoutml.models.download_weights`).

That export was produced with constant folding on, so each ``Conv -> BatchNorm`` pair has
collapsed into a single biased convolution and the BatchNorm statistics no longer exist
as separate tensors. Two consequences follow, and both are load-bearing:

* The original parameter *names* are gone for every folded convolution: they appear in
  the graph as numeric temporaries (``2167``, ``2168``, ...). Only ``side1..side6`` and
  ``outconv``, which have no BatchNorm after them, kept their names. So a name-based
  remapping - what :func:`cutoutml.models.u2net.adapter.remap_official_state_dict` does
  for a real ``.pth`` - cannot work here.
* The folded weights are not interchangeable with the unfolded ones. Loading them into
  an architecture that still has BatchNorm layers requires those layers to be neutral.

How the mapping is established
------------------------------
ONNX requires ``graph.node`` to be topologically sorted, so the Conv nodes appear in
execution order. The PyTorch execution order is recovered the same way - by *running*
the module under forward hooks rather than by assuming the order the source code implies.
Pairing the two sequences is then positional.

A positional mapping is a hypothesis, so it is checked three ways, and every check is a
hard failure rather than a warning:

1. **Counts and shapes.** Both sequences must have the same length and every paired
   tensor must have identical dimensions. A single inserted or reordered layer misaligns
   the tail and shows up as a shape conflict.
2. **The named tail.** The last seven convolutions are the six side heads and the fusion
   convolution, and those kept their upstream names. Requiring ``side1.weight`` to land
   on this implementation's ``side.0`` is an independent confirmation of the alignment
   that does not depend on shapes - the six side heads have interchangeable shapes in the
   lite variant, so shape checking alone would not catch a permutation of them.
3. **Numerical parity.** The assembled module is run against onnxruntime on the same
   input and the maximum absolute difference is required to be small. This is the check
   that actually matters: it can only pass if every one of the 119 tensors landed in the
   right place. Observed agreement is ~1.5e-6, which is roughly 400x finer than one
   8-bit alpha level (1/255), so the difference cannot survive quantisation to a PNG.

The BatchNorm layers are set to the identity - ``weight=1``, ``bias=0``, ``mean=0`` and
``var = 1 - eps`` so that ``sqrt(var + eps)`` is exactly 1 - because the folded
convolution has already absorbed them. The resulting checkpoint is therefore
mathematically equivalent to the official one in ``eval()`` mode but is **not** suitable
for fine-tuning: the BatchNorms would start re-learning statistics for activations that
have already been scaled.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cutoutml.core.logging import get_logger
from cutoutml.models.u2net.arch import U2Net, u2net_full, u2net_lite

log = get_logger(__name__)

#: One 8-bit alpha level is 1/255 ~= 3.9e-3. The default tolerance is two orders of
#: magnitude tighter, so a graph that passes cannot differ by a visible amount.
PARITY_TOLERANCE = 1e-4

#: The convolutions that keep their upstream names because no BatchNorm follows them.
_NAMED_TAIL: tuple[tuple[str, str], ...] = (
    ("side.0", "side1.weight"),
    ("side.1", "side2.weight"),
    ("side.2", "side3.weight"),
    ("side.3", "side4.weight"),
    ("side.4", "side5.weight"),
    ("side.5", "side6.weight"),
    ("outconv", "outconv.weight"),
)

VARIANTS = ("full", "lite")


class ConversionError(RuntimeError):
    """The ONNX graph does not correspond to this architecture."""


@dataclasses.dataclass(frozen=True, slots=True)
class ConversionResult:
    """Outcome of one conversion, recorded inside the checkpoint it produced."""

    variant: str
    source_path: str
    source_sha256: str
    convolutions: int
    parity_max_abs_diff: float
    parity_tolerance: float
    parity_samples: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def build_variant(variant: str) -> U2Net:
    if variant not in VARIANTS:
        raise ValueError(f"unknown U2Net variant {variant!r}; expected one of {VARIANTS}")
    return u2net_full() if variant == "full" else u2net_lite()


def torch_conv_order(module: nn.Module, input_size: tuple[int, int] = (320, 320)) -> list[str]:
    """Names of every ``Conv2d`` in the order a forward pass executes them.

    Recovered by running the module with hooks attached rather than by reading
    ``named_modules()``, whose order follows construction. For this architecture the two
    happen to agree, but relying on that would make the conversion silently wrong the
    first time a stage is built in a different order than it is used.
    """
    order: list[str] = []
    handles = [
        mod.register_forward_hook(lambda _m, _i, _o, name=name: order.append(name))
        for name, mod in module.named_modules()
        if isinstance(mod, nn.Conv2d)
    ]
    try:
        was_training = module.training
        module.eval()
        with torch.inference_mode():
            module(torch.zeros(1, 3, input_size[1], input_size[0]))
        module.train(was_training)
    finally:
        for handle in handles:
            handle.remove()
    return order


def _onnx_convolutions(graph: Any) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """``(weight, bias, weight_initialiser_name)`` per Conv node, in execution order."""
    from onnx import numpy_helper

    initialisers = {t.name: t for t in graph.initializer}
    convs: list[tuple[np.ndarray, np.ndarray, str]] = []
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) < 3:
            raise ConversionError(
                f"Conv node {node.name or '<unnamed>'} has no bias input, so this graph was "
                "not exported with its BatchNorms folded. Use "
                "remap_official_state_dict() with the original .pth instead."
            )
        weight_name, bias_name = node.input[1], node.input[2]
        try:
            weight = numpy_helper.to_array(initialisers[weight_name])
            bias = numpy_helper.to_array(initialisers[bias_name])
        except KeyError as exc:
            raise ConversionError(
                f"Conv input {exc.args[0]!r} is a computed tensor rather than an initialiser; "
                "this graph's weights are not constant and cannot be extracted."
            ) from None
        convs.append((weight, bias, weight_name))
    return convs


def materialize_state_dict(onnx_path: Path | str, variant: str = "full") -> dict[str, torch.Tensor]:
    """Build a state dict for :class:`U2Net` from a BN-folded ONNX graph.

    Performs the structural checks described in the module docstring. Numerical parity is
    *not* checked here - use :func:`verify_parity` or :func:`convert`, which does both.
    """
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - onnx is an optional extra
        raise RuntimeError(
            "onnx is required to convert a graph to a checkpoint; install the 'onnx' extra"
        ) from exc

    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"no ONNX graph at {path}")

    module = build_variant(variant)
    order = torch_conv_order(module)
    convs = _onnx_convolutions(onnx.load(str(path)).graph)

    if len(convs) != len(order):
        raise ConversionError(
            f"{path.name} has {len(convs)} convolutions but u2net_{variant} executes "
            f"{len(order)}. This graph is a different architecture (or a different "
            "variant - 'full' and 'lite' have the same topology and different widths)."
        )

    reference = module.state_dict()
    state: dict[str, torch.Tensor] = {}
    for index, (name, (weight, bias, _)) in enumerate(zip(order, convs, strict=True)):
        expected = tuple(reference[f"{name}.weight"].shape)
        if tuple(weight.shape) != expected:
            raise ConversionError(
                f"convolution {index} ({name}) expects weights of shape {expected} but the "
                f"graph supplies {tuple(weight.shape)}. The positional mapping is not valid "
                f"for this graph; check that variant={variant!r} is right."
            )
        state[f"{name}.weight"] = torch.from_numpy(np.array(weight, dtype=np.float32))
        state[f"{name}.bias"] = torch.from_numpy(np.array(bias, dtype=np.float32))

    tail = dict(zip(order[-len(_NAMED_TAIL) :], convs[-len(_NAMED_TAIL) :], strict=True))
    for torch_name, onnx_name in _NAMED_TAIL:
        actual = tail.get(torch_name)
        if actual is None or actual[2] != onnx_name:
            raise ConversionError(
                f"expected the graph's {onnx_name!r} to align with this implementation's "
                f"{torch_name!r}, but it did not. The convolution order differs from the "
                "upstream forward pass, so a positional mapping would put the side-output "
                "heads in the wrong places - which shapes alone would not detect."
            )

    # The folded convolutions have absorbed the BatchNorms, so every BatchNorm must now
    # be an exact identity. var = 1 - eps makes sqrt(var + eps) exactly 1.
    for name, mod in module.named_modules():
        if not isinstance(mod, nn.BatchNorm2d):
            continue
        channels = mod.num_features
        state[f"{name}.weight"] = torch.ones(channels)
        state[f"{name}.bias"] = torch.zeros(channels)
        state[f"{name}.running_mean"] = torch.zeros(channels)
        state[f"{name}.running_var"] = torch.full((channels,), 1.0 - mod.eps)
        state[f"{name}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    missing = set(reference) - set(state)
    if missing:
        raise ConversionError(
            f"conversion left {len(missing)} tensors unset: {sorted(missing)[:5]}"
        )
    return state


def verify_parity(
    onnx_path: Path | str,
    module: nn.Module,
    *,
    samples: int = 2,
    input_size: tuple[int, int] = (320, 320),
    seed: int = 0,
) -> float:
    """Max absolute difference between ``module`` and onnxruntime on random inputs.

    The graph emits ``sigmoid(fused)``; the module returns logits, so a sigmoid is
    applied before comparing. Random normal inputs are used rather than images on
    purpose: they exercise the whole dynamic range instead of the narrow band a natural
    image occupies, so a misplaced tensor cannot hide in an unexercised activation.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - onnxruntime is an optional extra
        raise RuntimeError("onnxruntime is required to verify conversion parity") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rng = np.random.default_rng(seed)
    worst = 0.0
    module.eval()
    for _ in range(max(1, samples)):
        batch = rng.standard_normal((1, 3, input_size[1], input_size[0])).astype(np.float32)
        (reference,) = session.run([output_name], {input_name: batch})
        with torch.inference_mode():
            logits = module(torch.from_numpy(batch))
        fused = logits[0] if isinstance(logits, (tuple, list)) else logits
        got = torch.sigmoid(fused.float()).numpy()
        worst = max(worst, float(np.abs(np.asarray(reference, dtype=np.float32) - got).max()))
    return worst


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    onnx_path: Path | str,
    output_path: Path | str,
    *,
    variant: str = "full",
    tolerance: float = PARITY_TOLERANCE,
    parity_samples: int = 2,
) -> ConversionResult:
    """Convert, verify, and write a checkpoint. Raises unless parity holds.

    The provenance record travels inside the checkpoint rather than beside it, because a
    sidecar file is exactly what goes missing when weights are copied between machines -
    and a set of U^2-Net weights whose origin is unknown is a licensing problem as much
    as a reproducibility one.
    """
    source = Path(onnx_path)
    state = materialize_state_dict(source, variant)

    module = build_variant(variant)
    module.load_state_dict(state, strict=True)
    input_size = (320, 320)
    worst = verify_parity(source, module, samples=parity_samples, input_size=input_size)
    if worst > tolerance:
        raise ConversionError(
            f"conversion of {source.name} disagrees with onnxruntime by {worst:.3e}, above "
            f"the {tolerance:.0e} tolerance. The weights were extracted but they are not "
            "equivalent to the graph, so the checkpoint has not been written."
        )

    result = ConversionResult(
        variant=variant,
        source_path=source.name,
        source_sha256=sha256_file(source),
        convolutions=sum(1 for k in state if k.endswith(".weight") and state[k].ndim == 4),
        parity_max_abs_diff=worst,
        parity_tolerance=tolerance,
        parity_samples=parity_samples,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": state,
        "provenance": {
            **result.as_dict(),
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "converter": "cutoutml.models.u2net.from_onnx",
            "batchnorm": "identity (folded into the preceding convolution)",
            "fine_tuning": "not supported; the BatchNorms are neutral, not calibrated",
            "license": "Apache-2.0 (Qin et al., U^2-Net)",
        },
    }
    # Written to a temporary sibling and renamed so an interrupted write cannot leave a
    # truncated checkpoint that a later run would load as if it were valid.
    staging = destination.with_suffix(destination.suffix + ".partial")
    torch.save(payload, staging)
    staging.replace(destination)

    log.info(
        "u2net_converted_from_onnx",
        source=str(source),
        destination=str(destination),
        variant=variant,
        parity_max_abs_diff=worst,
    )
    return result


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a BN-folded U^2-Net ONNX graph into a PyTorch checkpoint"
    )
    parser.add_argument("--onnx", type=Path, required=True, help="source .onnx graph")
    parser.add_argument("--output", type=Path, required=True, help="destination .pt file")
    parser.add_argument("--variant", default="full", choices=list(VARIANTS))
    parser.add_argument("--tolerance", type=float, default=PARITY_TOLERANCE)
    parser.add_argument("--parity-samples", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    from cutoutml.core.logging import configure_logging

    args = build_parser().parse_args(argv)
    configure_logging(fmt="console")
    result = convert(
        args.onnx,
        args.output,
        variant=args.variant,
        tolerance=args.tolerance,
        parity_samples=args.parity_samples,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
