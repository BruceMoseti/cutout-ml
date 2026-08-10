"""Which architectures the training loop can actually train, and how.

The training loop is deliberately architecture-agnostic: it knows about batches,
schedules and losses, not about networks. Everything a network needs in order to be
trained lives in one :class:`TrainableArch` entry here - how to build it, what input
normalisation its adapter expects at serving time, what resolution it is designed
for, and which of its outputs (if any) predicts an edge map rather than a mask.

The normalisation field is the important one. Train/serve normalisation mismatch is
the classic silent failure in this kind of project: the loss curve looks perfect, the
checkpoint saves, and then inference scores near chance because the adapter feeds
ImageNet-normalised tensors to a network trained on ``[-1, 1]``. Keeping the constant
next to the builder, and asserting it against the adapter in the test suite, is what
makes that mismatch impossible to introduce quietly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from torch import nn

from cutoutml.models.birefnet.arch import birefnet_compact, birefnet_tiny
from cutoutml.models.cutoutnet.arch import cutoutnet_base, cutoutnet_small, cutoutnet_tiny
from cutoutml.models.u2net.arch import u2net_full, u2net_lite

Normalization = tuple[tuple[float, float, float], tuple[float, float, float]]

SYMMETRIC: Normalization = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
IMAGENET: Normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@dataclasses.dataclass(frozen=True, slots=True)
class TrainableArch:
    """Everything the training loop needs to know about one architecture."""

    name: str
    build: Callable[[], nn.Module]
    #: Must equal the serving adapter's ``normalization`` property.
    normalization: Normalization
    #: Resolution the architecture is designed for; the default for ``--resolution``.
    default_resolution: int
    #: Registry model name that consumes the checkpoint this run produces.
    serves_as: str
    #: Path of the produced checkpoint, relative to the model weights directory.
    checkpoint: str
    #: Index of an output that predicts the target's *edge map* instead of the mask,
    #: or ``None``. BiRefNet's gradient head is the only such output here.
    gradient_output_index: int | None = None
    #: Rough guidance for CPU training, recorded in the run JSON so a reader can see
    #: which runs were expected to be feasible on this hardware.
    cpu_feasible: bool = True
    notes: str = ""


ARCHITECTURES: dict[str, TrainableArch] = {
    "cutoutnet-tiny": TrainableArch(
        name="cutoutnet-tiny",
        build=cutoutnet_tiny,
        normalization=SYMMETRIC,
        default_resolution=256,
        serves_as="cutoutnet-tiny",
        checkpoint="cutoutnet/cutoutnet-tiny.pt",
        notes="Latency floor: the smallest configuration worth serving.",
    ),
    "cutoutnet-small": TrainableArch(
        name="cutoutnet-small",
        build=cutoutnet_small,
        normalization=SYMMETRIC,
        default_resolution=256,
        serves_as="cutoutnet",
        checkpoint="cutoutnet/cutoutnet-small.pt",
        notes="The default served model.",
    ),
    "cutoutnet-base": TrainableArch(
        name="cutoutnet-base",
        build=cutoutnet_base,
        normalization=SYMMETRIC,
        default_resolution=256,
        serves_as="cutoutnet-base",
        checkpoint="cutoutnet/cutoutnet-base.pt",
        notes="4x the parameters of small; tests whether capacity or data is the limit.",
    ),
    "u2net-lite": TrainableArch(
        name="u2net-lite",
        build=u2net_lite,
        normalization=IMAGENET,
        default_resolution=256,
        serves_as="u2net-lite",
        checkpoint="u2net/u2net-lite.pt",
        notes=(
            "U^2-Net-P. Similar parameter count to cutoutnet-small but a very "
            "different compute profile, which is what makes the comparison useful."
        ),
    ),
    "u2net-full": TrainableArch(
        name="u2net-full",
        build=u2net_full,
        normalization=IMAGENET,
        default_resolution=320,
        serves_as="u2net",
        checkpoint="u2net/u2net-full.pt",
        cpu_feasible=False,
        notes=(
            "44M parameters. Included for completeness; a useful run needs a GPU, so "
            "no in-repo checkpoint exists for it."
        ),
    ),
    "birefnet-compact": TrainableArch(
        name="birefnet-compact",
        build=birefnet_compact,
        normalization=IMAGENET,
        default_resolution=512,
        serves_as="birefnet",
        checkpoint="birefnet/birefnet-compact.pt",
        gradient_output_index=5,
        cpu_feasible=False,
        notes=(
            "Designed for 512px input, where the bilateral reference pays off. "
            "Training it usefully needs a GPU."
        ),
    ),
    "birefnet-tiny": TrainableArch(
        name="birefnet-tiny",
        build=birefnet_tiny,
        normalization=IMAGENET,
        default_resolution=384,
        serves_as="birefnet",
        checkpoint="birefnet/birefnet-tiny.pt",
        gradient_output_index=5,
        cpu_feasible=False,
        notes="Smaller bilateral-reference variant; still GPU territory at 384px.",
    ),
}


def resolve_arch(name: str) -> TrainableArch:
    """Look up a trainable architecture, listing the alternatives on failure."""
    try:
        return ARCHITECTURES[name]
    except KeyError:
        raise ValueError(
            f"unknown architecture {name!r}; available: {', '.join(sorted(ARCHITECTURES))}"
        ) from None
