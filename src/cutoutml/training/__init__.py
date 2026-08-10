"""Training: losses and the CutoutNet training loop.

The trainer's names are resolved lazily (:pep:`562`) rather than imported eagerly. The
trainer is normally started as ``python -m cutoutml.training.train``, and runpy imports
the containing package before executing the module: an eager import here means
``train.py`` is imported once as ``cutoutml.training.train`` and then executed again as
``__main__``. Python warns about exactly that on every run ("found in sys.modules ...
prior to execution of"), and two live copies of a module is a hazard rather than noise.

``train`` deliberately resolves to the **submodule**, not to the ``train()`` function
inside it. The two cannot both own the name: importing the submodule binds it as an
attribute of this package, so the import system's value wins over anything this module
returns, and a re-export would resolve to the function or the module depending only on
what had already been imported. The function is
:func:`cutoutml.training.train.train`.
"""

import importlib
from typing import TYPE_CHECKING, Any

from cutoutml.training.losses import (
    LossWeights,
    SegmentationLoss,
    bce_loss,
    dice_loss,
    edge_loss,
    soft_iou_loss,
    ssim_loss,
)

if TYPE_CHECKING:  # import for type checkers only; see the module docstring
    from cutoutml.training.train import TrainConfig, evaluate

_LAZY = frozenset({"TrainConfig", "evaluate"})


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        # importlib rather than `from . import train`: the latter reaches this same hook
        # to resolve the name and recurses until the stack runs out.
        return getattr(importlib.import_module("cutoutml.training.train"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "LossWeights",
    "SegmentationLoss",
    "TrainConfig",
    "bce_loss",
    "dice_loss",
    "edge_loss",
    "evaluate",
    "soft_iou_loss",
    "ssim_loss",
    "train",
]
