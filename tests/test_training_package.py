"""The ``cutoutml.training`` package surface.

These are cheap guards on one specific mistake: re-exporting the trainer from the
package ``__init__``. It looks harmless and it means that ``python -m
cutoutml.training.train`` - the documented way to start a run, and the one every script
in ``scripts/`` uses - imports ``train.py`` twice, once as ``cutoutml.training.train``
and once as ``__main__``. Two module objects hold two copies of everything defined at
module level, and Python only reports it as a ``RuntimeWarning`` that is easy to scroll
past in a training log.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the trainer as ``python -m`` in a fresh interpreter.

    A subprocess is required rather than ``runpy.run_module``: the warning is emitted
    only when the package is imported before the module is executed, and inside pytest
    both are already in ``sys.modules``, so an in-process check would pass whatever the
    package does.
    """
    return subprocess.run(
        [sys.executable, "-m", "cutoutml.training.train", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
        timeout=120,
    )


def test_running_the_trainer_as_a_module_does_not_double_import_it():
    result = _run_module("--help")

    assert result.returncode == 0
    assert "found in sys.modules" not in result.stderr, (
        "cutoutml/training/__init__.py imports the trainer eagerly again; runpy then "
        "executes train.py a second time as __main__"
    )


def test_the_help_output_still_comes_from_the_trainer_itself():
    """Guards the other direction: silencing the warning by breaking ``-m`` entirely."""
    result = _run_module("--help")

    assert "--arch" in result.stdout
    assert "--epochs" in result.stdout


def test_the_configuration_and_evaluation_helpers_are_importable_from_the_package():
    from cutoutml.training import TrainConfig, evaluate

    assert TrainConfig(arch="cutoutnet-tiny").arch == "cutoutnet-tiny"
    assert callable(evaluate)


def test_train_resolves_to_the_submodule_and_the_function_lives_inside_it():
    """The name is ambiguous by construction, so which one wins is worth pinning."""
    from cutoutml.training import train as submodule

    assert submodule.__name__ == "cutoutml.training.train"
    assert callable(submodule.train)


def test_an_unknown_attribute_still_raises_rather_than_importing_something():
    import cutoutml.training as package

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        getattr(package, "nonexistent")  # noqa: B009 - attribute access is the subject


def test_everything_advertised_in_dunder_all_actually_resolves():
    """``__all__`` is what ``from ... import *`` promises; a lazy hook makes it possible
    to advertise a name that no longer exists without anything failing at import time."""
    import cutoutml.training as package

    for name in package.__all__:
        assert getattr(package, name) is not None, name
