"""The API process must not import a deep-learning framework.

``docs/architecture.md`` claims the API is a control plane that loads no models, and the
Dockerfile is sized on that basis. The claim is easy to make and easy to break: nothing
about ``from cutoutml.models.registry import resolve_spec`` looks like it imports torch,
but it did, because the package ``__init__`` re-exported an adapter base class and that
base class imports torch and OpenCV. Boot time went from 0.68 s to 1.49 s for a process
that never runs a kernel.

So the boundary is asserted rather than described. Each check runs in a fresh
interpreter: pytest itself imports torch for the model tests, so an in-process assertion
would pass no matter what the API does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Frameworks whose absence is the actual claim. numpy is included deliberately: it is
#: cheap next to torch, but the API deals in bytes and JSON and has no array to hold, so
#: importing it means something reached across the boundary.
FORBIDDEN = ("torch", "onnxruntime", "cv2", "numpy")


def _modules_after_importing(target: str) -> set[str]:
    """Top-level module names present after importing ``target`` in a clean interpreter."""
    script = (
        "import json, sys;"
        f"__import__({target!r});"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
        timeout=180,
        # A settings object is constructed at import time and refuses a weak secret, so
        # the environment has to be good enough to boot before the import can be judged.
        env={
            "PATH": "/usr/bin:/bin",
            "CUTOUTML_JWT_SECRET": "t" * 48,
            "HOME": str(REPO_ROOT),
        },
    )
    assert result.returncode == 0, f"importing {target} failed:\n{result.stderr}"
    return set(json.loads(result.stdout))


def test_importing_the_api_does_not_import_torch_or_opencv():
    loaded = _modules_after_importing("services.api.app.main")

    leaked = sorted(set(FORBIDDEN) & loaded)
    assert not leaked, (
        f"importing the API pulled in {leaked}. Something on the API's import path now "
        "reaches into the model layer; check package __init__ re-exports first, they are "
        "the usual cause. See cutoutml.models.spec and cutoutml.core.precision."
    )


def test_the_registry_can_be_read_without_a_framework():
    """The API's actual need: name a model, decide whether it could run, without torch."""
    loaded = _modules_after_importing("cutoutml.models.registry")

    leaked = sorted(set(FORBIDDEN) & loaded)
    assert not leaked, f"cutoutml.models.registry imported {leaked} at module scope"


def test_reading_a_recorded_benchmark_does_not_import_the_harness():
    loaded = _modules_after_importing("cutoutml.benchmarks.results")

    assert "torch" not in loaded
    assert "cutoutml" in loaded


@pytest.mark.parametrize(
    "package",
    ["cutoutml.core", "cutoutml.models", "cutoutml.benchmarks"],
)
def test_importing_a_package_does_not_import_its_heavy_submodules(package: str):
    """The lazy ``__getattr__`` hooks are the mechanism; this is the property."""
    loaded = _modules_after_importing(package)

    leaked = sorted(set(FORBIDDEN) & loaded)
    assert not leaked, f"import {package} eagerly pulled in {leaked}"


@pytest.mark.parametrize(
    ("package", "name"),
    [
        ("cutoutml.core", "resolve_device"),
        ("cutoutml.core", "Precision"),
        ("cutoutml.core", "get_settings"),
        ("cutoutml.models", "ModelSpec"),
        ("cutoutml.models", "SegmentationModel"),
        ("cutoutml.models", "get_model"),
        ("cutoutml.benchmarks", "load_report"),
        ("cutoutml.benchmarks", "BenchmarkHarness"),
    ],
)
def test_the_lazy_re_exports_still_resolve(package: str, name: str):
    """Laziness must not turn a broken re-export into a silent one: an eager import fails
    at import time, a lazy one fails only when someone finally asks for the name."""
    import importlib

    module = importlib.import_module(package)

    assert getattr(module, name) is not None


@pytest.mark.parametrize(
    "package",
    ["cutoutml.core", "cutoutml.models", "cutoutml.benchmarks"],
)
def test_everything_advertised_in_dunder_all_actually_resolves(package: str):
    import importlib

    module = importlib.import_module(package)

    for name in module.__all__:
        assert getattr(module, name) is not None, f"{package}.{name}"


@pytest.mark.parametrize(
    "package",
    ["cutoutml.core", "cutoutml.models", "cutoutml.benchmarks"],
)
def test_the_export_map_and_dunder_all_do_not_drift(package: str):
    """``__all__`` is spelled out literally so that ruff can see the re-exports, which
    leaves two lists to keep in step. Adding a name to one and not the other either hides
    an export or advertises one that cannot resolve."""
    import importlib

    module = importlib.import_module(package)

    assert sorted(module.__all__) == sorted(module._EXPORTS)


@pytest.mark.parametrize(
    "package",
    ["cutoutml.core", "cutoutml.models", "cutoutml.benchmarks"],
)
def test_an_unknown_attribute_raises_rather_than_importing_something(package: str):
    import importlib

    module = importlib.import_module(package)

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        getattr(module, "nonexistent")  # noqa: B009 - attribute access is the subject


def test_the_declarative_and_executable_halves_agree_on_the_spec_type():
    """``base`` re-exports ``spec``'s classes; two definitions of ``ModelSpec`` would let
    an ``isinstance`` check quietly fail across the seam."""
    from cutoutml.models import base, spec

    assert base.ModelSpec is spec.ModelSpec
    assert base.ModelMetadata is spec.ModelMetadata
    assert base.WeightsUnavailableError is spec.WeightsUnavailableError


def test_precision_has_one_definition():
    from cutoutml.core import devices, precision

    assert devices.Precision is precision.Precision
