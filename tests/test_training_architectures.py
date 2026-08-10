"""The training architecture registry against the serving registry.

Train/serve normalisation mismatch is the classic silent failure in a project shaped
like this one: the loss curve looks perfect, the checkpoint saves, and then inference
scores near chance because the adapter feeds ImageNet-normalised tensors to a network
trained on ``[-1, 1]``. Nothing raises, and the only symptom is a bad accuracy number
that reads as "the model did not learn".

``cutoutml.training.architectures`` and ``cutoutml.models.registry`` both say what an
architecture's normalisation, resolution and checkpoint path are, and the two are
written down separately because the trainer must not import an adapter. Two docstrings
in the training package claimed the pair was asserted against each other in the test
suite. It was not - the file they named did not exist - so this module makes the claim
true rather than removing it, since the property is worth more than the sentence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cutoutml.models.registry import get_model, resolve_spec
from cutoutml.training.architectures import ARCHITECTURES, TrainableArch

ARCH_IDS = sorted(ARCHITECTURES)


@pytest.fixture(params=ARCH_IDS)
def arch(request: pytest.FixtureRequest) -> TrainableArch:
    return ARCHITECTURES[request.param]


def test_every_trainable_architecture_names_a_registered_model(arch: TrainableArch) -> None:
    """``serves_as`` is what connects a training run to the entry that loads its output.
    A typo here produces a checkpoint nothing can serve."""
    assert resolve_spec(arch.serves_as).name == arch.serves_as


def test_training_normalisation_equals_the_serving_adapters(arch: TrainableArch) -> None:
    """The mismatch this whole module exists for.

    The adapter is constructed with ``load=False`` and random initialisation so the
    comparison needs no checkpoint: normalisation is a property of the architecture, not
    of the weights, which is exactly why it can disagree unnoticed.
    """
    spec = resolve_spec(arch.serves_as)
    model = get_model(
        arch.serves_as,
        load=False,
        random_init=spec.supports_random_init,
    )

    assert tuple(model.normalization) == tuple(arch.normalization), (
        f"{arch.name} trains with normalisation {arch.normalization} but "
        f"{arch.serves_as} serves with {model.normalization}; a checkpoint trained here "
        "would score near chance at inference with nothing raising"
    )


def _weight_candidates(name: str) -> set[Path]:
    spec = resolve_spec(name)
    return {Path(p) for p in (spec.default_weights, *spec.alt_weights) if p}


def test_the_trained_resolution_matches_what_the_spec_serves(arch: TrainableArch) -> None:
    """A network trained at 384 and served at 512 still runs, and still degrades."""
    if not arch.checkpoint_served:
        pytest.skip(f"{arch.name} produces no checkpoint {arch.serves_as} loads")
    spec = resolve_spec(arch.serves_as)

    assert spec.input_size == (arch.default_resolution, arch.default_resolution), (
        f"{arch.name} trains at {arch.default_resolution}px but {arch.serves_as} declares "
        f"input_size {spec.input_size}"
    )


def test_the_checkpoint_a_run_writes_is_where_its_spec_looks_for_weights(
    arch: TrainableArch,
) -> None:
    """Otherwise a finished run leaves a file the registry never finds, and the model
    keeps reporting ``weights_available: false`` after training succeeded."""
    if not arch.checkpoint_served:
        pytest.skip(f"{arch.name} is recorded as producing an unserved checkpoint")

    assert Path(arch.checkpoint) in _weight_candidates(arch.serves_as), (
        f"{arch.name} writes {arch.checkpoint}, but {arch.serves_as} only loads from "
        f"{sorted(str(c) for c in _weight_candidates(arch.serves_as))}"
    )


def test_an_unserved_checkpoint_cannot_shadow_the_weights_its_entry_does_load(
    arch: TrainableArch,
) -> None:
    """The reason the two exceptions are separated rather than wired up.

    `u2net-full` trains a 44M network from scratch and `u2net` serves the authors'
    published Apache-2.0 weights. If the training path were one the entry loads, a
    from-scratch run would take their place under a benchmark row labelled as theirs and
    under a licence line that describes them. Keeping the paths disjoint is what makes
    that impossible, so it is asserted rather than assumed.
    """
    if arch.checkpoint_served:
        pytest.skip(f"{arch.name} is served by {arch.serves_as} by design")

    assert Path(arch.checkpoint) not in _weight_candidates(arch.serves_as), (
        f"{arch.name} is recorded as unserved yet writes to a path {arch.serves_as} "
        "loads, so a from-scratch checkpoint could be served as that entry's weights"
    )


def test_no_two_architectures_write_to_the_same_checkpoint() -> None:
    """Two runs sharing a path means the second silently overwrites the first, and the
    committed run record then describes weights that are no longer on disk."""
    written: dict[str, str] = {}
    for name, entry in sorted(ARCHITECTURES.items()):
        clash = written.get(entry.checkpoint)
        assert clash is None, f"{name} and {clash} both write {entry.checkpoint}"
        written[entry.checkpoint] = name
