"""Failure classification and retry backoff.

This module decides, for every exception a worker sees, whether the job is retried,
retried at a smaller batch, or failed outright. Both mistakes are expensive - a poison
message retried three times starves real work, and a permanently-failed job for a
two-second Postgres restart is a user-visible error nobody needed to see - so the table
is tested case by case rather than trusted.

Most of these tests are really about *ordering*. The exception hierarchy overlaps
heavily: CUDA OOM is a ``RuntimeError``, ``FileNotFoundError`` and ``ConnectionError``
are both ``OSError``, and ``ModelNotFoundError`` is a ``KeyError``. Every one of those
pairs has one branch that would swallow the other if the checks were reordered, which is
exactly the kind of regression a refactor introduces silently.
"""

from __future__ import annotations

import random

import pytest
from services.inference.app.errors import (
    Classification,
    FailureKind,
    classify,
    retry_delay,
)

from cutoutml.models.base import WeightsUnavailableError
from cutoutml.models.registry import ModelNotFoundError
from cutoutml.pipelines.ffmpeg import UnsupportedCodecError
from cutoutml.storage.base import ObjectNotFoundError


class FakeCudaOutOfMemoryError(RuntimeError):
    """Stands in for ``torch.cuda.OutOfMemoryError``.

    Classification matches on the *name* rather than the type, precisely so it works on
    a CPU-only build where the real class may not be importable - which is the case on
    the machine this suite runs on. Naming the stand-in class the same thing is
    therefore a faithful test of the real code path, not a shortcut around it.
    """

    def __init__(self, message: str = "CUDA out of memory. Tried to allocate 2.00 GiB") -> None:
        super().__init__(message)


FakeCudaOutOfMemoryError.__name__ = "OutOfMemoryError"


# ============================================================== out of memory


def test_a_typed_cuda_oom_is_classified_as_oom():
    result = classify(FakeCudaOutOfMemoryError())
    assert result.kind is FailureKind.OOM
    assert result.code == "out_of_memory"


def test_oom_is_checked_before_the_generic_runtime_error_branches():
    """CUDA OOM *is* a RuntimeError, so a reordering that put the pattern-matched
    transient branch first would turn "retry at half the batch size" into "retry the
    identical batch", which reproduces the OOM exactly."""
    exc = FakeCudaOutOfMemoryError()
    assert isinstance(exc, RuntimeError)
    assert classify(exc).is_oom is True


def test_a_host_memory_error_is_oom_too():
    """The recovery is the same - shrink the inputs - whether it was the GPU or the
    host that ran out."""
    assert classify(MemoryError("cannot allocate 32 GiB")).kind is FailureKind.OOM


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 20.00 MiB",
        "Unable to find a valid cuDNN algorithm: CUDNN_STATUS_ALLOC_FAILED",
        "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
        "failed to allocate 4194304 bytes on device",
        "cannot allocate memory in static TLS block",
        "No space left on device: /dev/shm",
    ],
)
def test_oom_is_recognised_from_the_message_when_the_type_is_a_bare_runtime_error(message: str):
    """cuDNN, cuBLAS and ``pin_memory`` all report allocation failure as a plain
    ``RuntimeError``, so the type check alone would miss most real OOMs."""
    assert classify(RuntimeError(message)).kind is FailureKind.OOM


def test_oom_matching_is_case_insensitive():
    assert classify(RuntimeError("CUDA OUT OF MEMORY")).is_oom is True
    assert classify(RuntimeError("cuda out of memory")).is_oom is True


def test_an_oom_is_retryable_because_the_batch_size_shrinks_first():
    """Distinct from RETRYABLE: the handler must change something before trying again,
    and ``is_oom`` is how it knows to halve the batch rather than resubmit as-is."""
    result = classify(FakeCudaOutOfMemoryError())
    assert result.retryable is True
    assert result.is_oom is True


def test_a_full_disk_that_is_not_shared_memory_is_not_an_oom():
    """The pattern is anchored on ``shm`` on purpose. A genuinely full output volume is
    an I/O problem, and halving the batch size does not create disk space."""
    result = classify(OSError("No space left on device: /var/lib/cutoutml/out.png"))
    assert result.is_oom is False


# ========================================================== the request itself


def test_an_unsupported_codec_fails_immediately():
    """A ProRes file whose codec this build cannot decode will not decode on the third
    attempt either."""
    result = classify(UnsupportedCodecError("codec hap is not supported"))
    assert result.kind is FailureKind.NON_RETRYABLE
    assert result.code == "unsupported_codec"


def test_an_unknown_model_is_reported_as_unknown_model_not_as_a_bad_key():
    """``ModelNotFoundError`` subclasses ``KeyError``, so it has to be matched before
    the generic ValueError/TypeError/KeyError branch or the operator reading the failed
    job sees "invalid_request" for what is really a bad model name."""
    exc = ModelNotFoundError("segment-anything", ["cutoutnet", "u2net"])
    assert isinstance(exc, KeyError)

    result = classify(exc)
    assert result.kind is FailureKind.NON_RETRYABLE
    assert result.code == "unknown_model"


def test_missing_weights_are_not_retried():
    """A checkpoint does not appear on disk because a worker asked twice; this needs an
    operator to fetch or train it."""
    result = classify(WeightsUnavailableError("u2net", "models/u2net/u2net.pth"))
    assert result.code == "weights_unavailable"
    assert result.retryable is False


def test_a_missing_object_is_not_retried_even_though_storage_is_infrastructure():
    """The tempting classification is "storage problem, retry". But the asset row exists
    and its bytes do not, which means the upload never completed or the object is gone -
    neither heals on a retry."""
    result = classify(ObjectNotFoundError("uploads/ab/cd/ef.png"))
    assert result.kind is FailureKind.NON_RETRYABLE
    assert result.code == "object_missing"


def test_a_missing_file_is_not_retried_despite_being_an_oserror():
    """``FileNotFoundError`` is an ``OSError``, and the ``OSError`` fallback is
    retryable. Order is what keeps a missing temp file from being retried three times."""
    exc = FileNotFoundError("/tmp/frames/000001.png")
    assert isinstance(exc, OSError)

    result = classify(exc)
    assert result.kind is FailureKind.NON_RETRYABLE
    assert result.code == "file_missing"


@pytest.mark.parametrize("exc", [ValueError("bad alpha"), TypeError("expected ndarray")])
def test_programming_and_validation_errors_fail_immediately(exc: Exception):
    result = classify(exc)
    assert result.kind is FailureKind.NON_RETRYABLE
    assert result.code == "invalid_request"


def test_the_exception_type_wins_over_a_transient_looking_message():
    """ "Type first, message second" is the documented rule: a message is a string an
    upstream library can reword in a patch release, so a ``ValueError`` that happens to
    contain "timed out" must not become retryable."""
    result = classify(ValueError("the requested resolution timed out of the allowed range"))
    assert result.kind is FailureKind.NON_RETRYABLE


def test_an_unrecognised_exception_is_not_retried():
    """Deliberate: an unrecognised bug retried three times is three times the log noise
    and three times the latency for the same failure."""

    class NeverSeenBeforeError(Exception):
        pass

    result = classify(NeverSeenBeforeError("not in the table"))
    assert result.kind is FailureKind.NON_RETRYABLE
    assert result.code == "internal_error"
    assert "NeverSeenBeforeError" in result.message, (
        "the type name is what makes the failure searchable in the logs"
    )


# =============================================================== transient


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionResetError("connection reset by peer"),
        ConnectionRefusedError("connection refused"),
        TimeoutError("operation timed out"),
    ],
)
def test_connection_and_timeout_errors_are_retried(exc: Exception):
    result = classify(exc)
    assert result.kind is FailureKind.RETRYABLE
    assert result.retryable is True


def test_a_database_error_is_matched_by_type_name_without_importing_sqlalchemy():
    """Matching on the name keeps this module free of a driver import purely to write a
    classification table, and works for the DBAPI wrappers too."""

    class OperationalError(Exception):
        pass

    result = classify(OperationalError("server closed the connection unexpectedly"))
    assert result.kind is FailureKind.RETRYABLE
    assert result.code == "infrastructure_error"


@pytest.mark.parametrize(
    "message",
    [
        "deadlock detected",
        "could not serialize access due to concurrent update",
        "sorry, too many connections already",
        "503 Service Unavailable",
        "SlowDown: please reduce your request rate",
        "Throttling: rate exceeded",
        "resource temporarily unavailable",
    ],
)
def test_transient_infrastructure_is_recognised_from_the_message(message: str):
    result = classify(RuntimeError(message))
    assert result.kind is FailureKind.RETRYABLE
    assert result.code == "transient_error"


def test_a_generic_oserror_is_retried_as_io():
    result = classify(OSError("input/output error"))
    assert result.kind is FailureKind.RETRYABLE
    assert result.code == "io_error"


# ========================================================== the payload shape


def test_as_dict_carries_the_retry_decision_and_not_just_the_kind():
    """This dict is persisted on the run row, so a reader can see *why* a job was
    retried without having to reimplement the classification table to interpret it."""
    payload = classify(FakeCudaOutOfMemoryError()).as_dict()
    assert payload == {
        "kind": "oom",
        "code": "out_of_memory",
        "message": "CUDA out of memory. Tried to allocate 2.00 GiB",
        "retryable": True,
    }


def test_the_message_is_never_empty_even_for_an_exception_with_no_arguments():
    """An empty ``str(exc)`` is common for typed OOM errors, and a blank message on a
    failed job is useless to whoever is looking at it."""
    result = classify(FakeCudaOutOfMemoryError(""))
    assert result.message == "OutOfMemoryError"


def test_a_classification_is_immutable():
    """It is recorded on the run row and read afterwards; nothing should be able to
    rewrite the decision after the fact."""
    result = classify(ValueError("nope"))
    with pytest.raises(AttributeError):
        result.code = "something_else"  # type: ignore[misc]


def test_only_retryable_and_oom_are_retryable():
    assert Classification(FailureKind.RETRYABLE, "c", "m").retryable is True
    assert Classification(FailureKind.OOM, "c", "m").retryable is True
    assert Classification(FailureKind.NON_RETRYABLE, "c", "m").retryable is False


# ================================================================== backoff


def test_the_delay_grows_exponentially_with_the_attempt():
    """Compared on the centres rather than single samples, because the jitter bands of
    consecutive attempts overlap."""
    centres = [2.0, 4.0, 8.0, 16.0]
    for attempt, centre in enumerate(centres, start=1):
        samples = [retry_delay(attempt) for _ in range(200)]
        assert centre * 0.75 <= min(samples)
        assert max(samples) <= centre * 1.25


def test_the_delay_is_capped_so_a_long_outage_does_not_park_a_job_for_hours():
    """The cap bounds the centre, with jitter applied afterwards - so the ceiling is
    ``cap * 1.25``, not ``cap``. Asserted explicitly because "capped at 120" invites the
    reader to assume a hard maximum."""
    samples = [retry_delay(attempt) for attempt in range(8, 40) for _ in range(20)]
    assert max(samples) <= 120.0 * 1.25
    assert min(samples) >= 120.0 * 0.75


def test_the_delay_never_drops_to_zero():
    """A zero delay is a hot retry loop against whatever just failed."""
    assert retry_delay(1, base=0.0) >= 0.1
    assert retry_delay(1, base=0.001, jitter=1.0) >= 0.1


def test_a_first_or_zeroth_attempt_does_not_produce_a_negative_exponent():
    """Celery numbers retries from 0 in some code paths and 1 in others; neither may
    turn into ``base * 2**-1``."""
    for attempt in (-3, 0, 1):
        assert 1.5 <= retry_delay(attempt) <= 2.5


def test_jitter_actually_spreads_the_delay():
    """Without spread, every task that failed on the same dependency blip retries at the
    same instant and knocks it over again, which is the whole reason jitter is here."""
    samples = {retry_delay(4) for _ in range(50)}
    assert len(samples) > 40, "the delay is effectively deterministic"


def test_jitter_can_be_switched_off_for_a_deterministic_delay():
    assert retry_delay(3, jitter=0.0) == pytest.approx(8.0)


def test_the_backoff_is_reproducible_under_a_seeded_random_module():
    """It draws from the global ``random``, so a test that needs a fixed value can seed
    it - and a caller that needs independence from application code cannot."""
    random.seed(1234)
    first = [retry_delay(a) for a in range(1, 6)]
    random.seed(1234)
    assert [retry_delay(a) for a in range(1, 6)] == first
