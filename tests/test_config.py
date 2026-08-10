"""Settings, and the documentation that is supposed to describe them.

The drift test here is the point of the file. `.env.example` is the only thing telling an
operator which knobs exist, and it is a plain text file with no mechanism keeping it
honest: adding a setting and forgetting the example is silent, and the operator finds out
by not being able to configure something. Asserting the correspondence makes the omission
a test failure instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cutoutml.core.config import DEFAULT_JWT_SECRET, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

#: Read by `apps/web/next.config.mjs` to build the API rewrite, not by `Settings`. It
#: belongs in `.env.example` because an operator has to set it, so it is exempted here
#: rather than being allowed to weaken the check for everything else.
FRONTEND_ONLY = {"CUTOUTML_API_URL"}


def documented_variables() -> set[str]:
    """Every `CUTOUTML_*` key in `.env.example`, commented-out ones included.

    A commented default is still documentation - it tells the reader the knob exists - so
    the leading `#` is deliberately optional in the pattern.
    """
    return set(re.findall(r"^#?\s*(CUTOUTML_[A-Z0-9_]+)=", ENV_EXAMPLE.read_text(), re.M))


def setting_variables() -> set[str]:
    return {f"CUTOUTML_{name.upper()}" for name in Settings.model_fields}


def test_every_setting_appears_in_the_env_example():
    missing = sorted(setting_variables() - documented_variables())
    assert missing == [], (
        f"{len(missing)} setting(s) are undocumented in .env.example: {missing}. "
        "An operator has no other list of what is configurable."
    )


def test_the_env_example_documents_nothing_that_does_not_exist():
    """A stale key is worse than a missing one: it looks configurable and silently is not."""
    unknown = sorted(documented_variables() - setting_variables() - FRONTEND_ONLY)
    assert unknown == [], f"stale or misspelled keys in .env.example: {unknown}"


def test_the_example_ships_no_usable_secret():
    """`.env.example` is committed, so anything that looks like a real credential in it
    would end up deployed by someone who copied the file and only edited the database URL."""
    text = ENV_EXAMPLE.read_text()
    secret_line = next(
        line for line in text.splitlines() if line.strip().startswith("CUTOUTML_JWT_SECRET=")
    )
    value = secret_line.split("=", 1)[1].strip()
    assert value in {"", DEFAULT_JWT_SECRET} or "change" in value.lower(), (
        "the example JWT secret must be empty or an obvious placeholder"
    )


def test_production_refuses_the_development_signing_key():
    """The failure mode this prevents is the worst one available: a deployment that boots
    happily while signing tokens with a key published in this repository."""
    with pytest.raises(ValueError, match="CUTOUTML_JWT_SECRET is still the development default"):
        Settings(environment="prod", jwt_secret=DEFAULT_JWT_SECRET)


def test_development_tolerates_it_so_a_fresh_clone_runs():
    assert Settings(environment="dev", jwt_secret=DEFAULT_JWT_SECRET).jwt_secret == (
        DEFAULT_JWT_SECRET
    )
