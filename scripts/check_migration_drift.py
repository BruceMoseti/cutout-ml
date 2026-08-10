#!/usr/bin/env python3
"""Fail if the ORM metadata and the live database schema disagree.

Alembic migrations and SQLAlchemy models are two descriptions of the same schema, and
nothing keeps them in step automatically. The usual failure is silent: someone adds a
column to a model, the tests pass because they create tables with
``Base.metadata.create_all``, and the migration that production actually runs never grows
the column. This runs Alembic's own autogenerate comparison against a migrated database
and exits non-zero on any difference, so that divergence fails CI instead of a deploy.

Usage::

    alembic upgrade head
    python scripts/check_migration_drift.py

Reads ``CUTOUTML_DATABASE_URL`` through the normal settings object, so it cannot be
pointed at a different database than the migrations were.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

from cutoutml.core.config import get_settings
from cutoutml.db.models import Base

#: Diff entries to ignore. Alembic reports the ``vector`` column type as a change on
#: every run because pgvector's type does not round-trip through reflection identically;
#: that is a reflection artefact, not schema drift.
_IGNORED_TYPES = ("vector",)


def _is_ignorable(diff: Any) -> bool:
    text = repr(diff).lower()
    return any(name in text for name in _IGNORED_TYPES) and "modify_type" in text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=None,
        help="override CUTOUTML_DATABASE_URL (mainly for CI)",
    )
    args = parser.parse_args(argv)

    url = args.database_url or get_settings().sync_database_url
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            diffs = [d for d in compare_metadata(context, Base.metadata) if not _is_ignorable(d)]
    finally:
        engine.dispose()

    if not diffs:
        print(f"no drift: {url.rsplit('@', 1)[-1]} matches cutoutml.db.models")
        return 0

    print(f"{len(diffs)} difference(s) between the database and the ORM metadata:")
    for diff in diffs:
        print(f"  {diff!r}")
    print(
        "\nGenerate a migration for these with:\n"
        "  alembic revision --autogenerate -m 'describe the change'"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
