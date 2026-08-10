"""Alembic environment.

The URL comes from application settings rather than ``alembic.ini`` so there is
exactly one source of truth for "which database". ``compare_type`` and
``compare_server_default`` are enabled so autogenerate notices column type and
default changes, which it silently ignores by default - a common source of
migrations that pass review and then do nothing.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from cutoutml.core.config import get_settings
from cutoutml.db.models import Base
from cutoutml.db.session import build_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.sync_database_url)


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (``alembic upgrade head --sql``)."""
    context.configure(
        url=settings.sync_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    connectable = build_engine(settings)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
