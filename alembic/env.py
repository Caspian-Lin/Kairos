"""Run Kairos migrations from the configured PostgreSQL DSN."""

import asyncio
import os

from sqlalchemy import Connection, create_engine, pool
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

from alembic import context


def _connection_url() -> URL:
    dsn = os.environ.get("KAIROS_PG_DSN") or context.config.get_main_option("sqlalchemy.url")
    if not dsn:
        raise RuntimeError("Set KAIROS_PG_DSN to the target PostgreSQL connection URL.")
    try:
        url = make_url(dsn)
    except ArgumentError:
        raise RuntimeError("KAIROS_PG_DSN must be a valid PostgreSQL connection URL.") from None
    if url.drivername in {"postgres", "postgresql"}:
        return url.set(drivername="postgresql+asyncpg")
    if url.get_backend_name() != "postgresql":
        raise RuntimeError("Kairos migrations require PostgreSQL.")
    return url


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations(url: URL) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url, poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


def _run_online_migrations() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:
        _run_migrations(connection)
        return

    url = _connection_url()
    if url.get_dialect().is_async:
        asyncio.run(_run_async_migrations(url))
        return

    engine = create_engine(url, poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            _run_migrations(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    context.configure(dialect_name="postgresql", literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    _run_online_migrations()
