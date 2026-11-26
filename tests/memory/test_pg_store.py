"""Storage contracts for mocked asyncpg and a migrated test database."""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest
from alembic.command import downgrade, upgrade
from alembic.config import Config

from kairos.config import PostgresConfig
from kairos.memory.pg_store import PgStore
from kairos.models import AuditRecord, NormalizedMessage

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_MOCK_DSN = "postgresql://example.invalid/kairos"


def _message(
    msg_id: str = "100",
    *,
    bot_id: str = "lin_xiaoman",
    group_id: int = 200,
    created_at: datetime = _NOW,
) -> NormalizedMessage:
    return NormalizedMessage(
        bot_id=bot_id,
        group_id=group_id,
        msg_id=msg_id,
        last_msg_id=msg_id,
        sender_qq=300,
        sender_alias="U1:早安",
        content="群友说 '你好'",
        reply_to_msg_id="99",
        is_bot=False,
        is_at_bot=False,
        is_reply_to_bot=False,
        is_peer_bot=False,
        created_at=created_at,
        latest_at=created_at,
    )


def _audit(bot_id: str = "lin_xiaoman") -> AuditRecord:
    return AuditRecord(
        bot_id=bot_id,
        group_id=200,
        context_snapshot=[{"role": "user", "content": "你好"}],
        gate_output={"action": "pass", "labels": ["@"]},
        main_vetoed=False,
        final_reply="早安",
        stale_action="SEND_DIRECT",
        latency_ms=42,
        human_label=None,
        created_at=_NOW,
    )


@pytest.fixture
async def store_and_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[PgStore, AsyncMock]]:
    pool = AsyncMock(spec=asyncpg.Pool)
    monkeypatch.setattr(asyncpg, "create_pool", AsyncMock(return_value=pool))
    store = PgStore(PostgresConfig(dsn=_MOCK_DSN))
    await store.connect()
    try:
        yield store, pool
    finally:
        await store.close()


async def test_connect_and_close_share_one_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = AsyncMock(spec=asyncpg.Pool)
    create_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr(asyncpg, "create_pool", create_pool)
    store = PgStore(PostgresConfig(dsn=_MOCK_DSN, pool_min_size=1, pool_max_size=4))

    assert await store.health_check() is False
    await asyncio.gather(store.connect(), store.connect())
    create_pool.assert_awaited_once_with(dsn=_MOCK_DSN, min_size=1, max_size=4)
    await asyncio.gather(store.close(), store.close())
    pool.close.assert_awaited_once_with()
    assert await store.health_check() is False


async def test_save_message_uses_parameterized_idempotent_insert(
    store_and_pool: tuple[PgStore, AsyncMock],
) -> None:
    store, pool = store_and_pool
    message = _message()
    await store.save_message(message)

    query, *params = pool.execute.await_args.args
    assert "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)" in query
    assert "ON CONFLICT (bot_id, group_id, msg_id) DO NOTHING" in query
    assert message.content not in query
    assert params == [
        message.bot_id,
        message.group_id,
        message.msg_id,
        message.sender_qq,
        message.sender_alias,
        message.content,
        message.reply_to_msg_id,
        message.is_bot,
        message.created_at,
    ]


async def test_load_recent_messages_queries_bot_and_group_and_restores_in_order(
    store_and_pool: tuple[PgStore, AsyncMock],
) -> None:
    store, pool = store_and_pool
    first = _message()
    second = _message("101", created_at=_NOW + timedelta(seconds=1)).model_copy(
        update={"is_bot": True}
    )
    pool.fetch.return_value = [
        message.model_dump(
            exclude={"last_msg_id", "latest_at", "is_at_bot", "is_reply_to_bot", "is_peer_bot"}
        )
        for message in (second, first)
    ]

    assert await store.load_recent_messages(first.bot_id, first.group_id, limit=2) == [
        first,
        second,
    ]
    query, *params = pool.fetch.await_args.args
    assert "WHERE bot_id = $1 AND group_id = $2" in query
    assert "ORDER BY created_at DESC, id DESC" in query
    assert "LIMIT $3" in query
    assert params == [first.bot_id, first.group_id, 2]


async def test_save_audit_log_serializes_jsonb_and_preserves_metadata(
    store_and_pool: tuple[PgStore, AsyncMock],
) -> None:
    store, pool = store_and_pool
    audit = _audit()
    await store.save_audit_log(audit)

    query, bot_id, group_id, context, gate, *params = pool.execute.await_args.args
    assert "$3::jsonb" in query and "$4::jsonb" in query
    assert (bot_id, group_id) == (audit.bot_id, audit.group_id)
    assert json.loads(context) == audit.context_snapshot
    assert json.loads(gate) == audit.gate_output
    assert params == [
        audit.main_vetoed,
        audit.final_reply,
        audit.stale_action,
        audit.latency_ms,
        audit.human_label,
        audit.created_at,
    ]


async def test_health_check_reports_up_and_unavailable(
    store_and_pool: tuple[PgStore, AsyncMock],
) -> None:
    store, pool = store_and_pool
    pool.fetchval.return_value = 1

    assert await store.health_check() is True
    pool.fetchval.assert_awaited_once_with("SELECT 1")
    pool.fetchval.side_effect = OSError("connection lost")
    assert await store.health_check() is False


def test_offline_migration_creates_and_drops_core_tables_and_indexes() -> None:
    output = StringIO()
    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    config = Config(config_path, output_buffer=output)

    upgrade(config, "head", sql=True)
    upgrade_sql = output.getvalue()
    assert "CREATE TABLE group_messages" in upgrade_sql
    assert "CREATE TABLE decision_audit_logs" in upgrade_sql
    assert "uq_group_messages_bot_group_msgid UNIQUE (bot_id, group_id, msg_id)" in upgrade_sql
    assert "ON group_messages (bot_id, group_id, created_at DESC)" in upgrade_sql
    assert "ON decision_audit_logs (bot_id, group_id, created_at DESC)" in upgrade_sql

    output.seek(0)
    output.truncate()
    downgrade(config, "head:base", sql=True)
    downgrade_sql = output.getvalue()
    assert "DROP INDEX idx_audit_logs_bot_group_ts" in downgrade_sql
    assert "DROP TABLE decision_audit_logs" in downgrade_sql
    assert "DROP INDEX idx_group_messages_bot_group_ts" in downgrade_sql
    assert "DROP TABLE group_messages" in downgrade_sql


@pytest.mark.pg
async def test_real_postgres_round_trip() -> None:
    """Use a disposable database that has already been migrated to head."""
    dsn = os.environ.get("KAIROS_TEST_PG_DSN")
    if not dsn:
        pytest.skip("KAIROS_TEST_PG_DSN is unset")

    bot_id = f"test_{uuid4().hex}"
    other_bot_id = f"{bot_id}_other"
    first = _message(bot_id=bot_id)
    second = _message("101", bot_id=bot_id)
    third = _message("102", bot_id=bot_id, created_at=_NOW + timedelta(seconds=1))
    another_group = _message(bot_id=bot_id, group_id=201)
    another_bot = _message(bot_id=other_bot_id)
    audit = _audit(bot_id)
    store = PgStore(PostgresConfig(dsn=dsn, pool_min_size=1, pool_max_size=2))

    async with AsyncExitStack() as resources:
        observer = await asyncpg.connect(dsn)
        resources.push_async_callback(observer.close)
        migrated = await observer.fetchval(
            "SELECT to_regclass('group_messages') IS NOT NULL "
            "AND to_regclass('decision_audit_logs') IS NOT NULL"
        )
        assert migrated, "Run Alembic upgrade head on the disposable test database first."
        await store.connect()
        resources.push_async_callback(store.close)
        try:
            assert await store.health_check() is True
            for msg in (first, second, third, another_group, another_bot):
                await store.save_message(msg)
            await store.save_message(first.model_copy(update={"content": "retry"}))

            assert await store.load_recent_messages(bot_id, first.group_id) == [
                first,
                second,
                third,
            ]
            assert await store.load_recent_messages(bot_id, first.group_id, limit=2) == [
                second,
                third,
            ]
            assert await store.load_recent_messages(bot_id, another_group.group_id) == [
                another_group
            ]
            assert await store.load_recent_messages(other_bot_id, first.group_id) == [another_bot]

            await store.save_audit_log(audit)
            row = await observer.fetchrow(
                "SELECT context_snapshot, gate_output, stale_action, final_reply "
                "FROM decision_audit_logs WHERE bot_id = $1 AND group_id = $2",
                audit.bot_id,
                audit.group_id,
            )
            assert row is not None
            assert json.loads(row["context_snapshot"]) == audit.context_snapshot
            assert json.loads(row["gate_output"]) == audit.gate_output
            assert row["stale_action"] == audit.stale_action
            assert row["final_reply"] == audit.final_reply
        finally:
            await observer.execute("DELETE FROM decision_audit_logs WHERE bot_id = $1", bot_id)
            await observer.execute(
                "DELETE FROM group_messages WHERE bot_id = ANY($1::text[])", [bot_id, other_bot_id]
            )
