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
import sqlalchemy as sa
from alembic.command import downgrade, upgrade
from alembic.config import Config
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from kairos.config import PostgresConfig
from kairos.memory.pg_store import PgStore
from kairos.models import AuditRecord, ContextSummary, MemoryNote, NormalizedMessage, UserImpression

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_MOCK_DSN = "postgresql://example.invalid/kairos"
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


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


def _summary(bot_id: str, msg_id: str = "100", *, at: datetime = _NOW) -> ContextSummary:
    return ContextSummary(
        bot_id=bot_id,
        group_id=200,
        summary_text="聊了烹饪",
        covered_until_msg_id=msg_id,
        covered_until_at=at,
        token_estimate=6,
        created_at=at,
    )


def _impression(bot_id: str) -> UserImpression:
    return UserImpression(
        bot_id=bot_id,
        group_id=200,
        user_qq=300,
        nickname="早安",
        impression_summary="喜欢烹饪",
        affinity_score=0.5,
        updated_at=_NOW,
    )


def _note(bot_id: str) -> MemoryNote:
    return MemoryNote(
        bot_id=bot_id,
        scope_group_id=200,
        entity_qqs=[300],
        summary_text="Weeknight mushroom risotto with garlic",
        keywords=["cooking", "risotto"],
        importance=2,
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
    audit = _audit().model_copy(
        update={"action_type": "react", "target_msg_id": "100", "emoji_id": "76"}
    )
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
        audit.action_type,
        audit.target_msg_id,
        audit.emoji_id,
    ]


async def test_save_and_load_summaries(store_and_pool: tuple[PgStore, AsyncMock]) -> None:
    store, pool = store_and_pool
    first = _summary("bot")
    second = _summary("bot", "101", at=_NOW + timedelta(seconds=1))
    await store.save_summary(first)
    query, *params = pool.execute.await_args.args
    assert "ON CONFLICT (bot_id, group_id, covered_until_msg_id) DO NOTHING" in query
    assert params == list(first.model_dump().values())

    pool.fetch.return_value = [second.model_dump(), first.model_dump()]
    assert await store.load_recent_summaries(first.bot_id, first.group_id, limit=2) == [
        first,
        second,
    ]
    query, *params = pool.fetch.await_args.args
    assert "WHERE bot_id = $1 AND group_id = $2" in query
    assert "ORDER BY covered_until_at DESC, id DESC" in query and "LIMIT $3" in query
    assert params == [first.bot_id, first.group_id, 2]


async def test_upsert_and_load_impressions(store_and_pool: tuple[PgStore, AsyncMock]) -> None:
    store, pool = store_and_pool
    impression = _impression("bot")
    await store.upsert_user_impression(impression)
    query, *params = pool.execute.await_args.args
    assert "ON CONFLICT (bot_id, group_id, user_qq) DO UPDATE" in query
    assert params == list(impression.model_dump().values())

    pool.fetch.return_value = [impression.model_dump()]
    assert await store.load_user_impressions("bot", 200, [300, 300, 400]) == [impression]
    query, *params = pool.fetch.await_args.args
    assert "WHERE bot_id = $1 AND group_id = $2 AND user_qq = ANY($3::bigint[])" in query
    assert "LIMIT $4" in query
    assert params == ["bot", 200, [300, 400], 2]


async def test_save_and_recall_notes(store_and_pool: tuple[PgStore, AsyncMock]) -> None:
    store, pool = store_and_pool
    note = _note("bot")
    await store.save_memory_note(note)
    query, *params = pool.execute.await_args.args
    assert note.summary_text not in query
    assert params == list(note.model_dump().values())

    recalled = note.model_copy(update={"last_recalled_at": _NOW})
    pool.fetch.return_value = [recalled.model_dump()]
    assert await store.recall_memory_notes("bot", 200, ("cooking",), "  garlic  ", 1) == [recalled]
    query, *params = pool.fetch.await_args.args
    assert "WHERE bot_id = $1 AND scope_group_id = $2" in query
    assert "keywords && $3::text[]" in query and "summary_text % $4" in query
    assert "UPDATE memory_notes" in query and "LIMIT $5" in query
    assert params == ["bot", 200, ["cooking"], "garlic", 1]


async def test_empty_memory_reads_do_not_touch_pool(
    store_and_pool: tuple[PgStore, AsyncMock],
) -> None:
    store, pool = store_and_pool
    assert await store.load_user_impressions("bot", 200, []) == []
    assert await store.load_recent_summaries("bot", 200, limit=0) == []
    assert await store.recall_memory_notes("bot", 200, [], "  ") == []
    assert await store.recall_memory_notes("bot", 200, ["cooking"], limit=0) == []
    pool.fetch.assert_not_awaited()


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
    config = Config(_ALEMBIC_INI, output_buffer=output)

    upgrade(config, "head", sql=True)
    upgrade_sql = output.getvalue()
    assert "CREATE TABLE group_messages" in upgrade_sql
    assert "CREATE TABLE decision_audit_logs" in upgrade_sql
    assert "uq_group_messages_bot_group_msgid UNIQUE (bot_id, group_id, msg_id)" in upgrade_sql
    assert "ON group_messages (bot_id, group_id, created_at DESC)" in upgrade_sql
    assert "ON decision_audit_logs (bot_id, group_id, created_at DESC)" in upgrade_sql
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in upgrade_sql
    assert "CREATE TABLE context_summaries" in upgrade_sql
    assert "CREATE TABLE user_impressions" in upgrade_sql
    assert "CREATE TABLE memory_notes" in upgrade_sql
    assert "USING gin (summary_text gin_trgm_ops)" in upgrade_sql
    assert "ALTER TABLE decision_audit_logs ADD COLUMN action_type TEXT" in upgrade_sql

    output.seek(0)
    output.truncate()
    downgrade(config, "head:base", sql=True)
    downgrade_sql = output.getvalue()
    assert "DROP INDEX idx_audit_logs_bot_group_ts" in downgrade_sql
    assert "DROP TABLE decision_audit_logs" in downgrade_sql
    assert "DROP INDEX idx_group_messages_bot_group_ts" in downgrade_sql
    assert "DROP TABLE group_messages" in downgrade_sql
    assert "ALTER TABLE decision_audit_logs DROP COLUMN action_type" in downgrade_sql
    assert "DROP TABLE context_summaries" in downgrade_sql
    assert "DROP TABLE user_impressions" in downgrade_sql
    assert "DROP TABLE memory_notes" in downgrade_sql


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


@pytest.mark.pg
async def test_real_postgres_m2_storage_and_isolation() -> None:
    dsn = os.environ.get("KAIROS_TEST_PG_DSN")
    if not dsn:
        pytest.skip("KAIROS_TEST_PG_DSN is unset")

    bot_id = f"test_{uuid4().hex}"
    other_bot = f"{bot_id}_other"
    store = PgStore(PostgresConfig(dsn=dsn, pool_min_size=1, pool_max_size=2))

    async with AsyncExitStack() as resources:
        observer = await asyncpg.connect(dsn)
        resources.push_async_callback(observer.close)
        await store.connect()
        resources.push_async_callback(store.close)
        try:
            first = _summary(bot_id)
            second = _summary(bot_id, "101", at=_NOW + timedelta(seconds=1))
            third = _summary(bot_id, "102", at=second.covered_until_at)
            for summary in (first, second, third, first, _summary(other_bot)):
                await store.save_summary(summary)
            await store.save_summary(first.model_copy(update={"group_id": 201}))
            assert await store.load_recent_summaries(bot_id, 200) == [first, second, third]
            assert await store.load_recent_summaries(bot_id, 200, limit=2) == [second, third]
            assert await store.load_recent_summaries(bot_id, 201) == [
                first.model_copy(update={"group_id": 201})
            ]

            impression = _impression(bot_id)
            revised = impression.model_copy(
                update={
                    "impression_summary": "喜欢意大利菜",
                    "updated_at": _NOW + timedelta(seconds=1),
                }
            )
            for item in (impression, revised, impression, _impression(other_bot)):
                await store.upsert_user_impression(item)
            await store.upsert_user_impression(impression.model_copy(update={"group_id": 201}))
            assert await store.load_user_impressions(bot_id, 200, [300, 400]) == [revised]
            assert await store.load_user_impressions(bot_id, 201, [300]) == [
                impression.model_copy(update={"group_id": 201})
            ]

            note = _note(bot_id)
            unrelated = note.model_copy(
                update={"summary_text": "Bike ride", "keywords": ["cycling"]}
            )
            for item in (note, unrelated, _note(other_bot)):
                await store.save_memory_note(item)
            await store.save_memory_note(note.model_copy(update={"scope_group_id": 201}))
            for keywords, query in ((["cooking"], ""), ([], "mushroom risotto garlic")):
                (match,) = await store.recall_memory_notes(bot_id, 200, keywords, query)
                assert match.model_dump(exclude={"last_recalled_at"}) == note.model_dump(
                    exclude={"last_recalled_at"}
                )
                assert match.last_recalled_at is not None
            assert (
                await observer.fetchval(
                    "SELECT count(*) FROM memory_notes WHERE bot_id IN ($1, $2) "
                    "AND last_recalled_at IS NOT NULL",
                    bot_id,
                    other_bot,
                )
                == 1
            )

            audit = _audit(bot_id).model_copy(
                update={"action_type": "react", "target_msg_id": "101", "emoji_id": "76"}
            )
            await store.save_audit_log(audit)
            row = await observer.fetchrow(
                "SELECT action_type, target_msg_id, emoji_id FROM decision_audit_logs "
                "WHERE bot_id = $1 AND group_id = $2",
                bot_id,
                200,
            )
            assert row is not None and tuple(row) == ("react", "101", "76")
        finally:
            bot_ids = [bot_id, other_bot]
            await observer.execute(
                "DELETE FROM memory_notes WHERE bot_id = ANY($1::text[])", bot_ids
            )
            await observer.execute(
                "DELETE FROM user_impressions WHERE bot_id = ANY($1::text[])", bot_ids
            )
            await observer.execute(
                "DELETE FROM context_summaries WHERE bot_id = ANY($1::text[])", bot_ids
            )
            await observer.execute("DELETE FROM decision_audit_logs WHERE bot_id = $1", bot_id)


@pytest.mark.pg
async def test_real_postgres_m2_upgrade_and_downgrade() -> None:
    dsn = os.environ.get("KAIROS_TEST_PG_DSN")
    if not dsn:
        pytest.skip("KAIROS_TEST_PG_DSN is unset")

    schema = f"test_{uuid4().hex}"
    engine = create_async_engine(make_url(dsn).set(drivername="postgresql+asyncpg"))

    def migrate(connection: sa.Connection) -> None:
        config = Config(_ALEMBIC_INI)
        config.attributes["connection"] = connection
        upgrade(config, "0001_m1_core_tables")
        connection.execute(
            sa.text(
                "INSERT INTO group_messages "
                "(bot_id, group_id, msg_id, sender_qq, sender_alias, content, created_at) "
                "VALUES ('m1', 200, '100', 300, 'U1', 'message', now())"
            )
        )
        upgrade(config, "head")
        assert "action_type" in {
            column["name"]
            for column in sa.inspect(connection).get_columns("decision_audit_logs", schema=schema)
        }
        assert set(sa.inspect(connection).get_table_names(schema=schema)) >= {
            "memory_notes",
            "context_summaries",
            "user_impressions",
        }
        downgrade(config, "0001_m1_core_tables")
        inspector = sa.inspect(connection)
        assert set(inspector.get_table_names(schema=schema)) == {
            "alembic_version",
            "decision_audit_logs",
            "group_messages",
        }
        assert "action_type" not in {
            column["name"] for column in inspector.get_columns("decision_audit_logs", schema=schema)
        }
        assert connection.scalar(sa.text("SELECT content FROM group_messages")) == "message"

    try:
        async with engine.begin() as connection:
            await connection.execute(sa.schema.CreateSchema(schema))
            await connection.execute(
                sa.text("SELECT set_config('search_path', :schema_path, true)"),
                {"schema_path": f"{schema}, public"},
            )
            # Keep the migration version separate from the already migrated test database.
            version = sa.Table(
                "alembic_version",
                sa.MetaData(),
                sa.Column("version_num", sa.String(32), primary_key=True),
                schema=schema,
            )
            await connection.execute(sa.schema.CreateTable(version))
            await connection.run_sync(migrate)
            await connection.execute(sa.schema.DropSchema(schema, cascade=True))
    finally:
        await engine.dispose()
