"""PostgreSQL persistence shared by the bot runtimes."""

import asyncio
import json
from collections.abc import Sequence

import asyncpg

from kairos.config import PostgresConfig
from kairos.models import AuditRecord, ContextSummary, MemoryNote, NormalizedMessage, UserImpression

_INSERT_MESSAGE = """
    INSERT INTO group_messages (
        bot_id, group_id, msg_id, sender_qq, sender_alias, content,
        reply_to_msg_id, is_bot, created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    ON CONFLICT (bot_id, group_id, msg_id) DO NOTHING
"""

_SELECT_RECENT_MESSAGES = """
    SELECT bot_id, group_id, msg_id, sender_qq, sender_alias, content,
           reply_to_msg_id, is_bot, created_at
    FROM group_messages
    WHERE bot_id = $1 AND group_id = $2
    ORDER BY created_at DESC, id DESC
    LIMIT $3
"""

_INSERT_AUDIT_LOG = """
    INSERT INTO decision_audit_logs (
        bot_id, group_id, context_snapshot, gate_output, main_vetoed, final_reply,
        stale_action, latency_ms, human_label, created_at, action_type, target_msg_id, emoji_id
    )
    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6, $7, $8, $9, $10, $11, $12, $13)
"""

_INSERT_SUMMARY = """
    INSERT INTO context_summaries (
        bot_id, group_id, summary_text, covered_until_msg_id, covered_until_at,
        token_estimate, created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (bot_id, group_id, covered_until_msg_id) DO NOTHING
"""

_SELECT_RECENT_SUMMARIES = """
    SELECT bot_id, group_id, summary_text, covered_until_msg_id, covered_until_at,
           token_estimate, created_at
    FROM context_summaries
    WHERE bot_id = $1 AND group_id = $2
    ORDER BY covered_until_at DESC, id DESC
    LIMIT $3
"""

_UPSERT_IMPRESSION = """
    INSERT INTO user_impressions (
        bot_id, group_id, user_qq, nickname, impression_summary, affinity_score, updated_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (bot_id, group_id, user_qq) DO UPDATE
    SET nickname = EXCLUDED.nickname,
        impression_summary = EXCLUDED.impression_summary,
        affinity_score = EXCLUDED.affinity_score,
        updated_at = EXCLUDED.updated_at
    WHERE user_impressions.updated_at <= EXCLUDED.updated_at
"""

_SELECT_IMPRESSIONS = """
    SELECT bot_id, group_id, user_qq, nickname, impression_summary, affinity_score, updated_at
    FROM user_impressions
    WHERE bot_id = $1 AND group_id = $2 AND user_qq = ANY($3::bigint[])
    ORDER BY user_qq
    LIMIT $4
"""

_INSERT_NOTE = """
    INSERT INTO memory_notes (
        bot_id, scope_group_id, entity_qqs, summary_text, keywords, importance,
        last_recalled_at, created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""

_RECALL_NOTES = """
    WITH matches AS (
        SELECT id, (keywords && $3::text[])::int
                   + CASE WHEN $4::text <> '' THEN similarity(summary_text, $4) ELSE 0 END AS score
        FROM memory_notes
        WHERE bot_id = $1 AND scope_group_id = $2
          AND (keywords && $3::text[] OR ($4 <> '' AND summary_text % $4))
        ORDER BY score DESC, importance DESC, created_at DESC, id DESC
        LIMIT $5
    ), recalled AS (
        UPDATE memory_notes AS note
        SET last_recalled_at = GREATEST(note.last_recalled_at, statement_timestamp())
        FROM matches
        WHERE note.id = matches.id AND note.bot_id = $1 AND note.scope_group_id = $2
        RETURNING note.*, matches.score
    )
    SELECT bot_id, scope_group_id, entity_qqs, summary_text, keywords, importance,
           last_recalled_at, created_at
    FROM recalled
    ORDER BY score DESC, importance DESC, created_at DESC, id DESC
"""


class PgStore:
    """Implement MessageStore, SummaryStore and MemoryStore with one pool."""

    def __init__(self, config: PostgresConfig) -> None:
        self._config = config
        self._pool: asyncpg.Pool | None = None
        self._lifecycle_lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lifecycle_lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    dsn=self._config.dsn,
                    min_size=self._config.pool_min_size,
                    max_size=self._config.pool_max_size,
                )

    async def close(self) -> None:
        async with self._lifecycle_lock:
            pool = self._pool
            if pool is not None:
                self._pool = None
                await pool.close()

    def _connected_pool(self) -> asyncpg.Pool:
        pool = self._pool
        if pool is None:
            raise RuntimeError("PgStore has no open connection pool.")
        return pool

    async def save_message(self, msg: NormalizedMessage) -> None:
        await self._connected_pool().execute(
            _INSERT_MESSAGE,
            msg.bot_id,
            msg.group_id,
            msg.msg_id,
            msg.sender_qq,
            msg.sender_alias,
            msg.content,
            msg.reply_to_msg_id,
            msg.is_bot,
            msg.created_at,
        )

    async def load_recent_messages(
        self, bot_id: str, group_id: int, limit: int = 40
    ) -> list[NormalizedMessage]:
        """Restore individual arrivals in chronological order for session warmup."""
        rows = await self._connected_pool().fetch(_SELECT_RECENT_MESSAGES, bot_id, group_id, limit)
        return [
            NormalizedMessage(
                bot_id=row["bot_id"],
                group_id=row["group_id"],
                msg_id=row["msg_id"],
                last_msg_id=row["msg_id"],
                sender_qq=row["sender_qq"],
                sender_alias=row["sender_alias"],
                content=row["content"],
                reply_to_msg_id=row["reply_to_msg_id"],
                is_bot=row["is_bot"],
                is_at_bot=False,
                is_reply_to_bot=False,
                is_peer_bot=False,
                created_at=row["created_at"],
                latest_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    async def save_audit_log(self, record: AuditRecord) -> None:
        await self._connected_pool().execute(
            _INSERT_AUDIT_LOG,
            record.bot_id,
            record.group_id,
            json.dumps(record.context_snapshot, ensure_ascii=False, allow_nan=False),
            json.dumps(record.gate_output, ensure_ascii=False, allow_nan=False),
            record.main_vetoed,
            record.final_reply,
            record.stale_action,
            record.latency_ms,
            record.human_label,
            record.created_at,
            record.action_type,
            record.target_msg_id,
            record.emoji_id,
        )

    async def save_summary(self, summary: ContextSummary) -> None:
        await self._connected_pool().execute(
            _INSERT_SUMMARY,
            summary.bot_id,
            summary.group_id,
            summary.summary_text,
            summary.covered_until_msg_id,
            summary.covered_until_at,
            summary.token_estimate,
            summary.created_at,
        )

    async def load_recent_summaries(
        self, bot_id: str, group_id: int, limit: int = 3
    ) -> list[ContextSummary]:
        if limit <= 0:
            return []
        rows = await self._connected_pool().fetch(_SELECT_RECENT_SUMMARIES, bot_id, group_id, limit)
        return [ContextSummary.model_validate(dict(row)) for row in reversed(rows)]

    async def upsert_user_impression(self, impression: UserImpression) -> None:
        await self._connected_pool().execute(
            _UPSERT_IMPRESSION,
            impression.bot_id,
            impression.group_id,
            impression.user_qq,
            impression.nickname,
            impression.impression_summary,
            impression.affinity_score,
            impression.updated_at,
        )

    async def load_user_impressions(
        self, bot_id: str, group_id: int, user_qqs: Sequence[int]
    ) -> list[UserImpression]:
        ids = list(dict.fromkeys(user_qqs))
        if not ids:
            return []
        rows = await self._connected_pool().fetch(
            _SELECT_IMPRESSIONS, bot_id, group_id, ids, len(ids)
        )
        return [UserImpression.model_validate(dict(row)) for row in rows]

    async def save_memory_note(self, note: MemoryNote) -> None:
        await self._connected_pool().execute(
            _INSERT_NOTE,
            note.bot_id,
            note.scope_group_id,
            note.entity_qqs,
            note.summary_text,
            note.keywords,
            note.importance,
            note.last_recalled_at,
            note.created_at,
        )

    async def recall_memory_notes(
        self,
        bot_id: str,
        group_id: int,
        keywords: Sequence[str],
        query_text: str = "",
        limit: int = 5,
    ) -> list[MemoryNote]:
        query_text = query_text.strip()
        if limit <= 0 or (not keywords and not query_text):
            return []
        rows = await self._connected_pool().fetch(
            _RECALL_NOTES, bot_id, group_id, list(keywords), query_text, limit
        )
        return [MemoryNote.model_validate(dict(row)) for row in rows]

    async def health_check(self) -> bool:
        pool = self._pool
        if pool is None:
            return False
        try:
            result: object = await pool.fetchval("SELECT 1")
            return result == 1
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError):
            return False
