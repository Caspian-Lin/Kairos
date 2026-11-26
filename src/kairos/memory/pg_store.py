"""PostgreSQL persistence shared by the bot runtimes."""

import asyncio
import json

import asyncpg

from kairos.config import PostgresConfig
from kairos.models import AuditRecord, NormalizedMessage

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
        stale_action, latency_ms, human_label, created_at
    )
    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6, $7, $8, $9, $10)
"""


class PgStore:
    """Implement MessageStore with one pool for the process."""

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
        )

    async def health_check(self) -> bool:
        pool = self._pool
        if pool is None:
            return False
        try:
            result: object = await pool.fetchval("SELECT 1")
            return result == 1
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError):
            return False
