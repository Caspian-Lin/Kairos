"""Persist decision records independently of the chat pipeline."""

import asyncio

import structlog

from kairos.models import AuditRecord
from kairos.protocols import MessageStore

_log = structlog.get_logger(__name__)


class AuditLogger:
    def __init__(self, store: MessageStore) -> None:
        self._store = store
        self._tasks: set[asyncio.Task[None]] = set()

    def log_decision(self, record: AuditRecord) -> None:
        task = asyncio.create_task(self._persist(record))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*self._tasks)

    async def _persist(self, record: AuditRecord) -> None:
        try:
            await self._store.save_audit_log(record)
        except Exception as exc:  # noqa: BLE001 - Audit failures must remain non-blocking.
            _log.warning(
                "decision_audit_write_failed",
                bot_id=record.bot_id,
                group_id=record.group_id,
                error_type=type(exc).__name__,
            )
