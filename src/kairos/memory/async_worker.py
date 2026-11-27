"""Refresh the hot context and write derived memory outside the chat path."""

import asyncio
import json
from collections import OrderedDict
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

import structlog

from kairos.config import BotProfile
from kairos.models import (
    ChatMessage,
    ContextSummary,
    MemoryNote,
    NormalizedMessage,
    UserImpression,
    WarmMemory,
)
from kairos.protocols import ContextCompactor, LLMClient, MemoryProvider, MemoryStore, SummaryStore

_log = structlog.get_logger()
type _GroupKey = tuple[str, int]


class HotContextCache:
    def __init__(self, max_groups: int = 64) -> None:
        if max_groups < 1:
            raise ValueError("max_groups must be positive")
        self._max_groups = max_groups
        self._entries: OrderedDict[_GroupKey, WarmMemory] = OrderedDict()

    def get_warm_memory(self, bot_id: str, group_id: int) -> WarmMemory:
        key = (bot_id, group_id)
        memory = self._entries.get(key)
        if memory is None:
            return WarmMemory()
        self._entries.move_to_end(key)
        return memory

    def put(self, bot_id: str, group_id: int, memory: WarmMemory) -> None:
        key = (bot_id, group_id)
        self._entries[key] = memory
        self._entries.move_to_end(key)
        if len(self._entries) > self._max_groups:
            self._entries.popitem(last=False)

    def seed_summaries(
        self, bot_id: str, group_id: int, summaries: Sequence[ContextSummary]
    ) -> None:
        current = self.get_warm_memory(bot_id, group_id)
        self.put(bot_id, group_id, current.model_copy(update={"summaries": list(summaries)}))


@dataclass(slots=True)
class _GroupWork:
    last_msg_id: str | None = None
    warm_arrivals: int = 0
    note_arrivals: int = 0
    startup_scheduled: bool = False
    startup_pending: bool = False
    pending: tuple[NormalizedMessage, ...] | None = None
    task: asyncio.Task[None] | None = None


class AsyncMemoryWorker:
    def __init__(
        self,
        provider: MemoryProvider | None = None,
        memory_store: MemoryStore | None = None,
        summary_store: SummaryStore | None = None,
        compactor: ContextCompactor | None = None,
        llm: LLMClient | None = None,
        *,
        max_groups: int = 64,
    ) -> None:
        self._provider = provider
        self._memory_store = memory_store
        self._summary_store = summary_store
        self._compactor = compactor
        self._llm = llm
        self._cache = HotContextCache(max_groups)
        self._max_groups = max_groups
        self._groups: OrderedDict[_GroupKey, _GroupWork] = OrderedDict()
        self._closed = False

    def get_warm_memory(self, bot_id: str, group_id: int) -> WarmMemory:
        return self._cache.get_warm_memory(bot_id, group_id)

    def seed_summaries(
        self, bot_id: str, group_id: int, summaries: Sequence[ContextSummary]
    ) -> None:
        self._cache.seed_summaries(bot_id, group_id, summaries)

    def schedule_warmup(
        self, profile: BotProfile, group_id: int, messages: Sequence[NormalizedMessage]
    ) -> None:
        if self._closed or not (profile.memory.enabled or profile.compaction.enabled):
            return
        state = self._state((profile.bot_id, group_id))
        if state is None or state.startup_scheduled:
            return
        state.startup_scheduled = state.startup_pending = True
        if messages and state.last_msg_id is None:
            state.last_msg_id = messages[-1].last_msg_id
        self._dispatch(profile, group_id, messages, state)

    def on_messages_updated(
        self, profile: BotProfile, group_id: int, messages: Sequence[NormalizedMessage]
    ) -> None:
        if (
            self._closed
            or not messages
            or not (profile.memory.enabled or profile.compaction.enabled)
        ):
            return
        state = self._state((profile.bot_id, group_id))
        if state is None or state.last_msg_id == messages[-1].last_msg_id:
            return
        arrivals = self._new_arrivals(messages, state.last_msg_id)
        state.last_msg_id = messages[-1].last_msg_id
        state.warm_arrivals += arrivals
        state.note_arrivals += arrivals
        if (
            state.task is not None
            or self._warm_due(profile, state)
            or self._notes_due(profile, state)
            or self._compaction_due(profile, messages)
        ):
            self._dispatch(profile, group_id, messages, state)

    @staticmethod
    def _new_arrivals(messages: Sequence[NormalizedMessage], last_msg_id: str | None) -> int:
        if last_msg_id is None:
            return len(messages)
        # An evicted or coalesced tail still represents at least one new arrival.
        for count, message in enumerate(reversed(messages)):
            if message.last_msg_id == last_msg_id:
                return count
        return 1

    def _state(self, key: _GroupKey) -> _GroupWork | None:
        if key not in self._groups:
            if len(self._groups) == self._max_groups:
                evict = next(
                    (key for key, state in self._groups.items() if state.task is None), None
                )
                if evict is None:
                    return None
                del self._groups[evict]
            self._groups[key] = _GroupWork()
        self._groups.move_to_end(key)
        return self._groups[key]

    def _dispatch(
        self,
        profile: BotProfile,
        group_id: int,
        messages: Sequence[NormalizedMessage],
        state: _GroupWork,
    ) -> None:
        state.pending = tuple(messages)
        if state.task is None:
            state.task = asyncio.create_task(self._run(profile, group_id, state))

    def _warm_due(self, profile: BotProfile, state: _GroupWork) -> bool:
        return state.startup_pending or (
            profile.memory.enabled
            and self._provider is not None
            and state.warm_arrivals >= profile.memory.warm_trigger_messages
        )

    def _notes_due(self, profile: BotProfile, state: _GroupWork) -> bool:
        return (
            profile.memory.enabled
            and self._llm is not None
            and self._memory_store is not None
            and profile.worker_model_key is not None
            and state.note_arrivals >= profile.memory.note_extract_interval_messages
        )

    def _compaction_due(self, profile: BotProfile, messages: Sequence[NormalizedMessage]) -> bool:
        return (
            profile.compaction.enabled
            and self._compactor is not None
            and len(messages) > profile.compaction.keep_recent_messages
        )

    async def _run(self, profile: BotProfile, group_id: int, state: _GroupWork) -> None:
        try:
            while state.pending is not None:
                messages, state.pending = state.pending, None
                warm_due = self._warm_due(profile, state)
                notes_due = self._notes_due(profile, state)
                if warm_due:
                    state.startup_pending = False
                    state.warm_arrivals = 0
                if notes_due:
                    state.note_arrivals = 0

                key = (profile.bot_id, group_id)
                if warm_due:
                    await _run_job("warmup", self._refresh(profile, group_id, messages), key)
                if self._compaction_due(profile, messages):
                    await _run_job("compaction", self._compact(profile, group_id, messages), key)
                if notes_due:
                    await _run_job("extraction", self._extract(profile, group_id, messages), key)
        finally:
            state.task = None
            state.pending = None

    async def _refresh(
        self, profile: BotProfile, group_id: int, messages: Sequence[NormalizedMessage]
    ) -> None:
        bot_id = profile.bot_id
        if profile.memory.enabled and self._provider is not None:
            active_users = list(
                dict.fromkeys(
                    message.sender_qq
                    for message in reversed(messages)
                    if not (message.is_bot or message.is_peer_bot)
                )
            )
            recalled = await self._provider.recall(bot_id, group_id, active_users, messages)
        elif profile.compaction.enabled and self._summary_store is not None:
            summaries = await self._summary_store.load_recent_summaries(
                bot_id, group_id, profile.compaction.max_persisted_summaries
            )
            recalled = WarmMemory(summaries=summaries)
        else:
            return

        current = self.get_warm_memory(bot_id, group_id)
        summaries = _recent_summaries(
            [*current.summaries, *recalled.summaries], profile.compaction.max_persisted_summaries
        )
        self._cache.put(bot_id, group_id, recalled.model_copy(update={"summaries": summaries}))

    async def _compact(
        self, profile: BotProfile, group_id: int, messages: Sequence[NormalizedMessage]
    ) -> None:
        compactor = self._compactor
        current = self.get_warm_memory(profile.bot_id, group_id)
        if compactor is None or not compactor.should_compact(profile, messages, current.summaries):
            return
        summary = await compactor.compact(profile, group_id, messages, current.summaries)
        if summary is None:
            return
        if self._summary_store is not None:
            await self._summary_store.save_summary(summary)
        current = self.get_warm_memory(profile.bot_id, group_id)
        summaries = _recent_summaries(
            [*current.summaries, summary], profile.compaction.max_persisted_summaries
        )
        self.seed_summaries(profile.bot_id, group_id, summaries)

    async def _extract(
        self, profile: BotProfile, group_id: int, messages: Sequence[NormalizedMessage]
    ) -> None:
        if self._llm is None or self._memory_store is None or profile.worker_model_key is None:
            return
        recent = [
            message for message in messages[-16:] if not (message.is_bot or message.is_peer_bot)
        ]
        if not recent:
            return
        nicknames = {
            message.sender_qq: message.sender_alias.split(":", 1)[-1] for message in recent
        }
        dialogue = "\n".join(
            f"{message.sender_qq} {message.sender_alias}: {message.content[:240]}"
            for message in recent
        )
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    "从群聊提取有根据的短期印象与最多两条事件，忽略随口闲聊。只输出 JSON。"
                    '格式为 {"impressions":[{"user_qq":123,"impression_summary":"简短印象"}],'
                    '"notes":[{"summary_text":"简短事件","entity_qqs":[123],"keywords":["关键词"]}]}。'
                    "无内容时输出空数组。"
                ),
            ),
            ChatMessage(role="user", content=dialogue),
        ]
        result = await self._llm.chat_completion(
            profile.worker_model_key,
            prompt,
            session_id=f"{profile.bot_id}:{group_id}:memory",
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        payload = _parse_json_object(result.content[:8192])
        now = datetime.now(UTC)
        for row in _rows(payload.get("impressions"), len(nicknames)):
            try:
                impression = UserImpression.model_validate(
                    {
                        **row,
                        "bot_id": profile.bot_id,
                        "group_id": group_id,
                        "nickname": "",
                        "updated_at": now,
                    }
                )
            except ValueError:
                continue
            text = impression.impression_summary.strip()[:80]
            if text and impression.user_qq in nicknames and isfinite(impression.affinity_score):
                impression.nickname = nicknames[impression.user_qq]
                impression.impression_summary = text
                await self._memory_store.upsert_user_impression(impression)
        for row in _rows(payload.get("notes"), 2):
            try:
                note = MemoryNote.model_validate(
                    {
                        **row,
                        "bot_id": profile.bot_id,
                        "scope_group_id": group_id,
                        "created_at": now,
                        "last_recalled_at": None,
                    }
                )
            except ValueError:
                continue
            note.summary_text = note.summary_text.strip()[:160]
            if note.summary_text:
                note.entity_qqs = list(
                    dict.fromkeys(qq for qq in note.entity_qqs if qq in nicknames)
                )
                note.keywords = list(
                    dict.fromkeys(
                        key.strip().casefold()[:32] for key in note.keywords if key.strip()
                    )
                )[:8]
                note.importance = max(1, min(note.importance, 32767))
                await self._memory_store.save_memory_note(note)

    async def shutdown(self) -> None:
        self._closed = True
        tasks = [state.task for state in self._groups.values() if state.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # A task cancelled before its first turn never enters its finally block.
        self._groups.clear()


def _recent_summaries(summaries: Sequence[ContextSummary], limit: int) -> list[ContextSummary]:
    unique = {summary.covered_until_msg_id: summary for summary in summaries}
    return sorted(
        unique.values(), key=lambda summary: (summary.covered_until_at, summary.created_at)
    )[-limit:]


def _parse_json_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    first_object: dict[str, Any] | None = None
    index = 0
    while (start := content.find("{", index)) != -1:
        try:
            data, _ = decoder.raw_decode(content, start)
        except ValueError:
            index = start + 1
            continue
        if isinstance(data, dict):
            if "impressions" in data or "notes" in data:
                return data
            if first_object is None:
                first_object = data
        index = start + 1
    if first_object is not None:
        return first_object
    raise ValueError("Expected a JSON object from memory extraction")


def _rows(value: object, limit: int) -> list[dict[str, Any]]:
    return (
        [row for row in value[:limit] if isinstance(row, dict)] if isinstance(value, list) else []
    )


async def _run_job(name: str, operation: Awaitable[None], key: _GroupKey) -> None:
    try:
        await operation
    except Exception as exc:  # noqa: BLE001 - Other group jobs must continue after a failure.
        _log.warning(
            "memory_worker_job_failed",
            job=name,
            bot_id=key[0],
            group_id=key[1],
            error_type=type(exc).__name__,
        )
