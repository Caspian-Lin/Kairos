"""Measure Kairos Core RSS after a bounded, offline group-chat workload."""

import asyncio
import gc
import sys
import tracemalloc
from collections import deque
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import uvloop

from kairos.config import (
    BotProfile,
    CompactionConfig,
    DebounceConfig,
    GateConfig,
    MemoryConfig,
    RateLimitConfig,
)
from kairos.main import MultiBotManager
from kairos.models import (
    AuditRecord,
    ChatMessage,
    ContextSummary,
    LLMResult,
    MemoryNote,
    MessageSegment,
    NormalizedMessage,
    RawGroupEvent,
    UserImpression,
)
from kairos.telemetry.audit_logger import AuditLogger
from kairos.telemetry.log_buffer import LogRingBuffer
from kairos.telemetry.trace_store import PipelineTraceStore

_BOTS = 3
_GROUPS_PER_BOT = 10
_MESSAGES = 5_000
_BATCH_SIZE = 300
_BUFFER_LIMIT = 40
_THREAD_LIMIT = 20
_TRACKED_MESSAGE_LIMIT = 256
_HOT_CACHE_LIMIT = 64
_SUMMARY_LIMIT = 3
_LOG_LIMIT = 500
_TRACE_LIMIT = 100
_RSS_LIMIT = 180_000_000
_MB = 1_000_000
_REPLIES = ("今晚吃云吞", "我在上桥看河", "窗边的花开了", "这首歌循环了好久")


class CountingStore:
    """Simulate the database with bounded summary storage and representative memory."""

    def __init__(self, logs: LogRingBuffer) -> None:
        self._logs = logs
        self.inbound = 0
        self.replies = 0
        self.audits = 0
        self.summary_writes = 0
        self.note_writes = 0
        self.summaries: dict[tuple[str, int], deque[ContextSummary]] = {}
        self._progress = asyncio.Event()

    async def load_recent_messages(
        self, bot_id: str, group_id: int, limit: int = 40
    ) -> list[NormalizedMessage]:
        return []

    async def health_check(self) -> bool:
        return True

    async def save_message(self, msg: NormalizedMessage) -> None:
        if msg.is_bot:
            self.replies += 1
        else:
            self.inbound += 1

    async def save_audit_log(self, record: AuditRecord) -> None:
        self._logs.processor(
            None,
            "info",
            {
                "event": "decision_recorded",
                "bot_id": record.bot_id,
                "group_id": record.group_id,
                "outcome": record.stale_action,
            },
        )
        self.audits += 1
        self._progress.set()

    async def save_summary(self, summary: ContextSummary) -> None:
        self.summary_writes += 1
        self.summaries.setdefault(
            (summary.bot_id, summary.group_id), deque(maxlen=_SUMMARY_LIMIT)
        ).append(summary)

    async def load_recent_summaries(
        self, bot_id: str, group_id: int, limit: int = _SUMMARY_LIMIT
    ) -> list[ContextSummary]:
        return list(self.summaries.get((bot_id, group_id), ()))[-limit:]

    async def upsert_user_impression(self, impression: UserImpression) -> None:
        pass

    async def load_user_impressions(
        self, bot_id: str, group_id: int, user_qqs: Sequence[int]
    ) -> list[UserImpression]:
        return [
            UserImpression(
                bot_id=bot_id,
                group_id=group_id,
                user_qq=user_qq,
                nickname=f"群友{user_qq}",
                impression_summary="喜欢聊附近美食和休闲活动",
                updated_at=datetime.now(UTC),
            )
            for user_qq in user_qqs
        ]

    async def save_memory_note(self, note: MemoryNote) -> None:
        self.note_writes += 1

    async def recall_memory_notes(
        self,
        bot_id: str,
        group_id: int,
        keywords: Sequence[str],
        query_text: str = "",
        limit: int = 5,
    ) -> list[MemoryNote]:
        return [
            MemoryNote(
                bot_id=bot_id,
                scope_group_id=group_id,
                entity_qqs=[],
                summary_text="群友相约晚饭后散步。",
                keywords=["晚饭", "散步"],
                created_at=datetime.now(UTC),
            )
        ][:limit]

    async def wait_for_audits(self, target: int) -> None:
        try:
            async with asyncio.timeout(10):
                while self.audits < target:
                    self._progress.clear()
                    await self._progress.wait()
        except TimeoutError:
            raise RuntimeError(f"Audit progress stopped at {self.audits}/{target}") from None


class MockModel:
    def __init__(self) -> None:
        self.reply = _REPLIES[0]
        self.contexts = 0
        self.gates = 0
        self.summaries = 0
        self.with_memory = 0
        self.with_summary = 0

    async def chat_completion(
        self,
        model_key: str,
        messages: list[ChatMessage],
        *,
        session_id: str | None = None,
        **overrides: Any,
    ) -> LLMResult:
        if len(messages) < 2 or messages[0].role != "system" or messages[-1].role != "user":
            raise ValueError("Expected a persona and a user turn in the model context")
        if model_key == "gate":
            self.gates += 1
            content = '{"action":"pass","reason":"群友分享见闻"}'
        elif model_key == "worker" and session_id is not None and session_id.endswith(":summary"):
            self.summaries += 1
            content = "群友聊过晚饭安排，也分享了最近的见闻。"
        elif model_key == "worker":
            content = '{"notes":[{"summary_text":"群友约饭","entity_qqs":[],"keywords":["约饭"]}]}'
        else:
            self.contexts += 1
            self.with_memory += (
                "群友画像" in messages[0].content and "相关事件笔记" in messages[0].content
            )
            self.with_summary += "历史对话摘要" in messages[0].content
            content = self.reply
        return LLMResult(
            content=content,
            model=model_key,
            latency_ms=0,
            prompt_tokens=None,
            completion_tokens=None,
        )


class EchoSender:
    def __init__(self) -> None:
        self.sent = 0

    async def send_group_msg(
        self, self_id: int, group_id: int, text: str, reply_to_msg_id: str | None = None
    ) -> str:
        self.sent += 1
        return str(_MESSAGES + self.sent)

    async def set_msg_emoji_like(
        self, self_id: int, message_id: str, emoji_id: str, *, set_like: bool = True
    ) -> None:
        raise RuntimeError("The benchmark gate should pass each turn to the chat model")


@dataclass
class Benchmark:
    rss: int
    sessions: int
    max_buffer: int
    messages: int
    contexts: int
    replies: int
    gates: int
    summary_writes: int
    note_writes: int
    max_threads: int
    max_thread_messages: int
    hot_groups: int
    max_summaries: int
    logs: int
    traces: int


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is missing from /proc/self/status")


def _event(index: int, profiles: list[BotProfile]) -> RawGroupEvent:
    sessions = _BOTS * _GROUPS_PER_BOT
    bot, group = divmod(index % sessions, _GROUPS_PER_BOT)
    speaker = index // sessions % 20
    return RawGroupEvent(
        self_id=profiles[bot].self_id,
        group_id=group + 1,
        msg_id=str(index + 1),
        sender_qq=10_000 + speaker,
        sender_nickname=f"群友{speaker}",
        sender_card="",
        segments=[
            MessageSegment(
                type="text",
                data={"text": f"第 {index + 1} 条群聊回放。有人约饭，有人分享路上的见闻。"},
            )
        ],
        raw_message="",
        # Simulate independent conversations across a day to exercise thread eviction.
        timestamp=datetime.now(UTC) + timedelta(minutes=6 * (index // sessions)),
    )


async def _exercise() -> Benchmark:
    logs, traces = LogRingBuffer(_LOG_LIMIT), PipelineTraceStore(_TRACE_LIMIT)
    store, llm, sender = CountingStore(logs), MockModel(), EchoSender()
    profiles = [
        BotProfile(
            bot_id=f"benchmark_{bot}",
            self_id=bot + 1,
            nickname=f"机器人{bot}",
            persona_prompt="我是群友，说话简短。",
            chat_model_key="chat",
            gate_model_key="gate",
            worker_model_key="worker",
            trigger_mode="all_with_veto",
            gate=GateConfig(enabled=True, energy_max=1_000, energy_recover_per_min=0),
            compaction=CompactionConfig(
                enabled=True,
                input_token_budget=65536,
                trigger_tokens=65536,
                target_summary_tokens=16384,
                max_persisted_summaries=_SUMMARY_LIMIT,
            ),
            memory=MemoryConfig(enabled=True, max_impressions_recalled=3),
            debounce=DebounceConfig(normal_seconds=0, max_wait_seconds=0, mention_seconds=0),
            rate_limit=RateLimitConfig(cooldown_seconds=0),
        )
        for bot in range(_BOTS)
    ]
    audit = AuditLogger(store)
    manager = MultiBotManager(
        profiles,
        store,
        llm,
        sender,
        audit,
        summary_store=store,
        memory_store=store,
        trace_store=traces,
    )
    worker = manager.memory_worker
    if worker is None:
        raise RuntimeError("Expected a running memory worker")
    expected_turns = 0
    max_buffer = 0
    max_threads = 0
    max_thread_messages = 0
    max_summaries = 0
    hot_groups = 0
    expected_sessions = _BOTS * _GROUPS_PER_BOT
    async with AsyncExitStack() as resources:
        resources.push_async_callback(audit.drain)
        resources.push_async_callback(manager.shutdown)
        for batch, start in enumerate(range(0, _MESSAGES, _BATCH_SIZE)):
            llm.reply = _REPLIES[batch % len(_REPLIES)]
            stop = min(start + _BATCH_SIZE, _MESSAGES)
            for index in range(start, stop):
                await manager.handle_group_event(_event(index, profiles))
            expected_turns += min(stop - start, expected_sessions)
            await store.wait_for_audits(expected_turns)
            async with asyncio.timeout(10):
                await asyncio.gather(
                    *(state.task for state in worker._groups.values() if state.task is not None)
                )
            group_sessions = [
                session for bot in manager.runtimes.values() for session in bot.sessions.values()
            ]
            max_buffer = max(max_buffer, *(len(session.buffer) for session in group_sessions))
            if max_buffer > _BUFFER_LIMIT:
                raise RuntimeError(
                    f"Group buffer holds {max_buffer} messages (limit {_BUFFER_LIMIT})"
                )
            hot_groups = len(worker._cache._entries)
            for session in group_sessions:
                tracker = session.thread_tracker
                if tracker is None:
                    raise RuntimeError("Expected a thread tracker for each M2 session")
                max_threads = max(
                    max_threads,
                    len(tracker.active_threads(datetime.now(UTC), limit=_THREAD_LIMIT + 1)),
                )
                max_thread_messages = max(max_thread_messages, len(tracker._message_threads))
                memory = worker.get_warm_memory(session.profile.bot_id, session.group_id)
                max_summaries = max(max_summaries, len(memory.summaries), len(session.summaries))
                if not memory.impressions or not memory.notes:
                    raise RuntimeError("Group hot cache did not warm up")
            if max_threads > _THREAD_LIMIT or max_thread_messages > _TRACKED_MESSAGE_LIMIT:
                raise RuntimeError("Thread state exceeded its configured capacities")
            if hot_groups > _HOT_CACHE_LIMIT or max_summaries > _SUMMARY_LIMIT:
                raise RuntimeError("Hot memory or summaries exceeded their configured capacities")

        # Let completion callbacks remove finished writes before draining under uvloop.
        await asyncio.sleep(0)
        await audit.drain()
        sessions = len(group_sessions)
        if sessions != expected_sessions or store.inbound != _MESSAGES or hot_groups != sessions:
            raise RuntimeError(
                f"Incomplete workload: {sessions} sessions, {hot_groups} warmed, {store.inbound} arrivals"
            )
        if not llm.gates == llm.contexts == sender.sent == store.replies == expected_turns:
            raise RuntimeError(
                f"Incomplete chat turns: {llm.gates} gates, {llm.contexts} contexts, "
                f"{sender.sent} sends, {store.replies} reply writes (expected {expected_turns})"
            )
        if not (
            max_threads == _THREAD_LIMIT
            and llm.summaries == store.summary_writes > 0
            and llm.with_memory
            and llm.with_summary
            and store.note_writes
        ):
            raise RuntimeError("The workload did not exercise every M2 memory path")

        log_count = len(logs.recent(limit=_LOG_LIMIT + 1))
        trace_count = len(traces._ring.snapshots())
        latest = traces.list_recent(limit=1)
        if not (
            log_count == min(_LOG_LIMIT, expected_turns)
            and trace_count == min(_TRACE_LIMIT, expected_turns)
            and latest[0].request_context is not None
        ):
            raise RuntimeError("M2.1 telemetry did not cover the expected turns")

        # Sample the resident data after asynchronous writes settle, with caches still live.
        gc.collect()
        return Benchmark(
            _rss_bytes(),
            sessions,
            max_buffer,
            store.inbound,
            llm.contexts,
            store.replies,
            llm.gates,
            store.summary_writes,
            store.note_writes,
            max_threads,
            max_thread_messages,
            hot_groups,
            max_summaries,
            log_count,
            trace_count,
        )


def main() -> int:
    try:
        startup_rss = _rss_bytes()
        tracemalloc.start()
        try:
            result = uvloop.run(_exercise())
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    except (OSError, RuntimeError) as exc:
        print(f"Memory benchmark failed: {exc}", file=sys.stderr)
        return 1

    print(f"Startup RSS: {startup_rss / _MB:.2f} MB")
    print(f"Steady RSS: {result.rss / _MB:.2f} MB (limit < {_RSS_LIMIT / _MB:.0f} MB)")
    print(f"tracemalloc peak: {peak / _MB:.2f} MB")
    print(
        f"{_BOTS} bots, {result.sessions} sessions, {result.messages} arrivals, "
        f"{result.contexts} contexts, {result.replies} persisted replies, "
        f"max buffer {result.max_buffer}/{_BUFFER_LIMIT}"
    )
    print(
        f"{result.gates} gate calls, {result.summary_writes} summary writes, {result.note_writes} note writes, "
        f"max threads {result.max_threads}/{_THREAD_LIMIT}, "
        f"max tracked message IDs {result.max_thread_messages}/{_TRACKED_MESSAGE_LIMIT}, "
        f"hot groups {result.hot_groups}/{_HOT_CACHE_LIMIT}, "
        f"max cached summaries per group {result.max_summaries}/{_SUMMARY_LIMIT}"
    )
    print(f"M2.1 retained logs {result.logs}/{_LOG_LIMIT}, traces {result.traces}/{_TRACE_LIMIT}")
    if result.rss >= _RSS_LIMIT:
        print("Kairos Core exceeded the steady RSS limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
