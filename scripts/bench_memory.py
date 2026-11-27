"""Measure Kairos Core RSS after a bounded, offline group-chat workload."""

import asyncio
import gc
import sys
import tracemalloc
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvloop

from kairos.config import BotProfile, DebounceConfig, RateLimitConfig
from kairos.main import MultiBotManager
from kairos.models import (
    AuditRecord,
    ChatMessage,
    LLMResult,
    MessageSegment,
    NormalizedMessage,
    RawGroupEvent,
)
from kairos.telemetry.audit_logger import AuditLogger

_BOTS = 3
_GROUPS_PER_BOT = 10
_MESSAGES = 5_000
_BATCH_SIZE = 300
_BUFFER_LIMIT = 40
_RSS_LIMIT = 180_000_000
_MB = 1_000_000
_REPLIES = ("今晚吃云吞", "我在上桥看河", "窗边的花开了", "这首歌循环了好久")


class CountingStore:
    """Acknowledge writes and audit records without retaining their payloads."""

    def __init__(self) -> None:
        self.inbound = 0
        self.replies = 0
        self.audits = 0
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
        self.audits += 1
        self._progress.set()

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

    async def chat_completion(
        self, model_key: str, messages: list[ChatMessage], **overrides: Any
    ) -> LLMResult:
        if len(messages) < 2 or messages[0].role != "system" or messages[-1].role != "user":
            raise ValueError("Expected a persona and a user turn in the model context")
        self.contexts += 1
        return LLMResult(
            content=self.reply,
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


@dataclass
class Benchmark:
    rss: int
    sessions: int
    max_buffer: int
    messages: int
    contexts: int
    replies: int


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
        timestamp=datetime.now(UTC),
    )


async def _exercise() -> Benchmark:
    store, llm, sender = CountingStore(), MockModel(), EchoSender()
    profiles = [
        BotProfile(
            bot_id=f"benchmark_{bot}",
            self_id=bot + 1,
            nickname=f"机器人{bot}",
            persona_prompt="我是群友，说话简短。",
            chat_model_key="mock",
            trigger_mode="all_with_veto",
            debounce=DebounceConfig(normal_seconds=0, max_wait_seconds=0, mention_seconds=0),
            rate_limit=RateLimitConfig(cooldown_seconds=0),
        )
        for bot in range(_BOTS)
    ]
    audit = AuditLogger(store)
    manager = MultiBotManager(profiles, store, llm, sender, audit)
    expected_turns = 0
    max_buffer = 0
    async with AsyncExitStack() as resources:
        resources.push_async_callback(audit.drain)
        resources.push_async_callback(manager.shutdown)
        for batch, start in enumerate(range(0, _MESSAGES, _BATCH_SIZE)):
            llm.reply = _REPLIES[batch % len(_REPLIES)]
            stop = min(start + _BATCH_SIZE, _MESSAGES)
            for index in range(start, stop):
                await manager.handle_group_event(_event(index, profiles))
            expected_turns += min(stop - start, _BOTS * _GROUPS_PER_BOT)
            await store.wait_for_audits(expected_turns)
            buffers = [
                len(session.buffer)
                for bot in manager.runtimes.values()
                for session in bot.sessions.values()
            ]
            max_buffer = max(max_buffer, max(buffers, default=0))
            if max_buffer > _BUFFER_LIMIT:
                raise RuntimeError(
                    f"Group buffer holds {max_buffer} messages (limit {_BUFFER_LIMIT})"
                )

    sessions = sum(len(bot.sessions) for bot in manager.runtimes.values())
    if sessions != _BOTS * _GROUPS_PER_BOT or store.inbound != _MESSAGES:
        raise RuntimeError(f"Incomplete workload: {sessions} sessions, {store.inbound} arrivals")
    if not llm.contexts == sender.sent == store.replies == expected_turns:
        raise RuntimeError(
            f"Incomplete chat turns: {llm.contexts} contexts, {sender.sent} sends, "
            f"{store.replies} reply writes (expected {expected_turns})"
        )

    # Keep active sessions resident while sampling, after pending writes and timers settle.
    gc.collect()
    return Benchmark(_rss_bytes(), sessions, max_buffer, store.inbound, llm.contexts, store.replies)


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
    if result.rss >= _RSS_LIMIT:
        print("Kairos Core exceeded the steady RSS limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
