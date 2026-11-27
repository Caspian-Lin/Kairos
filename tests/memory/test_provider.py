import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from kairos.memory.provider import PgMemoryProvider, extract_keywords
from kairos.models import ContextSummary, MemoryNote, NormalizedMessage, UserImpression, WarmMemory
from kairos.protocols import MemoryStore, SummaryStore

_AT = datetime(2026, 11, 27, tzinfo=UTC)


def _message(text: str) -> NormalizedMessage:
    return NormalizedMessage(
        bot_id="bot",
        group_id=200,
        msg_id="1",
        last_msg_id="1",
        sender_qq=10,
        sender_alias="U1:小明",
        content=text,
        reply_to_msg_id=None,
        is_bot=False,
        is_peer_bot=False,
        is_at_bot=False,
        is_reply_to_bot=False,
        created_at=_AT,
        latest_at=_AT,
    )


def test_extract_keywords_supports_chinese_and_english_stop_words() -> None:
    messages = [_message("我今天在聊 显卡，和显卡有关。 The RISC-V and riscv group.")]
    result = extract_keywords(messages, max_keywords=3)
    assert len(result) == 3
    assert result[0] == "显卡"
    assert "risc-v" in result
    assert not {"今天", "the", "and"}.intersection(result)
    assert extract_keywords([_message("我们怎么了 the and")], 0) == []


async def test_recall_collects_sources_concurrently_and_applies_limits() -> None:
    store = AsyncMock(spec=MemoryStore)
    summaries = AsyncMock(spec=SummaryStore)
    impression = UserImpression(
        bot_id="bot",
        group_id=200,
        user_qq=10,
        nickname="小明",
        impression_summary="关注 GPU",
        updated_at=_AT,
    )
    note = MemoryNote(
        bot_id="bot",
        scope_group_id=200,
        entity_qqs=[10],
        summary_text="讨论 GPU",
        keywords=["gpu"],
        created_at=_AT,
    )
    summary = ContextSummary(
        bot_id="bot",
        group_id=200,
        summary_text="历史摘要",
        covered_until_msg_id="0",
        covered_until_at=_AT,
        token_estimate=4,
        created_at=_AT,
    )
    ready = asyncio.Event()
    starts: list[str] = []

    async def query(name: str, value: object) -> list[object]:
        starts.append(name)
        if len(starts) == 3:
            ready.set()
        await ready.wait()
        return [value, value]

    async def impressions(*_args: object) -> list[object]:
        return await query("impressions", impression)

    async def notes(*_args: object) -> list[object]:
        return await query("notes", note)

    async def history(*_args: object) -> list[object]:
        return await query("summaries", summary)

    store.load_user_impressions.side_effect = impressions
    store.recall_memory_notes.side_effect = notes
    summaries.load_recent_summaries.side_effect = history
    provider = PgMemoryProvider(
        store,
        summaries,
        max_impressions_recalled=1,
        max_notes_recalled=1,
        max_persisted_summaries=1,
    )
    async with asyncio.timeout(1):
        memory = await provider.recall("bot", 200, [10, 10, 20], [_message("The GPU GPU")])

    assert memory == WarmMemory(impressions=[impression], notes=[note], summaries=[summary])
    store.load_user_impressions.assert_awaited_once_with("bot", 200, [10])
    store.recall_memory_notes.assert_awaited_once_with("bot", 200, ["gpu"], "The GPU GPU", 1)
    summaries.load_recent_summaries.assert_awaited_once_with("bot", 200, 1)


async def test_source_failure_returns_only_that_empty_subset() -> None:
    store = AsyncMock(spec=MemoryStore)
    store.load_user_impressions.side_effect = OSError("unavailable")
    note = MemoryNote(
        bot_id="bot",
        scope_group_id=200,
        entity_qqs=[10],
        summary_text="讨论 GPU",
        keywords=["gpu"],
        created_at=_AT,
    )
    store.recall_memory_notes.return_value = [note]
    assert await PgMemoryProvider(store, None).recall(
        "bot", 200, [10], [_message("GPU")]
    ) == WarmMemory(notes=[note])


async def test_disabled_limits_avoid_unnecessary_queries() -> None:
    store = AsyncMock(spec=MemoryStore)
    provider = PgMemoryProvider(store, None, max_impressions_recalled=0, max_notes_recalled=0)
    assert await provider.recall("bot", 200, [10], [_message("显卡")]) == WarmMemory()
    assert store.mock_calls == []
