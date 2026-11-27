import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

from kairos.config import BotProfile, CompactionConfig, MemoryConfig
from kairos.memory.async_worker import AsyncMemoryWorker, HotContextCache
from kairos.models import ContextSummary, LLMResult, NormalizedMessage, WarmMemory
from kairos.protocols import ContextCompactor, LLMClient, MemoryProvider, MemoryStore, SummaryStore

_AT = datetime(2026, 11, 27, tzinfo=UTC)


def _profile(*, compaction: bool = False, worker_key: str | None = None) -> BotProfile:
    return BotProfile(
        bot_id="bot",
        self_id=100,
        nickname="小满",
        persona_prompt="爱聊天",
        worker_model_key=worker_key,
        memory=MemoryConfig(
            enabled=True, warm_trigger_messages=2, note_extract_interval_messages=2
        ),
        compaction=CompactionConfig(
            enabled=compaction, keep_recent_messages=1, max_persisted_summaries=2
        ),
    )


def _message(number: int) -> NormalizedMessage:
    at = _AT + timedelta(seconds=number)
    return NormalizedMessage(
        bot_id="bot",
        group_id=10,
        msg_id=str(number),
        last_msg_id=str(number),
        sender_qq=20,
        sender_alias="U1:小明",
        content=f"聊天 {number}",
        reply_to_msg_id=None,
        is_bot=False,
        is_peer_bot=False,
        is_at_bot=False,
        is_reply_to_bot=False,
        created_at=at,
        latest_at=at,
    )


def _summary(number: int, *, bot_id: str = "bot", group_id: int = 10) -> ContextSummary:
    at = _AT + timedelta(seconds=number)
    return ContextSummary(
        bot_id=bot_id,
        group_id=group_id,
        summary_text=f"摘要 {number}",
        covered_until_msg_id=str(number),
        covered_until_at=at,
        token_estimate=4,
        created_at=at,
    )


async def _finish_work(worker: AsyncMemoryWorker) -> None:
    tasks = [state.task for state in worker._groups.values() if state.task is not None]
    async with asyncio.timeout(1):
        await asyncio.gather(*tasks)


def test_hot_context_is_group_and_bot_scoped_with_lru_eviction() -> None:
    cache = HotContextCache(max_groups=2)
    first, other_bot = _summary(1), _summary(2, bot_id="other")
    cache.seed_summaries("bot", 10, [first])
    cache.seed_summaries("other", 10, [other_bot])
    assert cache.get_warm_memory("bot", 10).summaries == [first]
    cache.seed_summaries("bot", 11, [_summary(3, group_id=11)])
    assert cache.get_warm_memory("other", 10) == WarmMemory()
    assert cache.get_warm_memory("bot", 10).summaries == [first]
    assert len(cache._entries) == 2


async def test_warmup_and_threshold_count_coalesced_arrivals_without_queries_on_read() -> None:
    provider = AsyncMock(spec=MemoryProvider)
    summary = _summary(0)
    provider.recall.return_value = WarmMemory(summaries=[summary])
    worker = AsyncMemoryWorker(provider)
    profile = _profile()
    first = _message(1)
    worker.schedule_warmup(profile, 10, [first])
    assert worker.get_warm_memory("bot", 10) == WarmMemory()
    provider.recall.assert_not_awaited()
    await _finish_work(worker)
    assert worker.get_warm_memory("bot", 10).summaries == [summary]

    second = first.model_copy(update={"last_msg_id": "2"})
    worker.on_messages_updated(profile, 10, [second])
    worker.on_messages_updated(profile, 10, [second])
    await _finish_work(worker)
    provider.recall.assert_awaited_once_with("bot", 10, [20], (first,))
    third = _message(3)
    worker.on_messages_updated(profile, 10, [second, third])
    await _finish_work(worker)
    assert provider.recall.await_count == 2
    assert provider.recall.await_args.args == ("bot", 10, [20], (second, third))
    await worker.shutdown()


async def test_same_group_merges_updates_while_recall_is_in_flight() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    provider = AsyncMock(spec=MemoryProvider)

    async def recall(*_args: object) -> WarmMemory:
        entered.set()
        await release.wait()
        return WarmMemory()

    provider.recall.side_effect = recall
    worker = AsyncMemoryWorker(provider, max_groups=1)
    profile = _profile()
    worker.schedule_warmup(profile, 10, [_message(1)])
    await asyncio.wait_for(entered.wait(), timeout=1)
    try:
        task = worker._groups[("bot", 10)].task
        for number in range(2, 50):
            worker.on_messages_updated(profile, 10, [_message(number)])
        worker.schedule_warmup(profile, 11, [])
        assert len(worker._groups) == 1
        assert worker._groups[("bot", 10)].task is task
        assert worker.get_warm_memory("bot", 10) == WarmMemory()
        assert provider.recall.await_count == 1
    finally:
        release.set()
        await _finish_work(worker)
        await worker.shutdown()
    assert provider.recall.await_count == 2
    assert provider.recall.await_args.args[3] == (_message(49),)


async def test_compaction_persists_new_summary_even_if_recall_fails() -> None:
    provider = AsyncMock(spec=MemoryProvider)
    provider.recall.side_effect = OSError("recall unavailable")
    store = AsyncMock(spec=SummaryStore)
    compactor = Mock(spec=ContextCompactor)
    compactor.should_compact.return_value = True
    summary = _summary(3)
    compactor.compact.return_value = summary
    profile = _profile(compaction=True)
    worker = AsyncMemoryWorker(provider, summary_store=store, compactor=compactor)
    old = [_summary(1), _summary(2)]
    worker.seed_summaries("bot", 10, old)
    worker.on_messages_updated(profile, 10, [_message(3), _message(4)])
    await _finish_work(worker)

    provider.recall.assert_awaited_once()
    compactor.compact.assert_awaited_once_with(profile, 10, (_message(3), _message(4)), old)
    store.save_summary.assert_awaited_once_with(summary)
    assert worker.get_warm_memory("bot", 10).summaries == [old[-1], summary]
    await worker.shutdown()


async def test_notes_are_extracted_on_threshold_and_invalid_output_is_ignored() -> None:
    store = AsyncMock(spec=MemoryStore)
    llm = AsyncMock(spec=LLMClient)
    data = {
        "impressions": [{"user_qq": 20, "impression_summary": "聊显卡", "bot_id": "wrong"}],
        "notes": [
            {"summary_text": "一起聊显卡", "entity_qqs": [20, 99], "keywords": [" GPU ", "gpu"]}
        ],
    }
    llm.chat_completion.side_effect = [
        LLMResult(
            content=content,
            model="worker",
            latency_ms=1,
            prompt_tokens=None,
            completion_tokens=None,
        )
        for content in (json.dumps(data), "bad JSON")
    ]
    worker = AsyncMemoryWorker(memory_store=store, llm=llm)
    profile = _profile(worker_key="worker")
    worker.on_messages_updated(profile, 10, [_message(1)])
    llm.chat_completion.assert_not_awaited()
    worker.on_messages_updated(profile, 10, [_message(1), _message(2)])
    await _finish_work(worker)
    assert llm.chat_completion.await_args.args[0] == "worker"
    (impression,) = store.upsert_user_impression.await_args.args
    assert (impression.bot_id, impression.group_id, impression.nickname) == ("bot", 10, "小明")
    assert impression.impression_summary == "聊显卡"
    (note,) = store.save_memory_note.await_args.args
    assert (note.bot_id, note.scope_group_id, note.entity_qqs, note.keywords) == (
        "bot",
        10,
        [20],
        ["gpu"],
    )

    worker.on_messages_updated(profile, 10, [_message(2), _message(3), _message(4)])
    await _finish_work(worker)
    assert llm.chat_completion.await_count == 2
    assert store.upsert_user_impression.await_count == store.save_memory_note.await_count == 1
    await worker.shutdown()


async def test_notes_extraction_tolerates_markdown_fenced_json() -> None:
    store = AsyncMock(spec=MemoryStore)
    llm = AsyncMock(spec=LLMClient)
    payload = {
        "impressions": [{"user_qq": 20, "impression_summary": "喜欢做饭"}],
        "notes": [
            {"summary_text": "聊到番茄炒蛋做法", "entity_qqs": [20], "keywords": ["番茄炒蛋"]}
        ],
    }
    llm.chat_completion.return_value = LLMResult(
        content=f"提取结果如下：\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n完毕",
        model="worker",
        latency_ms=1,
        prompt_tokens=None,
        completion_tokens=None,
    )
    worker = AsyncMemoryWorker(memory_store=store, llm=llm)
    profile = _profile(worker_key="worker")
    worker.on_messages_updated(profile, 10, [_message(1), _message(2)])
    await _finish_work(worker)

    store.upsert_user_impression.assert_awaited_once()
    store.save_memory_note.assert_awaited_once()
    await worker.shutdown()


async def test_shutdown_cancels_active_and_queued_work() -> None:
    provider = AsyncMock(spec=MemoryProvider)
    started, stopped = asyncio.Event(), asyncio.Event()

    async def recall(*_args: object) -> WarmMemory:
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    provider.recall.side_effect = recall
    worker = AsyncMemoryWorker(provider)
    profile = _profile()
    worker.schedule_warmup(profile, 10, [])
    await asyncio.wait_for(started.wait(), timeout=1)
    worker.on_messages_updated(profile, 10, [_message(1), _message(2)])
    worker.schedule_warmup(profile, 11, [])
    await asyncio.wait_for(worker.shutdown(), timeout=1)
    assert stopped.is_set() and not worker._groups
    worker.schedule_warmup(profile, 12, [])
    worker.on_messages_updated(profile, 10, [_message(3)])
    assert not worker._groups and provider.recall.await_count == 1
