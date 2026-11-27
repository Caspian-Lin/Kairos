from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from kairos.config import BotProfile, CompactionConfig
from kairos.harness.compactor import DefaultContextCompactor, estimate_tokens
from kairos.models import ContextSummary, LLMResult, NormalizedMessage
from kairos.protocols import LLMClient

_START = datetime(2026, 11, 27, tzinfo=UTC)


def _profile(*, ring_buffer_size: int = 40) -> BotProfile:
    return BotProfile(
        bot_id="bot",
        self_id=100,
        nickname="小满",
        persona_prompt="喜欢群聊",
        worker_model_key="worker",
        ring_buffer_size=ring_buffer_size,
        compaction=CompactionConfig(enabled=True, input_token_budget=512, keep_recent_messages=2),
    )


def _message(msg_id: str, seconds: int, body: str = "你好") -> NormalizedMessage:
    at = _START + timedelta(seconds=seconds)
    return NormalizedMessage(
        bot_id="bot",
        group_id=1,
        msg_id=msg_id,
        last_msg_id=msg_id,
        sender_qq=101,
        sender_alias="U1",
        content=body,
        reply_to_msg_id=None,
        is_bot=False,
        is_at_bot=False,
        is_reply_to_bot=False,
        is_peer_bot=False,
        created_at=at,
        latest_at=at,
    )


def _summary(msg_id: str, seconds: int) -> ContextSummary:
    at = _START + timedelta(seconds=seconds)
    return ContextSummary(
        bot_id="bot",
        group_id=1,
        summary_text=f"摘要 {msg_id}",
        covered_until_msg_id=msg_id,
        covered_until_at=at,
        token_estimate=5,
        created_at=at,
    )


def test_token_estimate_weights_chinese_and_punctuation() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("你好，世界") == 5
    assert estimate_tokens("hello 世界！") == 5


def test_trigger_uses_budget_and_window_pressure() -> None:
    compactor = DefaultContextCompactor(AsyncMock(spec=LLMClient))
    profile = _profile()
    short = [_message(str(index), index) for index in range(4)]
    long = [_message(str(index), index, "消息" * 150) for index in range(3)]
    assert not compactor.should_compact(profile, short)
    assert not compactor.should_compact(profile, long[:2])
    assert not compactor.should_compact(profile, long, [_summary("0", 0)])
    assert compactor.should_compact(profile, long)
    assert compactor.should_compact(_profile(ring_buffer_size=4), short)
    profile.compaction.enabled = False
    assert not compactor.should_compact(profile, long)


async def test_compaction_preserves_recent_messages_and_burst_watermark() -> None:
    llm = AsyncMock(spec=LLMClient)
    llm.chat_completion.return_value = LLMResult(
        content=" 群友约吃饭。 ",
        model="worker",
        latency_ms=1,
        prompt_tokens=None,
        completion_tokens=None,
    )
    compactor = DefaultContextCompactor(llm)
    context = [_message("1", 1), _message("2", 2, "约饭"), _message("4", 4), _message("5", 5)]
    context[1] = context[1].model_copy(
        update={"last_msg_id": "3", "latest_at": _message("3", 3).latest_at}
    )
    result = await compactor.compact(_profile(), 1, context, [_summary("1", 1)])
    assert result is not None
    assert result.summary_text == "群友约吃饭。"
    assert result.covered_until_msg_id == "3"
    assert result.covered_until_at == context[1].latest_at
    assert result.token_estimate == estimate_tokens(result.summary_text)
    model, messages = llm.chat_completion.await_args.args
    assert model == "worker"
    assert llm.chat_completion.await_args.kwargs["max_tokens"] == 300
    assert "摘要 1" in messages[1].content and "约饭" in messages[1].content
    assert "[m4]" not in messages[1].content and "[m5]" not in messages[1].content


async def test_compaction_fails_open_on_missing_model_or_request_error() -> None:
    llm = AsyncMock(spec=LLMClient)
    llm.chat_completion.side_effect = TimeoutError
    profile = _profile()
    compactor = DefaultContextCompactor(llm)
    messages = [_message(str(index), index) for index in range(4)]
    assert await compactor.compact(profile, 1, messages) is None
    llm.chat_completion.assert_awaited_once()
    profile.worker_model_key = None
    assert await compactor.compact(profile, 1, messages) is None
    llm.chat_completion.assert_awaited_once()


def test_restore_sorts_and_deduplicates_summaries_and_messages() -> None:
    compactor = DefaultContextCompactor(AsyncMock(spec=LLMClient), max_persisted_summaries=2)
    older, middle, latest = _summary("1", 1), _summary("2", 2), _summary("4", 4)
    four = _message("4", 4)
    five = _message("5", 5)
    six = _message("6", 6)
    summaries, messages = compactor.merge_restored_context(
        [latest, older, middle, middle], [six, four, five, five, _message("3", 3)]
    )
    assert summaries == [middle, latest]
    assert messages == [five, six]
    assert compactor.merge_restored_context([], [six, five, six]) == ([], [five, six])


def test_restore_same_timestamp_uses_the_watermark_message_in_store_order() -> None:
    compactor = DefaultContextCompactor(AsyncMock(spec=LLMClient))
    boundary = _message("42", 4)
    after_boundary = _message("5", 4)
    assert compactor.merge_restored_context(
        [_summary("42", 4)], [_message("99", 4), boundary, after_boundary, boundary]
    )[1] == [after_boundary]
