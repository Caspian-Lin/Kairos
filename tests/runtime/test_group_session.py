import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kairos.config import BotProfile, DebounceConfig
from kairos.harness.compactor import DefaultContextCompactor
from kairos.models import ContextSummary, NormalizedMessage, RawGroupEvent
from kairos.protocols import LLMClient
from kairos.runtime import group_session
from kairos.runtime.group_session import GroupSession

_NOW = datetime(2026, 10, 1, tzinfo=UTC)
_GROUP = 2468
_SELF = 10001
_PEER = 10002
_USER = 10003
_HOSTED = {_SELF, _PEER}


@pytest.fixture
async def session() -> AsyncIterator[GroupSession]:
    profile = BotProfile(
        bot_id="companion",
        self_id=_SELF,
        nickname="助手",
        persona_prompt="",
        chat_model_key="chat",
        trigger_mode="all_with_veto",
        ring_buffer_size=3,
    )
    group = GroupSession(profile, _GROUP)
    try:
        yield group
    finally:
        group.close()
        await group.wait_closed()


def _event(msg_id: int, text: str, sender: int = _USER) -> RawGroupEvent:
    return RawGroupEvent(
        self_id=_SELF,
        group_id=_GROUP,
        msg_id=str(msg_id),
        sender_qq=sender,
        sender_nickname="群友",
        sender_card="",
        segments=[],
        raw_message=text,
        timestamp=_NOW + timedelta(seconds=msg_id),
    )


def _ingest(
    session: GroupSession, msg_id: int, text: str, sender: int = _USER
) -> NormalizedMessage:
    result, _ = session.ingest_event(_event(msg_id, text, sender), _HOSTED)
    assert result is not None
    return result


def test_ingest_coalesces_buffer_but_returns_individual_row(session: GroupSession) -> None:
    first = _ingest(session, 1, "继续")
    second, merged = session.ingest_event(_event(2, "下一句"), _HOSTED)

    assert merged and second is not None
    assert second.content == "下一句"
    assert second.msg_id == "2"
    assert session.buffer[-1].content == "继续 / 下一句"
    assert session.buffer[-1].sender_alias == first.sender_alias
    assert (session.buffer[-1].msg_id, session.buffer[-1].last_msg_id) == ("1", "2")
    assert session.seq == 2


async def test_ingest_advances_while_pipeline_lock_is_held(session: GroupSession) -> None:
    async with session.pipeline_lock:
        _ingest(session, 1, "推理期间也收到消息")
        assert session.seq == 1


def test_warmup_restores_aliases_and_rebuilds_bursts(session: GroupSession) -> None:
    other = GroupSession(session.profile, _GROUP)
    first = _ingest(other, 1, "第一条").model_copy(update={"sender_alias": "U12:历史"})
    second = _ingest(other, 2, "第二条").model_copy(update={"sender_alias": "U12:历史"})
    sent = other.record_bot_message("sent-1", "回复", _NOW + timedelta(seconds=3))
    different_group = first.model_copy(update={"group_id": _GROUP + 1})
    session.warmup([different_group, first, second, sent])

    assert [message.content for message in session.buffer] == ["第一条 / 第二条", "回复"]
    assert list(session.bot_msg_ids) == ["sent-1"]
    assert session.seq == 3
    reply = _ingest(session, 4, "[CQ:reply,id=sent-1]继续")
    assert reply.is_reply_to_bot
    assert reply.sender_alias == "U12:历史"
    assert _ingest(session, 5, "我是另一位", _USER + 1).sender_alias.startswith("U13:")


def test_record_bot_messages_bounds_history_and_echo_ids(session: GroupSession) -> None:
    for index in range(102):
        sent = session.record_bot_message(str(index), "已发出", _NOW, "previous")
        assert sent.is_bot and sent.reply_to_msg_id == "previous"

    assert session.seq == 102
    assert session.last_bot_reply_at is not None
    assert [message.msg_id for message in session.buffer] == ["99", "100", "101"]
    assert len(session.bot_msg_ids) == 100
    assert session.bot_msg_ids[0] == "2"
    assert session.bot_msg_ids[-1] == "101"


def test_l1a_cooldown_and_trigger_mode(session: GroupSession) -> None:
    plain = _ingest(session, 1, "群聊")
    at_bot = _ingest(session, 2, f"[CQ:at,qq={_SELF}]你好")
    sent = session.record_bot_message("mine", "你好", _NOW)
    reply = _ingest(session, 3, f"[CQ:reply,id={sent.msg_id}]继续")
    session.last_bot_reply_at = 100.0

    assert session.evaluate_l1a_gate(plain, 102) == (False, "cooldown")
    assert session.evaluate_l1a_gate(plain, 103) == (True, "pass")
    assert session.evaluate_l1a_gate(at_bot, 102) == (True, "pass")
    assert session.evaluate_l1a_gate(reply, 102) == (True, "pass")
    session.profile.trigger_mode = "mention_only"
    assert session.evaluate_l1a_gate(plain, 104) == (False, "mention_only")
    assert session.evaluate_l1a_gate(at_bot, 104) == (True, "pass")


def test_l1a_caps_peer_turns_until_human_message(session: GroupSession) -> None:
    for index in (1, 2):
        peer = _ingest(session, index, "另一个机器人", _PEER)
        assert session.evaluate_l1a_gate(peer, 100 + index) == (True, "pass")
    blocked_peer = _ingest(session, 3, f"[CQ:at,qq={_SELF}]又来了", _PEER)
    assert session.evaluate_l1a_gate(blocked_peer, 103) == (False, "peer_bot_limit")

    session.record_bot_message("mine", "我也说了一句", _NOW)
    assert session.consecutive_peer_bot_turns == 2
    session.last_bot_reply_at = None
    _ingest(session, 4, "群友插话")
    assert session.consecutive_peer_bot_turns == 0
    peer_after_human = _ingest(session, 5, "继续", _PEER)
    assert session.evaluate_l1a_gate(peer_after_human, 105) == (True, "pass")


def test_l1a_energy_recovers_lazily_and_direct_address_bypasses_exhaustion(
    session: GroupSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 100.0
    monkeypatch.setattr(group_session, "monotonic", lambda: now)
    profile = session.profile.model_copy(deep=True)
    profile.gate.enabled = True
    profile.gate.energy_max = 30
    profile.gate.energy_recover_per_min = 6
    group = GroupSession(profile, _GROUP)
    plain = _ingest(group, 1, "群聊")
    mention = plain.model_copy(update={"is_at_bot": True})

    assert group.energy == 30
    group.consume_energy(profile.gate.energy_cost_reply)
    group.consume_energy(profile.gate.energy_cost_react)
    assert group.energy == 0
    assert group.evaluate_l1a_gate(plain, now) == (False, "low_energy")
    assert group.evaluate_l1a_gate(mention, now) == (True, "pass")
    now += 60
    assert group.evaluate_l1a_gate(plain, now) == (True, "pass")
    assert group.energy == 6
    now += 360
    assert group.evaluate_l1a_gate(plain, now) == (True, "pass")
    assert group.energy == 30


def test_warmup_replays_only_uncovered_messages_and_tracks_threads(session: GroupSession) -> None:
    profile = session.profile.model_copy(deep=True)
    profile.compaction.enabled = True
    source = GroupSession(profile, _GROUP)
    older = _ingest(source, 1, "已覆盖")
    covered_bot = source.record_bot_message("2", "已覆盖的回复", _NOW + timedelta(seconds=2))
    tail = _ingest(source, 3, "需要回放")
    later = _ingest(source, 4, "继续")

    def summary(message: NormalizedMessage) -> ContextSummary:
        return ContextSummary(
            bot_id=profile.bot_id,
            group_id=_GROUP,
            summary_text=message.content,
            covered_until_msg_id=message.last_msg_id,
            covered_until_at=message.latest_at,
            created_at=message.latest_at,
            token_estimate=6,
        )

    first, second = summary(older), summary(covered_bot)
    unrelated = summary(later).model_copy(update={"group_id": _GROUP + 1})
    group = GroupSession(profile, _GROUP)
    group.warmup(
        [later, tail, covered_bot, older, tail],
        [second, unrelated, first],
        compactor=DefaultContextCompactor(AsyncMock(spec=LLMClient)),
    )
    assert group.summaries == [first, second]
    assert [message.content for message in group.buffer] == ["需要回放 / 继续"]
    assert list(group.bot_msg_ids) == ["2"]
    assert group.seq == 2
    reply = _ingest(group, 5, "[CQ:reply,id=2]好的")
    assert reply.is_reply_to_bot
    recorded = group.record_bot_message("6", "新回复", _NOW + timedelta(seconds=6), "5")
    assert group.thread_tracker is not None
    (thread,) = group.thread_tracker.active_threads(recorded.latest_at)
    assert (thread.root_msg_id, thread.latest_msg_id, thread.message_count) == ("3", "6", 4)


@pytest.mark.parametrize("direct_flag", ["is_at_bot", "is_reply_to_bot", "mentions_bot_name"])
async def test_debounce_deadlines(
    session: GroupSession, monkeypatch: pytest.MonkeyPatch, direct_flag: str
) -> None:
    # Keep real timer cancellation but advance a deterministic monotonic clock for assertions.
    loop = asyncio.get_running_loop()
    arm = Mock(wraps=loop.call_later)
    monkeypatch.setattr(group_session, "get_running_loop", lambda: SimpleNamespace(call_later=arm))
    now = 0.0
    monkeypatch.setattr(group_session, "monotonic", lambda: now)
    ordinary = _ingest(session, 1, "你好")
    addressed = ordinary.model_copy(update={direct_flag: True})

    async def callback(*_args: object) -> None:
        pass

    session.schedule_trigger(ordinary, callback)
    assert arm.call_args.args[0] == pytest.approx(1.8)
    now = 1.5
    session.schedule_trigger(ordinary, callback)
    assert arm.call_args.args[0] == pytest.approx(1.8)
    now = 3.0
    session.schedule_trigger(ordinary, callback)
    assert arm.call_args.args[0] == pytest.approx(1.5)
    now = 3.5
    session.schedule_trigger(addressed, callback)
    assert arm.call_args.args[0] == pytest.approx(0.5)
    now = 3.75
    session.schedule_trigger(ordinary, callback)
    assert arm.call_args.args[0] == pytest.approx(0.25)


async def test_debounce_fires_once_when_ordinary_stream_reaches_hard_deadline(
    session: GroupSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 0.0
    monkeypatch.setattr(group_session, "monotonic", lambda: now)
    msg = _ingest(session, 1, "连续普通消息")
    triggered = asyncio.Event()
    calls: list[float] = []

    async def callback(*_args: object) -> None:
        calls.append(now)
        triggered.set()

    for now in (0.0, 1.5, 3.0, 4.5):
        session.schedule_trigger(msg, callback)
    await asyncio.wait_for(triggered.wait(), timeout=1)
    await asyncio.sleep(0)
    assert calls == [4.5]


async def test_active_callback_coalesces_waiters_into_one_latest_trigger(
    session: GroupSession,
) -> None:
    session.profile.debounce = DebounceConfig(normal_seconds=0)
    msg = _ingest(session, 1, "触发推理")
    started = asyncio.Event()
    release = asyncio.Event()
    next_run = asyncio.Event()
    calls: list[str] = []

    async def first(*_args: object) -> None:
        calls.append("first")
        started.set()
        await release.wait()

    async def superseded(*_args: object) -> None:
        calls.append("superseded")

    async def latest(*_args: object) -> None:
        calls.append("latest")
        next_run.set()

    session.schedule_trigger(msg, first)
    await asyncio.wait_for(started.wait(), timeout=1)
    for _ in range(10):
        session.schedule_trigger(msg, superseded)
        await asyncio.sleep(0)
    session.schedule_trigger(msg, latest)
    assert calls == ["first"]

    release.set()
    await asyncio.wait_for(next_run.wait(), timeout=1)
    assert calls == ["first", "latest"]


async def test_close_cancels_waiting_and_running_triggers(session: GroupSession) -> None:
    session.profile.debounce = DebounceConfig(normal_seconds=0)
    msg = _ingest(session, 1, "触发后关闭")
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def running(*_args: object) -> None:
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    unexpected = Mock()

    async def waiting(*_args: object) -> None:
        unexpected()

    without_active_callback = GroupSession(session.profile, _GROUP)
    without_active_callback.schedule_trigger(msg, waiting)
    without_active_callback.close()
    await without_active_callback.wait_closed()

    session.schedule_trigger(msg, running)
    await asyncio.wait_for(started.wait(), timeout=1)
    session.schedule_trigger(msg, waiting)
    session.close()
    session.close()
    await session.wait_closed()
    session.schedule_trigger(msg, waiting)
    await asyncio.sleep(0)
    assert cleaned.is_set()
    unexpected.assert_not_called()


def test_reset_keeps_only_newer_messages_in_memory_and_after_restore(session: GroupSession) -> None:
    profile = session.profile.model_copy(deep=True)
    profile.compaction.enabled = True
    group = GroupSession(profile, _GROUP)
    old = _ingest(group, 1, "旧上下文")
    latest = _ingest(group, 2, "新上下文", _USER + 1)
    historical = ContextSummary(
        bot_id=profile.bot_id,
        group_id=_GROUP,
        summary_text="旧摘要",
        covered_until_msg_id="0",
        covered_until_at=_NOW,
        created_at=_NOW,
        token_estimate=3,
    )
    group.summaries.append(historical)
    session_id = group.llm_session_id

    marker = group.reset_context(keep_latest=1)

    assert marker is not None and marker.covered_until_msg_id == old.msg_id
    assert group.summaries == [] and list(group.buffer) == [latest]
    assert group.llm_session_id != session_id
    assert group.context_generation == 1 and group.seq == 2
    rebuilt = GroupSession(profile, _GROUP)
    rebuilt.warmup(
        [old, latest],
        [historical, marker],
        compactor=DefaultContextCompactor(AsyncMock(spec=LLMClient)),
    )
    assert rebuilt.summaries == [] and list(rebuilt.buffer) == [latest]
