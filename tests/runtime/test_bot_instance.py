import asyncio
import signal
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from itertools import count
from unittest.mock import AsyncMock, Mock

import pytest
from structlog.testing import capture_logs

from kairos import main as entry_point
from kairos.config import BotProfile, DebounceConfig, GlobalConfig, PostgresConfig, RateLimitConfig
from kairos.llm import LLMRequestError
from kairos.main import MultiBotManager
from kairos.models import AuditRecord, LLMResult, MessageSegment, NormalizedMessage, RawGroupEvent
from kairos.protocols import LLMClient, MessageStore, OneBotSender
from kairos.runtime import bot_instance
from kairos.telemetry.audit_logger import AuditLogger

_GROUP = 900
_TIME = datetime(2026, 1, 2, tzinfo=UTC)
_REPLY = "收到，下次见。"


@pytest.mark.parametrize(
    ("hour", "greeting"),
    [
        (0, "晚上好"),
        (4, "晚上好"),
        (5, "早上好"),
        (10, "早上好"),
        (11, "中午好"),
        (17, "中午好"),
        (18, "晚上好"),
        (23, "晚上好"),
    ],
)
def test_online_greeting_time_buckets(hour: int, greeting: str) -> None:
    assert bot_instance._online_greeting(hour) == greeting


def _profile(bot_id: str = "first", self_id: int = 1) -> BotProfile:
    return BotProfile(
        bot_id=bot_id,
        self_id=self_id,
        nickname=f"昵称 {bot_id}",
        persona_prompt=f"独立人设 {bot_id}",
        chat_model_key=f"{bot_id}-model",
        trigger_mode="all_with_veto",
        debounce=DebounceConfig(normal_seconds=0, mention_seconds=0, max_wait_seconds=0),
        rate_limit=RateLimitConfig(cooldown_seconds=0),
    )


def _event(
    msg_id: str,
    *,
    self_id: int = 1,
    group_id: int = _GROUP,
    sender: int = 42,
    text: str = "周五见。",
    mention: bool = False,
) -> RawGroupEvent:
    segments = [MessageSegment(type="text", data={"text": text})]
    if mention:
        segments.insert(0, MessageSegment(type="at", data={"qq": str(self_id)}))
    return RawGroupEvent(
        self_id=self_id,
        group_id=group_id,
        msg_id=msg_id,
        sender_qq=sender,
        sender_nickname=f"群友{sender}",
        sender_card="",
        segments=segments,
        raw_message="",
        timestamp=_TIME,
    )


def _result(text: str = _REPLY) -> LLMResult:
    return LLMResult(
        content=text, model="mock", latency_ms=1, prompt_tokens=None, completion_tokens=None
    )


def _history(msg_id: str, sender: int, alias: str, text: str) -> NormalizedMessage:
    timestamp = _TIME - timedelta(seconds=5)
    return NormalizedMessage(
        bot_id="first",
        group_id=_GROUP,
        msg_id=msg_id,
        last_msg_id=msg_id,
        sender_qq=sender,
        sender_alias=alias,
        content=text,
        reply_to_msg_id=None,
        is_bot=sender == 1,
        is_at_bot=False,
        is_reply_to_bot=False,
        is_peer_bot=False,
        created_at=timestamp,
        latest_at=timestamp,
    )


class _RuntimeCase:
    def __init__(self, profiles: Sequence[BotProfile]) -> None:
        self.store = Mock(spec=MessageStore)
        self.store.load_recent_messages.return_value = []
        self.audit_records: asyncio.Queue[AuditRecord] = asyncio.Queue()
        self.store.save_audit_log.side_effect = self.audit_records.put_nowait
        self.llm = Mock(spec=LLMClient)
        self.llm.chat_completion.return_value = _result()
        self.sender = Mock(spec=OneBotSender)
        self.sends: asyncio.Queue[tuple[int, int, str, str | None]] = asyncio.Queue()
        echo_ids = count(1001)

        def acknowledge(
            self_id: int, group_id: int, text: str, reply_to_msg_id: str | None = None
        ) -> str:
            self.sends.put_nowait((self_id, group_id, text, reply_to_msg_id))
            return str(next(echo_ids))

        self.sender.send_group_msg.side_effect = acknowledge
        self.audit = AuditLogger(self.store)
        self.manager = MultiBotManager(profiles, self.store, self.llm, self.sender, self.audit)

    async def next_audit(self) -> AuditRecord:
        return await asyncio.wait_for(self.audit_records.get(), timeout=2)


@pytest.fixture
async def make_case() -> AsyncIterator[Callable[[Sequence[BotProfile]], _RuntimeCase]]:
    async with AsyncExitStack() as resources:

        def create(profiles: Sequence[BotProfile]) -> _RuntimeCase:
            case = _RuntimeCase(profiles)
            resources.push_async_callback(case.audit.drain)
            resources.push_async_callback(case.manager.shutdown)
            return case

        yield create


async def test_reply_echo_updates_history_store_and_audit(make_case) -> None:
    case = make_case([_profile()])
    await case.manager.handle_group_event(_event("10"))
    record = await case.next_audit()
    await case.manager.shutdown()

    case.sender.send_group_msg.assert_awaited_once_with(1, _GROUP, _REPLY, reply_to_msg_id=None)
    history = case.manager.runtimes[1].sessions[_GROUP].buffer
    assert [(m.msg_id, m.is_bot) for m in history] == [("10", False), ("1001", True)]
    assert [call.args[0] for call in case.store.save_message.await_args_list] == list(history)
    key, messages = case.llm.chat_completion.await_args.args
    assert key == "first-model"
    assert case.llm.chat_completion.await_args.kwargs["session_id"] == (
        case.manager.runtimes[1].sessions[_GROUP].llm_session_id
    )
    assert [message.role for message in messages] == ["system", "user"]
    assert record.context_snapshot == [message.model_dump() for message in messages]
    assert (record.stale_action, record.final_reply, record.main_vetoed) == (
        "SEND_DIRECT",
        _REPLY,
        False,
    )
    assert record.latency_ms >= 0


@pytest.mark.parametrize("mention", [False, True])
async def test_veto_leaves_no_group_notice_outside_debug_groups(make_case, mention: bool) -> None:
    profile = _profile()
    profile.trigger_mode = None
    profile.nickname = "asuka"
    profile.allowed_groups = [_GROUP, _GROUP + 1]
    profile.debug_groups = [_GROUP]
    case = make_case([profile])
    case.llm.chat_completion.return_value = _result("<|act:silent|>")

    await case.manager.handle_group_event(
        _event("10", text="AsUkA，你觉得呢？", mention=mention, group_id=_GROUP + 1)
    )
    assert (await case.next_audit()).main_vetoed
    case.sender.send_group_msg.assert_not_called()
    _, messages = case.llm.chat_completion.await_args.args
    assert "仅输出 <|act:silent|>" in messages[0].content


async def test_model_error_is_reported_even_when_group_debug_is_off(make_case) -> None:
    profile = _profile()
    profile.trigger_mode = None
    profile.nickname = "asuka"
    case = make_case([profile])
    case.llm.chat_completion.side_effect = LLMRequestError("private upstream diagnostic", 429)

    await case.manager.handle_group_event(_event("10", text="AsUkA，在吗？"))

    report = await asyncio.wait_for(case.sends.get(), timeout=2)
    assert report == (1, _GROUP, "【Kairos 错误】模型接口返回 HTTP 429。", "10")
    case.sender.send_group_msg.assert_awaited_once()
    assert [message.msg_id for message in case.manager.runtimes[1].sessions[_GROUP].buffer] == [
        "10"
    ]


async def test_concurrent_first_events_warm_up_once_and_resume_aliases(make_case) -> None:
    first = _profile()
    first.trigger_mode = "mention_only"
    case = make_case([first, _profile("second", 2)])
    loading = asyncio.Event()
    release = asyncio.Event()

    async def load(*_args, **_kwargs) -> list[NormalizedMessage]:
        loading.set()
        await release.wait()
        return [
            _history("old-1", 90, "U7:熟人", "可以"),
            _history("old-2", 90, "U7:熟人", "周五见"),
            _history("old-own", 1, "昵称 first", "我稍后看看"),
            _history("old-peer", 2, "U8:同伴", "好的"),
        ]

    case.store.load_recent_messages.side_effect = load
    async with asyncio.TaskGroup() as events:
        events.create_task(case.manager.handle_group_event(_event("new-1", sender=90)))
        await asyncio.wait_for(loading.wait(), timeout=2)
        events.create_task(case.manager.handle_group_event(_event("new-2", sender=91)))
        await asyncio.sleep(0)
        assert case.store.load_recent_messages.await_count == 1
        release.set()
    await case.manager.shutdown()

    case.store.load_recent_messages.assert_awaited_once_with("first", _GROUP, limit=40)
    session = case.manager.runtimes[1].sessions[_GROUP]
    assert [message.msg_id for message in session.buffer] == [
        "old-1",
        "old-own",
        "old-peer",
        "new-1",
        "new-2",
    ]
    assert session.buffer[0].last_msg_id == "old-2"
    assert session.buffer[2].is_peer_bot
    assert list(session.bot_msg_ids) == ["old-own"]
    assert session.buffer[3].sender_alias.startswith("U7:")
    assert session.buffer[4].sender_alias.startswith("U9:")
    assert [call.args[0].msg_id for call in case.store.save_message.await_args_list] == [
        "new-1",
        "new-2",
    ]


async def test_bots_share_infrastructure_with_isolated_same_group_state(make_case) -> None:
    case = make_case([_profile(), _profile("second", 2)])
    await asyncio.gather(
        case.manager.handle_group_event(_event("a", self_id=1, sender=31, text="只有甲看到")),
        case.manager.handle_group_event(_event("b", self_id=2, sender=32, text="只有乙看到")),
    )
    records = {
        record.bot_id: record for record in [await case.next_audit(), await case.next_audit()]
    }
    await case.manager.shutdown()

    for self_id, bot_id, event_id, prompt_text in (
        (1, "first", "a", "只有甲看到"),
        (2, "second", "b", "只有乙看到"),
    ):
        runtime = case.manager.runtimes[self_id]
        history = runtime.sessions[_GROUP].buffer
        assert [message.bot_id for message in history] == [bot_id, bot_id]
        assert history[0].msg_id == event_id
        assert history[0].sender_alias.startswith("U1:")
        system_turn, user_turn = records[bot_id].context_snapshot
        assert system_turn["content"].startswith(f"独立人设 {bot_id}\n")
        assert user_turn["content"].endswith(prompt_text)
    assert {call.args[0] for call in case.sender.send_group_msg.await_args_list} == {1, 2}


async def test_unknown_bot_disallowed_group_and_self_echo_do_not_create_sessions(make_case) -> None:
    profile = _profile()
    profile.allowed_groups = [_GROUP]
    case = make_case([profile])
    for event in (_event("1", self_id=2), _event("2", group_id=_GROUP + 1), _event("3", sender=1)):
        await case.manager.handle_group_event(event)

    assert not case.manager.runtimes[1].sessions
    case.store.load_recent_messages.assert_not_called()
    case.store.save_message.assert_not_called()
    case.llm.chat_completion.assert_not_called()


@pytest.mark.parametrize(
    ("generated", "expected_action", "veto"),
    [("<|act:silent|>", "VETOED", True), (_REPLY, "DROPPED_REPEAT", False)],
)
async def test_postprocess_stops_send_and_saves_audit(
    make_case, generated: str, expected_action: str, veto: bool
) -> None:
    case = make_case([_profile()])
    case.store.load_recent_messages.return_value = [_history("own", 1, "昵称 first", _REPLY)]
    case.llm.chat_completion.return_value = _result(generated)
    await case.manager.handle_group_event(_event("10"))
    record = await case.next_audit()

    case.sender.send_group_msg.assert_not_called()
    assert (record.stale_action, record.main_vetoed, record.final_reply) == (
        expected_action,
        veto,
        None,
    )


@pytest.mark.parametrize(("delta", "action"), [(3, "SEND_WITH_REPLY"), (6, "DROP_STALE")])
async def test_ingress_during_inference_drives_stale_decision(
    make_case, delta: int, action: str
) -> None:
    profile = _profile()
    profile.trigger_mode = "mention_only"
    case = make_case([profile])
    inference_started = asyncio.Event()
    release_inference = asyncio.Event()
    generated = f"{_REPLY}\n\n我还有个想法。"

    async def reply(*_args, **_kwargs) -> LLMResult:
        inference_started.set()
        await release_inference.wait()
        return _result(generated)

    case.llm.chat_completion.side_effect = reply
    await case.manager.handle_group_event(_event("first-part", mention=True))
    await case.manager.handle_group_event(_event("last-part"))
    await asyncio.wait_for(inference_started.wait(), timeout=2)
    session = case.manager.runtimes[1].sessions[_GROUP]
    assert session.seq == 2
    assert session.buffer[-1].last_msg_id == "last-part"
    for index in range(delta):
        await asyncio.wait_for(
            case.manager.handle_group_event(_event(str(index), sender=100 + index)), timeout=2
        )
    assert session.seq == 2 + delta
    release_inference.set()
    record = await case.next_audit()
    await case.manager.shutdown()

    assert record.stale_action == action
    if delta == 3:
        case.sender.send_group_msg.assert_awaited_once_with(
            1, _GROUP, generated, reply_to_msg_id="last-part"
        )
        assert record.final_reply == generated
        assert session.buffer[-1].reply_to_msg_id == "last-part"
    else:
        case.sender.send_group_msg.assert_not_called()
        assert record.final_reply is None
    assert [call.args[0].msg_id for call in case.store.save_message.await_args_list][:2] == [
        "first-part",
        "last-part",
    ]


async def test_follow_up_gates_on_latest_inbound_message_after_bot_sends(make_case) -> None:
    profile = _profile()
    profile.trigger_mode = "mention_only"
    case = make_case([profile])
    inference_started = asyncio.Event()
    release_inference = asyncio.Event()

    async def reply(*_args, **_kwargs) -> LLMResult:
        if case.llm.chat_completion.await_count == 1:
            inference_started.set()
            await release_inference.wait()
            return _result()
        return _result("下周我有空。")

    case.llm.chat_completion.side_effect = reply
    await case.manager.handle_group_event(_event("10", mention=True))
    await asyncio.wait_for(inference_started.wait(), timeout=2)
    await case.manager.handle_group_event(_event("11", sender=99, mention=True))
    release_inference.set()
    assert (await case.next_audit()).final_reply == _REPLY
    assert (await case.next_audit()).final_reply == "下周我有空。"
    assert case.sender.send_group_msg.await_count == 2


async def test_slow_failing_audit_does_not_block_other_group_sends(make_case) -> None:
    case = make_case([_profile()])
    audit_started = asyncio.Event()
    release_audit = asyncio.Event()

    async def fail(_record: AuditRecord) -> None:
        audit_started.set()
        await release_audit.wait()
        raise OSError("private database diagnostic")

    case.store.save_audit_log.side_effect = fail
    with capture_logs() as logs:
        try:
            await case.manager.handle_group_event(_event("10"))
            assert (await asyncio.wait_for(case.sends.get(), timeout=2))[1] == _GROUP
            await asyncio.wait_for(audit_started.wait(), timeout=2)
            await case.manager.handle_group_event(_event("11", group_id=_GROUP + 1))
            assert (await asyncio.wait_for(case.sends.get(), timeout=2))[1] == _GROUP + 1
        finally:
            release_audit.set()
        await case.audit.drain()

    warnings = [log for log in logs if log["event"] == "decision_audit_write_failed"]
    assert len(warnings) == 2
    assert all(log["log_level"] == "warning" for log in warnings)
    assert "private database diagnostic" not in str(warnings)


@pytest.mark.parametrize("start_fails", [False, True])
async def test_main_cleans_up_in_order_after_signal_or_start_failure(
    monkeypatch: pytest.MonkeyPatch, start_fails: bool
) -> None:
    order: list[str] = []
    signals: dict[signal.Signals, Callable[[], None]] = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", signals.__setitem__)
    monkeypatch.setattr(loop, "remove_signal_handler", signals.pop)
    monkeypatch.setattr(entry_point.os, "name", "posix")

    def operation(name: str) -> AsyncMock:
        mock = AsyncMock(side_effect=lambda: order.append(name))
        return mock

    def start() -> None:
        order.append("ws.start")
        if start_fails:
            raise OSError("Cannot bind mock gateway")
        signals[signal.SIGTERM]()

    def router(*_args) -> Mock:
        order.append("llm.create")
        return Mock(aclose=operation("llm.close"))

    def manager(*_args, **_kwargs) -> Mock:
        order.append("manager.create")
        return Mock(runtimes={1: Mock()}, shutdown=operation("manager.shutdown"))

    monkeypatch.setattr(
        entry_point,
        "PgStore",
        Mock(return_value=Mock(connect=operation("pg.connect"), close=operation("pg.close"))),
    )
    monkeypatch.setattr(entry_point, "LLMRouter", router)
    monkeypatch.setattr(entry_point, "MultiBotManager", manager)
    monkeypatch.setattr(
        entry_point, "AuditLogger", Mock(return_value=Mock(drain=operation("audit.drain")))
    )
    monkeypatch.setattr(
        entry_point,
        "OneBotWSGateway",
        Mock(
            return_value=Mock(
                start=AsyncMock(side_effect=start), stop=operation("ws.stop"), port=8080
            )
        ),
    )
    config = GlobalConfig(postgres=PostgresConfig(dsn="postgresql://mock"), llm_models={})

    if start_fails:
        with pytest.raises(OSError, match="Cannot bind mock gateway"):
            await entry_point.run(config, [_profile()])
    else:
        await entry_point.run(config, [_profile()])

    assert order == [
        "pg.connect",
        "llm.create",
        "manager.create",
        "ws.start",
        "ws.stop",
        "manager.shutdown",
        "audit.drain",
        "llm.close",
        "pg.close",
    ]
    assert not signals
