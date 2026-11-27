import asyncio
import json
import signal
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from itertools import count
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientSession
from conftest import FakeSender, FakeStore
from structlog.testing import capture_logs

from kairos import main as entry_point
from kairos.config import (
    BotProfile,
    DebounceConfig,
    GlobalConfig,
    OneBotWSConfig,
    PostgresConfig,
    RateLimitConfig,
)
from kairos.gateway.onebot_ws import OneBotSendError, OneBotWSGateway
from kairos.harness.tool_registry import ToolRegistry
from kairos.llm import LLMRequestError
from kairos.main import MultiBotManager
from kairos.models import (
    AuditRecord,
    ContextSummary,
    LLMResult,
    MemoryNote,
    MessageSegment,
    NormalizedMessage,
    RawGroupEvent,
    ToolDefinition,
    ToolSpec,
)
from kairos.protocols import LLMClient
from kairos.runtime import bot_instance
from kairos.telemetry.audit_logger import AuditLogger
from kairos.telemetry.trace_store import PipelineTraceStore

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
    def __init__(
        self,
        profiles: Sequence[BotProfile],
        bot_clients: Mapping[str, LLMClient] | None = None,
        *,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.store = FakeStore()
        self.audit_records: asyncio.Queue[AuditRecord] = asyncio.Queue()
        self.store.save_audit_log.side_effect = self.audit_records.put_nowait
        self.llm = Mock(spec=LLMClient)
        self.llm.chat_completion.return_value = _result()
        self.sender = FakeSender()
        self.sends: asyncio.Queue[tuple[int, int, str, str | None]] = asyncio.Queue()
        echo_ids = count(1001)

        def acknowledge(
            self_id: int, group_id: int, text: str, reply_to_msg_id: str | None = None
        ) -> str:
            self.sends.put_nowait((self_id, group_id, text, reply_to_msg_id))
            return str(next(echo_ids))

        self.sender.send_group_msg.side_effect = acknowledge
        self.audit = AuditLogger(self.store)
        self.traces = PipelineTraceStore()
        self.manager = MultiBotManager(
            profiles,
            self.store,
            self.llm,
            self.sender,
            self.audit,
            bot_llm_clients=bot_clients,
            summary_store=self.store,
            memory_store=self.store,
            tool_registry=registry,
            trace_store=self.traces,
        )

    async def next_audit(self) -> AuditRecord:
        return await asyncio.wait_for(self.audit_records.get(), timeout=2)


@pytest.fixture
async def make_case() -> AsyncIterator[Callable[[Sequence[BotProfile]], _RuntimeCase]]:
    async with AsyncExitStack() as resources:

        def create(
            profiles: Sequence[BotProfile],
            bot_clients: Mapping[str, LLMClient] | None = None,
            *,
            registry: ToolRegistry | None = None,
        ) -> _RuntimeCase:
            case = _RuntimeCase(profiles, bot_clients, registry=registry)
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


def _gated_profile() -> BotProfile:
    profile = _profile()
    profile.gate.enabled = True
    profile.gate.energy_recover_per_min = 0
    profile.gate_model_key = "gate"
    return profile


@pytest.mark.parametrize(
    ("action", "candidate", "expected_target"),
    [("silent", None, None), ("react", "m9", "9"), ("react", None, "10"), ("react", "m999", "10")],
)
async def test_model_gate_short_circuits_the_main_model_with_audit(
    make_case, action: str, candidate: str | None, expected_target: str | None
) -> None:
    case = make_case([_gated_profile()])
    case.store.load_recent_messages.return_value = [_history("9", 43, "U1", "另一条")]
    emoji_id = "76" if action == "react" else None
    case.llm.chat_completion.return_value = _result(
        json.dumps(
            {
                "action": action,
                "candidate_target": candidate,
                "emoji_id": emoji_id,
                "reason": "初筛理由",
            }
        )
    )
    await case.manager.handle_group_event(_event("10"))
    audit = await case.next_audit()
    await case.manager.shutdown()

    assert [call.args[0] for call in case.llm.chat_completion.await_args_list] == ["gate"]
    assert audit.stale_action == f"GATE_{action.upper()}"
    assert audit.gate_output["action"] == audit.action_type == action
    assert audit.gate_output["reason"] == "初筛理由"
    assert audit.gate_output["latency_ms"] >= 0
    assert (audit.target_msg_id, audit.emoji_id) == (expected_target, emoji_id)
    case.sender.send_group_msg.assert_not_awaited()
    if expected_target is None:
        case.sender.set_msg_emoji_like.assert_not_awaited()
    else:
        case.sender.set_msg_emoji_like.assert_awaited_once_with(1, expected_target, emoji_id)
    session = case.manager.runtimes[1].sessions[_GROUP]
    assert session.energy == (100 if action == "silent" else 95)
    assert [message.msg_id for message in session.buffer] == ["9", "10"]
    assert [call.args[0].msg_id for call in case.store.save_message.await_args_list] == ["10"]


@pytest.mark.parametrize(
    ("answer", "action", "outcome", "remaining", "target"),
    [
        (_REPLY, "reply", "SEND_DIRECT", 75, None),
        ("<|act:silent|>", "silent", "VETOED", 100, None),
        ("<|act:react:m10:76|>", "react", "MAIN_REACT", 95, "10"),
        ("<|act:react:m999:76|>", "react", "MAIN_REACT_INVALID", 100, "999"),
    ],
)
async def test_gate_pass_keeps_the_main_models_action_choice(
    make_case, answer: str, action: str, outcome: str, remaining: float, target: str | None
) -> None:
    case = make_case([_gated_profile()])
    case.llm.chat_completion.side_effect = [
        _result('{"action":"pass","candidate_target":"m10","reason":"可参与"}'),
        _result(answer),
    ]
    await case.manager.handle_group_event(_event("10"))
    audit = await case.next_audit()
    await case.manager.shutdown()

    calls = case.llm.chat_completion.await_args_list
    assert [call.args[0] for call in calls] == ["gate", "first-model"]
    prompt = calls[1].args[1][0].content
    assert "初筛建议目标 m10" in prompt and "仅供参考" in prompt
    assert "<|act:react:mID:emoji_id|>" in prompt and "活跃线程板" in prompt
    assert (audit.action_type, audit.stale_action, audit.target_msg_id) == (action, outcome, target)
    assert audit.main_vetoed is (action == "silent")
    session = case.manager.runtimes[1].sessions[_GROUP]
    assert session.energy == remaining
    ids = [call.args[0].msg_id for call in case.store.save_message.await_args_list]
    assert ids == (["10", "1001"] if action == "reply" else ["10"])
    if outcome == "MAIN_REACT":
        case.sender.set_msg_emoji_like.assert_awaited_once_with(1, "10", "76")
    else:
        case.sender.set_msg_emoji_like.assert_not_awaited()


@pytest.mark.parametrize(
    ("responses", "expected_outcome"),
    [
        (
            [_result('{"action":"react","candidate_target":"m10","emoji_id":"76"}')],
            "GATE_REACT_FAILED",
        ),
        (
            [
                _result('{"action":"pass","candidate_target":"m10","reason":"可参与"}'),
                _result("<|act:react:m10:76|>"),
            ],
            "MAIN_REACT_FAILED",
        ),
    ],
)
async def test_failed_reaction_records_audit_without_group_error_notice(
    make_case, responses: list[LLMResult], expected_outcome: str
) -> None:
    case = make_case([_gated_profile()])
    case.llm.chat_completion.side_effect = responses
    case.sender.set_msg_emoji_like.side_effect = OneBotSendError("unsupported emoji")
    with capture_logs() as logs:
        await case.manager.handle_group_event(_event("10"))
        audit = await case.next_audit()
        await case.manager.shutdown()

    assert (audit.action_type, audit.stale_action, audit.target_msg_id, audit.emoji_id) == (
        "react",
        expected_outcome,
        "10",
        "76",
    )
    case.sender.send_group_msg.assert_not_awaited()
    assert case.manager.runtimes[1].sessions[_GROUP].energy == 100
    assert any(log["event"] == "group_reaction_failed" for log in logs)


async def test_compacted_session_preserves_newer_same_timestamp_rows_after_warmup(
    make_case,
) -> None:
    profile = _profile()
    profile.compaction.enabled = True
    case = make_case([profile])
    earlier = _history("42", 43, "U1", "已被覆盖的原文")
    later = _history("5", 44, "U2", "同一秒续接的原文")
    summary = ContextSummary(
        bot_id=profile.bot_id,
        group_id=_GROUP,
        summary_text="恢复的持久化摘要",
        covered_until_msg_id="42",
        covered_until_at=earlier.latest_at,
        created_at=_TIME,
        token_estimate=10,
    )
    case.store.load_recent_messages.return_value = [earlier, later]
    case.store.load_recent_summaries.return_value = [summary]
    await case.manager.handle_group_event(_event("10"))
    audit = await case.next_audit()
    await case.manager.shutdown()

    system, user = [turn["content"] for turn in audit.context_snapshot]
    assert "恢复的持久化摘要" in system
    assert "同一秒续接的原文" in user and "已被覆盖的原文" not in user
    assert case.manager.runtimes[1].sessions[_GROUP].summaries == [summary]
    case.store.load_recent_summaries.assert_awaited_once_with(profile.bot_id, _GROUP, limit=3)


async def test_main_uses_hot_memory_while_recall_is_blocked(make_case) -> None:
    profile = _profile()
    profile.trigger_mode = "mention_only"
    profile.memory.enabled = True
    profile.memory.warm_trigger_messages = 1
    case = make_case([profile])
    note = MemoryNote(
        bot_id=profile.bot_id,
        scope_group_id=_GROUP,
        entity_qqs=[42],
        summary_text="上周约好周五见",
        keywords=["周五"],
        created_at=_TIME,
    )
    case.store.recall_memory_notes.return_value = [note]
    await case.manager.handle_group_event(_event("10"))
    worker = case.manager.memory_worker
    assert worker is not None
    async with asyncio.timeout(1):
        while not worker.get_warm_memory(profile.bot_id, _GROUP).notes:
            await asyncio.sleep(0)

    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def blocking_recall(*_args: object) -> list[MemoryNote]:
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
        return []

    case.store.recall_memory_notes.side_effect = blocking_recall
    await case.manager.handle_group_event(_event("11", sender=43, mention=True))
    await asyncio.wait_for(entered.wait(), timeout=1)
    audit = await case.next_audit()
    assert audit.final_reply == _REPLY
    assert "上周约好周五见" in audit.context_snapshot[0]["content"]
    assert not cancelled.is_set()
    await case.manager.shutdown()
    assert cancelled.is_set()


async def test_shared_memory_worker_uses_each_bots_own_llm_client(make_case) -> None:
    first, second = _profile(), _profile("second", 2)
    for profile in (first, second):
        profile.trigger_mode = "mention_only"
        profile.memory.enabled = True
        profile.memory.note_extract_interval_messages = 1
        profile.worker_model_key = "worker"
    clients = {profile.bot_id: Mock(spec=LLMClient) for profile in (first, second)}
    for bot_id, client in clients.items():
        client.chat_completion.return_value = _result(
            json.dumps(
                {"notes": [{"summary_text": bot_id, "entity_qqs": [42], "keywords": ["test"]}]}
            )
        )
    case = make_case([first, second], clients)
    await case.manager.handle_group_event(_event("10"))
    await case.manager.handle_group_event(_event("11", self_id=2))
    async with asyncio.timeout(1):
        while case.store.save_memory_note.await_count < 2:
            await asyncio.sleep(0)
    for bot_id, client in clients.items():
        client.chat_completion.assert_awaited_once()
        assert client.chat_completion.await_args.kwargs["session_id"] == f"{bot_id}:{_GROUP}:memory"
    assert {
        call.args[0].bot_id: call.args[0].summary_text
        for call in case.store.save_memory_note.await_args_list
    } == {
        first.bot_id: first.bot_id,
        second.bot_id: second.bot_id,
    }


async def test_m1_uses_only_existing_calls_and_preserves_literal_action_text(make_case) -> None:
    case = make_case([_profile()])
    marker = "<|act:react:10:76|>"
    case.llm.chat_completion.return_value = _result(marker)
    await case.manager.handle_group_event(_event("10"))
    audit = await case.next_audit()
    assert case.manager.memory_worker is None
    assert audit.final_reply == marker and audit.action_type is None
    case.llm.chat_completion.assert_awaited_once()
    case.store.load_recent_messages.assert_awaited_once()
    case.store.load_recent_summaries.assert_not_awaited()
    case.store.recall_memory_notes.assert_not_awaited()
    case.store.load_user_impressions.assert_not_awaited()
    case.sender.set_msg_emoji_like.assert_not_awaited()


@pytest.mark.parametrize("start_fails, restart", [(False, False), (True, False), (False, True)])
async def test_main_cleans_up_in_order_after_signal_or_start_failure(
    monkeypatch: pytest.MonkeyPatch, start_fails: bool, restart: bool
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

    controller = entry_point.RestartController(lambda: order.append("restart"))

    def start() -> None:
        order.append("ws.start")
        if start_fails:
            raise OSError("Cannot bind mock gateway")
        if restart:
            controller.request_restart()
        else:
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
            await entry_point.run(config, [_profile()], restart_controller=controller)
    else:
        await entry_point.run(config, [_profile()], restart_controller=controller)

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
    ] + (["restart"] if restart else [])
    assert not signals


@pytest.mark.parametrize("requested", [False, True])
async def test_dynamic_tool_gating_and_trace_request_context(make_case, requested: bool) -> None:
    profile = _gated_profile()
    profile.enabled_tools = ["lookup"]
    tool = ToolDefinition(
        name="lookup", description="Look up a note", parameters={"type": "object"}
    )
    registry = ToolRegistry([ToolSpec(definition=tool, catalog_hint="Find a stored note")])
    case = make_case([profile], registry=registry)
    need_tools = ["lookup", "unexpected"] if requested else []
    case.llm.chat_completion.side_effect = [
        _result(json.dumps({"action": "pass", "need_tools": need_tools})),
        _result(),
    ]
    await case.manager.handle_group_event(_event("10"))
    await case.next_audit()
    calls = case.llm.chat_completion.await_args_list
    assert "lookup: Find a stored note" in calls[0].args[1][0].content
    if requested:
        assert calls[1].kwargs["tools"] == [tool]
    else:
        assert "tools" not in calls[1].kwargs
    (trace,) = case.traces.list_recent()
    assert trace.outcome == "SEND_DIRECT"
    assert [step.stage for step in trace.steps] == list(bot_instance._TRACE_STAGES)
    assert all(step.status == "ok" and step.latency_ms >= 0 for step in trace.steps)
    assert trace.request_context is not None
    assert trace.request_context["attached_tools"] == ([tool.model_dump()] if requested else [])
    assert trace.request_context["turns"][0]["role"] == "user"


@pytest.mark.parametrize("mode", ["filtered", "silent", "react", "error"])
async def test_tracing_keeps_short_circuits_and_errors(make_case, mode: str) -> None:
    profile = _gated_profile()
    if mode == "filtered":
        profile.trigger_mode = "mention_only"
    case = make_case([profile])
    if mode in {"silent", "react"}:
        case.llm.chat_completion.return_value = _result(
            json.dumps({"action": mode, "candidate_target": "10", "emoji_id": "76"})
        )
    if mode == "error":
        case.llm.chat_completion.side_effect = [_result('{"action":"pass"}'), OSError("offline")]
    await case.manager.handle_group_event(_event("10"))
    async with asyncio.timeout(2):
        while not case.traces.list_recent():
            await asyncio.sleep(0)
    (trace,) = case.traces.list_recent()
    assert (
        trace.outcome
        == {
            "filtered": "FILTERED",
            "silent": "GATE_SILENT",
            "react": "GATE_REACT",
            "error": "ERROR",
        }[mode]
    )
    assert [step.stage for step in trace.steps] == list(bot_instance._TRACE_STAGES)
    assert (trace.request_context is not None) is (mode == "error")


async def test_reload_switches_profile_on_same_websocket_and_rolls_back(make_case) -> None:
    old, new = _profile(), _profile("second", 1)
    case = make_case([old])
    gateway = OneBotWSGateway(OneBotWSConfig(port=0), case.manager.handle_group_event)
    await gateway.start()
    try:
        async with (
            ClientSession() as client,
            client.ws_connect(
                f"http://127.0.0.1:{gateway.port}{gateway.config.path}",
                headers={"X-Self-ID": "1"},
            ) as socket,
        ):
            original = case.manager.runtimes[1]
            with pytest.raises(ValueError, match="distinct"):
                await case.manager.reload_profiles([old, new], {})
            assert case.manager.runtimes[1] is original
            await case.manager.reload_profiles([new], {})
            assert case.manager.runtimes[1].profile.bot_id == "second"
            assert gateway.connected_self_ids == {1}
            await socket.send_json(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "group_id": _GROUP,
                    "message_id": 10,
                    "user_id": 42,
                    "time": int(_TIME.timestamp()),
                    "message": "你好",
                }
            )
            audit = await case.next_audit()
            assert audit.bot_id == "second" and audit.stale_action == "SEND_DIRECT"
            assert not socket.closed
    finally:
        await gateway.stop()


async def test_three_compaction_failures_reset_the_runtime_context(make_case) -> None:
    profile = _profile()
    profile.trigger_mode = "mention_only"
    profile.ring_buffer_size = 2
    profile.worker_model_key = "worker"
    profile.compaction.enabled = True
    profile.compaction.keep_recent_messages = 1
    case = make_case([profile])

    async def reply(model: str, *_args: object, **_kwargs: object) -> LLMResult:
        return _result("") if model == "worker" else _result()

    case.llm.chat_completion.side_effect = reply
    await case.manager.handle_group_event(_event("11", text="旧对话甲"))
    await case.manager.handle_group_event(_event("12", sender=43, text="旧对话乙"))
    async with asyncio.timeout(2):
        while not case.store.save_summary.await_count:
            await asyncio.sleep(0)
    session = case.manager.runtimes[1].sessions[_GROUP]
    assert session.context_generation == 1 and not session.buffer and not session.summaries
    (marker,) = case.store.save_summary.await_args.args
    assert marker.covered_until_msg_id == "12" and marker.summary_text == ""
    assert [call.args[0] for call in case.llm.chat_completion.await_args_list] == ["worker"] * 3

    await case.manager.handle_group_event(_event("13", mention=True, text="新对话"))
    audit = await case.next_audit()
    assert audit.final_reply == _REPLY
    assert "新对话" in audit.context_snapshot[-1]["content"]
    assert all("旧对话" not in turn["content"] for turn in audit.context_snapshot)
