import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from kairos.config import BotProfile, GateConfig
from kairos.harness.gate import ModelGate
from kairos.models import GateAction, LLMResult, NormalizedMessage, ThreadSummary
from kairos.protocols import LLMClient

_AT = datetime(2026, 11, 27, tzinfo=UTC)


def _profile(*, fail_open: bool = False) -> BotProfile:
    return BotProfile(
        bot_id="bot",
        self_id=100,
        nickname="小满",
        persona_prompt="完整人设" * 300,
        gate_model_key="gate",
        gate=GateConfig(enabled=True, timeout_seconds=0.01, fail_open=fail_open),
    )


def _message(msg_id: int, content: str, **updates: bool) -> NormalizedMessage:
    return NormalizedMessage(
        bot_id="bot",
        group_id=1,
        msg_id=str(msg_id),
        last_msg_id=str(msg_id),
        sender_qq=101,
        sender_alias="U1",
        content=content,
        reply_to_msg_id=None,
        is_bot=False,
        is_peer_bot=False,
        is_at_bot=updates.get("is_at_bot", False),
        is_reply_to_bot=updates.get("is_reply_to_bot", False),
        created_at=_AT,
        latest_at=_AT,
    )


def _llm(content: str) -> AsyncMock:
    llm = AsyncMock(spec=LLMClient)
    llm.chat_completion.return_value = LLMResult(
        content=content, model="gate", latency_ms=1, prompt_tokens=None, completion_tokens=None
    )
    return llm


@pytest.mark.parametrize(
    ("response", "action", "target", "emoji", "reason"),
    [
        (
            '{"action":"pass","candidate_target":"m123","reason":"接着聊"}',
            GateAction.PASS,
            "123",
            None,
            "接着聊",
        ),
        ('{"action":"silent"}', GateAction.SILENT, None, None, ""),
        (
            '门控结果\n```json\n{"action":"REACT","candidate_target":"m-123","emoji_id":76}\n```\n完毕',
            GateAction.REACT,
            "-123",
            "76",
            "",
        ),
        (
            '{"action":"react","emoji_id":""}',
            GateAction.SILENT,
            None,
            None,
            "gate_invalid_reaction",
        ),
    ],
)
async def test_gate_parses_actions_and_requests_small_json(
    response: str, action: GateAction, target: str | None, emoji: str | None, reason: str
) -> None:
    llm = _llm(response)
    decision = await ModelGate(llm).evaluate(_profile(), [_message(123, "你好")])
    assert (decision.action, decision.candidate_target, decision.emoji_id, decision.reason) == (
        action,
        target,
        emoji,
        reason,
    )
    assert decision.latency_ms >= 0
    assert llm.chat_completion.await_args.args[0] == "gate"
    assert llm.chat_completion.await_args.kwargs == {
        "session_id": "bot:1:gate",
        "temperature": 0.1,
        "max_tokens": 160,
        "response_format": {"type": "json_object"},
    }


@pytest.mark.parametrize("fail_open", [False, True])
@pytest.mark.parametrize("failure", ["timeout", "request", "parse"])
async def test_gate_failures_use_configured_fallback(fail_open: bool, failure: str) -> None:
    llm = _llm("前缀 {无法解析}")

    async def wait_forever(*_args: object, **_kwargs: object) -> None:
        await asyncio.Future()

    if failure == "timeout":
        llm.chat_completion.side_effect = wait_forever
    elif failure == "request":
        llm.chat_completion.side_effect = OSError("request failed")
    result = await ModelGate(llm).evaluate(_profile(fail_open=fail_open), [_message(1, "聊天")])
    assert result.action is (GateAction.PASS if fail_open else GateAction.SILENT)
    assert result.reason == ("gate_fallback_open" if fail_open else "gate_fallback_closed")


@pytest.mark.parametrize("flag", ["is_at_bot", "is_reply_to_bot"])
async def test_direct_address_does_not_call_gate_model(flag: str) -> None:
    llm = _llm('{"action":"silent"}')
    message = _message(123, "在吗", **{flag: True})
    decision = await ModelGate(llm).evaluate(_profile(), [], latest_msg=message)
    assert decision.action is GateAction.PASS
    assert decision.candidate_target == "123"
    llm.chat_completion.assert_not_awaited()


@pytest.mark.parametrize("missing_key", [False, True])
async def test_disabled_gate_returns_pass_without_llm(missing_key: bool) -> None:
    llm = _llm('{"action":"silent"}')
    profile = _profile()
    if missing_key:
        profile.gate_model_key = None
    else:
        profile.gate.enabled = False
    decision = await ModelGate(llm).evaluate(profile, [])
    assert decision.action is GateAction.PASS and decision.reason == "gate_disabled"
    llm.chat_completion.assert_not_awaited()


async def test_gate_only_sends_recent_messages_short_persona_and_threads() -> None:
    profile = _profile()
    profile.gate.persona_card = "爱聊拉面"
    profile.gate.recent_messages_limit = 2
    messages = [_message(1, "旧消息"), _message(2, "新话题"), _message(3, "拉面" * 2000)]
    thread = ThreadSummary(
        thread_id="2",
        root_msg_id="2",
        latest_msg_id="3",
        message_count=2,
        participants=["U1"],
        heat=2,
        last_active_at=_AT,
    )
    llm = _llm('{"action":"pass"}')
    await ModelGate(llm).evaluate(profile, messages, [thread])
    system, user = llm.chat_completion.await_args.args[1]
    assert "爱聊拉面" in system.content and "完整人设" not in system.content
    assert "新话题" in user.content and "m3(2)" in user.content
    assert "旧消息" not in user.content and len(user.content) < 250

    profile.gate.persona_card = ""
    await ModelGate(llm).evaluate(profile, messages)
    assert profile.persona_prompt[:150] in llm.chat_completion.await_args.args[1][0].content
    assert profile.persona_prompt[:151] not in llm.chat_completion.await_args.args[1][0].content


async def test_gate_includes_only_a_compact_catalog_and_parses_tool_requests() -> None:
    profile = _profile()
    llm = _llm('{"action":"pass","need_tools":["status","notes","status",4," "]}')
    gate = ModelGate(llm)
    messages = [_message(123, "查看状态。")]

    unassisted = await gate.evaluate(profile, messages)
    unassisted_prompt = llm.chat_completion.await_args.args[1][0].content
    assisted = await gate.evaluate(profile, messages, tool_catalog="status: 查询当前状态")
    assisted_prompt = llm.chat_completion.await_args.args[1][0].content

    assert unassisted.need_tools == []
    assert "need_tools" not in unassisted_prompt and "可用工具" not in unassisted_prompt
    assert assisted.need_tools == ["status", "notes"]
    assert assisted_prompt.startswith(unassisted_prompt)
    assert "status: 查询当前状态" in assisted_prompt
    assert "need_tools" in assisted_prompt
    assert "tools" not in llm.chat_completion.await_args.kwargs


@pytest.mark.parametrize(
    ("response", "need_tools"),
    [
        ('{"action":"pass","need_tools":["status"]}', ["status"]),
        ('{"action":"silent","need_tools":["status"]}', []),
        ('{"action":"react","candidate_target":"m123","emoji_id":"76"}', []),
        ("invalid", []),
    ],
)
async def test_direct_address_with_catalog_can_route_tools_and_still_pass(
    response: str, need_tools: list[str]
) -> None:
    llm = _llm(response)
    gate = ModelGate(llm)
    message = _message(123, "查一下状态。", is_at_bot=True)
    decision = await gate.evaluate(
        _profile(), [message], latest_msg=message, tool_catalog="status: 查询状态"
    )

    llm.chat_completion.assert_awaited_once()
    assert "直接提问" in llm.chat_completion.await_args.args[1][0].content
    assert (decision.action, decision.reason, decision.candidate_target) == (
        GateAction.PASS,
        "direct_address",
        message.last_msg_id,
    )
    assert decision.need_tools == need_tools
