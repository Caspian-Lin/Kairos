"""Classify recent group activity with the optional lightweight model."""

import asyncio
import json
import re
from collections.abc import Sequence
from time import perf_counter

import structlog

from kairos.config import BotProfile
from kairos.models import ChatMessage, GateAction, GateDecision, NormalizedMessage, ThreadSummary
from kairos.protocols import GateHook, LLMClient

_log = structlog.get_logger()
_TARGET_ID = re.compile(r"m?(-?[0-9]+)")
_EMOJI_ID = re.compile(r"([0-9]+)")


def _gate_prompt(
    profile: BotProfile,
    messages: Sequence[NormalizedMessage],
    threads: Sequence[ThreadSummary],
    tool_catalog: str,
) -> list[ChatMessage]:
    persona = profile.gate.persona_card or profile.persona_prompt[:150]
    system = (
        f"群聊初筛。人设：{persona}\n"
        '只输出 JSON，如 {"action":"pass","candidate_target":"m108",'
        '"emoji_id":null,"reason":"简短理由"}。'
        "动作可选 pass、silent、react。无目标或表情用 null，"
        "贴表情时填 QQ 表情 ID（常用：76赞、66爱心、13呲牙、79胜利、63玫瑰）。"
    )
    if tool_catalog:
        system += (
            f"\n可用工具：{tool_catalog}。"
            '确有需要时才在 JSON 中加入 "need_tools": ["工具名"]，根据目录选择名称。'
        )
    board = " ".join(
        f"m{thread.latest_msg_id[:16]}({thread.message_count})" for thread in threads[:4]
    )
    remaining = 216
    lines = []
    for message in reversed(messages[-profile.gate.recent_messages_limit :]):
        if remaining < 24:
            break
        sender = "我" if message.is_bot else message.sender_alias[:16]
        content = message.content[:64].replace("\n", " ").replace("\r", " ")
        line = f"m{message.last_msg_id[:20]} {sender}: {content}"[:remaining]
        lines.append(line)
        remaining -= len(line) + 1
    recent = "\n".join(reversed(lines))
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(
            role="user", content=f"线程 mID(消息数)：{board[:64] or '无'}\n近期消息：\n{recent}"
        ),
    ]


def _parse_id(value: object, pattern: re.Pattern[str]) -> str | None:
    if not isinstance(value, str) and type(value) is not int:
        return None
    match = pattern.fullmatch(str(value).strip())
    return match[1] if match else None


def _parse_decision(content: str, *, allow_tools: bool) -> GateDecision:
    decoder = json.JSONDecoder()
    text = content[:4096]
    for opening in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, opening.start())
            raw_action = data.get("action")
            if not isinstance(raw_action, str):
                continue
            action = GateAction(raw_action.strip().lower())
        except ValueError:
            continue

        target = _parse_id(data.get("candidate_target"), _TARGET_ID)
        emoji = _parse_id(data.get("emoji_id"), _EMOJI_ID)
        if action is GateAction.REACT and emoji is None:
            return GateDecision(action=GateAction.SILENT, reason="gate_invalid_reaction")
        reason = data.get("reason")
        names = data.get("need_tools") if allow_tools and action is GateAction.PASS else None
        requested = (
            list(
                dict.fromkeys(
                    name.strip() for name in names if isinstance(name, str) and name.strip()
                )
            )
            if isinstance(names, list)
            else []
        )
        return GateDecision(
            action=action,
            candidate_target=target,
            emoji_id=emoji if action is GateAction.REACT else None,
            reason=reason.strip()[:120] if isinstance(reason, str) else "",
            need_tools=requested,
        )
    raise ValueError("Invalid gate JSON")


class ModelGate(GateHook):
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def evaluate(
        self,
        profile: BotProfile,
        messages: Sequence[NormalizedMessage],
        threads: Sequence[ThreadSummary] = (),
        *,
        latest_msg: NormalizedMessage | None = None,
        tool_catalog: str = "",
    ) -> GateDecision:
        model_key = profile.gate_model_key
        if not profile.gate.enabled or model_key is None:
            return GateDecision(action=GateAction.PASS, reason="gate_disabled")

        latest = latest_msg or (messages[-1] if messages else None)
        direct = (
            GateDecision(
                action=GateAction.PASS, candidate_target=latest.last_msg_id, reason="direct_address"
            )
            if latest is not None and (latest.is_at_bot or latest.is_reply_to_bot)
            else None
        )
        if direct is not None and not tool_catalog:
            return direct

        prompt = _gate_prompt(profile, messages, threads, tool_catalog)
        if direct is not None:
            prompt[0].content += "\n本条为直接提问，请放行并按需选择工具。"
        started = perf_counter()
        try:
            async with asyncio.timeout(profile.gate.timeout_seconds):
                reply = await self._llm.chat_completion(
                    model_key,
                    prompt,
                    session_id=f"{profile.bot_id}:{latest.group_id}:gate" if latest else None,
                    temperature=0.1,
                    max_tokens=160,
                    response_format={"type": "json_object"},
                )
            decision = _parse_decision(reply.content, allow_tools=bool(tool_catalog))
        except Exception as exc:  # noqa: BLE001 - Gate adapter and parse failures follow the configured fallback.
            _log.warning("model_gate_failed", bot_id=profile.bot_id, error_type=type(exc).__name__)
            decision = (
                GateDecision(action=GateAction.PASS, reason="gate_fallback_open")
                if profile.gate.fail_open
                else GateDecision(action=GateAction.SILENT, reason="gate_fallback_closed")
            )
        if direct is not None:
            direct.need_tools = decision.need_tools
            decision = direct
        decision.latency_ms = int((perf_counter() - started) * 1000)
        return decision
