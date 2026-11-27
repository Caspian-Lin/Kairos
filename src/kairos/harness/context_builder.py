"""Build role-separated model input from a group conversation."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from kairos.config import BotProfile
from kairos.harness.compactor import clip_to_tokens, estimate_tokens
from kairos.harness.postprocess import SILENT_ACTION
from kairos.harness.thread_tracker import format_thread_board
from kairos.models import ChatMessage, GateDecision, NormalizedMessage, WarmMemory

_TURN_TOKENS = 4


def format_relative_time(delta_seconds: float) -> str:
    """Render an age in the largest applicable whole unit."""
    seconds = max(0, int(delta_seconds))
    for scale, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= scale:
            return f"[-{seconds // scale}{unit}]"
    return f"[-{seconds}s]"


def _utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _silence_line(seconds: float) -> str:
    hours, minutes = divmod(int(seconds // 60), 60)
    duration = f"{hours} 小时 {minutes} 分钟" if hours else f"{minutes} 分钟"
    return f"--- [静默 {duration}] ---"


class ContextBuilder:
    @staticmethod
    def build_messages(
        profile: BotProfile,
        messages: Sequence[NormalizedMessage],
        roster: Sequence[str],
        now: datetime | None = None,
        *,
        warm_memory: WarmMemory | None = None,
        gate_decision: GateDecision | None = None,
        token_budget: int | None = None,
        enable_react: bool = False,
    ) -> list[ChatMessage]:
        """Keep each speaker's historical role and render timing in group-chat turns."""
        reference_time = datetime.now(UTC) if now is None else _utc(now)
        roster_text = "、".join(roster) or "暂无"
        prompt = (
            f"{profile.persona_prompt}\n\n"
            f"我的群聊名是 {profile.nickname}。"
            "我正在群聊中。user 是群友的发言，assistant 是我此前发出的消息。"
            "[mID] 是消息 ID，[-Xs] 等标记表示距现在的时间，"
            "[U1:昵称] 标注发言人，(回复 mID) 指向被引用的消息。\n"
            f"当前活跃群友：{roster_text}\n"
            f"如果决定沉默，仅输出 {SILENT_ACTION}。\n"
            "回复时直接用第一人称写要发到群里的正文。按话题决定篇幅，开头省略 [mID]、[-Xs]、[我] 等标签。"
        )
        if enable_react:
            prompt += "\n若适合贴表情，仅输出 <|act:react:mID:emoji_id|>，使用群聊中的消息 ID。"
        if gate_decision is not None and (gate_decision.candidate_target or gate_decision.reason):
            target = (
                f"m{gate_decision.candidate_target}" if gate_decision.candidate_target else "无"
            )
            prompt += (
                f"\n初筛建议目标 {target}，理由：{gate_decision.reason or '无'}。"
                "仅供参考，可自行修正对象、保持沉默或改贴表情。"
            )
        if warm_memory is not None:
            prompt += _memory_sections(warm_memory)

        ordered = sorted(messages, key=lambda item: _utc(item.created_at))
        if token_budget is None:
            return ContextBuilder._dialogue_turns(prompt, ordered, reference_time)
        if token_budget < _TURN_TOKENS:
            return []

        latest_user = next((message for message in reversed(ordered) if not message.is_bot), None)
        latest_turn = (
            ContextBuilder._dialogue_turns("", (latest_user,), reference_time)[1]
            if latest_user is not None
            else None
        )
        reserve = min(_turn_cost(latest_turn), token_budget // 4) if latest_turn is not None else 0
        prompt = clip_to_tokens(prompt, token_budget - _TURN_TOKENS - reserve)
        turns = ContextBuilder._dialogue_turns(prompt, ordered, reference_time)
        if sum(map(_turn_cost, turns)) <= token_budget:
            return turns

        # A cut can land inside a user block. Rebuild the longest suffix that fits
        # so a leading assistant turn and its orphaned timeline markers are removed.
        low, high = 1, len(ordered)
        while low < high:
            middle = (low + high) // 2
            candidate = ContextBuilder._dialogue_turns(prompt, ordered[middle:], reference_time)
            if sum(map(_turn_cost, candidate)) <= token_budget:
                high = middle
            else:
                low = middle + 1
        turns = ContextBuilder._dialogue_turns(prompt, ordered[low:], reference_time)
        if len(turns) == 1 and latest_turn is not None:
            available = token_budget - _turn_cost(turns[0]) - _TURN_TOKENS
            if content := clip_to_tokens(latest_turn.content, available):
                turns.append(ChatMessage(role="user", content=content))
        return turns

    @staticmethod
    def _dialogue_turns(
        prompt: str, messages: Sequence[NormalizedMessage], reference_time: datetime
    ) -> list[ChatMessage]:
        turns = [ChatMessage(role="system", content=prompt)]
        previous_activity: datetime | None = None

        for message in messages:
            previous_turn = turns[-1]
            if message.is_bot and previous_turn.role == "system":
                continue

            created_at = _utc(message.created_at)
            silence = ""
            if previous_activity is not None:
                gap = (created_at - previous_activity).total_seconds()
                if gap > 300:
                    silence = _silence_line(gap)

            role: Literal["user", "assistant"]
            if message.is_bot:
                role = "assistant"
                content = message.content
                if silence:
                    # Keep timeline markers in the preceding group-chat block.
                    user_turn = previous_turn if previous_turn.role == "user" else turns[-2]
                    user_turn.content += f"\n{silence}"
            else:
                role = "user"
                relative_time = format_relative_time((reference_time - created_at).total_seconds())
                reply_tag = (
                    f" (回复 m{message.reply_to_msg_id})"
                    if message.reply_to_msg_id is not None
                    else ""
                )
                content = (
                    f"[m{message.msg_id}] {relative_time} "
                    f"[{message.sender_alias}]{reply_tag}: {message.content}"
                )
                if silence:
                    content = f"{silence}\n{content}"

            if previous_turn.role == role:
                previous_turn.content += f"\n{content}"
            else:
                turns.append(ChatMessage(role=role, content=content))
            previous_activity = _utc(message.latest_at)

        return turns


def _turn_cost(turn: ChatMessage) -> int:
    return _TURN_TOKENS + estimate_tokens(turn.content)


def _memory_sections(memory: WarmMemory) -> str:
    sections = []
    if memory.summaries:
        summaries = sorted(memory.summaries, key=lambda item: _utc(item.covered_until_at))
        sections.append(
            "历史对话摘要\n"
            + "\n".join(
                f"截至 {_utc(summary.covered_until_at).isoformat()} [m{summary.covered_until_msg_id}]："
                f"{summary.summary_text}"
                for summary in summaries
            )
        )
    if memory.impressions:
        sections.append(
            "群友画像\n"
            + "\n".join(
                f"{item.nickname}：{item.impression_summary}" for item in memory.impressions
            )
        )
    if memory.notes:
        sections.append("相关事件笔记\n" + "\n".join(note.summary_text for note in memory.notes))
    if memory.threads:
        sections.append("活跃线程板\n" + format_thread_board(memory.threads))
    return "\n\n" + "\n\n".join(sections) if sections else ""
