"""Build role-separated model input from a group conversation."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from kairos.config import BotProfile
from kairos.harness.postprocess import SILENT_ACTION
from kairos.models import ChatMessage, NormalizedMessage


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
        turns = [ChatMessage(role="system", content=prompt)]
        previous_activity: datetime | None = None

        for message in sorted(messages, key=lambda item: _utc(item.created_at)):
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
