"""Interpret the model's speaking protocol and filter repeated replies."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from kairos.models import ActionType, BotAction

SILENT_ACTION = "<|act:silent|>"
REACT_ACTION = re.compile(r"<\|act:react:m?(-?[0-9]+):([0-9]+)\|>")
_INVALID_REACT_ACTION = re.compile(r"<\|act:react:[^\r\n|]*\|>")


@dataclass(slots=True)
class PostProcessResult:
    content: str
    vetoed: bool = False
    repeated_dropped: bool = False
    action: BotAction = field(default_factory=lambda: BotAction(action=ActionType.SILENT))


def _character_grams(text: str) -> set[str]:
    if len(text) < 2:
        return set(text)
    return {text[index : index + 2] for index in range(len(text) - 1)}


def process_llm_output(
    raw_text: str,
    recent_bot_replies: Sequence[str],
    similarity_threshold: float = 0.70,
    *,
    enable_react: bool = True,
) -> PostProcessResult:
    """Recognize standalone actions and preserve ordinary reply text."""
    text = raw_text.strip()
    if text == SILENT_ACTION:
        return PostProcessResult(content="", vetoed=True)
    if not text:
        return PostProcessResult(content="")
    if enable_react:
        if reaction := REACT_ACTION.fullmatch(text):
            return PostProcessResult(
                content="",
                action=BotAction(
                    action=ActionType.REACT, target_msg_id=reaction[1], emoji_id=reaction[2]
                ),
            )
        if _INVALID_REACT_ACTION.fullmatch(text):
            return PostProcessResult(content="")

    generated_grams = _character_grams(text)
    for recent_reply in recent_bot_replies[-2:]:
        recent_grams = _character_grams(recent_reply.strip())
        similarity = len(generated_grams & recent_grams) / len(generated_grams | recent_grams)
        if similarity > similarity_threshold:
            return PostProcessResult(content="", repeated_dropped=True)
    return PostProcessResult(content=text, action=BotAction(action=ActionType.REPLY, content=text))
