"""Interpret the model's speaking protocol and filter repeated replies."""

from collections.abc import Sequence
from dataclasses import dataclass

SILENT_ACTION = "<|act:silent|>"


@dataclass(slots=True)
class PostProcessResult:
    content: str
    vetoed: bool = False
    repeated_dropped: bool = False


def _character_grams(text: str) -> set[str]:
    if len(text) < 2:
        return set(text)
    return {text[index : index + 2] for index in range(len(text) - 1)}


def process_llm_output(
    raw_text: str,
    recent_bot_replies: Sequence[str],
    similarity_threshold: float = 0.70,
) -> PostProcessResult:
    """Recognize a standalone silence marker and preserve the reply body."""
    text = raw_text.strip()
    if text == SILENT_ACTION:
        return PostProcessResult(content="", vetoed=True)
    if not text:
        return PostProcessResult(content="")

    generated_grams = _character_grams(text)
    for recent_reply in recent_bot_replies[-2:]:
        recent_grams = _character_grams(recent_reply.strip())
        similarity = len(generated_grams & recent_grams) / len(generated_grams | recent_grams)
        if similarity > similarity_threshold:
            return PostProcessResult(content="", repeated_dropped=True)
    return PostProcessResult(content=text)
