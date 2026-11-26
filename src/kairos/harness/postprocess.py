"""Interpret the model's speaking protocol and filter repeated replies."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

_VETO = "<|act:silent|>"
_METADATA_PREFIX = re.compile(
    r"^[\[（(【](?:m(?:-?\d+|ID)|-(?:\d+|X)[smhd]|我|U\d+(?:[:：][^\]）)】\r\n]*)?"
    r"|回复\s+m-?\d+)[\]）)】]\s*[:：]?\s*"
)


@dataclass(slots=True)
class PostProcessResult:
    vetoed: bool
    repeated_dropped: bool
    segments: list[str]
    raw_cleaned: str


def _strip_metadata(text: str) -> str:
    text = text.strip()
    while prefix := _METADATA_PREFIX.match(text):
        text = text[prefix.end() :]
    return text


def _character_grams(text: str) -> set[str]:
    if len(text) < 2:
        return set(text)
    return {text[index : index + 2] for index in range(len(text) - 1)}


def process_llm_output(
    raw_text: str,
    recent_bot_replies: Sequence[str],
    max_segments: int = 2,
    similarity_threshold: float = 0.70,
) -> PostProcessResult:
    """Prepare sendable segments, with recent replies ordered oldest to newest."""
    if max_segments < 1:
        raise ValueError("max_segments must be positive")

    text = raw_text.strip()
    if _VETO in text:
        return PostProcessResult(vetoed=True, repeated_dropped=False, segments=[], raw_cleaned=text)

    segments = [cleaned for part in text.split("||") if (cleaned := _strip_metadata(part))]
    if not segments:
        return PostProcessResult(vetoed=True, repeated_dropped=False, segments=[], raw_cleaned="")

    raw_cleaned = " || ".join(segments)
    generated_grams = _character_grams(raw_cleaned)
    for recent_reply in recent_bot_replies[-2:]:
        recent_grams = _character_grams(recent_reply.strip())
        similarity = len(generated_grams & recent_grams) / len(generated_grams | recent_grams)
        if similarity > similarity_threshold:
            return PostProcessResult(
                vetoed=False, repeated_dropped=True, segments=[], raw_cleaned=raw_cleaned
            )

    if len(segments) > max_segments:
        last_segment = max_segments - 1
        segments = [*segments[:last_segment], " ".join(segments[last_segment:])]
    return PostProcessResult(
        vetoed=False, repeated_dropped=False, segments=segments, raw_cleaned=raw_cleaned
    )
