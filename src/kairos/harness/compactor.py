"""Estimate prompt size and summarize eligible history outside the chat path."""

from collections.abc import Sequence
from datetime import UTC, datetime

import structlog

from kairos.config import BotProfile, CompactionConfig
from kairos.models import ChatMessage, CompactionOutcome, ContextSummary, NormalizedMessage
from kairos.protocols import ContextCompactor, LLMClient

_log = structlog.get_logger()


def estimate_tokens(text: str) -> int:
    """Approximate Latin words at four characters per token and other glyphs at one."""
    latin = sum(char.isspace() or (char.isascii() and char.isalnum()) for char in text)
    return (latin + 3) // 4 + len(text) - latin


def clip_to_tokens(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()


def _utc(timestamp: datetime) -> datetime:
    return timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp


def _merge_context(
    summaries: Sequence[ContextSummary], messages: Sequence[NormalizedMessage], limit: int
) -> tuple[list[ContextSummary], list[NormalizedMessage]]:
    unique_summaries: dict[tuple[datetime, str], ContextSummary] = {}
    for summary in sorted(summaries, key=lambda item: _utc(item.created_at)):
        unique_summaries[(_utc(summary.covered_until_at), summary.covered_until_msg_id)] = summary
    ordered_summaries = sorted(
        unique_summaries.values(), key=lambda item: _utc(item.covered_until_at)
    )[-limit:]

    by_id: dict[str, NormalizedMessage] = {}
    for message in messages:
        previous = by_id.get(message.msg_id)
        if previous is None or _utc(message.latest_at) > _utc(previous.latest_at):
            by_id[message.msg_id] = message
    burst_tails = {
        message.last_msg_id for message in by_id.values() if message.last_msg_id != message.msg_id
    }
    ordered_messages = sorted(
        (message for message in by_id.values() if message.msg_id not in burst_tails),
        key=lambda item: _utc(item.created_at),
    )
    if not ordered_summaries:
        return [], ordered_messages

    latest = ordered_summaries[-1]
    watermark = _utc(latest.covered_until_at)
    # Store order settles timestamp ties when the raw watermark message is still present.
    anchor = next(
        (
            index
            for index, message in enumerate(ordered_messages)
            if latest.covered_until_msg_id in (message.msg_id, message.last_msg_id)
        ),
        None,
    )
    uncovered = [
        message
        for index, message in enumerate(ordered_messages)
        if _utc(message.latest_at) > watermark
        or (anchor is not None and index > anchor and _utc(message.latest_at) == watermark)
    ]
    return ordered_summaries, uncovered


def _uses_legacy_budget(settings: CompactionConfig) -> bool:
    fields = settings.model_fields_set
    return bool(fields & {"input_token_budget", "trigger_ratio", "summary_max_tokens"}) and not (
        fields & {"trigger_tokens", "target_summary_tokens", "max_retries"}
    )


class DefaultContextCompactor(ContextCompactor):
    def __init__(self, llm: LLMClient, *, max_persisted_summaries: int = 3) -> None:
        if max_persisted_summaries < 1:
            raise ValueError("max_persisted_summaries must be positive")
        self._llm = llm
        self._max_persisted_summaries = max_persisted_summaries

    def estimate_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def should_compact(
        self,
        profile: BotProfile,
        messages: Sequence[NormalizedMessage],
        summaries: Sequence[ContextSummary] = (),
    ) -> bool:
        settings = profile.compaction
        if not settings.enabled or len(messages) <= settings.keep_recent_messages:
            return False
        ordered, uncovered = _merge_context(summaries, messages, settings.max_persisted_summaries)
        if len(uncovered) <= settings.keep_recent_messages:
            return False
        estimated = estimate_tokens(profile.persona_prompt) + sum(
            estimate_tokens(summary.summary_text) for summary in ordered
        )
        estimated += sum(
            10 + estimate_tokens(f"{message.sender_alias} {message.content}")
            for message in uncovered
        )
        if len(uncovered) >= profile.ring_buffer_size:
            return True
        if _uses_legacy_budget(settings):
            return estimated > settings.input_token_budget * settings.trigger_ratio
        return estimated >= settings.trigger_tokens

    async def compact(
        self,
        profile: BotProfile,
        group_id: int,
        messages: Sequence[NormalizedMessage],
        existing_summaries: Sequence[ContextSummary] = (),
    ) -> ContextSummary | None:
        outcome = await self.compact_with_outcome(profile, group_id, messages, existing_summaries)
        return outcome.summary

    async def compact_with_outcome(
        self,
        profile: BotProfile,
        group_id: int,
        messages: Sequence[NormalizedMessage],
        existing_summaries: Sequence[ContextSummary] = (),
    ) -> CompactionOutcome:
        settings = profile.compaction
        if not settings.enabled:
            return CompactionOutcome(status="skipped", reason="disabled")
        summaries, uncovered = _merge_context(
            existing_summaries, messages, settings.max_persisted_summaries
        )
        to_summarize = uncovered[: -settings.keep_recent_messages]
        if not to_summarize:
            return CompactionOutcome(status="skipped", reason="no_old_context")
        model_key = profile.worker_model_key
        if model_key is None:
            _log.warning("context_summary_model_missing", bot_id=profile.bot_id, group_id=group_id)
            return CompactionOutcome(status="skipped", reason="worker_model_missing")

        legacy_budget = _uses_legacy_budget(settings)
        max_attempts = 1 if legacy_budget else settings.max_retries
        target = settings.summary_max_tokens if legacy_budget else settings.target_summary_tokens
        history = "\n".join(
            f"截至 m{summary.covered_until_msg_id}：{summary.summary_text}" for summary in summaries
        )
        dialogue = "\n".join(
            f"[m{message.msg_id}] {message.sender_alias}：{message.content}"
            for message in to_summarize
        )
        instruction = (
            "用简洁中文总结群聊，只保留当前仍在推进的核心事实、观点归属与未决事项。"
            "主动丢弃无关闲聊细节，不要重复已记入长期记忆（群友画像、事件笔记）的稳定事实。"
            f"已有摘要仅作背景。只输出摘要正文，最多 {target} tokens。"
        )
        source = ChatMessage(
            role="user", content=f"已有摘要：\n{history}\n\n新增群聊：\n{dialogue}"
        )
        reason = ""
        for attempt in range(1, max_attempts + 1):
            constraint = ""
            if attempt > 1:
                constraint = (
                    f"\n第 {attempt} 次尝试。严格精简：只保留关键短句和未决事项，删除所有次要内容。"
                )
            try:
                reply = await self._llm.chat_completion(
                    model_key,
                    [ChatMessage(role="system", content=instruction + constraint), source],
                    session_id=f"{profile.bot_id}:{group_id}:summary",
                    max_tokens=target,
                )
            except Exception as exc:  # noqa: BLE001 - Worker adapter failures follow the retry policy.
                _log.warning(
                    "context_summary_failed",
                    bot_id=profile.bot_id,
                    group_id=group_id,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                )
                reason = "request_failed"
                continue

            text = reply.content.strip()
            tokens = estimate_tokens(text)
            if not tokens or tokens > target:
                reason = "empty_summary" if not tokens else "summary_over_target"
                continue

            tail = to_summarize[-1]
            summary = ContextSummary(
                bot_id=profile.bot_id,
                group_id=group_id,
                summary_text=text,
                covered_until_msg_id=tail.last_msg_id,
                covered_until_at=tail.latest_at,
                token_estimate=tokens,
                created_at=datetime.now(UTC),
            )
            return CompactionOutcome(status="succeeded", summary=summary, attempts=attempt)

        return CompactionOutcome(
            status="skipped" if legacy_budget else "reset_required",
            attempts=max_attempts,
            reason=reason,
        )

    def merge_restored_context(
        self, summaries: Sequence[ContextSummary], messages: Sequence[NormalizedMessage]
    ) -> tuple[list[ContextSummary], list[NormalizedMessage]]:
        return _merge_context(summaries, messages, self._max_persisted_summaries)
