"""Interfaces shared across independently implemented modules."""

from collections.abc import Sequence
from typing import Any, Protocol

from kairos.config import BotProfile
from kairos.models import (
    AuditRecord,
    ChatMessage,
    ContextSummary,
    GateDecision,
    LLMResult,
    MemoryNote,
    NormalizedMessage,
    ThreadSummary,
    UserImpression,
    WarmMemory,
)


class OneBotSender(Protocol):
    async def send_group_msg(
        self, self_id: int, group_id: int, text: str, reply_to_msg_id: str | None = None
    ) -> str | None: ...

    async def set_msg_emoji_like(
        self, self_id: int, message_id: str, emoji_id: str, *, set_like: bool = True
    ) -> None: ...


class MessageStore(Protocol):
    async def save_message(self, msg: NormalizedMessage) -> None: ...

    async def load_recent_messages(
        self, bot_id: str, group_id: int, limit: int = 40
    ) -> list[NormalizedMessage]: ...

    async def save_audit_log(self, record: AuditRecord) -> None: ...

    async def health_check(self) -> bool: ...


class SummaryStore(Protocol):
    async def save_summary(self, summary: ContextSummary) -> None: ...

    async def load_recent_summaries(
        self, bot_id: str, group_id: int, limit: int = 3
    ) -> list[ContextSummary]: ...


class MemoryStore(Protocol):
    async def upsert_user_impression(self, impression: UserImpression) -> None: ...

    async def load_user_impressions(
        self, bot_id: str, group_id: int, user_qqs: Sequence[int]
    ) -> list[UserImpression]: ...

    async def save_memory_note(self, note: MemoryNote) -> None: ...

    async def recall_memory_notes(
        self,
        bot_id: str,
        group_id: int,
        keywords: Sequence[str],
        query_text: str = "",
        limit: int = 5,
    ) -> list[MemoryNote]: ...


class LLMClient(Protocol):
    async def chat_completion(
        self,
        model_key: str,
        messages: list[ChatMessage],
        *,
        session_id: str | None = None,
        **overrides: Any,
    ) -> LLMResult: ...


class GateHook(Protocol):
    async def evaluate(
        self,
        profile: BotProfile,
        messages: Sequence[NormalizedMessage],
        threads: Sequence[ThreadSummary] = (),
        *,
        latest_msg: NormalizedMessage | None = None,
    ) -> GateDecision: ...


class MemoryProvider(Protocol):
    async def recall(
        self,
        bot_id: str,
        group_id: int,
        active_user_qqs: Sequence[int],
        recent_messages: Sequence[NormalizedMessage],
    ) -> WarmMemory: ...


class ContextCompactor(Protocol):
    def estimate_tokens(self, text: str) -> int: ...

    def should_compact(
        self,
        profile: BotProfile,
        messages: Sequence[NormalizedMessage],
        summaries: Sequence[ContextSummary] = (),
    ) -> bool: ...

    async def compact(
        self,
        profile: BotProfile,
        group_id: int,
        messages: Sequence[NormalizedMessage],
        existing_summaries: Sequence[ContextSummary] = (),
    ) -> ContextSummary | None: ...

    def merge_restored_context(
        self, summaries: Sequence[ContextSummary], messages: Sequence[NormalizedMessage]
    ) -> tuple[list[ContextSummary], list[NormalizedMessage]]: ...
