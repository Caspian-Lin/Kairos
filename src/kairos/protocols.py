"""Interfaces shared across independently implemented modules."""

from typing import Any, Protocol

from kairos.models import AuditRecord, ChatMessage, LLMResult, NormalizedMessage


class OneBotSender(Protocol):
    async def send_group_msg(
        self, self_id: int, group_id: int, text: str, reply_to_msg_id: str | None = None
    ) -> str | None: ...


class MessageStore(Protocol):
    async def save_message(self, msg: NormalizedMessage) -> None: ...

    async def load_recent_messages(
        self, bot_id: str, group_id: int, limit: int = 40
    ) -> list[NormalizedMessage]: ...

    async def save_audit_log(self, record: AuditRecord) -> None: ...

    async def health_check(self) -> bool: ...


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
    """Reserved interface for the M2 model gate."""


class MemoryProvider(Protocol):
    """Reserved interface for M2 memory retrieval."""
