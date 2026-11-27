"""Data exchanged by the gateway, harness, runtime, and persistence layers."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

GROUP_SILENCE_NOTICE = "这条触发成功但是模型选择沉默"
GROUP_ERROR_PREFIX = "【Kairos 错误】"


class MessageSegment(BaseModel):
    type: str
    data: dict[str, Any]


class RawGroupEvent(BaseModel):
    self_id: int
    group_id: int
    msg_id: str
    sender_qq: int
    sender_nickname: str
    sender_card: str
    segments: list[MessageSegment]
    raw_message: str
    timestamp: datetime


class NormalizedMessage(BaseModel):
    """A group message or burst carrying its first and latest OneBot message IDs.

    is_bot identifies this runtime's own output. Other hosted bots use is_peer_bot.
    """

    bot_id: str
    group_id: int
    msg_id: str
    last_msg_id: str
    sender_qq: int
    sender_alias: str
    content: str
    reply_to_msg_id: str | None
    is_bot: bool
    is_at_bot: bool
    is_reply_to_bot: bool
    mentions_bot_name: bool = False
    is_peer_bot: bool
    created_at: datetime
    latest_at: datetime

    @property
    def addresses_bot(self) -> bool:
        return self.is_at_bot or self.is_reply_to_bot or self.mentions_bot_name

    @property
    def is_kairos_notice(self) -> bool:
        return (self.is_bot or self.is_peer_bot) and (
            self.content == GROUP_SILENCE_NOTICE
            or self.content.startswith(("【Kairos 调试】", GROUP_ERROR_PREFIX))
        )


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel):
    content: str
    model: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None


class StaleAction(StrEnum):
    SEND_DIRECT = "SEND_DIRECT"
    SEND_WITH_REPLY = "SEND_WITH_REPLY"
    DROP_STALE = "DROP_STALE"


class StaleDecision(BaseModel):
    action: StaleAction
    delta: int
    reply_to_msg_id: str | None


class AuditRecord(BaseModel):
    bot_id: str
    group_id: int
    context_snapshot: list[dict[str, Any]]
    gate_output: dict[str, Any]
    main_vetoed: bool
    final_reply: str | None
    stale_action: str
    latency_ms: int
    human_label: str | None = None
    created_at: datetime
