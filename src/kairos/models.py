"""Data exchanged by the gateway, harness, runtime, and persistence layers."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

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


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = True


class ToolSpec(BaseModel):
    definition: ToolDefinition
    catalog_hint: str
    allowed_profiles: tuple[str, ...] = ("social_companion", "utility_assistant")


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LLMResult(BaseModel):
    content: str
    model: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ActionType(StrEnum):
    REPLY = "reply"
    SILENT = "silent"
    REACT = "react"


class BotAction(BaseModel):
    action: ActionType
    content: str = ""
    target_msg_id: str | None = None
    emoji_id: str | None = None


class GateAction(StrEnum):
    PASS = "pass"
    SILENT = "silent"
    REACT = "react"


class GateDecision(BaseModel):
    action: GateAction
    candidate_target: str | None = None
    emoji_id: str | None = None
    reason: str = ""
    latency_ms: int = 0
    need_tools: list[str] = Field(default_factory=list)


class PipelineStep(BaseModel):
    stage: str
    status: Literal["ok", "skipped", "filtered", "error"]
    latency_ms: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    trace_id: str
    bot_id: str
    group_id: int
    msg_id: str
    sender_alias: str
    input_preview: str
    started_at: datetime
    total_latency_ms: int = 0
    outcome: str = ""
    steps: list[PipelineStep] = Field(default_factory=list)
    request_context: dict[str, Any] | None = None


class ThreadSummary(BaseModel):
    thread_id: str
    root_msg_id: str
    latest_msg_id: str
    participants: list[str]
    message_count: int
    heat: float
    last_active_at: datetime


class ContextSummary(BaseModel):
    bot_id: str
    group_id: int
    summary_text: str
    covered_until_msg_id: str
    covered_until_at: datetime
    token_estimate: int
    created_at: datetime


class CompactionOutcome(BaseModel):
    status: Literal["skipped", "succeeded", "reset_required"]
    summary: ContextSummary | None = None
    attempts: int = 0
    reason: str = ""


class UserImpression(BaseModel):
    bot_id: str
    group_id: int
    user_qq: int
    nickname: str
    impression_summary: str
    affinity_score: float = 0.0
    updated_at: datetime


class MemoryNote(BaseModel):
    bot_id: str
    scope_group_id: int
    entity_qqs: list[int]
    summary_text: str
    keywords: list[str]
    importance: int = 1
    last_recalled_at: datetime | None = None
    created_at: datetime


class WarmMemory(BaseModel):
    summaries: list[ContextSummary] = Field(default_factory=list)
    impressions: list[UserImpression] = Field(default_factory=list)
    notes: list[MemoryNote] = Field(default_factory=list)
    threads: list[ThreadSummary] = Field(default_factory=list)


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
    action_type: str | None = None
    target_msg_id: str | None = None
    emoji_id: str | None = None
