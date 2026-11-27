"""Run the group-chat pipeline for a single bot profile."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from time import monotonic
from zoneinfo import ZoneInfo

import structlog

from kairos.config import BotProfile
from kairos.gateway.onebot_ws import OneBotSendError
from kairos.harness.context_builder import ContextBuilder
from kairos.harness.normalizer import build_roster
from kairos.harness.postprocess import process_llm_output
from kairos.llm import LLMConfigError, LLMRequestError, LLMResponseError
from kairos.models import (
    GROUP_ERROR_PREFIX,
    GROUP_SILENCE_NOTICE,
    AuditRecord,
    NormalizedMessage,
    RawGroupEvent,
    StaleAction,
)
from kairos.protocols import LLMClient, MessageStore, OneBotSender
from kairos.runtime.group_session import GroupSession
from kairos.runtime.stale_check import evaluate_stale
from kairos.telemetry.audit_logger import AuditLogger

_log = structlog.get_logger(__name__)
_GREETING_TIMEZONE = ZoneInfo("Asia/Shanghai")

type RuntimeGateHook = Callable[[BotProfile, Sequence[NormalizedMessage]], Awaitable[bool]]


def _online_greeting(hour: int) -> str:
    if 5 <= hour < 11:
        return "早上好"
    if 11 <= hour < 18:
        return "中午好"
    return "晚上好"


def _group_error_detail(error: Exception) -> str:
    if isinstance(error, LLMRequestError):
        if error.status_code is not None:
            return f"模型接口返回 HTTP {error.status_code}。"
        return "模型请求失败（连接、超时或协议错误）。"
    if isinstance(error, LLMResponseError):
        return "模型返回了无法解析的回复。"
    if isinstance(error, LLMConfigError):
        return "模型配置有误。"
    if isinstance(error, OneBotSendError):
        return "OneBot 消息发送失败。"
    return f"群消息处理失败（{type(error).__name__}）。"


class BotRuntime:
    def __init__(
        self,
        profile: BotProfile,
        hosted_bot_ids: set[int],
        store: MessageStore,
        llm_client: LLMClient,
        sender: OneBotSender,
        audit_logger: AuditLogger,
        gate_hook: RuntimeGateHook | None = None,
    ) -> None:
        self.profile = profile
        self.sessions: dict[int, GroupSession] = {}
        self._hosted_bot_ids = set(hosted_bot_ids)
        self._store = store
        self._llm = llm_client
        self._sender = sender
        self._audit_logger = audit_logger
        self._gate_hook = gate_hook
        self._initializing: dict[int, asyncio.Lock] = {}
        self._write_locks: dict[int, asyncio.Lock] = {}
        self._writes: set[asyncio.Task[None]] = set()
        self._closing = False

    async def handle_online(self) -> None:
        if not self._closing and self.profile.allowed_groups:
            hour = datetime.now(_GREETING_TIMEZONE).hour
            await self._send_status(
                self.profile.allowed_groups[0], _online_greeting(hour), "online"
            )

    async def handle_group_poke(self, group_id: int) -> None:
        if not self._closing and self._allows_group(group_id):
            await self._send_status(group_id, "QAQ", "poke")

    async def _send_status(
        self, group_id: int, text: str, reason: str, *, reply_to_msg_id: str | None = None
    ) -> None:
        message_id = await self._sender.send_group_msg(
            self.profile.self_id, group_id, text, reply_to_msg_id=reply_to_msg_id
        )
        if message_id is not None:
            _log.info(
                "bot_status_message_sent",
                bot_id=self.profile.bot_id,
                group_id=group_id,
                reason=reason,
                message_id=message_id,
            )

    async def _report_group_error(self, group_id: int, message_id: str, error: Exception) -> None:
        try:
            await self._send_status(
                group_id,
                f"{GROUP_ERROR_PREFIX}{_group_error_detail(error)}",
                "group_error",
                reply_to_msg_id=message_id,
            )
        except Exception as send_error:  # noqa: BLE001 - A failed error notice only goes to logs.
            _log.warning(
                "group_error_notice_failed",
                bot_id=self.profile.bot_id,
                group_id=group_id,
                error_type=type(send_error).__name__,
            )

    def _allows_group(self, group_id: int) -> bool:
        groups = self.profile.allowed_groups
        return not groups or group_id in groups

    async def handle_group_event(self, event: RawGroupEvent) -> None:
        if (
            self._closing
            or event.self_id != self.profile.self_id
            or event.sender_qq == self.profile.self_id
            or not self._allows_group(event.group_id)
        ):
            return

        received_at = monotonic()
        try:
            session = await self._get_session(event.group_id)
        except Exception as error:
            await self._report_group_error(event.group_id, event.msg_id, error)
            raise
        if session is None:
            return
        message, _ = session.ingest_event(event, self._hosted_bot_ids)
        if message is not None:
            self._schedule_write(message)
            session.schedule_trigger(
                message,
                partial(self._run_pipeline, session, received_at),
                buffered=session.buffer[-1],
            )

    async def shutdown(self) -> None:
        self._closing = True
        for session in self.sessions.values():
            session.close()
        await asyncio.gather(*(session.wait_closed() for session in self.sessions.values()))
        await asyncio.gather(*self._writes)

    async def _get_session(self, group_id: int) -> GroupSession | None:
        if (session := self.sessions.get(group_id)) is not None:
            return session
        lock = self._initializing.setdefault(group_id, asyncio.Lock())
        async with lock:
            if self._closing:
                return None
            if (session := self.sessions.get(group_id)) is not None:
                return session

            history = await self._store.load_recent_messages(
                self.profile.bot_id, group_id, limit=self.profile.ring_buffer_size
            )
            if self._closing:
                return None
            # PostgreSQL stores each arrival, while peer status depends on current hosting.
            restored = []
            for message in history:
                is_peer = not message.is_bot and message.sender_qq in self._hosted_bot_ids
                if message.is_peer_bot != is_peer:
                    message = message.model_copy(update={"is_peer_bot": is_peer})
                restored.append(message)
            session = GroupSession(self.profile, group_id, self._gate_hook)
            session.warmup(restored)
            self._write_locks[group_id] = asyncio.Lock()
            self.sessions[group_id] = session
            self._initializing.pop(group_id)
            return session

    def _schedule_write(self, message: NormalizedMessage) -> None:
        task = asyncio.create_task(self._persist_message(message))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _persist_message(self, message: NormalizedMessage) -> None:
        # Preserve insertion order when arrivals share a OneBot timestamp.
        async with self._write_locks[message.group_id]:
            try:
                await self._store.save_message(message)
            except Exception as exc:  # noqa: BLE001 - Keep store failures off the chat path.
                _log.warning(
                    "group_message_write_failed",
                    bot_id=message.bot_id,
                    group_id=message.group_id,
                    error_type=type(exc).__name__,
                )

    async def _run_pipeline(
        self,
        session: GroupSession,
        received_at: float,
        trigger_message: NormalizedMessage,
    ) -> None:
        async with session.pipeline_lock:
            try:
                await self._process_trigger(session, received_at, trigger_message)
            except Exception as error:
                await self._report_group_error(session.group_id, trigger_message.last_msg_id, error)
                raise

    async def _process_trigger(
        self, session: GroupSession, received_at: float, trigger_message: NormalizedMessage
    ) -> None:
        allowed, gate_reason = session.evaluate_l1a_gate(trigger_message, monotonic())
        if not allowed:
            _log.debug(
                "group_trigger_filtered",
                bot_id=self.profile.bot_id,
                group_id=session.group_id,
                reason=gate_reason,
            )
            return

        history = list(session.buffer)
        if trigger_message.addresses_bot and all(
            message.msg_id != trigger_message.msg_id for message in history
        ):
            history = [trigger_message, *history[1:]]
        trigger_seq = session.seq
        gate_output: dict[str, bool | str] = {"allowed": allowed, "reason": gate_reason}
        if callable(hook := session.gate_hook):
            hook_allowed = bool(await hook(self.profile, history))
            gate_output["hook_allowed"] = hook_allowed
            if not hook_allowed:
                return

        chat_messages = ContextBuilder.build_messages(self.profile, history, build_roster(history))
        result = await self._llm.chat_completion(
            self.profile.chat_model_key,
            chat_messages,
            session_id=session.llm_session_id,
        )
        processed = process_llm_output(
            result.content, [message.content for message in history if message.is_bot]
        )

        reply_to = None
        if processed.vetoed:
            action = "VETOED"
        elif processed.repeated_dropped:
            action = "DROPPED_REPEAT"
        elif not processed.content:
            raise LLMResponseError("Model returned empty chat output")
        else:
            stale = self.profile.stale_check
            decision = evaluate_stale(
                trigger_seq,
                session.seq,
                trigger_message.last_msg_id,
                stale.direct_max_delta,
                stale.reply_max_delta,
            )
            action = decision.action
            reply_to = decision.reply_to_msg_id

        sent: str | None = None
        try:
            if action in {StaleAction.SEND_DIRECT, StaleAction.SEND_WITH_REPLY}:
                message_id = await self._sender.send_group_msg(
                    self.profile.self_id,
                    session.group_id,
                    processed.content,
                    reply_to_msg_id=reply_to,
                )
                if message_id is None:
                    raise OneBotSendError("Reply has no message receipt")
                sent = processed.content
                self._schedule_write(
                    session.record_bot_message(message_id, sent, datetime.now(UTC), reply_to)
                )
        finally:
            self._audit_logger.log_decision(
                AuditRecord(
                    bot_id=self.profile.bot_id,
                    group_id=session.group_id,
                    context_snapshot=[message.model_dump() for message in chat_messages],
                    gate_output=gate_output,
                    main_vetoed=processed.vetoed,
                    final_reply=sent,
                    stale_action=action,
                    latency_ms=int((monotonic() - received_at) * 1000),
                    created_at=datetime.now(UTC),
                )
            )
        if processed.vetoed and session.group_id in self.profile.debug_groups:
            await self._send_status(
                session.group_id,
                GROUP_SILENCE_NOTICE,
                "silence_debug",
                reply_to_msg_id=trigger_message.last_msg_id,
            )
