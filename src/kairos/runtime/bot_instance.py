"""Run the group-chat pipeline for a single bot profile."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from time import monotonic

import structlog

from kairos.config import BotProfile
from kairos.harness.context_builder import ContextBuilder
from kairos.harness.normalizer import build_roster
from kairos.harness.postprocess import process_llm_output
from kairos.models import AuditRecord, NormalizedMessage, RawGroupEvent, StaleAction
from kairos.protocols import LLMClient, MessageStore, OneBotSender
from kairos.runtime.group_session import GroupSession
from kairos.runtime.stale_check import compute_typing_delay, evaluate_stale
from kairos.telemetry.audit_logger import AuditLogger

_log = structlog.get_logger(__name__)

type RuntimeGateHook = Callable[[BotProfile, Sequence[NormalizedMessage]], Awaitable[bool]]


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

    async def handle_group_event(self, event: RawGroupEvent) -> None:
        if (
            self._closing
            or event.self_id != self.profile.self_id
            or event.sender_qq == self.profile.self_id
            or (self.profile.allowed_groups and event.group_id not in self.profile.allowed_groups)
        ):
            return

        received_at = monotonic()
        session = await self._get_session(event.group_id)
        if session is None:
            return
        message, _ = session.ingest_event(event, self._hosted_bot_ids)
        if message is not None:
            self._schedule_write(message)
            session.schedule_trigger(
                session.buffer[-1], partial(self._run_pipeline, session, received_at)
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

    async def _run_pipeline(self, session: GroupSession, received_at: float) -> None:
        async with session.pipeline_lock:
            history = list(session.buffer)
            latest = next((message for message in reversed(history) if not message.is_bot), None)
            if latest is None:
                return
            allowed, reason = session.evaluate_l1a_gate(latest, monotonic())
            if not allowed:
                _log.debug(
                    "group_trigger_filtered",
                    bot_id=self.profile.bot_id,
                    group_id=session.group_id,
                    reason=reason,
                )
                return

            trigger_seq = session.seq
            gate_output: dict[str, bool | str] = {"allowed": allowed, "reason": reason}
            if callable(hook := session.gate_hook):
                hook_allowed = bool(await hook(self.profile, history))
                gate_output["hook_allowed"] = hook_allowed
                if not hook_allowed:
                    return

            chat_messages = ContextBuilder.build_messages(
                self.profile, history, build_roster(history)
            )
            result = await self._llm.chat_completion(self.profile.chat_model_key, chat_messages)
            processed = process_llm_output(
                result.content, [message.content for message in history if message.is_bot]
            )

            def audit(action: str, reply: str | None = None, *, vetoed: bool = False) -> None:
                self._audit_logger.log_decision(
                    AuditRecord(
                        bot_id=self.profile.bot_id,
                        group_id=session.group_id,
                        context_snapshot=[message.model_dump() for message in chat_messages],
                        gate_output=gate_output,
                        main_vetoed=vetoed,
                        final_reply=reply,
                        stale_action=action,
                        latency_ms=int((monotonic() - received_at) * 1000),
                        created_at=datetime.now(UTC),
                    )
                )

            if processed.vetoed:
                audit("VETOED", vetoed=True)
                return
            if processed.repeated_dropped:
                audit("DROPPED_REPEAT")
                return

            stale = self.profile.stale_check
            decision = evaluate_stale(
                trigger_seq,
                session.seq,
                latest.last_msg_id,
                stale.direct_max_delta,
                stale.reply_max_delta,
            )
            if decision.action is StaleAction.DROP_STALE:
                audit(decision.action)
                return

            sent: list[str] = []
            try:
                for index, text in enumerate(processed.segments):
                    if index:
                        await asyncio.sleep(compute_typing_delay(text))
                    reply_to = None if index else decision.reply_to_msg_id
                    message_id = await self._sender.send_group_msg(
                        self.profile.self_id, session.group_id, text, reply_to_msg_id=reply_to
                    )
                    if message_id is None:
                        break
                    self._schedule_write(
                        session.record_bot_message(message_id, text, datetime.now(UTC), reply_to)
                    )
                    sent.append(text)
            finally:
                audit(decision.action, " || ".join(sent) if sent else None)
