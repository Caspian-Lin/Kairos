"""Maintain one bot's conversation state in one group."""

from asyncio import Lock, Task, TimerHandle, create_task, gather, get_running_loop
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from uuid import uuid4

import structlog

from kairos.config import BotProfile
from kairos.harness.normalizer import MessageNormalizer, UidAliasTable, coalesce_message
from kairos.models import NormalizedMessage, RawGroupEvent
from kairos.protocols import GateHook

_log = structlog.get_logger(__name__)

type TriggerCallback = Callable[[NormalizedMessage], Awaitable[None]]


@dataclass(slots=True)
class _PendingTrigger:
    first_at: float
    deadline: float
    message: NormalizedMessage
    callback: TriggerCallback


class GroupSession:
    """Buffer group messages and debounce the serialized pipeline callbacks."""

    def __init__(
        self, profile: BotProfile, group_id: int, gate_hook: GateHook | None = None
    ) -> None:
        self.profile = profile
        self.group_id = group_id
        self.llm_session_id = f"ses_{uuid4().hex}"
        self.gate_hook = gate_hook
        self.buffer: deque[NormalizedMessage] = deque(maxlen=profile.ring_buffer_size)
        self.seq = 0
        self.alias_table = UidAliasTable()
        self.bot_msg_ids: deque[str] = deque(maxlen=100)
        self.pipeline_lock = Lock()
        self.last_bot_reply_at: float | None = None
        self.consecutive_peer_bot_turns = 0
        self._normalizer = MessageNormalizer(
            profile.bot_id, profile.self_id, self.alias_table, bot_name=profile.nickname
        )
        self._pending: _PendingTrigger | None = None
        self._timer: TimerHandle | None = None
        self._trigger_task: Task[None] | None = None
        self._closed = False

    def warmup(self, history: Sequence[NormalizedMessage]) -> None:
        """Rebuild a fresh session from persisted individual arrivals in time order."""
        for message in history:
            if (
                message.bot_id != self.profile.bot_id
                or message.group_id != self.group_id
                or message.is_kairos_notice
            ):
                continue
            if not message.is_bot:
                self.alias_table.restore_alias(message.sender_qq, message.sender_alias)
            self._append_message(message)

    def ingest_event(
        self, event: RawGroupEvent, hosted_bot_ids: set[int]
    ) -> tuple[NormalizedMessage | None, bool]:
        """Ingest synchronously and return the individual row for persistence."""
        if event.self_id != self.profile.self_id or event.group_id != self.group_id:
            return None, False

        message = self._normalizer.normalize(event, self.bot_msg_ids, hosted_bot_ids)
        if message is None or message.is_kairos_notice:
            return None, False
        if not message.is_bot and not message.is_peer_bot:
            self.consecutive_peer_bot_turns = 0
        return message, self._append_message(message)

    def record_bot_message(
        self,
        msg_id: str,
        content: str,
        created_at: datetime,
        reply_to_msg_id: str | None = None,
    ) -> NormalizedMessage:
        message = NormalizedMessage(
            bot_id=self.profile.bot_id,
            group_id=self.group_id,
            msg_id=msg_id,
            last_msg_id=msg_id,
            sender_qq=self.profile.self_id,
            sender_alias=self.profile.nickname,
            content=content,
            reply_to_msg_id=reply_to_msg_id,
            is_bot=True,
            is_at_bot=False,
            is_reply_to_bot=False,
            is_peer_bot=False,
            created_at=created_at,
            latest_at=created_at,
        )
        self._append_message(message)
        self.last_bot_reply_at = monotonic()
        return message

    def evaluate_l1a_gate(self, latest_msg: NormalizedMessage, now_mono: float) -> tuple[bool, str]:
        """Apply group-level rules and count a peer turn on admission."""
        rate = self.profile.rate_limit
        addressed = latest_msg.addresses_bot
        if latest_msg.is_peer_bot and self.consecutive_peer_bot_turns >= rate.peer_bot_max_turns:
            return False, "peer_bot_limit"
        if (
            not addressed
            and self.last_bot_reply_at is not None
            and now_mono - self.last_bot_reply_at < rate.cooldown_seconds
        ):
            return False, "cooldown"
        if not addressed and self.profile.trigger_mode != "all_with_veto":
            return False, self.profile.trigger_mode or "not_addressed"
        if latest_msg.is_peer_bot:
            self.consecutive_peer_bot_turns += 1
        return True, "pass"

    def schedule_trigger(
        self,
        msg: NormalizedMessage,
        callback: TriggerCallback,
        *,
        buffered: NormalizedMessage | None = None,
    ) -> None:
        """Coalesce waiting work and preserve the earliest addressed-message deadline."""
        if self._closed:
            return

        now = monotonic()
        debounce = self.profile.debounce
        pending = self._pending
        first_at = pending.first_at if pending is not None else now
        deadline = min(now + debounce.normal_seconds, first_at + debounce.max_wait_seconds)
        addressed = msg.addresses_bot
        message = (buffered or msg) if addressed else msg
        if addressed:
            deadline = min(deadline, now + debounce.mention_seconds)
        if pending is not None:
            if addressed or pending.message.addresses_bot or pending.deadline <= now:
                deadline = min(deadline, pending.deadline)
            if pending.message.addresses_bot and not addressed:
                message = pending.message
                if buffered is not None and buffered.msg_id == message.msg_id:
                    message = buffered

        self._pending = _PendingTrigger(first_at, deadline, message, callback)
        self._arm_timer()

    def close(self) -> None:
        """Cancel waiting and running triggers so ingress can be shut down safely."""
        if self._closed:
            return
        self._closed = True
        self._pending = None
        self._arm_timer()
        if self._trigger_task is not None:
            self._trigger_task.cancel()

    async def wait_closed(self) -> None:
        """Wait for cancellation cleanup after close()."""
        if self._trigger_task is not None:
            await gather(self._trigger_task, return_exceptions=True)

    def _append_message(self, incoming: NormalizedMessage) -> bool:
        message, merged = coalesce_message(self.buffer, incoming)
        if merged:
            self.buffer[-1] = message
        else:
            self.buffer.append(message)
        if incoming.is_bot:
            self.bot_msg_ids.append(incoming.msg_id)
        self.seq += 1
        return merged

    def _arm_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        # An active pipeline needs only one waiting trigger, however long inference takes.
        if self._pending is not None and self._trigger_task is None:
            delay = max(0.0, self._pending.deadline - monotonic())
            self._timer = get_running_loop().call_later(delay, self._start_trigger)

    def _start_trigger(self) -> None:
        self._timer = None
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        task = create_task(self._run_trigger(pending))
        self._trigger_task = task
        task.add_done_callback(self._trigger_done)

    async def _run_trigger(self, pending: _PendingTrigger) -> None:
        await pending.callback(pending.message)

    def _trigger_done(self, task: Task[None]) -> None:
        self._trigger_task = None
        if not task.cancelled() and (error := task.exception()) is not None:
            _log.error(
                "group_trigger_failed",
                bot_id=self.profile.bot_id,
                group_id=self.group_id,
                error_type=type(error).__name__,
            )
        self._arm_timer()
