"""Track bounded, reply-aware conversation threads for one group session."""

from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime

from kairos.models import NormalizedMessage, ThreadSummary


class ThreadTracker:
    def __init__(self, max_threads: int = 20, max_tracked_messages: int = 256) -> None:
        if max_threads < 1 or max_tracked_messages < 1:
            raise ValueError("Thread and message capacities must be positive")
        self._max_threads = max_threads
        self._max_tracked_messages = max_tracked_messages
        self._threads: dict[str, ThreadSummary] = {}
        self._message_threads: OrderedDict[str, str] = OrderedDict()

    def observe(self, message: NormalizedMessage) -> None:
        """Assign the message to its reply parent or the latest active thread."""
        if self._thread_for(message.last_msg_id) is not None:
            return

        at = message.latest_at
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        thread = self._thread_for(message.reply_to_msg_id)
        if thread is None and self._threads:
            latest = max(self._threads.values(), key=lambda item: item.last_active_at)
            if (at - latest.last_active_at).total_seconds() <= 300:
                thread = latest
        if thread is None:
            if len(self._threads) == self._max_threads:
                oldest = min(self._threads.values(), key=lambda item: item.last_active_at)
                del self._threads[oldest.thread_id]
            thread = ThreadSummary(
                thread_id=message.msg_id,
                root_msg_id=message.msg_id,
                latest_msg_id=message.last_msg_id,
                participants=[],
                message_count=0,
                heat=0.0,
                last_active_at=at,
            )
            self._threads[thread.thread_id] = thread

        if at >= thread.last_active_at:
            thread.latest_msg_id = message.last_msg_id
            thread.last_active_at = at
        thread.message_count += 1
        alias = message.sender_alias[:24]
        if alias not in thread.participants and len(thread.participants) < 8:
            thread.participants.append(alias)

        for msg_id in (message.msg_id, message.last_msg_id):
            self._message_threads[msg_id] = thread.thread_id
            self._message_threads.move_to_end(msg_id)
        while len(self._message_threads) > self._max_tracked_messages:
            self._message_threads.popitem(last=False)

    def _thread_for(self, message_id: str | None) -> ThreadSummary | None:
        if message_id is None:
            return None
        thread_id = self._message_threads.get(message_id)
        return self._threads.get(thread_id) if thread_id is not None else None

    def active_threads(self, now: datetime, limit: int = 5) -> list[ThreadSummary]:
        if limit <= 0:
            return []
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        snapshots = []
        for thread in self._threads.values():
            age = max(0, (now - thread.last_active_at).total_seconds())
            heat = thread.message_count * 0.5 ** (age / 600)
            snapshots.append(
                thread.model_copy(update={"heat": heat, "participants": thread.participants.copy()})
            )
        snapshots.sort(
            key=lambda item: (item.heat, item.last_active_at, item.thread_id), reverse=True
        )
        return snapshots[:limit]


def format_thread_board(threads: Sequence[ThreadSummary]) -> str:
    return "\n".join(
        f"m{thread.root_msg_id} → m{thread.latest_msg_id}（{thread.message_count} 条，"
        f"{', '.join(thread.participants)}，热度 {thread.heat:.1f}）"
        for thread in threads
    )
