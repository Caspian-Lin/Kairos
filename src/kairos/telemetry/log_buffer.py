"""Bounded, redacted event snapshots for the local admin interface."""

import asyncio
import json
import math
import re
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from sys import getsizeof
from threading import Lock
from typing import Any, TypedDict

_SENSITIVE_KEYS = ("key", "token", "dsn", "password", "secret", "authorization")
_REDACTED = "[REDACTED]"
_TRUNCATED = "[TRUNCATED]"
_VALUE_PATTERNS = (
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/'\"<>]+@[^\s'\"<>]+"),
    re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|redis)(?:\+[a-z0-9_]+)?://[^\s'\"<>]+"),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[^\s,;\"'<>]+"),
    re.compile(
        r"""(?i)\b(?:api[-_]?key|access[-_]?token|authorization|password|secret|dsn)["']?\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;&<>]+)"""
    ),
)
_MAX_SUBSCRIBERS = 32
_SUBSCRIBER_ITEMS = 128
_SUBSCRIBER_BYTES = 512 * 1024


class LogEntry(TypedDict):
    id: int
    timestamp: str
    level: str
    event: str
    fields: dict[str, Any]


def _secrets_tuple(secrets: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({item for item in secrets if item}, key=len, reverse=True))


class _Redactor:
    def __init__(self, secrets: Iterable[str], max_chars: int) -> None:
        self._secrets = _secrets_tuple(secrets)
        self._pattern = (
            re.compile("|".join(map(re.escape, self._secrets))) if self._secrets else None
        )
        self._lookahead = max(512, len(self._secrets[0]) if self._secrets else 0)
        self.remaining = max_chars

    def text(self, raw: str) -> str:
        available = max(0, self.remaining)
        candidate = raw[: available + self._lookahead]
        if self._pattern is not None:
            candidate = self._pattern.sub(_REDACTED, candidate)
        for pattern in _VALUE_PATTERNS:
            candidate = pattern.sub(_REDACTED, candidate)
        candidate = candidate.encode("utf-8", errors="replace").decode("utf-8")
        width = _json_size(candidate) - 2
        if width > available or len(raw) > available + self._lookahead:
            low, high = 0, min(len(candidate), available)
            while low < high:
                middle = (low + high + 1) // 2
                if _json_size(candidate[:middle]) - 2 <= available - len(_TRUNCATED):
                    low = middle
                else:
                    high = middle - 1
            self.remaining = 0
            return candidate[:low] + _TRUNCATED
        self.remaining -= width
        return candidate

    def visit(self, value: Any, depth: int = 0) -> Any:
        if self.remaining < 64 or depth >= 12:
            return _TRUNCATED
        # Account for resident containers as well as serialized text.
        self.remaining -= 64
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if self.remaining < 64:
                    result["_truncated"] = True
                    break
                name = self.text(key) if isinstance(key, str) else "<non-string key>"
                if isinstance(key, str) and any(part in key.casefold() for part in _SENSITIVE_KEYS):
                    result[name] = _REDACTED
                else:
                    result[name] = self.visit(item, depth + 1)
            return result
        if isinstance(value, (list, tuple)):
            items = []
            for item in value:
                if self.remaining < 64:
                    items.append(_TRUNCATED)
                    break
                items.append(self.visit(item, depth + 1))
            return items
        if isinstance(value, str):
            return self.text(value)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value if value.bit_length() <= 128 else _TRUNCATED
        if isinstance(value, float):
            return value if math.isfinite(value) else _TRUNCATED
        if isinstance(value, datetime):
            return self.text(value.isoformat())
        return f"<{type(value).__name__}>"


def redact_payload(value: Any, *, secrets: Iterable[str] = (), max_chars: int = 256 * 1024) -> Any:
    """Return an independent JSON-compatible snapshot with credential values masked."""
    return _Redactor(secrets, max_chars).visit(value)


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _storage_size(value: Any) -> int:
    """Charge the larger of the JSON body and its resident Python objects."""
    pending = [value]
    resident = 0
    while pending:
        item = pending.pop()
        resident += getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return max(resident, _json_size(value))


class _EventQueue[Event](asyncio.Queue[Event]):
    def __init__(self) -> None:
        super().__init__(maxsize=_SUBSCRIBER_ITEMS)
        self._sizes: deque[int] = deque()
        self._bytes = 0

    def _get(self) -> Event:
        self._bytes -= self._sizes.popleft()
        return super()._get()

    def get_nowait(self) -> Event:
        return deepcopy(super().get_nowait())

    def offer(self, event: Event, size: int) -> None:
        if size > _SUBSCRIBER_BYTES:
            return
        while self.full() or (self._sizes and self._bytes + size > _SUBSCRIBER_BYTES):
            super().get_nowait()
            self.task_done()
        self.put_nowait(event)
        self._sizes.append(size)
        self._bytes += size


@dataclass(eq=False)
class _Subscriber[Event]:
    loop: asyncio.AbstractEventLoop
    queue: _EventQueue[Event]
    delivered: int
    scheduled: bool = False


class _EventRing[Event]:
    """Keep bounded snapshots and coalesce cross-thread deliveries per subscriber."""

    def __init__(self, maxlen: int, max_bytes: int) -> None:
        if maxlen < 1:
            raise ValueError("Event capacity must be positive")
        self._maxlen = maxlen
        self._max_bytes = max_bytes
        self._events: deque[tuple[int, Event, int]] = deque()
        self._bytes = 0
        self._sequence = 0
        self._lock = Lock()
        self._subscribers: set[_Subscriber[Event]] = set()

    def append(self, build_event: Callable[[int], Event], size: int) -> None:
        if size > self._max_bytes:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            while self._events and (
                len(self._events) == self._maxlen or self._bytes + size > self._max_bytes
            ):
                _, _, evicted = self._events.popleft()
                self._bytes -= evicted
            self._sequence += 1
            event = build_event(self._sequence)
            self._events.append((self._sequence, event, size))
            self._bytes += size
            for subscriber in self._subscribers:
                if subscriber.loop is loop:
                    self._deliver(subscriber)
                elif not subscriber.scheduled and not subscriber.loop.is_closed():
                    subscriber.scheduled = True
                    try:
                        subscriber.loop.call_soon_threadsafe(self._flush, subscriber)
                    except RuntimeError:
                        subscriber.scheduled = False

    def _deliver(self, subscriber: _Subscriber[Event]) -> None:
        if subscriber.delivered == self._sequence - 1:
            _, event, size = self._events[-1]
            subscriber.queue.offer(event, size)
        elif subscriber.delivered < self._sequence:
            unread: list[tuple[int, Event, int]] = []
            for entry in reversed(self._events):
                if entry[0] <= subscriber.delivered or len(unread) == _SUBSCRIBER_ITEMS:
                    break
                unread.append(entry)
            for _, event, size in reversed(unread):
                subscriber.queue.offer(event, size)
        subscriber.delivered = self._sequence

    def _flush(self, subscriber: _Subscriber[Event]) -> None:
        with self._lock:
            subscriber.scheduled = False
            if subscriber in self._subscribers:
                self._deliver(subscriber)

    def snapshots(self) -> list[Event]:
        with self._lock:
            return [item for _, item, _ in reversed(self._events)]

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Event]]:
        with self._lock:
            if len(self._subscribers) == _MAX_SUBSCRIBERS:
                raise RuntimeError("Too many event subscribers")
            subscriber = _Subscriber(
                asyncio.get_running_loop(), _EventQueue[Event](), self._sequence
            )
            self._subscribers.add(subscriber)
        try:
            yield subscriber.queue
        finally:
            with self._lock:
                self._subscribers.discard(subscriber)


class LogRingBuffer:
    def __init__(self, maxlen: int = 500, *, secrets: Iterable[str] = ()) -> None:
        self._ring = _EventRing[LogEntry](maxlen, max_bytes=4 * 1024 * 1024)
        self.set_secrets(secrets)

    def set_secrets(self, secrets: Iterable[str]) -> None:
        self._secrets = _secrets_tuple(secrets)

    def processor(
        self, _logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        event = event_dict.get("event", "")
        timestamp = event_dict.get("timestamp")
        fields = {
            key: value
            for key, value in event_dict.items()
            if key not in {"timestamp", "level", "event"}
        }
        metadata = {
            "timestamp": timestamp if isinstance(timestamp, str) else datetime.now(UTC).isoformat(),
            "level": method_name.casefold(),
            "event": event if isinstance(event, str) else "",
        }
        clean = {
            key: redact_payload(value, secrets=self._secrets, max_chars=1024)
            for key, value in metadata.items()
        }
        clean_fields = redact_payload(fields, secrets=self._secrets, max_chars=16 * 1024)

        def make_entry(event_id: int) -> LogEntry:
            return {
                "id": event_id,
                "timestamp": clean["timestamp"],
                "level": clean["level"],
                "event": clean["event"],
                "fields": clean_fields,
            }

        self._ring.append(make_entry, _storage_size(clean) + _storage_size(clean_fields))
        return event_dict

    def recent(
        self,
        limit: int = 200,
        level: str | None = None,
        bot_id: str | None = None,
        query: str | None = None,
    ) -> list[LogEntry]:
        if limit <= 0:
            return []
        results = []
        for entry in self._ring.snapshots():
            if level is not None and level.casefold() != entry["level"]:
                continue
            if bot_id is not None and str(entry["fields"].get("bot_id", "")) != bot_id:
                continue
            if query and query.casefold() not in json.dumps(entry, ensure_ascii=False).casefold():
                continue
            results.append(deepcopy(entry))
            if len(results) == limit:
                break
        return results

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[LogEntry]]:
        async with self._ring.subscribe() as queue:
            yield queue
