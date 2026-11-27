import asyncio
from contextlib import AsyncExitStack

import pytest

from kairos.telemetry.log_buffer import LogRingBuffer


def test_redaction_and_filters_use_sanitized_snapshots() -> None:
    ring = LogRingBuffer(maxlen=2, secrets=("literal-value",))
    ring.processor(None, "info", {"event": "evicted"})
    original = {
        "event": "connection literal-value failed",
        "bot_id": "1",
        "nested": [{"X-Authorization": "Bearer unlisted", "public": "kept"}],
        "postgres_DSN": "sensitive-dsn",
        "description": "postgresql://person:pass@localhost/db",
    }
    ring.processor(None, "warning", original)
    ring.processor(None, "info", {"event": "ready", "bot_id": "2"})

    assert [item["id"] for item in ring.recent()] == [3, 2]
    (entry,) = ring.recent(level="WARNING", bot_id="1", query="KEPT")
    assert entry["event"] == "connection [REDACTED] failed"
    assert entry["fields"]["nested"] == [{"X-Authorization": "[REDACTED]", "public": "kept"}]
    assert entry["fields"]["postgres_DSN"] == entry["fields"]["description"] == "[REDACTED]"
    assert "unlisted" in original["nested"][0]["X-Authorization"]
    entry["fields"]["nested"].clear()
    assert ring.recent(1, bot_id="1")[0]["fields"]["nested"]
    assert ring.recent(limit=0) == []


async def test_slow_subscriber_keeps_only_recent_events() -> None:
    ring = LogRingBuffer()
    async with ring.subscribe() as queue:
        assert queue.maxsize == 128
        for item in range(150):
            ring.processor(None, "info", {"event": str(item), "api_key": "private"})
        assert queue.qsize() == 128
        first = queue.get_nowait()
        assert first["id"] == 23 and first["fields"]["api_key"] == "[REDACTED]"
    ring.processor(None, "info", {"event": "after"})
    assert queue.qsize() == 127


async def test_threaded_publication_is_delivered_on_subscriber_loop() -> None:
    ring = LogRingBuffer()
    async with ring.subscribe() as queue:
        await asyncio.to_thread(ring.processor, None, "info", {"event": "thread published"})
        entry = await asyncio.wait_for(queue.get(), timeout=1)
        assert entry["event"] == "thread published"


async def test_record_bytes_and_subscriber_count_are_bounded() -> None:
    ring = LogRingBuffer(maxlen=500)
    async with AsyncExitStack() as resources:
        queue = await resources.enter_async_context(ring.subscribe())
        for item in range(300):
            ring.processor(None, "info", {"event": str(item), "data": "x" * 32_768})
        assert queue.qsize() < 128
        assert len(ring.recent(limit=500)) < 300
        assert ring.recent(1)[0]["fields"]["data"].endswith("[TRUNCATED]")
        for _ in range(31):
            await resources.enter_async_context(ring.subscribe())
        with pytest.raises(RuntimeError, match="Too many"):
            await resources.enter_async_context(ring.subscribe())
