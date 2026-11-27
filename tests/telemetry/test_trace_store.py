from datetime import UTC, datetime

from kairos.models import PipelineStep, PipelineTrace
from kairos.telemetry.trace_store import PipelineTraceStore


def _trace(
    index: int,
    *,
    bot: str = "1",
    group: int = 42,
    outcome: str = "REPLY",
) -> PipelineTrace:
    return PipelineTrace(
        trace_id=str(index),
        bot_id=bot,
        group_id=group,
        msg_id=str(index),
        sender_alias="user",
        input_preview="literal-value",
        started_at=datetime.now(UTC),
        outcome=outcome,
        steps=[PipelineStep(stage="L2", status="ok", detail={"password": "nested-secret"})],
        request_context={"system": "literal-value", "total_tokens": 50, "budget_tokens": None},
    )


def test_ring_filters_and_redacts_without_losing_public_counters() -> None:
    store = PipelineTraceStore(maxlen=3, secrets=("literal-value",))
    store.record(_trace(1))
    store.record(_trace(2, group=43))
    store.record(_trace(3, bot="2", outcome="SILENT"))
    store.record(_trace(4, outcome="SILENT"))
    assert store.get("1") is None
    assert [item.trace_id for item in store.list_recent()] == ["4", "3", "2"]
    (trace,) = store.list_recent(bot_id="1", group_id=42, outcome="SILENT")
    assert trace.trace_id == "4"
    assert trace.input_preview == "[REDACTED]"
    assert trace.steps[0].detail["password"] == "[REDACTED]"
    assert trace.request_context == {
        "system": "[REDACTED]",
        "total_tokens": 50,
        "budget_tokens": None,
    }
    trace.steps[0].detail["password"] = "user mutation"
    assert store.get("4").steps[0].detail["password"] == "[REDACTED]"


async def test_stream_drops_old_items_and_copies_input() -> None:
    store = PipelineTraceStore()
    async with store.subscribe() as queue:
        for index in range(130):
            trace = _trace(index)
            store.record(trace)
            trace.steps[0].detail["password"] = "modified"
        assert queue.qsize() == queue.maxsize == 128
        first = queue.get_nowait()
        assert first.trace_id == "2"
        assert first.steps[0].detail["password"] == "[REDACTED]"


async def test_trace_context_and_subscriber_bytes_are_bounded() -> None:
    store = PipelineTraceStore(maxlen=1000)
    async with store.subscribe() as queue:
        for index in range(80):
            trace = _trace(index)
            trace.request_context = {
                "body": "\0" * (300 * 1024),
                "total_tokens": 40,
                "token": "private",
            }
            store.record(trace)
        assert queue.qsize() < 128
        snapshots = store.list_recent(limit=1000)
        assert len(snapshots) < 80
        assert snapshots[0].request_context["body"].endswith("[TRUNCATED]")
        assert snapshots[0].request_context["total_tokens"] == 40


def test_trace_rejects_text_in_public_counter_slots() -> None:
    store = PipelineTraceStore()
    trace = _trace(1)
    trace.request_context = {"total_tokens": "credentials", "budget_tokens": "credentials"}
    store.record(trace)
    assert store.get("1").request_context == {
        "total_tokens": "[REDACTED]",
        "budget_tokens": "[REDACTED]",
    }
