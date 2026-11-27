"""Bounded, searchable pipeline traces for the local admin interface."""

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager

from kairos.models import PipelineStep, PipelineTrace
from kairos.telemetry.log_buffer import (
    _EventRing,
    _secrets_tuple,
    _storage_size,
    redact_payload,
)

_CONTEXT_BYTES = 256 * 1024


class PipelineTraceStore:
    def __init__(self, maxlen: int = 100, *, secrets: Iterable[str] = ()) -> None:
        self._ring = _EventRing[PipelineTrace](maxlen, max_bytes=16 * 1024 * 1024)
        self.set_secrets(secrets)

    def set_secrets(self, secrets: Iterable[str]) -> None:
        self._secrets = _secrets_tuple(secrets)

    def record(self, trace: PipelineTrace) -> None:
        context = trace.request_context
        safe_context = (
            redact_payload(context, secrets=self._secrets, max_chars=_CONTEXT_BYTES)
            if context is not None
            else None
        )
        if context is not None and safe_context is not None:
            # These numeric schema fields are counters, never credentials.
            for name in ("total_tokens", "budget_tokens"):
                if name in context and (
                    (type(context[name]) is int and context[name].bit_length() <= 128)
                    or (name == "budget_tokens" and context[name] is None)
                ):
                    safe_context[name] = context[name]
        max_steps = 128
        original_steps = trace.steps[:max_steps]
        step_data = redact_payload(
            [{"stage": step.stage, "detail": step.detail} for step in original_steps],
            secrets=self._secrets,
            max_chars=32 * 1024,
        )
        steps = []
        for original, safe in zip(original_steps, step_data, strict=False):
            if not isinstance(safe, dict):
                break
            detail = safe.get("detail")
            steps.append(
                PipelineStep(
                    stage=safe.get("stage", "[TRUNCATED]"),
                    status=original.status,
                    latency_ms=original.latency_ms,
                    detail=detail if isinstance(detail, dict) else {"truncated": True},
                )
            )
        if len(steps) < len(trace.steps):
            steps.append(PipelineStep(stage="trace", status="skipped", detail={"truncated": True}))
        identifiers = {
            field: redact_payload(getattr(trace, field), secrets=self._secrets, max_chars=1024)
            for field in (
                "trace_id",
                "bot_id",
                "msg_id",
                "sender_alias",
                "input_preview",
                "outcome",
            )
        }
        safe_trace = trace.model_copy(
            update={**identifiers, "steps": steps, "request_context": safe_context}
        )
        self._ring.append(lambda _: safe_trace, _storage_size(safe_trace.model_dump(mode="json")))

    def list_recent(
        self,
        limit: int = 50,
        bot_id: str | None = None,
        group_id: int | None = None,
        outcome: str | None = None,
    ) -> list[PipelineTrace]:
        if limit <= 0:
            return []
        matches = (
            trace
            for trace in self._ring.snapshots()
            if (bot_id is None or trace.bot_id == bot_id)
            and (group_id is None or trace.group_id == group_id)
            and (outcome is None or trace.outcome == outcome)
        )
        results = []
        for trace in matches:
            results.append(trace.model_copy(deep=True))
            if len(results) == limit:
                break
        return results

    def get(self, trace_id: str) -> PipelineTrace | None:
        for trace in self._ring.snapshots():
            if trace.trace_id == trace_id:
                return trace.model_copy(deep=True)
        return None

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[PipelineTrace]]:
        async with self._ring.subscribe() as queue:
            yield queue
