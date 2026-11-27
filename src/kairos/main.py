"""Compose the shared Kairos infrastructure and serve configured bots."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any

import structlog

from kairos.config import (
    BotProfile,
    GlobalConfig,
    LoggingConfig,
    load_bot_profiles,
    load_global_config,
)
from kairos.gateway.onebot_ws import OneBotWSGateway
from kairos.harness.compactor import DefaultContextCompactor
from kairos.harness.gate import ModelGate
from kairos.harness.tool_registry import ToolRegistry
from kairos.llm.router import LLMRouter
from kairos.memory.async_worker import AsyncMemoryWorker
from kairos.memory.pg_store import PgStore
from kairos.memory.provider import PgMemoryProvider
from kairos.models import (
    ChatMessage,
    ContextSummary,
    LLMResult,
    NormalizedMessage,
    RawGroupEvent,
    WarmMemory,
)
from kairos.protocols import (
    LLMClient,
    MemoryProvider,
    MemoryStore,
    MessageStore,
    OneBotSender,
    SummaryStore,
    ToolRegistryProtocol,
)
from kairos.runtime.bot_instance import BotRuntime
from kairos.telemetry.audit_logger import AuditLogger
from kairos.telemetry.log_buffer import LogRingBuffer
from kairos.telemetry.trace_store import PipelineTraceStore

_log = structlog.get_logger(__name__)


class _WorkerLLM(LLMClient):
    def __init__(self, clients: Mapping[str, LLMClient]) -> None:
        self._clients = clients

    async def chat_completion(
        self,
        model_key: str,
        messages: list[ChatMessage],
        *,
        session_id: str | None = None,
        **overrides: Any,
    ) -> LLMResult:
        # Summary and memory jobs identify their bot in the bot:group:job session ID.
        bot_id, _, _ = (session_id or "").rsplit(":", 2)
        return await self._clients[bot_id].chat_completion(
            model_key, messages, session_id=session_id, **overrides
        )


class _BotMemoryProviders(MemoryProvider):
    def __init__(self, providers: Mapping[str, MemoryProvider]) -> None:
        self._providers = providers

    async def recall(
        self,
        bot_id: str,
        group_id: int,
        active_user_qqs: Sequence[int],
        recent_messages: Sequence[NormalizedMessage],
    ) -> WarmMemory:
        return await self._providers[bot_id].recall(
            bot_id, group_id, active_user_qqs, recent_messages
        )


class MultiBotManager:
    def __init__(
        self,
        profiles: Sequence[BotProfile],
        store: MessageStore,
        llm_client: LLMClient,
        sender: OneBotSender,
        audit_logger: AuditLogger,
        *,
        bot_llm_clients: Mapping[str, LLMClient] | None = None,
        summary_store: SummaryStore | None = None,
        memory_store: MemoryStore | None = None,
        tool_registry: ToolRegistryProtocol | None = None,
        trace_store: PipelineTraceStore | None = None,
    ) -> None:
        self._store = store
        self._llm_client = llm_client
        self._sender = sender
        self._audit_logger = audit_logger
        self._summary_store = summary_store
        self._memory_store = memory_store
        self.tool_registry = ToolRegistry() if tool_registry is None else tool_registry
        self.trace_store = trace_store
        self._reload_lock = asyncio.Lock()
        self._available = asyncio.Event()
        self._available.set()
        self._closing = False
        self.runtimes, self.memory_worker = self._build_runtimes(profiles, bot_llm_clients or {})

    def _build_runtimes(
        self, profiles: Sequence[BotProfile], bot_llm_clients: Mapping[str, LLMClient]
    ) -> tuple[dict[int, BotRuntime], AsyncMemoryWorker | None]:
        if not 1 <= len(profiles) <= 3:
            raise ValueError("Kairos requires between 1 and 3 active bot profiles.")
        hosted_bot_ids = {profile.self_id for profile in profiles}
        if len(hosted_bot_ids) != len(profiles) or len({p.bot_id for p in profiles}) != len(
            profiles
        ):
            raise ValueError("Active bot profiles must have distinct self_id and bot_id values.")
        clients = {
            profile.bot_id: bot_llm_clients.get(profile.bot_id, self._llm_client)
            for profile in profiles
        }
        providers = {
            profile.bot_id: PgMemoryProvider(
                self._memory_store,
                self._summary_store if profile.compaction.enabled else None,
                max_impressions_recalled=profile.memory.max_impressions_recalled,
                max_notes_recalled=profile.memory.max_notes_recalled,
                max_persisted_summaries=profile.compaction.max_persisted_summaries,
            )
            for profile in profiles
            if profile.memory.enabled
        }
        runtimes: dict[int, BotRuntime] = {}
        self_ids = {profile.bot_id: profile.self_id for profile in profiles}

        def reset_context(
            bot_id: str,
            group_id: int,
            batch: Sequence[NormalizedMessage],
        ) -> ContextSummary | None:
            runtime = runtimes.get(self_ids[bot_id])
            return runtime.reset_group_context(group_id, batch) if runtime is not None else None

        worker = None
        if providers or any(profile.compaction.enabled for profile in profiles):
            worker_llm = _WorkerLLM(clients)
            worker = AsyncMemoryWorker(
                _BotMemoryProviders(providers),
                self._memory_store,
                self._summary_store,
                DefaultContextCompactor(worker_llm),
                worker_llm,
                on_context_reset=reset_context,
            )
        runtimes.update(
            {
                profile.self_id: BotRuntime(
                    profile,
                    hosted_bot_ids,
                    self._store,
                    clients[profile.bot_id],
                    self._sender,
                    self._audit_logger,
                    gate_hook=ModelGate(clients[profile.bot_id]) if profile.gate.enabled else None,
                    memory_worker=worker
                    if profile.memory.enabled or profile.compaction.enabled
                    else None,
                    summary_store=self._summary_store,
                    compactor=DefaultContextCompactor(
                        clients[profile.bot_id],
                        max_persisted_summaries=profile.compaction.max_persisted_summaries,
                    )
                    if profile.compaction.enabled
                    else None,
                    owns_memory_worker=False,
                    tool_registry=self.tool_registry,
                    trace_store=self.trace_store,
                )
                for profile in profiles
            }
        )
        return runtimes, worker

    async def reload_profiles(
        self, new_profiles: Sequence[BotProfile], bot_llm_clients: Mapping[str, LLMClient]
    ) -> None:
        async with self._reload_lock:
            if self._closing:
                raise RuntimeError("Kairos is shutting down")
            checked = [
                BotProfile.model_validate(profile.model_dump(exclude_unset=True))
                for profile in new_profiles
            ]
            for profile in checked:
                if profile.gate.enabled and profile.gate_model_key is None:
                    raise ValueError("gate.enabled requires gate_model_key")
                if profile.llm and any(
                    key is not None and key not in profile.llm
                    for key in (
                        profile.chat_model_key,
                        profile.gate_model_key,
                        profile.worker_model_key,
                    )
                ):
                    raise ValueError("Unknown model slot in Bot profile")
            replacement = self._build_runtimes(checked, bot_llm_clients)
            previous = self.runtimes, self.memory_worker
            self._available.clear()
            self.runtimes, self.memory_worker = replacement
            cleanup = asyncio.create_task(self._stop_runtimes(*previous))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            finally:
                self._available.set()

    async def handle_group_event(self, event: RawGroupEvent) -> None:
        await self._available.wait()
        if (runtime := self.runtimes.get(event.self_id)) is not None:
            await runtime.handle_group_event(event)

    async def handle_bot_online(self, self_id: int) -> None:
        await self._available.wait()
        if (runtime := self.runtimes.get(self_id)) is not None:
            await runtime.handle_online()

    async def handle_group_poke(self, self_id: int, group_id: int) -> None:
        await self._available.wait()
        if (runtime := self.runtimes.get(self_id)) is not None:
            await runtime.handle_group_poke(group_id)

    async def shutdown(self) -> None:
        async with self._reload_lock:
            self._closing = True
            await self._stop_runtimes(self.runtimes, self.memory_worker)

    @staticmethod
    async def _stop_runtimes(
        runtimes: Mapping[int, BotRuntime], worker: AsyncMemoryWorker | None
    ) -> None:
        async with AsyncExitStack() as resources:
            if worker is not None:
                resources.push_async_callback(worker.shutdown)
            await asyncio.gather(*(runtime.shutdown() for runtime in runtimes.values()))


def _exec_current_process() -> None:
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])


class RestartController:
    def __init__(self, restart: Callable[[], None] = _exec_current_process) -> None:
        self.stop = asyncio.Event()
        self._restart = restart
        self._requested = False

    def request_restart(self) -> None:
        self._requested = True
        self.stop.set()

    def restart_after_shutdown(self) -> None:
        if self._requested:
            self._restart()


async def run(
    config: GlobalConfig,
    profiles: Sequence[BotProfile],
    *,
    restart_controller: RestartController | None = None,
    trace_store: PipelineTraceStore | None = None,
) -> None:
    """Serve the configured bots until a shutdown signal."""
    loop = asyncio.get_running_loop()
    controller = RestartController() if restart_controller is None else restart_controller
    stop = controller.stop
    async with AsyncExitStack() as resources:
        if os.name == "posix":
            for signum in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(signum, stop.set)
                resources.callback(loop.remove_signal_handler, signum)

        store = PgStore(config.postgres)
        resources.push_async_callback(store.close)
        await store.connect()

        llm_client = LLMRouter(config.llm_models)
        resources.push_async_callback(llm_client.aclose)
        bot_llm_clients = {
            profile.bot_id: llm_client.for_models(profile.llm)
            for profile in profiles
            if profile.llm
        }
        audit_logger = AuditLogger(store)
        resources.push_async_callback(audit_logger.drain)

        gateway = OneBotWSGateway(config.onebot_ws)
        manager = MultiBotManager(
            profiles,
            store,
            llm_client,
            gateway,
            audit_logger,
            bot_llm_clients=bot_llm_clients,
            summary_store=store,
            memory_store=store,
            trace_store=trace_store,
        )
        resources.push_async_callback(manager.shutdown)
        gateway.on_group_event = manager.handle_group_event
        gateway.on_bot_online = manager.handle_bot_online
        gateway.on_group_poke = manager.handle_group_poke
        resources.push_async_callback(gateway.stop)
        await gateway.start()

        _log.info(
            "kairos_ready",
            host=config.onebot_ws.host,
            port=gateway.port,
            bot_count=len(manager.runtimes),
        )
        await stop.wait()
    controller.restart_after_shutdown()


def _configure_logging(config: LoggingConfig, log_buffer: LogRingBuffer | None = None) -> None:
    level = logging.getLevelNamesMapping().get(config.level.upper())
    if level is None:
        raise ValueError(f"Unknown logging level: {config.level}")
    logging.basicConfig(level=level, format="%(message)s")
    processors: list[structlog.types.Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if log_buffer is not None:
        processors.append(
            lambda logger, method, event: log_buffer.processor(logger, method, dict(event))
        )
    processors.append(
        structlog.processors.JSONRenderer()
        if config.json_format
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Kairos OneBot gateway.")
    parser.add_argument("--config", default="config/global.yaml", help="Path to global YAML config")
    parser.add_argument("--bots-dir", default="config/bots", help="Directory of bot YAML profiles")
    args = parser.parse_args(argv)
    try:
        config = load_global_config(args.config)
        profiles = load_bot_profiles(args.bots_dir, config)
        _configure_logging(config.logging)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if sys.platform.startswith("linux"):
        import uvloop

        uvloop.run(run(config, profiles))
    else:
        asyncio.run(run(config, profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
