"""Compose the shared Kairos infrastructure and serve configured bots."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections.abc import Mapping, Sequence
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
from kairos.llm.router import LLMRouter
from kairos.memory.async_worker import AsyncMemoryWorker
from kairos.memory.pg_store import PgStore
from kairos.memory.provider import PgMemoryProvider
from kairos.models import ChatMessage, LLMResult, NormalizedMessage, RawGroupEvent, WarmMemory
from kairos.protocols import (
    LLMClient,
    MemoryProvider,
    MemoryStore,
    MessageStore,
    OneBotSender,
    SummaryStore,
)
from kairos.runtime.bot_instance import BotRuntime
from kairos.telemetry.audit_logger import AuditLogger

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
    ) -> None:
        if not 1 <= len(profiles) <= 3:
            raise ValueError("Kairos requires between 1 and 3 active bot profiles.")
        hosted_bot_ids = {profile.self_id for profile in profiles}
        if len(hosted_bot_ids) != len(profiles) or len({p.bot_id for p in profiles}) != len(
            profiles
        ):
            raise ValueError("Active bot profiles must have distinct self_id and bot_id values.")
        bot_llm_clients = bot_llm_clients or {}
        clients = {
            profile.bot_id: bot_llm_clients.get(profile.bot_id, llm_client) for profile in profiles
        }
        providers = {
            profile.bot_id: PgMemoryProvider(
                memory_store,
                summary_store if profile.compaction.enabled else None,
                max_impressions_recalled=profile.memory.max_impressions_recalled,
                max_notes_recalled=profile.memory.max_notes_recalled,
                max_persisted_summaries=profile.compaction.max_persisted_summaries,
            )
            for profile in profiles
            if profile.memory.enabled
        }
        worker = None
        if providers or any(profile.compaction.enabled for profile in profiles):
            worker_llm = _WorkerLLM(clients)
            worker = AsyncMemoryWorker(
                _BotMemoryProviders(providers),
                memory_store,
                summary_store,
                DefaultContextCompactor(worker_llm),
                worker_llm,
            )
        self.memory_worker = worker
        self.runtimes = {
            profile.self_id: BotRuntime(
                profile,
                hosted_bot_ids,
                store,
                clients[profile.bot_id],
                sender,
                audit_logger,
                gate_hook=ModelGate(clients[profile.bot_id]) if profile.gate.enabled else None,
                memory_worker=worker
                if profile.memory.enabled or profile.compaction.enabled
                else None,
                summary_store=summary_store,
                compactor=DefaultContextCompactor(
                    clients[profile.bot_id],
                    max_persisted_summaries=profile.compaction.max_persisted_summaries,
                )
                if profile.compaction.enabled
                else None,
                owns_memory_worker=False,
            )
            for profile in profiles
        }

    async def handle_group_event(self, event: RawGroupEvent) -> None:
        if (runtime := self.runtimes.get(event.self_id)) is not None:
            await runtime.handle_group_event(event)

    async def handle_bot_online(self, self_id: int) -> None:
        if (runtime := self.runtimes.get(self_id)) is not None:
            await runtime.handle_online()

    async def handle_group_poke(self, self_id: int, group_id: int) -> None:
        if (runtime := self.runtimes.get(self_id)) is not None:
            await runtime.handle_group_poke(group_id)

    async def shutdown(self) -> None:
        async with AsyncExitStack() as resources:
            if self.memory_worker is not None:
                resources.push_async_callback(self.memory_worker.shutdown)
            await asyncio.gather(*(runtime.shutdown() for runtime in self.runtimes.values()))


async def run(config: GlobalConfig, profiles: Sequence[BotProfile]) -> None:
    """Serve the configured bots until a shutdown signal."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
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


def _configure_logging(config: LoggingConfig) -> None:
    level = logging.getLevelNamesMapping().get(config.level.upper())
    if level is None:
        raise ValueError(f"Unknown logging level: {config.level}")
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer()
            if config.json_format
            else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
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
