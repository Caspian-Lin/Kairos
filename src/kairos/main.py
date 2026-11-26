"""Compose the shared Kairos infrastructure and serve configured bots."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections.abc import Sequence
from contextlib import AsyncExitStack

import structlog

from kairos.config import (
    BotProfile,
    GlobalConfig,
    LoggingConfig,
    load_bot_profiles,
    load_global_config,
)
from kairos.gateway.onebot_ws import OneBotWSGateway
from kairos.llm.router import LLMRouter
from kairos.memory.pg_store import PgStore
from kairos.models import RawGroupEvent
from kairos.protocols import LLMClient, MessageStore, OneBotSender
from kairos.runtime.bot_instance import BotRuntime
from kairos.telemetry.audit_logger import AuditLogger

_log = structlog.get_logger(__name__)


class MultiBotManager:
    def __init__(
        self,
        profiles: Sequence[BotProfile],
        store: MessageStore,
        llm_client: LLMClient,
        sender: OneBotSender,
        audit_logger: AuditLogger,
    ) -> None:
        if not 1 <= len(profiles) <= 3:
            raise ValueError("Kairos requires between 1 and 3 active bot profiles.")
        hosted_bot_ids = {profile.self_id for profile in profiles}
        if len(hosted_bot_ids) != len(profiles) or len({p.bot_id for p in profiles}) != len(
            profiles
        ):
            raise ValueError("Active bot profiles must have distinct self_id and bot_id values.")
        self.runtimes = {
            profile.self_id: BotRuntime(
                profile, hosted_bot_ids, store, llm_client, sender, audit_logger
            )
            for profile in profiles
        }

    async def handle_group_event(self, event: RawGroupEvent) -> None:
        if (runtime := self.runtimes.get(event.self_id)) is not None:
            await runtime.handle_group_event(event)

    async def shutdown(self) -> None:
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
        audit_logger = AuditLogger(store)
        resources.push_async_callback(audit_logger.drain)

        gateway = OneBotWSGateway(config.onebot_ws)
        manager = MultiBotManager(profiles, store, llm_client, gateway, audit_logger)
        resources.push_async_callback(manager.shutdown)
        gateway.on_group_event = manager.handle_group_event
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
