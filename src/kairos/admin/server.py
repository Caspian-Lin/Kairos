"""Loopback-only Admin REST and event streams."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from secrets import compare_digest
from typing import TYPE_CHECKING, Any

from aiohttp import web

from kairos.admin.config_editor import BotConfigEditor
from kairos.config import AdminUIConfig
from kairos.gateway.onebot_ws import OneBotWSGateway
from kairos.harness.context_builder import ContextBuilder
from kairos.harness.normalizer import build_roster
from kairos.models import NormalizedMessage
from kairos.telemetry.log_buffer import LogRingBuffer, redact_payload
from kairos.telemetry.trace_store import PipelineTraceStore

if TYPE_CHECKING:
    from kairos.main import MultiBotManager

_STATIC = Path(__file__).with_name("static") / "index.html"


class AdminServer:
    def __init__(
        self,
        config: AdminUIConfig,
        manager: "MultiBotManager",
        gateway: OneBotWSGateway,
        editor: BotConfigEditor,
        logs: LogRingBuffer,
        traces: PipelineTraceStore,
        *,
        reload_profiles: Callable[[], Awaitable[None]],
        request_restart: Callable[[], None],
        secrets: Callable[[], Iterable[str]] = tuple,
    ) -> None:
        self.config = config
        self.manager = manager
        self.gateway = gateway
        self.editor = editor
        self.logs = logs
        self.traces = traces
        self._reload_profiles = reload_profiles
        self._request_restart = request_restart
        self._secrets = secrets
        self._runner: web.AppRunner | None = None

    @property
    def port(self) -> int:
        return (
            int(self._runner.addresses[0][1])
            if self._runner and self._runner.addresses
            else self.config.port
        )

    async def start(self) -> None:
        if not self.config.enabled or self._runner is not None:
            return
        if self.config.host != "127.0.0.1":
            raise ValueError("Admin UI requires host 127.0.0.1")
        runner = web.AppRunner(self.create_app(), access_log=None, shutdown_timeout=1)
        try:
            await runner.setup()
            await web.TCPSite(runner, self.config.host, self.config.port).start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner

    async def stop(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            await runner.cleanup()

    def create_app(self) -> web.Application:
        app = web.Application(middlewares=[self._guard], client_max_size=256 * 1024)
        app.on_response_prepare.append(self._response_headers)
        app.add_routes(
            [
                web.get("/", self._index),
                web.get("/api/status", self._status),
                web.get("/api/logs", self._logs),
                web.get("/api/logs/stream", self._log_stream),
                web.get("/api/bots/{bot_id}/config", self._read_config),
                web.put("/api/bots/{bot_id}/config", self._write_config),
                web.post("/api/bots/{bot_id}/context-preview", self._preview),
                web.get("/api/traces", self._traces),
                web.get("/api/traces/stream", self._trace_stream),
                web.get("/api/traces/{trace_id}", self._trace),
                web.post("/api/control/reload", self._reload),
                web.post("/api/control/restart", self._restart),
            ]
        )
        return app

    @web.middleware
    async def _guard(
        self, request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        # Check the requested host as well as the socket bind to prevent DNS rebinding.
        host = request.host.partition(":")[0].lower()
        if host not in {"127.0.0.1", "localhost"}:
            raise web.HTTPForbidden()
        if request.path.startswith("/api/"):
            token = self.config.access_token
            if token is not None:
                scheme, _, credential = request.headers.get("Authorization", "").partition(" ")
                if scheme.lower() != "bearer" or not compare_digest(
                    credential.encode(errors="surrogatepass"), token.encode(errors="surrogatepass")
                ):
                    raise web.HTTPUnauthorized(headers={"WWW-Authenticate": "Bearer"})
            if request.method in {"POST", "PUT"}:
                origin = request.headers.get("Origin")
                if origin is not None and origin != f"http://{request.host}":
                    raise web.HTTPForbidden()
                if request.content_type != "application/json":
                    raise web.HTTPUnsupportedMediaType()
        return await handler(request)

    @staticmethod
    async def _response_headers(_request: web.Request, response: web.StreamResponse) -> None:
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        )

    async def _index(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(_STATIC)

    async def _status(self, _request: web.Request) -> web.Response:
        online = self.gateway.connected_self_ids
        bots = []
        for runtime in self.manager.runtimes.values():
            profile = runtime.profile
            bots.append(
                {
                    "bot_id": profile.bot_id,
                    "self_id": profile.self_id,
                    "nickname": redact_payload(profile.nickname, secrets=self._secrets()),
                    "online": profile.self_id in online,
                    "active_sessions": len(runtime.sessions),
                    "groups": [
                        {"group_id": session.group_id, "energy": round(session.energy, 2)}
                        for session in runtime.sessions.values()
                    ],
                    "features": {
                        "gate": profile.gate.enabled,
                        "compaction": profile.compaction.enabled,
                        "memory": profile.memory.enabled,
                        "prompt_style": profile.prompt_style,
                        "tools": len(profile.enabled_tools),
                        "trace": self.manager.trace_store is not None,
                    },
                }
            )
        return web.json_response(
            {
                "bots": bots,
                "connected_self_ids": sorted(online),
                "configurable_bots": await asyncio.to_thread(self.editor.list_bot_ids),
            }
        )

    async def _logs(self, request: web.Request) -> web.Response:
        options = _log_filters(request)
        return web.json_response(self.logs.recent(limit=_limit(request, 200, 500), **options))

    async def _log_stream(self, request: web.Request) -> web.StreamResponse:
        options = _log_filters(request)

        def matches(entry: Any) -> bool:
            query = options["query"]
            level = options["level"]
            bot_id = options["bot_id"]
            return (
                (not level or entry["level"].lower() == level.lower())
                and (not bot_id or str(entry["fields"].get("bot_id", "")) == bot_id)
                and (
                    not query
                    or query.casefold() in json.dumps(entry, ensure_ascii=False).casefold()
                )
            )

        async with self.logs.subscribe() as queue:
            return await _stream(request, queue, matches, lambda item: item)

    async def _read_config(self, request: web.Request) -> web.Response:
        try:
            value = await asyncio.to_thread(self.editor.read_safe, request.match_info["bot_id"])
        except FileNotFoundError:
            raise web.HTTPNotFound() from None
        except (ValueError, OSError):
            raise web.HTTPBadRequest(text="无法读取该 Bot 配置") from None
        return web.json_response(value)

    async def _write_config(self, request: web.Request) -> web.Response:
        patch = await _object_body(request)
        try:
            value = await asyncio.to_thread(
                self.editor.update_safe, request.match_info["bot_id"], patch
            )
        except FileNotFoundError:
            raise web.HTTPNotFound() from None
        except (ValueError, OSError):
            raise web.HTTPBadRequest(text="配置未通过校验，或文件无法保存") from None
        return web.json_response(value)

    async def _preview(self, request: web.Request) -> web.Response:
        body = await _object_body(request)
        patch, group_id = body.get("patch", {}), body.get("group_id")
        if not isinstance(patch, dict) or (group_id is not None and type(group_id) is not int):
            raise web.HTTPBadRequest(text="草稿或群号格式无效")
        bot_id = request.match_info["bot_id"]
        try:
            profile = await asyncio.to_thread(self.editor.build_preview_profile, bot_id, patch)
            secrets = await asyncio.to_thread(self.editor.preview_secrets, bot_id, patch)
        except FileNotFoundError:
            raise web.HTTPNotFound() from None
        except (ValueError, OSError):
            raise web.HTTPBadRequest(text="草稿未通过配置校验") from None
        runtime = next(
            (item for item in self.manager.runtimes.values() if item.profile.bot_id == bot_id), None
        )
        active = runtime.sessions if runtime is not None else {}
        session = (
            active.get(group_id) if group_id is not None else next(iter(active.values()), None)
        )
        now = datetime.now(UTC)
        warm_memory = None
        tools = []
        if session is not None and runtime is not None:
            history = list(session.buffer)
            if profile.memory.enabled or profile.compaction.enabled or profile.gate.enabled:
                threads = (
                    session.thread_tracker.active_threads(now) if session.thread_tracker else []
                )
                warm_memory, history = runtime._hot_context(session, history, threads)
                warm_memory = warm_memory.model_copy(
                    update={
                        "summaries": warm_memory.summaries if profile.compaction.enabled else [],
                        "impressions": warm_memory.impressions if profile.memory.enabled else [],
                        "notes": warm_memory.notes if profile.memory.enabled else [],
                    }
                )
            if profile.gate.enabled and profile.enabled_tools:
                latest = self.traces.list_recent(limit=1, bot_id=bot_id, group_id=session.group_id)
                for step in latest[0].steps if latest else ():
                    names = step.detail.get("need_tools")
                    if step.stage == "L1b_model_gate" and isinstance(names, list):
                        tools = self.manager.tool_registry.resolve_definitions(
                            profile, [name for name in names if isinstance(name, str)]
                        )
        else:
            history = [_preview_message(bot_id, group_id, now)]
        inspected = ContextBuilder.inspect_context(
            profile,
            history,
            build_roster(history),
            now,
            warm_memory=warm_memory,
            attached_tools=tools,
            enable_react=profile.gate.enabled
            or profile.compaction.enabled
            or profile.memory.enabled,
            token_budget=profile.compaction.input_token_budget,
        )
        safe = redact_payload(inspected, secrets=(*secrets, *self._secrets()))
        for field in ("total_tokens", "tool_tokens", "budget_tokens"):
            safe[field] = inspected[field]
        safe["sample"] = session is None
        return web.json_response(safe)

    async def _traces(self, request: web.Request) -> web.Response:
        bot_id, group_id, outcome = _trace_filters(request)
        records = self.traces.list_recent(
            limit=_limit(request, 50, 100), bot_id=bot_id, group_id=group_id, outcome=outcome
        )
        # The list and stream carry previews. Clients fetch one full request at a time.
        return web.json_response(
            [trace.model_dump(mode="json", exclude={"request_context"}) for trace in records]
        )

    async def _trace(self, request: web.Request) -> web.Response:
        trace = self.traces.get(request.match_info["trace_id"])
        if trace is None:
            raise web.HTTPNotFound()
        return web.json_response(trace.model_dump(mode="json"))

    async def _trace_stream(self, request: web.Request) -> web.StreamResponse:
        bot_id, group_id, outcome = _trace_filters(request)

        def matches(trace: Any) -> bool:
            return (
                (bot_id is None or bot_id == trace.bot_id)
                and (group_id is None or group_id == trace.group_id)
                and (outcome is None or outcome == trace.outcome)
            )

        async with self.traces.subscribe() as queue:
            return await _stream(
                request,
                queue,
                matches,
                lambda trace: trace.model_dump(mode="json", exclude={"request_context"}),
            )

    async def _reload(self, request: web.Request) -> web.Response:
        await _object_body(request)
        try:
            await self._reload_profiles()
        except (ValueError, OSError, RuntimeError):
            raise web.HTTPBadRequest(text="Bot 配置重载失败") from None
        return web.json_response({"ok": True})

    async def _restart(self, request: web.Request) -> web.Response:
        await _object_body(request)
        asyncio.get_running_loop().call_soon(self._request_restart)
        return web.json_response({"ok": True}, status=202)


def _limit(request: web.Request, default: int, maximum: int) -> int:
    try:
        return max(1, min(maximum, int(request.query.get("limit", default))))
    except ValueError:
        raise web.HTTPBadRequest(text="limit 无效") from None


def _log_filters(request: web.Request) -> dict[str, str | None]:
    return {key: request.query.get(key) or None for key in ("level", "bot_id", "query")}


def _trace_filters(request: web.Request) -> tuple[str | None, int | None, str | None]:
    text = request.query.get("group_id")
    try:
        group = int(text) if text else None
    except ValueError:
        raise web.HTTPBadRequest(text="群号无效") from None
    return request.query.get("bot_id") or None, group, request.query.get("outcome") or None


async def _object_body(request: web.Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except (ValueError, RecursionError):
        raise web.HTTPBadRequest(text="请求必须是 JSON 对象") from None
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(text="请求必须是 JSON 对象")
    return data


async def _stream[Event](
    request: web.Request,
    queue: asyncio.Queue[Event],
    matches: Callable[[Event], bool],
    serialize: Callable[[Event], Any],
) -> web.StreamResponse:
    response = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "X-Accel-Buffering": "no"}
    )
    await response.prepare(request)
    try:
        async with asyncio.timeout(5):
            await response.write(b": connected\n\n")
        while request.transport is not None and not request.transport.is_closing():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=10)
            except TimeoutError:
                data = b": heartbeat\n\n"
            else:
                if not matches(event):
                    continue
                data = (
                    "data: " + json.dumps(serialize(event), ensure_ascii=False) + "\n\n"
                ).encode()
            async with asyncio.timeout(5):
                await response.write(data)
    except (ConnectionError, TimeoutError):
        pass
    return response


def _preview_message(bot_id: str, group_id: int | None, now: datetime) -> NormalizedMessage:
    return NormalizedMessage(
        bot_id=bot_id,
        group_id=group_id or 0,
        msg_id="preview",
        last_msg_id="preview",
        sender_qq=0,
        sender_alias="U1:预览",
        content="晚上聊什么？",
        reply_to_msg_id=None,
        is_bot=False,
        is_at_bot=False,
        is_reply_to_bot=False,
        is_peer_bot=False,
        created_at=now,
        latest_at=now,
    )
