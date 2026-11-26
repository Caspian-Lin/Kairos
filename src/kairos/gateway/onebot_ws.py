"""Single-port OneBot v11 reverse WebSocket server and action sender."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from secrets import compare_digest
from typing import Any
from uuid import uuid4

import structlog
from aiohttp import WSCloseCode, WSMsgType, web

from kairos.config import OneBotWSConfig
from kairos.models import MessageSegment, RawGroupEvent

_log = structlog.get_logger()


class OneBotSendError(RuntimeError):
    """A OneBot action failed or did not return a usable receipt."""


class BotOfflineError(OneBotSendError):
    """The bot has no live connection for this action."""


@dataclass(slots=True)
class _PendingEcho:
    socket: web.WebSocketResponse
    future: asyncio.Future[dict[str, Any]]


class OneBotWSGateway:
    def __init__(
        self,
        config: OneBotWSConfig,
        on_group_event: Callable[[RawGroupEvent], Awaitable[None]] | None = None,
    ) -> None:
        self.config = config
        self.on_group_event = on_group_event
        self.pending_echoes: dict[str, _PendingEcho] = {}
        self._connections: dict[int, web.WebSocketResponse] = {}
        self._sockets: set[web.WebSocketResponse] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._runner: web.AppRunner | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False

    @property
    def connected_self_ids(self) -> set[int]:
        return {self_id for self_id, socket in self._connections.items() if not socket.closed}

    @property
    def port(self) -> int:
        """Return the bound port while running, including for a port of zero."""
        if self._runner is not None and self._runner.addresses:
            return int(self._runner.addresses[0][1])
        return self.config.port

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get(self.config.path, self._handle_socket, allow_head=False)
        app.on_startup.append(self._open)
        app.on_shutdown.append(self._close_sockets)
        app.on_cleanup.append(self._cancel_tasks)
        return app

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._runner is not None:
                return
            # HTTP request logs would include credentials provided via the query string.
            runner = web.AppRunner(self.create_app(), access_log=None)
            try:
                await runner.setup()
                await web.TCPSite(runner, self.config.host, self.config.port).start()
            except BaseException:
                await runner.cleanup()
                raise
            self._runner = runner

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if self._runner is not None:
                await self._runner.cleanup()
                self._runner = None

    async def _open(self, _app: web.Application) -> None:
        self._closing = False

    async def _close_sockets(self, _app: web.Application) -> None:
        self._closing = True
        for self_id, socket in self._connections.items():
            self._fail_pending(self_id, socket)
        self._connections.clear()
        await asyncio.gather(*(_close_socket(socket) for socket in self._sockets))

    async def _cancel_tasks(self, _app: web.Application) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _track(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _authorize(self, request: web.Request) -> None:
        token = self.config.access_token
        if token is None:
            return
        header = request.headers.get("Authorization")
        if header is None:
            provided = request.query.get("access_token", "")
        else:
            scheme, _, credential = header.partition(" ")
            provided = credential if scheme.lower() == "bearer" else ""
        if not provided or not compare_digest(
            provided.encode(errors="surrogatepass"), token.encode(errors="surrogatepass")
        ):
            raise web.HTTPUnauthorized(headers={"WWW-Authenticate": "Bearer"})

    def _register(self, self_id: int, socket: web.WebSocketResponse) -> None:
        old_socket = self._connections.get(self_id)
        self._connections[self_id] = socket
        if old_socket is not None:
            self._fail_pending(self_id, old_socket)
            self._track(_close_socket(old_socket))

    def _unregister(self, self_id: int, socket: web.WebSocketResponse) -> None:
        if self._connections.get(self_id) is socket:
            del self._connections[self_id]
        self._fail_pending(self_id, socket)

    def _fail_pending(self, self_id: int, socket: web.WebSocketResponse) -> None:
        for echo, pending in tuple(self.pending_echoes.items()):
            if pending.socket is socket:
                del self.pending_echoes[echo]
                if not pending.future.done():
                    pending.future.set_exception(BotOfflineError(f"Bot {self_id} disconnected"))

    async def _handle_socket(self, request: web.Request) -> web.WebSocketResponse:
        self._authorize(request)
        if self._closing:
            raise web.HTTPServiceUnavailable()
        header_id = request.headers.get("X-Self-ID")
        try:
            self_id = int(header_id) if header_id is not None else None
        except ValueError:
            raise web.HTTPBadRequest(text="Invalid X-Self-ID") from None

        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self._sockets.add(socket)
        try:
            if self._closing:
                await _close_socket(socket)
                return socket
            if self_id is not None:
                self._register(self_id, socket)
            async for frame in socket:
                if self._closing or (
                    self_id is not None and self._connections.get(self_id) is not socket
                ):
                    break
                if frame.type == WSMsgType.ERROR:
                    break
                if frame.type != WSMsgType.TEXT:
                    continue
                payload = _decode(frame.data)
                if payload is None:
                    continue
                if "self_id" in payload:
                    try:
                        claimed_id = _integer(payload["self_id"])
                    except ValueError:
                        _log.warning("onebot_invalid_self_id")
                        continue
                    if self_id is None:
                        self_id = claimed_id
                        self._register(self_id, socket)
                    elif claimed_id != self_id:
                        _log.warning("onebot_mismatched_self_id", self_id=self_id)
                        continue
                if self_id is not None:
                    self._handle_payload(self_id, socket, payload)
        finally:
            self._sockets.discard(socket)
            if self_id is not None:
                self._unregister(self_id, socket)
        return socket

    def _handle_payload(
        self, self_id: int, socket: web.WebSocketResponse, payload: dict[str, Any]
    ) -> None:
        echo = payload.get("echo")
        if isinstance(echo, str) and (pending := self.pending_echoes.get(echo)) is not None:
            if pending.socket is socket and not pending.future.done():
                pending.future.set_result(payload)
            return
        if payload.get("post_type") != "message" or payload.get("message_type") != "group":
            _log.debug("onebot_event_ignored", self_id=self_id)
            return
        try:
            event = _group_event(self_id, payload)
        except (KeyError, TypeError, ValueError, OSError, OverflowError):
            _log.warning("onebot_invalid_group_event", self_id=self_id)
            return
        callback = self.on_group_event
        if callback is not None:
            self._track(self._dispatch(callback, event))

    async def _dispatch(
        self, callback: Callable[[RawGroupEvent], Awaitable[None]], event: RawGroupEvent
    ) -> None:
        try:
            await callback(event)
        except Exception as exc:  # noqa: BLE001 - Callback failures must stay off the socket.
            _log.warning(
                "onebot_group_callback_failed",
                self_id=event.self_id,
                group_id=event.group_id,
                error_type=type(exc).__name__,
            )

    async def send_group_msg(
        self, self_id: int, group_id: int, text: str, reply_to_msg_id: str | None = None
    ) -> str | None:
        socket = self._connections.get(self_id)
        if socket is None or socket.closed:
            _log.warning("onebot_send_offline", self_id=self_id, group_id=group_id)
            raise BotOfflineError(f"Bot {self_id} is offline")

        segments = [{"type": "text", "data": {"text": text}}]
        if reply_to_msg_id:
            segments.insert(0, {"type": "reply", "data": {"id": str(reply_to_msg_id)}})
        echo = uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.pending_echoes[echo] = _PendingEcho(socket, future)
        try:
            async with asyncio.timeout(self.config.send_timeout_seconds):
                await socket.send_json(
                    {
                        "action": "send_group_msg",
                        "params": {"group_id": group_id, "message": segments},
                        "echo": echo,
                    }
                )
                response = await future
        except BotOfflineError:
            _log.warning("onebot_send_disconnected", self_id=self_id, group_id=group_id)
            raise
        except TimeoutError:
            _log.warning("onebot_send_timeout", self_id=self_id, group_id=group_id)
            raise OneBotSendError(
                f"Timed out waiting for bot {self_id} to send a message"
            ) from None
        except (OSError, RuntimeError):
            _log.warning("onebot_send_transport_failed", self_id=self_id, group_id=group_id)
            raise OneBotSendError(f"Could not send a message through bot {self_id}") from None
        finally:
            self.pending_echoes.pop(echo, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                # A disconnect can fail the receipt while send_json is still in progress.
                future.exception()

        if response.get("status") != "ok":
            _log.warning("onebot_send_rejected", self_id=self_id, group_id=group_id)
            raise OneBotSendError(f"Bot {self_id} rejected the send_group_msg action")
        try:
            return _message_id(response["data"]["message_id"])
        except (KeyError, TypeError, ValueError):
            _log.warning("onebot_send_invalid_receipt", self_id=self_id, group_id=group_id)
            raise OneBotSendError(
                f"Bot {self_id} returned a receipt without a message ID"
            ) from None


async def _close_socket(socket: web.WebSocketResponse) -> None:
    await socket.close(code=WSCloseCode.GOING_AWAY, drain=False)


def _decode(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except (ValueError, RecursionError):
        pass
    _log.warning("onebot_invalid_json")
    return None


def _integer(value: Any) -> int:
    if type(value) is int or isinstance(value, str):
        return int(value)
    raise ValueError("Expected an integer")


def _message_id(value: Any) -> str:
    if type(value) is int or (isinstance(value, str) and value):
        return str(value)
    raise ValueError("Expected a message ID")


def _group_event(self_id: int, payload: dict[str, Any]) -> RawGroupEvent:
    sender = payload.get("sender")
    if not isinstance(sender, dict):
        sender = {}
    message = payload["message"]
    raw_message = _text(payload.get("raw_message"))
    if isinstance(message, list):
        segments = [MessageSegment.model_validate(segment) for segment in message]
    elif isinstance(message, str):
        segments = []
        raw_message = raw_message or message
    else:
        raise TypeError("Expected OneBot message segments or a CQ string")
    return RawGroupEvent(
        self_id=self_id,
        group_id=_integer(payload["group_id"]),
        msg_id=_message_id(payload["message_id"]),
        sender_qq=_integer(payload.get("user_id", sender.get("user_id"))),
        sender_nickname=_text(sender.get("nickname")),
        sender_card=_text(sender.get("card")),
        segments=segments,
        raw_message=raw_message,
        timestamp=datetime.fromtimestamp(_integer(payload["time"]), UTC),
    )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""
