"""Replay groups across OneBot v11 WebSockets and OpenAI-compatible HTTP."""

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import aiohttp
import httpx
import pytest
import yaml
from aiohttp import web

from kairos.config import load_bot_profiles, load_global_config
from kairos.gateway.onebot_ws import OneBotWSGateway
from kairos.llm.router import LLMRouter
from kairos.main import MultiBotManager
from kairos.models import AuditRecord, NormalizedMessage
from kairos.telemetry.audit_logger import AuditLogger

_LIN_ID = 10_001
_CHEN_ID = 10_002
_GROUP = 80_001
_HUMAN = 30_001
_TOKEN = "loopback-onebot-token"
_MODEL_KEY = "loopback-model-key"
_WAIT = 4.0


class MemoryStore:
    def __init__(self) -> None:
        self.messages: dict[tuple[str, int, str], NormalizedMessage] = {}
        self.audits: asyncio.Queue[AuditRecord] = asyncio.Queue()
        self._changed = asyncio.Condition()

    async def load_recent_messages(
        self, bot_id: str, group_id: int, limit: int = 40
    ) -> list[NormalizedMessage]:
        history = [
            item
            for (bot, group, _), item in self.messages.items()
            if (bot, group) == (bot_id, group_id)
        ]
        return history[-limit:]

    async def save_message(self, msg: NormalizedMessage) -> None:
        async with self._changed:
            self.messages.setdefault((msg.bot_id, msg.group_id, msg.msg_id), msg)
            self._changed.notify_all()

    async def save_audit_log(self, record: AuditRecord) -> None:
        self.audits.put_nowait(record)

    async def health_check(self) -> bool:
        return True

    async def written(self, bot_id: str, *ids: int) -> list[NormalizedMessage]:
        keys = [(bot_id, _GROUP, str(msg_id)) for msg_id in ids]
        async with asyncio.timeout(_WAIT), self._changed:
            await self._changed.wait_for(lambda: all(key in self.messages for key in keys))
        return [self.messages[key] for key in keys]

    async def next_audit(self) -> AuditRecord:
        return await asyncio.wait_for(self.audits.get(), _WAIT)


@dataclass
class Completion:
    text: str
    delay: float = 0
    release: asyncio.Event | None = None


class OpenAIStub:
    def __init__(self) -> None:
        self.planned: deque[Completion] = deque()
        self.received: list[dict[str, Any]] = []
        self._requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def complete(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {_MODEL_KEY}":
            raise web.HTTPUnauthorized()
        payload = await request.json()
        self.received.append(payload)
        self._requests.put_nowait(payload)
        completion = self.planned.popleft() if self.planned else Completion("unexpected reply")
        deadline = asyncio.get_running_loop().time() + completion.delay
        if completion.release is not None:
            try:
                await asyncio.wait_for(completion.release.wait(), _WAIT)
            except TimeoutError:
                raise web.HTTPGatewayTimeout() from None
        await asyncio.sleep(max(0, deadline - asyncio.get_running_loop().time()))
        return web.json_response({"choices": [{"message": {"content": completion.text}}]})

    async def next_request(self) -> dict[str, Any]:
        return await asyncio.wait_for(self._requests.get(), _WAIT)


@dataclass
class SentMessage:
    msg_id: int
    segments: list[dict[str, Any]]


class OneBotPeer:
    def __init__(self, self_id: int, socket: aiohttp.ClientWebSocketResponse) -> None:
        self.self_id = self_id
        self._socket = socket
        self._sends: asyncio.Queue[SentMessage] = asyncio.Queue()
        self._ids = count(self_id * 1000 + 1)

    async def receive_actions(self) -> None:
        async for frame in self._socket:
            assert frame.type is aiohttp.WSMsgType.TEXT
            action = frame.json()
            assert action["action"] == "send_group_msg"
            assert action["params"]["group_id"] == _GROUP
            msg_id = next(self._ids)
            await self._socket.send_json(
                {
                    "status": "ok",
                    "retcode": 0,
                    "data": {"message_id": msg_id},
                    "echo": action["echo"],
                }
            )
            self._sends.put_nowait(SentMessage(msg_id, action["params"]["message"]))

    async def group_message(
        self,
        msg_id: int,
        message: str | list[dict[str, Any]],
        *,
        sender: int = _HUMAN,
        timestamp: int | None = None,
    ) -> None:
        await self._socket.send_json(
            {
                "self_id": self.self_id,
                "post_type": "message",
                "message_type": "group",
                "group_id": _GROUP,
                "message_id": msg_id,
                "user_id": sender,
                "time": int(time.time()) if timestamp is None else timestamp,
                "sender": {"nickname": f"用户{sender}", "card": ""},
                "message": message,
            }
        )

    async def next_send(self, *, timeout: float = _WAIT) -> SentMessage:
        return await asyncio.wait_for(self._sends.get(), timeout)

    async def expect_no_send(self) -> None:
        with pytest.raises(TimeoutError):
            await self.next_send(timeout=0.25)


@dataclass
class Replay:
    model: OpenAIStub
    store: MemoryStore
    manager: MultiBotManager
    lin: OneBotPeer
    chen: OneBotPeer


def _config_files(tmp_path: Path, api_port: int, normal: float, trigger_mode: str) -> Path:
    global_path = tmp_path / "settings.yaml"
    global_path.write_text(
        yaml.safe_dump(
            {
                "postgres": {"dsn": "postgresql://unused"},
                "onebot_ws": {"port": 0, "access_token": "${KAIROS_E2E_WS_TOKEN}"},
                "llm_models": {
                    name: {
                        "base_url": f"http://127.0.0.1:{api_port}/v1",
                        "api_key": "${KAIROS_E2E_MODEL_KEY}",
                        "model": name,
                        "max_retries": 0,
                    }
                    for name in ("lin_xiaoman", "chen_adai")
                },
            }
        ),
        encoding="utf-8",
    )
    bots = tmp_path / "bots"
    bots.mkdir()
    for bot_id, self_id, nickname in (
        ("lin_xiaoman", _LIN_ID, "林小满"),
        ("chen_adai", _CHEN_ID, "陈阿呆"),
    ):
        (bots / f"{bot_id}.yaml").write_text(
            yaml.safe_dump(
                {
                    "bot_id": bot_id,
                    "self_id": self_id,
                    "nickname": nickname,
                    "persona_prompt": f"我是{nickname}。",
                    "chat_model_key": bot_id,
                    "allowed_groups": [_GROUP],
                    "trigger_mode": trigger_mode,
                    "debounce": {"normal_seconds": normal, "mention_seconds": 0.015},
                    "rate_limit": {"cooldown_seconds": 0, "peer_bot_max_turns": 2},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    (bots / "example_unused.yaml").write_text("self_id: ${KAIROS_UNUSED_BOT}", encoding="utf-8")
    return global_path


async def _cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@pytest.fixture
async def make_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[..., Awaitable[Replay]]]:
    monkeypatch.setenv("KAIROS_E2E_WS_TOKEN", _TOKEN)
    monkeypatch.setenv("KAIROS_E2E_MODEL_KEY", _MODEL_KEY)
    async with AsyncExitStack() as resources:

        async def start(normal: float = 0.08, trigger_mode: str = "all_with_veto") -> Replay:
            model = OpenAIStub()
            app = web.Application()
            app.router.add_post("/v1/chat/completions", model.complete)
            runner = web.AppRunner(app, shutdown_timeout=1)
            await runner.setup()
            resources.push_async_callback(runner.cleanup)
            await web.TCPSite(runner, "127.0.0.1", 0).start()

            config = load_global_config(
                _config_files(tmp_path, runner.addresses[0][1], normal, trigger_mode)
            )
            profiles = load_bot_profiles(tmp_path / "bots", config)
            assert {profile.self_id for profile in profiles} == {_LIN_ID, _CHEN_ID}
            store = MemoryStore()
            llm = LLMRouter(
                config.llm_models, transport=httpx.AsyncHTTPTransport(http2=True, trust_env=False)
            )
            resources.push_async_callback(llm.aclose)
            audit = AuditLogger(store)
            resources.push_async_callback(audit.drain)
            gateway = OneBotWSGateway(config.onebot_ws)
            manager = MultiBotManager(profiles, store, llm, gateway, audit)
            resources.push_async_callback(manager.shutdown)
            gateway.on_group_event = manager.handle_group_event
            resources.push_async_callback(gateway.stop)
            await gateway.start()

            client = await resources.enter_async_context(aiohttp.ClientSession())
            peers = []
            for self_id in (_LIN_ID, _CHEN_ID):
                socket = await resources.enter_async_context(
                    client.ws_connect(
                        f"http://127.0.0.1:{gateway.port}{config.onebot_ws.path}",
                        headers={"X-Self-ID": str(self_id), "Authorization": f"Bearer {_TOKEN}"},
                    )
                )
                peer = OneBotPeer(self_id, socket)
                resources.push_async_callback(_cancel, asyncio.create_task(peer.receive_actions()))
                peers.append(peer)
            assert gateway.connected_self_ids == {_LIN_ID, _CHEN_ID}
            return Replay(model, store, manager, *peers)

        yield start


def _text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "data": {"text": content}}]


async def test_burst_role_split_and_two_bots_on_one_gateway(make_replay) -> None:
    case = await make_replay(normal=0.35)
    first_reply = "我带了伞。"
    case.model.planned.extend(
        Completion(text) for text in (first_reply, "门口见。", "屋里很凉快。")
    )
    start = int(time.time()) - 4
    await case.lin.group_message(
        21,
        _text("要下雨了") + [{"type": "image", "data": {"file": "private-fixture.png"}}],
        timestamp=start,
    )
    await case.lin.group_message(22, f"[CQ:at,qq={_LIN_ID}]去散步吗？", timestamp=start + 1)

    request = await case.model.next_request()
    assert request["model"] == "lin_xiaoman"
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    turn = request["messages"][1]["content"]
    assert turn.startswith("[m21]") and "[m22]" not in turn
    assert "要下雨了[图片] / @我去散步吗？" in turn
    assert "private-fixture.png" not in turn and "[CQ:" not in turn
    first = await case.lin.next_send()
    assert first.segments == _text(first_reply)
    audit = await case.store.next_audit()
    assert (audit.bot_id, audit.final_reply) == ("lin_xiaoman", first_reply)
    raw_a, raw_b, own = await case.store.written("lin_xiaoman", 21, 22, first.msg_id)
    assert [message.content for message in (raw_a, raw_b)] == ["要下雨了[图片]", "@我去散步吗？"]
    assert own.is_bot and own.sender_qq == _LIN_ID
    assert str(first.msg_id) in case.manager.runtimes[_LIN_ID].sessions[_GROUP].bot_msg_ids

    await case.lin.group_message(23, _text("我到楼下了"), timestamp=int(time.time()) + 1)
    request = await case.model.next_request()
    messages = request["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[2] == {"role": "assistant", "content": first_reply}
    assert "[m23]" in messages[3]["content"]
    assert (await case.lin.next_send()).segments == _text("门口见。")
    assert (await case.store.next_audit()).bot_id == "lin_xiaoman"

    await case.chen.group_message(24, _text("屋里开空调了吗？"))
    request = await case.model.next_request()
    assert request["model"] == "chen_adai"
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert "我是陈阿呆。" in request["messages"][0]["content"]
    assert "[m24]" in request["messages"][1]["content"]
    assert "[m21]" not in request["messages"][1]["content"]
    other = await case.chen.next_send()
    assert other.segments == _text("屋里很凉快。")
    assert (await case.store.next_audit()).bot_id == "chen_adai"
    (own_other,) = await case.store.written("chen_adai", other.msg_id)
    assert own_other.is_bot and own_other.sender_qq == _CHEN_ID
    assert ("chen_adai", _GROUP, str(first.msg_id)) not in case.store.messages


async def test_mention_uses_fast_path(make_replay) -> None:
    case = await make_replay(normal=2.0)
    case.model.planned.append(Completion("听见啦。"))
    await case.lin.group_message(31, f"[CQ:at,qq={_LIN_ID}]在吗？")
    assert (await case.lin.next_send(timeout=1.25)).segments == _text("听见啦。")
    request = await case.model.next_request()
    assert "@我在吗？" in request["messages"][1]["content"]
    assert (await case.store.next_audit()).stale_action == "SEND_DIRECT"


@pytest.mark.parametrize("arrivals", [3, 6])
async def test_stale_uses_last_burst_id_or_drops(make_replay, arrivals: int) -> None:
    case = await make_replay(trigger_mode="mention_only")
    release = asyncio.Event()
    case.model.planned.append(Completion("回一下刚才那句。", delay=0.2, release=release))
    start = int(time.time()) - 2
    await case.lin.group_message(41, _text("先问一句"), timestamp=start)
    await case.lin.group_message(42, f"[CQ:at,qq={_LIN_ID}]能看见吗？", timestamp=start + 1)
    request = await case.model.next_request()
    assert "[m41]" in request["messages"][1]["content"]
    ids = range(43, 43 + arrivals)
    for msg_id in ids:
        await case.lin.group_message(msg_id, _text(f"后来发言{msg_id}"), sender=_HUMAN + msg_id)
    await case.store.written("lin_xiaoman", *ids)
    release.set()

    audit = await case.store.next_audit()
    if arrivals == 3:
        assert audit.stale_action == "SEND_WITH_REPLY"
        sent = await case.lin.next_send()
        assert sent.segments == [{"type": "reply", "data": {"id": "42"}}] + _text(
            "回一下刚才那句。"
        )
        (saved,) = await case.store.written("lin_xiaoman", sent.msg_id)
        assert saved.reply_to_msg_id == "42"
    else:
        assert audit.stale_action == "DROP_STALE"
        assert audit.final_reply is None
    await case.lin.expect_no_send()
    assert len(case.model.received) == 1


async def test_model_veto_has_no_send(make_replay) -> None:
    case = await make_replay()
    case.model.planned.append(Completion("<|act:silent|>"))
    await case.lin.group_message(51, _text("大家晚安。"))
    audit = await case.store.next_audit()
    assert audit.main_vetoed and audit.stale_action == "VETOED"
    assert audit.final_reply is None
    await case.lin.expect_no_send()
    assert len(case.model.received) == 1


async def test_peer_bot_turn_limit_stops_full_network_loop(make_replay) -> None:
    case = await make_replay()
    replies = [
        "今晚去喂流浪猫",
        "西瓜切完放冰箱",
        "我已经把门修好了",
        "外面飘过一阵雨",
        "周末坐船看灯塔",
    ]
    case.model.planned.extend(Completion(text) for text in replies)
    await case.lin.group_message(61, _text("今天有什么趣事？"))
    sender, receiver = case.lin, case.chen
    for idx, text in enumerate(replies):
        sent = await sender.next_send()
        assert sent.segments == _text(text)
        request = await case.model.next_request()
        assert request["model"] == ("lin_xiaoman" if idx % 2 == 0 else "chen_adai")
        await case.store.next_audit()
        await receiver.group_message(sent.msg_id, sent.segments, sender=sender.self_id)
        sender, receiver = receiver, sender

    (peer_input,) = await case.store.written("chen_adai", sent.msg_id)
    assert peer_input.is_peer_bot and not peer_input.is_bot
    await asyncio.gather(case.lin.expect_no_send(), case.chen.expect_no_send())
    assert len(case.model.received) == len(replies)
    assert case.manager.runtimes[_CHEN_ID].sessions[_GROUP].consecutive_peer_bot_turns == 2
