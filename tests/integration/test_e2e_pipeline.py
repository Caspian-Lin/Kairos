"""Replay groups across OneBot v11 WebSockets and model HTTP APIs."""

import asyncio
import json
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any

import aiohttp
import httpx
import pytest
import yaml
from aiohttp import web
from structlog.testing import capture_logs

from kairos.config import load_bot_profiles, load_global_config
from kairos.gateway.onebot_ws import OneBotWSGateway
from kairos.llm.router import LLMRouter
from kairos.main import MultiBotManager
from kairos.models import AuditRecord, ContextSummary, MemoryNote, NormalizedMessage, UserImpression
from kairos.runtime import bot_instance
from kairos.telemetry.audit_logger import AuditLogger

_LIN_ID = 10_001
_CHEN_ID = 10_002
_LIN_BOT_ID = "1"
_CHEN_BOT_ID = "2"
_GROUP = 80_001
_HUMAN = 30_001
_TOKEN = "loopback-onebot-token"
_MODEL_KEY = "loopback-model-key"
_GEMINI_KEY = "loopback-gemini-key"
_GEMINI_PATH = "/v1beta/models/gemini-3.5-flash-lite:generateContent"
_CHAT_MODEL = "lin_xiaoman"
_GATE_MODEL = "lin_gate"
_WORKER_MODEL = "lin_worker"
_WAIT = 4.0


class MemoryStore:
    def __init__(self) -> None:
        self.messages: dict[tuple[str, int, str], NormalizedMessage] = {}
        self.audits: asyncio.Queue[AuditRecord] = asyncio.Queue()
        self.summaries: dict[tuple[str, int, str], ContextSummary] = {}
        self.impressions: dict[tuple[str, int, int], UserImpression] = {}
        self.notes: list[MemoryNote] = []
        self.recall_started = asyncio.Event()
        self.release_recall: asyncio.Event | None = None
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

    async def save_summary(self, summary: ContextSummary) -> None:
        key = (summary.bot_id, summary.group_id, summary.covered_until_msg_id)
        async with self._changed:
            self.summaries.setdefault(key, summary)
            self._changed.notify_all()

    async def load_recent_summaries(
        self, bot_id: str, group_id: int, limit: int = 3
    ) -> list[ContextSummary]:
        scoped = [
            item
            for (bot, group, _), item in self.summaries.items()
            if (bot, group) == (bot_id, group_id)
        ]
        return sorted(scoped, key=lambda item: item.covered_until_at)[-limit:] if limit > 0 else []

    async def upsert_user_impression(self, impression: UserImpression) -> None:
        self.impressions[(impression.bot_id, impression.group_id, impression.user_qq)] = impression

    async def load_user_impressions(
        self, bot_id: str, group_id: int, user_qqs: Sequence[int]
    ) -> list[UserImpression]:
        return [
            item
            for (bot, group, user), item in self.impressions.items()
            if (bot, group) == (bot_id, group_id) and user in user_qqs
        ]

    async def save_memory_note(self, note: MemoryNote) -> None:
        self.notes.append(note)

    async def recall_memory_notes(
        self,
        bot_id: str,
        group_id: int,
        keywords: Sequence[str],
        query_text: str = "",
        limit: int = 5,
    ) -> list[MemoryNote]:
        if self.release_recall is not None:
            self.recall_started.set()
            await self.release_recall.wait()
        return [
            note for note in self.notes if (note.bot_id, note.scope_group_id) == (bot_id, group_id)
        ][:limit]

    async def first_summary(self, bot_id: str, group_id: int) -> ContextSummary:
        async with asyncio.timeout(_WAIT), self._changed:
            await self._changed.wait_for(
                lambda: any(key[:2] == (bot_id, group_id) for key in self.summaries)
            )
            return (await self.load_recent_summaries(bot_id, group_id, 1))[0]

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
    status: int = 200
    reasoning: str | None = None


class ModelStub:
    def __init__(self) -> None:
        self.planned: deque[Completion] = deque()
        self._per_model: dict[str, deque[Completion]] = defaultdict(deque)
        self.received: list[dict[str, Any]] = []
        self._requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def complete(self, request: web.Request) -> web.Response:
        gemini = request.path == _GEMINI_PATH
        header, expected = (
            ("x-goog-api-key", _GEMINI_KEY) if gemini else ("Authorization", f"Bearer {_MODEL_KEY}")
        )
        if request.headers.get(header) != expected:
            raise web.HTTPUnauthorized()
        payload = await request.json()
        self.received.append(payload)
        self._requests.put_nowait(payload)
        planned = self._per_model.get(payload.get("model"), self.planned)
        completion = planned.popleft() if planned else Completion("unexpected reply")
        deadline = asyncio.get_running_loop().time() + completion.delay
        if completion.release is not None:
            try:
                await asyncio.wait_for(completion.release.wait(), _WAIT)
            except TimeoutError:
                raise web.HTTPGatewayTimeout() from None
        await asyncio.sleep(max(0, deadline - asyncio.get_running_loop().time()))
        if completion.status != 200:
            return web.json_response({"error": completion.text}, status=completion.status)
        if gemini:
            parts = []
            if completion.reasoning:
                parts.append({"thought": True, "text": completion.reasoning})
            parts.append({"text": completion.text})
            return web.json_response({"candidates": [{"content": {"parts": parts}}]})
        return web.json_response({"choices": [{"message": {"content": completion.text}}]})

    async def next_request(self) -> dict[str, Any]:
        return await asyncio.wait_for(self._requests.get(), _WAIT)

    def plan(self, model: str, *replies: str) -> None:
        self._per_model[model].extend(Completion(reply) for reply in replies)


@dataclass
class SentMessage:
    msg_id: int
    segments: list[dict[str, Any]]


class OneBotPeer:
    def __init__(self, self_id: int, socket: aiohttp.ClientWebSocketResponse) -> None:
        self.self_id = self_id
        self._socket = socket
        self._sends: asyncio.Queue[SentMessage] = asyncio.Queue()
        self._reactions: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._ids = count(self_id * 1000 + 1)

    async def receive_actions(self) -> None:
        async for frame in self._socket:
            assert frame.type is aiohttp.WSMsgType.TEXT
            action = frame.json()
            if action["action"] == "set_msg_emoji_like":
                await self._socket.send_json(
                    {"status": "ok", "retcode": 0, "data": {"result": True}, "echo": action["echo"]}
                )
                self._reactions.put_nowait(action)
                continue
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

    async def send_event(self, event: dict[str, Any]) -> None:
        await self._socket.send_json({"self_id": self.self_id, **event})

    async def group_message(
        self,
        msg_id: int,
        message: str | list[dict[str, Any]],
        *,
        sender: int = _HUMAN,
        timestamp: int | None = None,
    ) -> None:
        await self.send_event(
            {
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

    async def next_reaction(self) -> dict[str, Any]:
        return await asyncio.wait_for(self._reactions.get(), _WAIT)

    async def expect_no_send(self) -> None:
        with pytest.raises(TimeoutError):
            await self.next_send(timeout=0.25)


@dataclass
class Replay:
    model: ModelStub
    store: MemoryStore
    manager: MultiBotManager
    lin: OneBotPeer
    chen: OneBotPeer
    close: Callable[[], Awaitable[None]]


def _config_files(
    tmp_path: Path,
    api_port: int,
    normal: float,
    trigger_mode: str | None,
    *,
    gemini: bool,
    debug: bool,
    lin_options: dict[str, Any],
) -> Path:
    global_path = tmp_path / "settings.yaml"
    global_path.write_text(
        yaml.safe_dump(
            {
                "postgres": {"dsn": "postgresql://unused"},
                "onebot_ws": {"port": 0, "access_token": "${KAIROS_E2E_WS_TOKEN}"},
            }
        ),
        encoding="utf-8",
    )
    bots = tmp_path / "bots"
    bots.mkdir()
    for bot_id, self_id, nickname, model_key in (
        (_LIN_BOT_ID, _LIN_ID, "林小满", _CHAT_MODEL),
        (_CHEN_BOT_ID, _CHEN_ID, "陈阿呆", "chen_adai"),
    ):
        model = {
            "provider": "openai_compatible",
            "base_url": f"http://127.0.0.1:{api_port}/v1",
            "api_key": "${KAIROS_E2E_MODEL_KEY}",
            "model": model_key,
            "max_retries": 0,
        }
        models = {"chat": model}
        options = lin_options if bot_id == _LIN_BOT_ID else {}
        for key_field, worker_model in (
            ("gate_model_key", _GATE_MODEL),
            ("worker_model_key", _WORKER_MODEL),
        ):
            if key_field in options:
                models[options[key_field]] = {**model, "model": worker_model}
        if gemini and bot_id == _LIN_BOT_ID:
            model.update(
                provider="google_ai_studio",
                base_url=f"http://127.0.0.1:{api_port}/v1beta",
                api_key="${KAIROS_E2E_GEMINI_KEY}",
                model="gemini-3.5-flash-lite",
                thinking_level="minimal",
            )
        (bots / f"{bot_id}.yaml").write_text(
            yaml.safe_dump(
                {
                    "self_id": self_id,
                    "nickname": nickname,
                    "persona_prompt": f"我是{nickname}。",
                    "llm": models,
                    "allowed_groups": [_GROUP],
                    **({"trigger_mode": trigger_mode} if trigger_mode is not None else {}),
                    "debug_groups": [_GROUP] if debug and bot_id == _LIN_BOT_ID else [],
                    "debounce": {"normal_seconds": normal, "mention_seconds": 0.015},
                    "rate_limit": {"cooldown_seconds": 0, "peer_bot_max_turns": 2},
                    **options,
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    (bots / "3.yaml").write_text("self_id: ${KAIROS_UNUSED_BOT}", encoding="utf-8")
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
    monkeypatch.setenv("KAIROS_E2E_GEMINI_KEY", _GEMINI_KEY)
    monkeypatch.setenv("KAIROS_ENABLED_BOTS", f"{_LIN_BOT_ID},{_CHEN_BOT_ID}")
    case_numbers = count()
    async with AsyncExitStack() as fixture_resources:

        async def start(
            normal: float = 0.08,
            trigger_mode: str | None = "all_with_veto",
            *,
            status_messages: bool = False,
            gemini: bool = False,
            debug: bool = False,
            lin_options: dict[str, Any] | None = None,
            store: MemoryStore | None = None,
        ) -> Replay:
            resources = await fixture_resources.enter_async_context(AsyncExitStack())
            model = ModelStub()
            app = web.Application()
            app.router.add_post("/v1/chat/completions", model.complete)
            app.router.add_post(_GEMINI_PATH, model.complete)
            runner = web.AppRunner(app, shutdown_timeout=1)
            await runner.setup()
            resources.push_async_callback(runner.cleanup)
            await web.TCPSite(runner, "127.0.0.1", 0).start()

            directory = tmp_path / f"replay-{next(case_numbers)}"
            directory.mkdir()
            config = load_global_config(
                _config_files(
                    directory,
                    runner.addresses[0][1],
                    normal,
                    trigger_mode,
                    gemini=gemini,
                    debug=debug,
                    lin_options=lin_options or {},
                )
            )
            profiles = load_bot_profiles(directory / "bots", config)
            assert {profile.self_id for profile in profiles} == {_LIN_ID, _CHEN_ID}
            store = store if store is not None else MemoryStore()
            llm = LLMRouter(
                config.llm_models, transport=httpx.AsyncHTTPTransport(http2=True, trust_env=False)
            )
            resources.push_async_callback(llm.aclose)
            audit = AuditLogger(store)
            resources.push_async_callback(audit.drain)
            gateway = OneBotWSGateway(config.onebot_ws)
            manager = MultiBotManager(
                profiles,
                store,
                llm,
                gateway,
                audit,
                bot_llm_clients={
                    profile.bot_id: llm.for_models(profile.llm) for profile in profiles
                },
                summary_store=store,
                memory_store=store,
            )
            resources.push_async_callback(manager.shutdown)
            gateway.on_group_event = manager.handle_group_event
            if status_messages:
                gateway.on_bot_online = manager.handle_bot_online
                gateway.on_group_poke = manager.handle_group_poke
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
            return Replay(model, store, manager, *peers, resources.aclose)

        yield start


def _text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "data": {"text": content}}]


async def test_bots_can_share_a_slot_name_with_different_formats_and_keys(make_replay) -> None:
    case = await make_replay(trigger_mode="mention_only", gemini=True)
    case.model.planned.extend([Completion("我在呢"), Completion("我也来了")])

    await case.lin.group_message(11, f"[CQ:at,qq={_LIN_ID}]来打个招呼")
    gemini = await case.model.next_request()
    assert "我是林小满" in gemini["systemInstruction"]["parts"][0]["text"]
    assert gemini["contents"][0]["role"] == "user"
    assert gemini["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}
    assert (await case.lin.next_send()).segments == _text("我在呢")

    await case.chen.group_message(12, f"[CQ:at,qq={_CHEN_ID}]你也来打个招呼")
    chat = await case.model.next_request()
    assert "我是陈阿呆" in chat["messages"][0]["content"]
    assert chat["model"] == "chen_adai"
    assert (await case.chen.next_send()).segments == _text("我也来了")


async def test_online_and_group_poke_send_status_messages(
    make_replay, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bot_instance, "_online_greeting", lambda _hour: "中午好")
    case = await make_replay(status_messages=True, trigger_mode="mention_only")
    assert (await case.lin.next_send()).segments == _text("中午好")
    assert (await case.chen.next_send()).segments == _text("中午好")

    await case.lin.send_event({"post_type": "meta_event", "meta_event_type": "lifecycle"})
    poke = {"post_type": "notice", "notice_type": "notify", "sub_type": "poke", "user_id": _HUMAN}
    await case.lin.send_event({**poke, "target_id": _LIN_ID})
    await case.lin.send_event({**poke, "group_id": _GROUP, "target_id": _CHEN_ID})
    await case.lin.send_event({**poke, "group_id": _GROUP + 1, "target_id": _LIN_ID})
    await case.lin.send_event(
        {**poke, "group_id": _GROUP, "target_id": _LIN_ID, "user_id": _LIN_ID}
    )
    await case.lin.expect_no_send()

    for _ in range(2):
        await case.lin.send_event({**poke, "group_id": _GROUP, "target_id": _LIN_ID})
        assert (await case.lin.next_send()).segments == _text("QAQ")
    await case.chen.expect_no_send()
    assert case.model.received == []


@pytest.mark.parametrize("mention", [False, True])
async def test_name_and_at_share_reply_and_veto_debugging(make_replay, mention: bool) -> None:
    case = await make_replay(trigger_mode=None, gemini=True, debug=True)
    case.model.planned.extend([Completion("我在。"), Completion("<|act:silent|>")])
    await case.lin.group_message(61, _text("大家猜猜几点了"))
    await case.lin.expect_no_send()
    assert case.model.received == []

    await case.lin.group_message(62, _text("林小满，你在吗？"))
    named_request = await case.model.next_request()
    assert "<|reason|>" not in named_request["systemInstruction"]["parts"][0]["text"]
    assert (await case.lin.next_send()).segments == _text("我在。")
    assert (await case.store.next_audit()).final_reply == "我在。"

    text = f"[CQ:at,qq={_LIN_ID}]觉得呢？" if mention else _text("林小满觉得呢？")
    await case.lin.group_message(63, text)
    await case.lin.group_message(64, _text("刚才已经有人回答了"), sender=_HUMAN + 1)
    debug_request = await case.model.next_request()
    instruction = debug_request["systemInstruction"]["parts"][0]["text"]
    assert "仅输出 <|act:silent|>" in instruction
    sent = await case.lin.next_send()
    assert sent.segments == [{"type": "reply", "data": {"id": "63"}}] + _text(
        "这条触发成功但是模型选择沉默"
    )
    decision = await case.store.next_audit()
    assert decision.main_vetoed and decision.final_reply is None
    assert (_LIN_BOT_ID, _GROUP, str(sent.msg_id)) not in case.store.messages
    await case.lin.expect_no_send()
    assert len(case.model.received) == 2


async def test_model_endpoint_error_is_delivered_to_a_group_without_debug(make_replay) -> None:
    case = await make_replay(trigger_mode=None)
    case.model.planned.append(Completion("upstream detail stays private", status=503))

    await case.lin.group_message(65, _text("林小满，现在方便回答吗？"))

    send = await case.lin.next_send()
    assert send.segments == [{"type": "reply", "data": {"id": "65"}}] + _text(
        "【Kairos 错误】模型接口返回 HTTP 503。"
    )
    await case.lin.expect_no_send()


async def test_long_gemini_reply_and_silence_notice_keep_context_clean(make_replay) -> None:
    case = await make_replay(trigger_mode=None, gemini=True, debug=True)
    long_reply = "我在。\n\n" + "这是一段可以完整发出去的说明。" * 120
    reply_thought = "直接问候适合回一句。"
    silent_thought = "话题已经由另一位群友回答。"
    case.model.planned.extend(
        [
            Completion(long_reply, reasoning=reply_thought),
            Completion("<|act:silent|>", reasoning=silent_thought),
            Completion("我也在。"),
        ]
    )

    await case.lin.group_message(71, _text("林小满，你在吗？"))
    first = await case.model.next_request()
    assert first["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": "MINIMAL"},
    }
    reply = await case.lin.next_send()
    assert reply.segments == _text(long_reply)
    (saved,) = await case.store.written(_LIN_BOT_ID, reply.msg_id)
    assert saved.content == long_reply

    await case.lin.group_message(72, _text("林小满，再确认一下？"))
    later = await case.model.next_request()
    assert [item["parts"] for item in later["contents"] if item["role"] == "model"] == [
        [{"text": long_reply}]
    ]
    assert all(reply_thought not in str(item) for item in later["contents"])
    silent_debug = await case.lin.next_send()
    assert silent_debug.segments == [{"type": "reply", "data": {"id": "72"}}] + _text(
        "这条触发成功但是模型选择沉默"
    )

    await case.chen.group_message(silent_debug.msg_id, silent_debug.segments, sender=_LIN_ID)
    await case.chen.group_message(73, _text("陈阿呆，你在吗？"))
    peer = await case.model.next_request()
    assert all(
        "这条触发成功但是模型选择沉默" not in message["content"] for message in peer["messages"]
    )
    assert (await case.chen.next_send()).segments == _text("我也在。")
    assert not any(key[2] == str(silent_debug.msg_id) for key in case.store.messages)


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
    assert (audit.bot_id, audit.final_reply) == (_LIN_BOT_ID, first_reply)
    raw_a, raw_b, own = await case.store.written(_LIN_BOT_ID, 21, 22, first.msg_id)
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
    assert (await case.store.next_audit()).bot_id == _LIN_BOT_ID

    await case.chen.group_message(24, _text("屋里开空调了吗？"))
    request = await case.model.next_request()
    assert request["model"] == "chen_adai"
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert "我是陈阿呆。" in request["messages"][0]["content"]
    assert "[m24]" in request["messages"][1]["content"]
    assert "[m21]" not in request["messages"][1]["content"]
    other = await case.chen.next_send()
    assert other.segments == _text("屋里很凉快。")
    assert (await case.store.next_audit()).bot_id == _CHEN_BOT_ID
    (own_other,) = await case.store.written(_CHEN_BOT_ID, other.msg_id)
    assert own_other.is_bot and own_other.sender_qq == _CHEN_ID
    assert (_CHEN_BOT_ID, _GROUP, str(first.msg_id)) not in case.store.messages


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
    await case.store.written(_LIN_BOT_ID, *ids)
    release.set()

    audit = await case.store.next_audit()
    if arrivals == 3:
        assert audit.stale_action == "SEND_WITH_REPLY"
        sent = await case.lin.next_send()
        assert sent.segments == [{"type": "reply", "data": {"id": "42"}}] + _text(
            "回一下刚才那句。"
        )
        (saved,) = await case.store.written(_LIN_BOT_ID, sent.msg_id)
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

    (peer_input,) = await case.store.written(_CHEN_BOT_ID, sent.msg_id)
    assert peer_input.is_peer_bot and not peer_input.is_bot
    await asyncio.gather(case.lin.expect_no_send(), case.chen.expect_no_send())
    assert len(case.model.received) == len(replies)
    assert case.manager.runtimes[_CHEN_ID].sessions[_GROUP].consecutive_peer_bot_turns == 2


def _gate_options(**settings: Any) -> dict[str, Any]:
    return {
        "gate_model_key": "gate",
        "gate": {
            "enabled": True,
            "persona_card": "爱聊做饭",
            "energy_recover_per_min": 0,
            **settings,
        },
    }


async def test_m2_model_gate_silence_bypasses_chat_and_onebot_actions(make_replay) -> None:
    case = await make_replay(lin_options=_gate_options())
    case.model.plan(_GATE_MODEL, '{"action":"silent","reason":"随意闲聊"}')

    await case.lin.group_message(101, _text("我在等车。"))
    audit = await case.store.next_audit()

    assert (audit.stale_action, audit.action_type) == ("GATE_SILENT", "silent")
    assert audit.gate_output["reason"] == "随意闲聊"
    assert [request["model"] for request in case.model.received] == [_GATE_MODEL]
    assert case.model.received[0]["response_format"] == {"type": "json_object"}
    await case.lin.expect_no_send()
    assert case.lin._reactions.empty()


async def test_m2_gate_reacts_over_websocket_and_persists_only_the_action_audit(
    make_replay,
) -> None:
    case = await make_replay(lin_options=_gate_options())
    case.model.plan(_GATE_MODEL, '{"action":"react","candidate_target":"m101","emoji_id":"76"}')

    await case.lin.group_message(101, _text("今晚终于吃到云吞了！"))
    reaction = await case.lin.next_reaction()
    audit = await case.store.next_audit()
    await case.store.written(_LIN_BOT_ID, 101)

    assert reaction["action"] == "set_msg_emoji_like" and reaction["echo"]
    assert reaction["params"] == {"message_id": "101", "emoji_id": "76", "set": True}
    assert (audit.stale_action, audit.action_type, audit.target_msg_id, audit.emoji_id) == (
        "GATE_REACT",
        "react",
        "101",
        "76",
    )
    assert [request["model"] for request in case.model.received] == [_GATE_MODEL]
    assert [key[2] for key in case.store.messages] == ["101"]
    assert [
        message.msg_id for message in case.manager.runtimes[_LIN_ID].sessions[_GROUP].buffer
    ] == ["101"]
    await case.lin.expect_no_send()


@pytest.mark.parametrize("answer", ["<|act:silent|>", "<|act:react:m102:66|>"])
async def test_m2_chat_model_can_veto_or_react_after_gate_pass(make_replay, answer: str) -> None:
    case = await make_replay(lin_options=_gate_options(), debug=True)
    case.model.plan(_GATE_MODEL, '{"action":"pass","candidate_target":"m102","reason":"可以接话"}')
    case.model.plan(_CHAT_MODEL, answer)

    await case.lin.group_message(102, _text("你们想吃什么？"))
    audit = await case.store.next_audit()
    if answer == "<|act:silent|>":
        notice = await case.lin.next_send()
        assert notice.segments == [
            {"type": "reply", "data": {"id": "102"}},
            *_text("这条触发成功但是模型选择沉默"),
        ]
        assert audit.main_vetoed and audit.stale_action == "VETOED"
        assert case.lin._reactions.empty()
    else:
        reaction = await case.lin.next_reaction()
        assert reaction["action"] == "set_msg_emoji_like"
        assert reaction["params"] == {"message_id": "102", "emoji_id": "66", "set": True}
        assert (audit.stale_action, audit.action_type, audit.target_msg_id, audit.emoji_id) == (
            "MAIN_REACT",
            "react",
            "102",
            "66",
        )
        await case.lin.expect_no_send()
    assert [request["model"] for request in case.model.received] == [_GATE_MODEL, _CHAT_MODEL]
    prompt = case.model.received[1]["messages"][0]["content"]
    assert "初筛建议目标 m102" in prompt and "仅供参考" in prompt
    await case.store.written(_LIN_BOT_ID, 102)
    assert [key[2] for key in case.store.messages] == ["102"]


async def test_m2_over_budget_context_is_summarized_and_restored_across_restart(
    make_replay,
) -> None:
    options = {
        "worker_model_key": "worker",
        "compaction": {
            "enabled": True,
            "input_token_budget": 512,
            "trigger_ratio": 1.0,
            "summary_max_tokens": 64,
            "keep_recent_messages": 2,
            "max_persisted_summaries": 2,
        },
    }
    first_run = await make_replay(trigger_mode="mention_only", lin_options=options)
    summary_text = "群友讨论了两件旧事，随后约好周五见。"
    first_run.model.plan(_WORKER_MODEL, summary_text)
    timestamp = int(time.time()) - 60
    for offset, content in enumerate(
        (
            "旧记录甲" + "甲" * 182,
            "旧记录乙" + "乙" * 182,
            "续接第一条，订好时间。",
            "续接第二条，" + "丙" * 70,
        )
    ):
        msg_id = 91 + offset
        await first_run.lin.group_message(
            msg_id, _text(content), sender=_HUMAN + offset, timestamp=timestamp + offset
        )
        await first_run.store.written(_LIN_BOT_ID, msg_id)

    summary = await first_run.store.first_summary(_LIN_BOT_ID, _GROUP)
    assert summary.summary_text == summary_text and summary.covered_until_msg_id == "92"
    assert summary.covered_until_at == datetime.fromtimestamp(timestamp + 1, UTC)
    (worker_request,) = first_run.model.received
    assert worker_request["model"] == _WORKER_MODEL and worker_request["max_tokens"] == 64
    assert "旧记录甲" in worker_request["messages"][1]["content"]
    assert "旧记录乙" in worker_request["messages"][1]["content"]
    assert "续接第一条" not in worker_request["messages"][1]["content"]
    await first_run.close()

    rebuilt = await make_replay(
        trigger_mode="mention_only", lin_options=options, store=first_run.store
    )
    rebuilt.model.plan(_CHAT_MODEL, "好呀，周五见。")
    await rebuilt.lin.group_message(
        105, f"[CQ:at,qq={_LIN_ID}]最后核对一下。", timestamp=timestamp + 10
    )
    sent = await rebuilt.lin.next_send()
    assert sent.segments == _text("好呀，周五见。")
    audit = await rebuilt.store.next_audit()
    prompt = audit.context_snapshot
    assert prompt[0]["role"] == "system" and summary_text in prompt[0]["content"]
    transcript = "\n".join(message["content"] for message in prompt[1:])
    assert "旧记录甲" not in transcript and "旧记录乙" not in transcript
    assert (
        transcript.index("续接第一条")
        < transcript.index("续接第二条")
        < transcript.index("最后核对一下")
    )
    assert rebuilt.manager.runtimes[_LIN_ID] is not first_run.manager.runtimes[_LIN_ID]


async def test_m2_chat_reads_warmed_impressions_and_notes_during_slow_recall(make_replay) -> None:
    case = await make_replay(
        trigger_mode="mention_only",
        lin_options={"memory": {"enabled": True, "warm_trigger_messages": 1}},
    )
    await case.store.upsert_user_impression(
        UserImpression(
            bot_id=_LIN_BOT_ID,
            group_id=_GROUP,
            user_qq=_HUMAN,
            nickname="阿明",
            impression_summary="喜欢记录食谱",
            updated_at=datetime.now(UTC),
        )
    )
    await case.store.save_memory_note(
        MemoryNote(
            bot_id=_LIN_BOT_ID,
            scope_group_id=_GROUP,
            entity_qqs=[_HUMAN],
            summary_text="大家约过周五试做云吞",
            keywords=["云吞"],
            created_at=datetime.now(UTC),
        )
    )
    await case.lin.group_message(110, _text("今天来聊云吞。"))
    worker = case.manager.memory_worker
    assert worker is not None
    async with asyncio.timeout(_WAIT):
        while not worker.get_warm_memory(_LIN_BOT_ID, _GROUP).notes:
            await asyncio.sleep(0)
    assert case.model.received == []

    case.store.release_recall = asyncio.Event()
    case.model.plan(_CHAT_MODEL, "要不要一起试试？")
    try:
        await case.lin.group_message(111, f"[CQ:at,qq={_LIN_ID}]来聊聊云吞。")
        await asyncio.wait_for(case.store.recall_started.wait(), _WAIT)
        sent = await case.lin.next_send()
        assert sent.segments == _text("要不要一起试试？")
        (request,) = case.model.received
        prompt = request["messages"][0]["content"]
        assert "阿明：喜欢记录食谱" in prompt and "大家约过周五试做云吞" in prompt
        assert (await case.store.next_audit()).final_reply is not None
    finally:
        case.store.release_recall.set()


async def test_m2_energy_limits_ambient_chat_and_direct_mentions_still_pass(make_replay) -> None:
    case = await make_replay(normal=0.03, lin_options=_gate_options(energy_max=50))
    gate_pass = json.dumps({"action": "pass", "reason": "接得上"})
    case.model.plan(_GATE_MODEL, gate_pass, gate_pass)
    case.model.plan(_CHAT_MODEL, "我去买云吞。", "今天改骑车吧。", "我在。")
    for msg_id, text in ((120, "聊晚饭。"), (121, "聊出行。")):
        await case.lin.group_message(msg_id, _text(text))
        await case.lin.next_send()
        assert (await case.store.next_audit()).stale_action == "SEND_DIRECT"
    assert case.manager.runtimes[_LIN_ID].sessions[_GROUP].energy == 0

    with capture_logs() as logs:
        await case.lin.group_message(122, _text("有人想聊天吗？"))
        await case.store.written(_LIN_BOT_ID, 122)
        await case.lin.expect_no_send()
    assert any(
        log["event"] == "group_trigger_filtered" and log["reason"] == "low_energy" for log in logs
    )
    assert [request["model"] for request in case.model.received] == [_GATE_MODEL, _CHAT_MODEL] * 2

    await case.lin.group_message(123, f"[CQ:at,qq={_LIN_ID}]在吗？")
    assert (await case.lin.next_send()).segments == _text("我在。")
    audit = await case.store.next_audit()
    assert audit.gate_output["reason"] == "direct_address" and audit.final_reply == "我在。"
    assert [request["model"] for request in case.model.received] == [
        _GATE_MODEL,
        _CHAT_MODEL,
    ] * 2 + [_CHAT_MODEL]
