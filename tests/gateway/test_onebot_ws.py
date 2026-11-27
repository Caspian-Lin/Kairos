import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from aiohttp import ClientSession, WSMsgType, WSServerHandshakeError

from kairos.config import OneBotWSConfig
from kairos.gateway.onebot_ws import BotOfflineError, OneBotSendError, OneBotWSGateway
from kairos.models import MessageSegment, RawGroupEvent


@asynccontextmanager
async def _gateway(
    callback: Callable[[RawGroupEvent], Awaitable[None]] | None = None,
    **config: Any,
) -> AsyncIterator[tuple[OneBotWSGateway, ClientSession, str]]:
    settings = OneBotWSConfig(port=0, **config)
    gateway = OneBotWSGateway(settings, callback)
    await gateway.start()
    try:
        async with ClientSession() as client:
            yield gateway, client, f"http://127.0.0.1:{gateway.port}{settings.path}"
    finally:
        await gateway.stop()


def _message(self_id: int, msg_id: int, **overrides: Any) -> dict[str, Any]:
    return {
        "self_id": self_id,
        "post_type": "message",
        "message_type": "group",
        "group_id": 300,
        "message_id": msg_id,
        "user_id": 201,
        "sender": {"nickname": "测试群友", "card": "测试名片"},
        "time": 1_700_000_000,
        "message": [{"type": "text", "data": {"text": "hello"}}],
        **overrides,
    }


def _receipt(request: dict[str, Any], msg_id: int) -> dict[str, Any]:
    return {"echo": request["echo"], "status": "ok", "data": {"message_id": msg_id}}


async def test_authentication_with_header_and_query() -> None:
    async with _gateway(access_token="fake-test-token", path="/reverse") as (
        gateway,
        client,
        url,
    ):
        for headers in ({}, {"Authorization": "Bearer wrong-token"}):
            with pytest.raises(WSServerHandshakeError) as refused:
                await client.ws_connect(url, headers=headers)
            assert refused.value.status == 401

        async with (
            client.ws_connect(
                url, headers={"Authorization": "Bearer fake-test-token", "X-Self-ID": "101"}
            ),
            client.ws_connect(f"{url}?access_token=fake-test-token", headers={"X-Self-ID": "102"}),
        ):
            assert gateway.connected_self_ids == {101, 102}


async def test_two_bots_share_a_port_and_accept_array_and_cq_frames() -> None:
    events: asyncio.Queue[RawGroupEvent] = asyncio.Queue()
    async with (
        _gateway(events.put) as (gateway, client, url),
        client.ws_connect(url, headers={"X-Self-ID": "101"}) as peer_a,
        client.ws_connect(url, headers={"X-Self-ID": "102"}) as peer_b,
    ):
        cq = "[CQ:reply,id=89] 回应"
        await peer_a.send_json(_message(101, 501, raw_message="hello"))
        await peer_b.send_json(_message(102, 502, message=cq))
        async with asyncio.timeout(1):
            by_bot = {event.self_id: event for event in [await events.get(), await events.get()]}

        assert gateway.connected_self_ids == {101, 102}
        assert by_bot[101] == RawGroupEvent(
            self_id=101,
            group_id=300,
            msg_id="501",
            sender_qq=201,
            sender_nickname="测试群友",
            sender_card="测试名片",
            segments=[MessageSegment(type="text", data={"text": "hello"})],
            raw_message="hello",
            timestamp=datetime.fromtimestamp(1_700_000_000, UTC),
        )
        assert by_bot[102].msg_id == "502"
        assert by_bot[102].raw_message == cq
        assert by_bot[102].segments == []


async def test_bot_online_callback_runs_after_each_connection_registers() -> None:
    online: asyncio.Queue[int] = asyncio.Queue()
    async with _gateway() as (gateway, client, url):
        gateway.on_bot_online = online.put
        async with client.ws_connect(url, headers={"X-Self-ID": "101"}) as first:
            assert await asyncio.wait_for(online.get(), 1) == 101
            await first.send_json({"self_id": 101, "post_type": "meta_event"})
        async with client.ws_connect(url) as second:
            assert online.empty()
            await second.send_json({"self_id": 101, "post_type": "meta_event"})
            assert await asyncio.wait_for(online.get(), 1) == 101


async def test_first_frame_registration_and_invalid_events_are_isolated() -> None:
    events: asyncio.Queue[RawGroupEvent] = asyncio.Queue()
    failed_callback = asyncio.Event()

    async def consume(event: RawGroupEvent) -> None:
        if event.msg_id == "506":
            failed_callback.set()
            raise RuntimeError("event handler failed")
        await events.put(event)

    async with _gateway(consume) as (gateway, client, url), client.ws_connect(url) as peer:
        assert gateway.connected_self_ids == set()
        await peer.send_str("{invalid")
        await peer.send_json(["not an object"])
        await peer.send_json({"post_type": "meta_event", "self_id": 101})
        await peer.send_json(_message(102, 503))
        await peer.send_json(_message(101, 504, message_type="private"))
        await peer.send_json(_message(101, 505, time="bad timestamp"))
        await peer.send_json(_message(101, 506))
        await peer.send_json(_message(101, 507))
        async with asyncio.timeout(1):
            await failed_callback.wait()
            assert (await events.get()).msg_id == "507"
        assert gateway.connected_self_ids == {101}
        assert events.empty()
        assert not peer.closed


async def test_concurrent_echoes_match_in_reverse_order_and_ignore_foreign_peer() -> None:
    events: asyncio.Queue[RawGroupEvent] = asyncio.Queue()
    async with (
        _gateway(events.put) as (gateway, client, url),
        client.ws_connect(url, headers={"X-Self-ID": "101"}) as peer_a,
        client.ws_connect(url, headers={"X-Self-ID": "102"}) as peer_b,
        asyncio.TaskGroup() as tasks,
    ):
        direct = tasks.create_task(gateway.send_group_msg(101, 300, "direct"))
        reply = tasks.create_task(gateway.send_group_msg(101, 301, "quoted", "505"))
        requests = [await peer_a.receive_json(timeout=1), await peer_a.receive_json(timeout=1)]
        by_group = {request["params"]["group_id"]: request for request in requests}
        assert by_group[300]["echo"] != by_group[301]["echo"]
        assert by_group[300]["action"] == by_group[301]["action"] == "send_group_msg"
        assert by_group[300]["params"]["message"] == [{"type": "text", "data": {"text": "direct"}}]
        assert by_group[301]["params"]["message"] == [
            {"type": "reply", "data": {"id": "505"}},
            {"type": "text", "data": {"text": "quoted"}},
        ]

        await peer_b.send_json(_receipt(by_group[300], 999))
        await peer_b.send_json(_message(102, 509))
        assert (await asyncio.wait_for(events.get(), timeout=1)).self_id == 102
        assert not direct.done()

        await peer_a.send_json(_receipt(by_group[301], 702))
        assert await asyncio.wait_for(reply, timeout=1) == "702"
        assert not direct.done()
        await peer_a.send_json(_receipt(by_group[300], 701))
        assert await asyncio.wait_for(direct, timeout=1) == "701"
        assert not gateway.pending_echoes


@pytest.mark.parametrize("reconnect", [False, True])
async def test_disconnect_and_reconnect_fail_only_old_pending_echoes(reconnect: bool) -> None:
    async with (
        _gateway() as (gateway, client, url),
        client.ws_connect(url, headers={"X-Self-ID": "101"}) as old_peer,
        asyncio.TaskGroup() as tasks,
    ):

        async def send_on_old_socket() -> None:
            with pytest.raises(BotOfflineError, match="disconnected"):
                await gateway.send_group_msg(101, 300, "old connection")

        async def verify_disconnected() -> None:
            await asyncio.wait_for(disconnected, timeout=1)
            assert not gateway.pending_echoes

        disconnected = tasks.create_task(send_on_old_socket())
        await old_peer.receive_json(timeout=1)
        if reconnect:
            async with client.ws_connect(url, headers={"X-Self-ID": "101"}) as replacement:
                assert (await old_peer.receive(timeout=1)).type == WSMsgType.CLOSE
                await verify_disconnected()
                assert gateway.connected_self_ids == {101}
                sent = tasks.create_task(gateway.send_group_msg(101, 300, "new connection"))
                await replacement.send_json(
                    _receipt(await replacement.receive_json(timeout=1), 703)
                )
                assert await asyncio.wait_for(sent, timeout=1) == "703"
        else:
            await old_peer.close()
            await verify_disconnected()
            assert not gateway.connected_self_ids
            with pytest.raises(BotOfflineError, match="offline"):
                await gateway.send_group_msg(101, 300, "offline")


async def test_timeout_and_cancellation_clean_pending_echoes() -> None:
    async with (
        _gateway(send_timeout_seconds=0.1) as (gateway, client, url),
        client.ws_connect(url, headers={"X-Self-ID": "101"}) as peer,
        asyncio.TaskGroup() as tasks,
    ):

        async def send_without_receipt() -> None:
            with pytest.raises(OneBotSendError, match="Timed out"):
                await gateway.send_group_msg(101, 300, "timeout")

        timed_out = tasks.create_task(send_without_receipt())
        await asyncio.wait_for(timed_out, timeout=1)
        await peer.receive_json(timeout=1)
        assert not gateway.pending_echoes

        cancelled = tasks.create_task(gateway.send_group_msg(101, 300, "cancelled"))
        await peer.receive_json(timeout=1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert not gateway.pending_echoes


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({"status": "failed", "retcode": 100}, "rejected"),
        ({"status": "ok", "data": {}}, "without a message ID"),
    ],
)
async def test_failed_or_invalid_receipts_clean_pending_echoes(
    response: dict[str, Any], error: str
) -> None:
    async with (
        _gateway() as (gateway, client, url),
        client.ws_connect(url, headers={"X-Self-ID": "101"}) as peer,
        asyncio.TaskGroup() as tasks,
    ):

        async def send_bad_receipt() -> None:
            with pytest.raises(OneBotSendError, match=error):
                await gateway.send_group_msg(101, 300, "bad receipt")

        send = tasks.create_task(send_bad_receipt())
        action = await peer.receive_json(timeout=1)
        await peer.send_json({**response, "echo": action["echo"]})
        await asyncio.wait_for(send, timeout=1)
        assert not gateway.pending_echoes


async def test_stop_cleans_connections_and_waits_for_cancellation() -> None:
    started = asyncio.Event()
    finished = asyncio.Event()

    async def consume(_event: RawGroupEvent) -> None:
        started.set()
        try:
            await asyncio.Future()
        finally:
            finished.set()

    async with (
        _gateway(consume) as (gateway, client, url),
        client.ws_connect(url, headers={"X-Self-ID": "101"}) as peer,
        client.ws_connect(url) as unregistered_peer,
        asyncio.TaskGroup() as tasks,
    ):
        await peer.send_json(_message(101, 511))
        await asyncio.wait_for(started.wait(), timeout=1)
        stop = tasks.create_task(gateway.stop())
        assert (await peer.receive(timeout=1)).type == WSMsgType.CLOSE
        assert (await unregistered_peer.receive(timeout=1)).type == WSMsgType.CLOSE
        await asyncio.wait_for(stop, timeout=1)
        assert finished.is_set()
        assert not gateway.connected_self_ids
        assert not gateway.pending_echoes
