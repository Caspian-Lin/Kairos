import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientSession

from kairos import main as entry_point
from kairos.admin.config_editor import BotConfigEditor
from kairos.admin.server import AdminServer
from kairos.config import AdminUIConfig, OneBotWSConfig, load_bot_profiles, load_global_config
from kairos.gateway.onebot_ws import OneBotWSGateway
from kairos.main import MultiBotManager
from kairos.models import PipelineStep, PipelineTrace
from kairos.protocols import LLMClient, MessageStore
from kairos.telemetry.audit_logger import AuditLogger
from kairos.telemetry.log_buffer import LogRingBuffer
from kairos.telemetry.trace_store import PipelineTraceStore

_TOKEN = "test-admin-private-token"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


@pytest.fixture
async def admin(config_tree: tuple[Path, Path]) -> AsyncIterator[tuple[AdminServer, str, Path]]:
    global_path, bots_dir = config_tree
    config = load_global_config(global_path)
    gateway = OneBotWSGateway(OneBotWSConfig(port=0))
    store = Mock(spec=MessageStore)
    traces = PipelineTraceStore(secrets=[_TOKEN, config.postgres.dsn])
    logs = LogRingBuffer(secrets=[_TOKEN, config.postgres.dsn])
    manager = MultiBotManager(
        load_bot_profiles(bots_dir, config),
        store,
        Mock(spec=LLMClient),
        gateway,
        AuditLogger(store),
        trace_store=traces,
    )
    editor = BotConfigEditor(bots_dir, config)
    server = AdminServer(
        AdminUIConfig(enabled=True, port=0, access_token=_TOKEN),
        manager,
        gateway,
        editor,
        logs,
        traces,
        reload_profiles=AsyncMock(),
        request_restart=Mock(),
        secrets=lambda: [_TOKEN, config.postgres.dsn],
    )
    await server.start()
    try:
        yield server, f"http://127.0.0.1:{server.port}", bots_dir
    finally:
        await server.stop()
        await manager.shutdown()


async def test_loopback_auth_and_cross_origin_controls(admin) -> None:
    server, url, _ = admin
    async with ClientSession() as client:
        async with client.get(url) as response:
            assert response.status == 200 and "Kairos 管理台" in await response.text()
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        async with client.get(f"{url}/api/status?access_token={_TOKEN}") as response:
            assert response.status == 401
        async with client.get(f"{url}/api/status", headers=_AUTH) as response:
            payload = await response.json()
            assert payload["bots"][0]["bot_id"] == "1"
            assert payload["connected_self_ids"] == []
            assert payload["configurable_bots"] == ["1"]
            assert _TOKEN not in json.dumps(payload)
        async with client.post(
            f"{url}/api/control/restart",
            headers={**_AUTH, "Origin": "http://foreign.invalid"},
            json={},
        ) as response:
            assert response.status == 403
        async with client.get(
            f"{url}/api/status", headers={**_AUTH, "Host": "foreign.invalid"}
        ) as response:
            assert response.status == 403
    server._request_restart.assert_not_called()


async def test_admin_config_preserves_comments_and_secret_and_validates_draft(admin) -> None:
    _, url, bots_dir = admin
    path = bots_dir / "1.yaml"
    path.write_text(path.read_text().replace("${GEMINI_API_KEY}", "plaintext-private-model-key"))
    initial = path.read_text()
    async with ClientSession(headers=_AUTH) as client:
        async with client.get(f"{url}/api/bots/1/config") as response:
            config = await response.json()
            assert config["llm"]["chat"]["api_key"] == "***"
            assert "plaintext-private-model-key" not in json.dumps(config)
        async with client.put(
            f"{url}/api/bots/1/config", json={"llm": {"chat": {"api_key": "bad"}}}
        ) as response:
            assert response.status == 400
        assert path.read_text() == initial
        draft = {
            "prompt_style": "strict",
            "instruction_reminder": "核对格式",
            "persona_prompt": "草稿示例",
        }
        async with client.post(
            f"{url}/api/bots/1/context-preview", json={"patch": draft}
        ) as response:
            preview = await response.json()
            assert preview["reminder"] == "核对格式"
            assert preview["budget_tokens"] == 65536 and preview["total_tokens"] > 0
            assert {"system_rules", "persona", "runtime_context"} <= {
                part["name"] for part in preview["system_sections"]
            }
            assert "plaintext-private-model-key" not in json.dumps(preview)
        async with client.post(
            f"{url}/api/bots/1/context-preview",
            json={"patch": {"instruction_reminder": "${GEMINI_API_KEY}"}},
        ) as response:
            assert response.status == 200
            payload = await response.text()
            assert "test-gemini-key" not in payload and "[REDACTED]" in payload
        assert path.read_text() == initial
        async with client.put(f"{url}/api/bots/1/config", json=draft) as response:
            assert response.status == 200
            assert (await response.json())["prompt_style"] == "strict"
    updated = path.read_text()
    assert "plaintext-private-model-key" in updated and "# gate_model_key: gate" in updated
    assert "草稿示例" in updated


def _trace() -> PipelineTrace:
    return PipelineTrace(
        trace_id="trace-1",
        bot_id="1",
        group_id=42,
        msg_id="101",
        sender_alias="U1",
        input_preview=_TOKEN,
        started_at=datetime.now(UTC),
        outcome="SEND_DIRECT",
        steps=[PipelineStep(stage="main_llm", status="ok", latency_ms=7)],
        request_context={"turns": [{"role": "user", "content": _TOKEN}], "total_tokens": 10},
    )


async def _next_event(response) -> dict:
    async with asyncio.timeout(2):
        while line := await response.content.readline():
            if line.startswith(b"data: "):
                return json.loads(line.removeprefix(b"data: "))
    raise AssertionError("SSE ended before emitting a record")


async def test_admin_trace_detail_and_bounded_sse_feeds(admin) -> None:
    server, url, _ = admin
    async with ClientSession(headers=_AUTH) as client:
        async with client.get(
            f"{url}/api/logs/stream?level=warning&bot_id=1&query=memory"
        ) as response:
            assert response.headers["Content-Type"] == "text/event-stream"
            server.logs.processor(None, "info", {"event": "memory ignored", "bot_id": "1"})
            server.logs.processor(
                None, "warning", {"event": "memory read", "bot_id": "1", "dsn": _TOKEN}
            )
            event = await _next_event(response)
            assert event["event"] == "memory read" and event["fields"]["dsn"] == "[REDACTED]"
        async with client.get(f"{url}/api/traces/stream?group_id=42") as response:
            server.traces.record(_trace())
            assert (await _next_event(response))["trace_id"] == "trace-1"
        async with client.get(f"{url}/api/traces?bot_id=1&group_id=42") as response:
            (trace,) = await response.json()
            assert trace["outcome"] == "SEND_DIRECT" and trace["steps"][0]["latency_ms"] == 7
        async with client.get(f"{url}/api/traces/trace-1") as response:
            detail = await response.json()
            assert detail["request_context"]["total_tokens"] == 10
            assert _TOKEN not in json.dumps(detail)


async def test_admin_controls_call_reload_and_graceful_restart_hooks(admin) -> None:
    server, url, _ = admin
    async with ClientSession(headers=_AUTH) as client:
        async with client.post(f"{url}/api/control/reload", json={}) as response:
            assert response.status == 200
            assert (await response.json())["ok"]
        async with client.post(f"{url}/api/control/restart", json={}) as response:
            assert response.status == 202
    server._reload_profiles.assert_awaited_once()
    server._request_restart.assert_called_once()


async def test_run_mounts_admin_reloads_yaml_and_shuts_down_before_restart(
    config_tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    global_path, bots_dir = config_tree
    config = load_global_config(global_path)
    config.admin_ui = AdminUIConfig(enabled=True, port=0)
    config.onebot_ws.port = 0
    order: list[str] = []
    store = Mock(
        connect=AsyncMock(), close=AsyncMock(side_effect=lambda: order.append("postgres_closed"))
    )
    monkeypatch.setattr(entry_point, "PgStore", Mock(return_value=store))
    started: asyncio.Queue[AdminServer] = asyncio.Queue()
    original_start = AdminServer.start

    async def capture_server(server: AdminServer) -> None:
        await original_start(server)
        started.put_nowait(server)

    monkeypatch.setattr(AdminServer, "start", capture_server)
    controller = entry_point.RestartController(lambda: order.append("restart"))
    logs = LogRingBuffer()
    service = asyncio.create_task(
        entry_point.run(
            config,
            load_bot_profiles(bots_dir, config),
            bots_dir=bots_dir,
            log_buffer=logs,
            restart_controller=controller,
        )
    )
    try:
        server = await asyncio.wait_for(started.get(), 3)
        url = f"http://127.0.0.1:{server.port}"
        async with ClientSession() as client:
            async with client.put(
                f"{url}/api/bots/1/config", json={"nickname": "重载后的 Bot"}
            ) as response:
                assert response.status == 200
            async with client.post(f"{url}/api/control/reload", json={}) as response:
                assert response.status == 200
            async with client.get(f"{url}/api/status") as response:
                assert (await response.json())["bots"][0]["nickname"] == "重载后的 Bot"
            async with client.post(f"{url}/api/control/restart", json={}) as response:
                assert response.status == 202
        await asyncio.wait_for(service, 3)
        assert order == ["postgres_closed", "restart"]
        assert server._runner is None
    finally:
        if not service.done():
            controller.stop.set()
            await asyncio.wait_for(service, 3)
