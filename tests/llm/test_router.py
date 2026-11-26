import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack

import httpx
import pytest

from kairos.config import LLMModelConfig
from kairos.llm import LLMConfigError, LLMRequestError, LLMResponseError, LLMRouter, router
from kairos.models import ChatMessage

_API_KEY = "dummy-test-secret"
_MESSAGES = [
    ChatMessage(role="system", content="简短回复"),
    ChatMessage(role="user", content="你好"),
]
_CHAT = LLMModelConfig(
    base_url="https://chat.invalid/v1/",
    api_key=_API_KEY,
    model="configured-chat",
    timeout_connect=1.25,
    timeout_read=3,
)

type MockHandler = (
    Callable[[httpx.Request], httpx.Response] | Callable[[httpx.Request], Awaitable[httpx.Response]]
)
type RouterFactory = Callable[..., LLMRouter]


def _completion() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "在呢"}}]})


@pytest.fixture
async def make_router() -> AsyncIterator[RouterFactory]:
    async with AsyncExitStack() as stack:

        def make(
            handler: MockHandler, models: dict[str, LLMModelConfig] | None = None
        ) -> LLMRouter:
            client = LLMRouter(
                {"chat_model": _CHAT} if models is None else models,
                transport=httpx.MockTransport(handler),
            )
            stack.push_async_callback(client.aclose)
            return client

        yield make


@pytest.fixture
def backoffs(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def record_delay(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(router, "sleep", record_delay)
    return delays


async def test_routes_to_models_and_parses_completion(make_router: RouterFactory) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "gate.invalid":
            return _completion()
        return httpx.Response(
            200,
            json={
                "model": "resolved-chat",
                "choices": [{"message": {"content": "接住了"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    gate = LLMModelConfig(base_url="https://gate.invalid", api_key="gate-key", model="fast")
    client = make_router(handle, {"chat_model": _CHAT, "gate_model": gate})
    result = await client.chat_completion(
        "chat_model",
        _MESSAGES,
        temperature=0.2,
        stop=["STOP"],
        response_format={"type": "json_object"},
    )
    gate_result = await client.chat_completion("gate_model", _MESSAGES)

    assert [str(request.url) for request in requests] == [
        "https://chat.invalid/v1/chat/completions",
        "https://gate.invalid/chat/completions",
    ]
    chat_request, gate_request = requests
    assert chat_request.headers["Authorization"] == f"Bearer {_API_KEY}"
    assert gate_request.headers["Authorization"] == "Bearer gate-key"
    assert chat_request.extensions["timeout"]["connect"] == 1.25
    assert chat_request.extensions["timeout"]["read"] == 3
    assert json.loads(chat_request.content) == {
        "model": "configured-chat",
        "messages": [message.model_dump() for message in _MESSAGES],
        "temperature": 0.2,
        "max_tokens": 300,
        "stop": ["STOP"],
        "response_format": {"type": "json_object"},
    }
    assert json.loads(gate_request.content)["model"] == "fast"
    assert result.content == "接住了"
    assert result.model == "resolved-chat"
    assert result.latency_ms >= 0
    assert (result.prompt_tokens, result.completion_tokens) == (7, 3)
    assert gate_result.model == "fast"
    assert gate_result.prompt_tokens is None
    assert gate_result.completion_tokens is None


async def test_retries_rate_limit_and_service_unavailable(
    make_router: RouterFactory, backoffs: list[float]
) -> None:
    statuses = iter([429, 503, 200])

    def handle(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return _completion() if status == 200 else httpx.Response(status)

    client = make_router(handle)
    assert (await client.chat_completion("chat_model", _MESSAGES)).content == "在呢"
    assert len(backoffs) == 2
    assert 0.5 <= backoffs[0] <= 0.6
    assert 1.0 <= backoffs[1] <= 1.1


async def test_stops_after_max_retries(make_router: RouterFactory, backoffs: list[float]) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    client = make_router(handle)
    with pytest.raises(LLMRequestError, match="HTTP 503") as error:
        await client.chat_completion("chat_model", _MESSAGES)
    assert error.value.status_code == 503
    assert attempts == 3
    assert len(backoffs) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 501])
async def test_does_not_retry_other_http_errors(
    make_router: RouterFactory, backoffs: list[float], caplog: pytest.LogCaptureFixture, status: int
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": f"Rejected token {_API_KEY}"})

    client = make_router(handle)
    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(LLMRequestError, match=f"HTTP {status}") as error,
    ):
        await client.chat_completion("chat_model", _MESSAGES)
    assert error.value.status_code == status
    assert backoffs == []
    assert _API_KEY not in str(error.value)
    assert _API_KEY not in caplog.text


@pytest.mark.parametrize(
    ("exception", "message", "retry_count"),
    [
        (httpx.ReadTimeout, "timed out", 2),
        (httpx.ConnectError, "Could not connect", 2),
        (httpx.LocalProtocolError, "Could not complete", 0),
    ],
)
async def test_transport_errors_are_sanitized(
    make_router: RouterFactory,
    backoffs: list[float],
    caplog: pytest.LogCaptureFixture,
    exception: type[httpx.RequestError],
    message: str,
    retry_count: int,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise exception(f"Transport error for {request.headers['Authorization']}", request=request)

    client = make_router(handle)
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMRequestError, match=message) as error:
        await client.chat_completion("chat_model", _MESSAGES)
    assert len(backoffs) == retry_count
    assert error.value.status_code is None
    assert error.value.__context__ is None
    assert _API_KEY not in str(error.value)
    assert _API_KEY not in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        b"not JSON",
        b"[]",
        b"{}",
        b'{"choices":[]}',
        b'{"choices":[{"message":{"content":null}}]}',
        b'{"choices":[{"message":{"content":42}}]}',
    ],
)
async def test_rejects_invalid_completion(make_router: RouterFactory, body: bytes) -> None:
    client = make_router(lambda request: httpx.Response(200, content=body))
    with pytest.raises(LLMResponseError, match=r"choices\[0\]\.message\.content"):
        await client.chat_completion("chat_model", _MESSAGES)


async def test_rejects_invalid_configuration_before_request(make_router: RouterFactory) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid configuration sent a request")

    client = make_router(handle)
    with pytest.raises(LLMConfigError, match="Unknown model_key"):
        await client.chat_completion("unconfigured", _MESSAGES)
    with pytest.raises(LLMConfigError, match="Unsupported chat completion override"):
        await client.chat_completion("chat_model", _MESSAGES, model="unconfigured")


async def test_limits_concurrent_requests_independently_by_model(
    make_router: RouterFactory,
) -> None:
    release_chat = asyncio.Event()
    chat_limit_reached = asyncio.Event()
    active_chat = 0
    peak_chat = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == _CHAT.model:
            nonlocal active_chat, peak_chat
            active_chat += 1
            peak_chat = max(peak_chat, active_chat)
            if active_chat == 2:
                chat_limit_reached.set()
            try:
                await release_chat.wait()
            finally:
                active_chat -= 1
        return _completion()

    chat = _CHAT.model_copy(update={"max_concurrency": 2})
    gate = _CHAT.model_copy(update={"model": "gate", "max_concurrency": 1})
    client = make_router(handle, {"chat_model": chat, "gate_model": gate})
    async with asyncio.timeout(5), asyncio.TaskGroup() as group:
        for _ in range(6):
            group.create_task(client.chat_completion("chat_model", _MESSAGES))
        await chat_limit_reached.wait()
        assert (await client.chat_completion("gate_model", _MESSAGES)).content == "在呢"
        release_chat.set()
    assert peak_chat == 2
    assert active_chat == 0


async def test_cancellation_releases_the_model_slot(make_router: RouterFactory) -> None:
    first_attempt_started = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        if not first_attempt_started.is_set():
            first_attempt_started.set()
            await asyncio.Event().wait()
        return _completion()

    chat = _CHAT.model_copy(update={"max_concurrency": 1})
    client = make_router(handle, {"chat_model": chat})
    async with asyncio.timeout(5), asyncio.TaskGroup() as group:
        pending = group.create_task(client.chat_completion("chat_model", _MESSAGES))
        await first_attempt_started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (await client.chat_completion("chat_model", _MESSAGES)).content == "在呢"
