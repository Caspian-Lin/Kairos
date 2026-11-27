import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack

import httpx
import pytest
from hpack import Encoder

from kairos.config import LLMAPIFormat, LLMModelConfig, LLMProvider
from kairos.llm import LLMConfigError, LLMRequestError, LLMResponseError, LLMRouter, router
from kairos.models import ChatMessage, ToolCall, ToolDefinition

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


def test_wire_debug_logs_are_filtered(
    make_router: RouterFactory, caplog: pytest.LogCaptureFixture
) -> None:
    make_router(lambda _: _completion())
    with caplog.at_level(logging.DEBUG, logger="hpack.hpack"):
        Encoder().encode([(b"authorization", f"Bearer {_API_KEY}".encode(), True)])
    for name in ("httpcore.http11", "httpcore.http2"):
        with caplog.at_level(logging.DEBUG, logger=name):
            logging.getLogger(name).debug("wire headers: %s", _API_KEY)
    with caplog.at_level(logging.DEBUG, logger="kairos.llm.router"):
        logging.getLogger("kairos.llm.router").debug("Kairos debug message")

    assert [record.getMessage() for record in caplog.records] == ["Kairos debug message"]


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
        max_tokens=600,
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
        "max_tokens": 600,
        "stop": ["STOP"],
        "response_format": {"type": "json_object"},
    }
    assert json.loads(gate_request.content)["model"] == "fast"
    assert "max_tokens" not in json.loads(gate_request.content)
    assert result.content == "接住了"
    assert result.model == "resolved-chat"
    assert result.latency_ms >= 0
    assert (result.prompt_tokens, result.completion_tokens) == (7, 3)
    assert gate_result.model == "fast"
    assert gate_result.prompt_tokens is None
    assert gate_result.completion_tokens is None


async def test_openai_presets_encode_chat_completions_and_responses(
    make_router: RouterFactory,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        payload = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            assert payload["messages"] == [message.model_dump() for message in _MESSAGES]
            assert "max_completion_tokens" not in payload
            assert "temperature" not in payload
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "在呢", "reasoning_content": "聊天思考"}}]
                },
            )

        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert payload["input"] == [message.model_dump() for message in _MESSAGES]
        assert "max_output_tokens" not in payload
        assert payload["store"] is False
        assert "reasoning" not in payload
        return httpx.Response(
            200,
            json={
                "model": "resolved-response",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "公开摘要"}],
                        "encrypted_content": "opaque-data",
                    },
                    {"type": "message", "content": [{"type": "output_text", "text": "在呢"}]},
                ],
                "usage": {"input_tokens": 8, "output_tokens": 4},
            },
        )

    chat = LLMModelConfig(provider="openai", api_key=_API_KEY, model="gpt-4o-mini")
    responses = LLMModelConfig(
        provider="openai",
        api_format="openai_responses",
        api_key=_API_KEY,
        model="reasoning-model",
    )
    client = make_router(handle, {"chat": chat, "responses": responses})
    assert (await client.chat_completion("chat", _MESSAGES)).content == "在呢"
    result = await client.chat_completion("responses", _MESSAGES)
    assert result.content == "在呢"
    assert result.model == "resolved-response"
    assert (result.prompt_tokens, result.completion_tokens) == (8, 4)


@pytest.mark.parametrize(
    ("provider", "url", "auth_header", "auth_value"),
    [
        ("anthropic", "https://api.anthropic.com/v1/messages", "x-api-key", _API_KEY),
        (
            "opencode_go",
            "https://opencode.ai/inference/go/anthropic/v1/messages",
            "authorization",
            f"Bearer {_API_KEY}",
        ),
    ],
)
async def test_messages_presets(
    make_router: RouterFactory, provider: LLMProvider, url: str, auth_header: str, auth_value: str
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == url
        assert request.headers[auth_header] == auth_value
        if provider == "anthropic":
            assert request.headers["anthropic-version"]
        else:
            assert request.headers["x-opencode-session"].startswith("ses_")
            assert request.headers["user-agent"] == "Kairos"
        assert json.loads(request.content) == {
            "model": "configured-model",
            "system": "简短回复",
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 4096,
            "stop_sequences": ["END"],
        }
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "公开思考", "signature": "opaque-signature"},
                    {"type": "redacted_thinking", "data": "opaque-data"},
                    {"type": "text", "text": "在"},
                    {"type": "text", "text": "呢"},
                ],
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        )

    model = LLMModelConfig(
        provider=provider,
        api_format="anthropic_messages",
        api_key=_API_KEY,
        model="configured-model",
    )
    result = await make_router(handle, {"chat": model}).chat_completion(
        "chat", _MESSAGES, stop="END"
    )
    assert result.content == "在呢"
    assert result.model == "configured-model"
    assert (result.prompt_tokens, result.completion_tokens) == (7, 3)


async def test_go_uses_conversation_id_only_for_its_own_requests(
    make_router: RouterFactory,
) -> None:
    sessions: list[str | None] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers.get("x-opencode-session"))
        if request.url.host == "opencode.ai":
            assert request.url.path == "/inference/go/openai/v1/chat/completions"
            assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        return _completion()

    go = LLMModelConfig(provider="opencode_go", api_key=_API_KEY, model="kimi-k3")
    client = make_router(handle, {"go": go, "legacy": _CHAT})
    for group in ("ses_group_a", "ses_group_a", "ses_group_b"):
        await client.chat_completion("go", _MESSAGES, session_id=group)
    await client.chat_completion("legacy", _MESSAGES, session_id="ses_group_a")
    assert sessions == ["ses_group_a", "ses_group_a", "ses_group_b", None]


async def test_gemini_preset_maps_history_and_excludes_thought_parts(
    make_router: RouterFactory,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
        )
        assert request.headers["x-goog-api-key"] == _API_KEY
        payload = json.loads(request.content)
        assert payload["systemInstruction"] == {"parts": [{"text": "简短回复"}]}
        assert [turn["role"] for turn in payload["contents"]] == ["user", "model", "user"]
        assert payload["generationConfig"] == {
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        }
        return httpx.Response(
            200,
            json={
                "modelVersion": "gemini-3.5-flash-lite",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thought": True,
                                    "text": "公开思考",
                                    "thoughtSignature": "opaque-data",
                                },
                                {"text": "在呢"},
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2},
            },
        )

    model = LLMModelConfig(
        provider="google_ai_studio",
        api_key=_API_KEY,
        model="gemini-3.5-flash-lite",
        thinking_level="minimal",
    )
    history = [*_MESSAGES, ChatMessage(role="assistant", content="早")]
    result = await make_router(handle, {"chat": model}).chat_completion("chat", history)
    assert result.content == "在呢"
    assert (result.prompt_tokens, result.completion_tokens) == (9, 2)


@pytest.mark.parametrize(
    ("provider", "api_format"),
    [
        ("openai", "openai_responses"),
        ("openai", "openai_chat_completions"),
        ("openai_compatible", "openai_chat_completions"),
        ("opencode_go", "openai_responses"),
        ("opencode_go", "gemini_generate_content"),
        ("google_ai_studio", "gemini_generate_content"),
        ("anthropic", "anthropic_messages"),
    ],
)
async def test_json_mode_maps_to_the_provider_wire_format(
    make_router: RouterFactory, provider: LLMProvider, api_format: LLMAPIFormat
) -> None:
    response_format = {"type": "json_object"}
    reply = '{"action":"pass"}'

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if api_format == "openai_chat_completions":
            assert body["response_format"] == response_format
            response = {"choices": [{"message": {"content": reply}}]}
        elif api_format == "openai_responses":
            assert body["text"] == {"format": response_format}
            assert "response_format" not in body
            response = {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": reply}]}]
            }
        elif api_format == "gemini_generate_content":
            assert body["generationConfig"] == {"responseMimeType": "application/json"}
            response = {"candidates": [{"content": {"parts": [{"text": reply}]}}]}
        else:
            assert "response_format" not in body and "text" not in body
            response = {"content": [{"type": "text", "text": reply}]}
        return httpx.Response(200, json=response)

    model = LLMModelConfig(
        provider=provider,
        api_format=api_format,
        api_key=_API_KEY,
        model="test-gate",
        base_url="https://test.invalid/v1" if provider == "openai_compatible" else None,
    )
    result = await make_router(handle, {"gate": model}).chat_completion(
        "gate", _MESSAGES, response_format=response_format
    )
    assert result.content == reply


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


@pytest.mark.parametrize(
    ("provider", "api_format"),
    [
        ("openai", "openai_chat_completions"),
        ("openai", "openai_responses"),
        ("anthropic", "anthropic_messages"),
        ("google_ai_studio", "gemini_generate_content"),
    ],
)
async def test_tool_only_responses_and_native_tool_schema(
    make_router: RouterFactory, provider: LLMProvider, api_format: LLMAPIFormat
) -> None:
    name = "group_status"
    parameters = {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
        "additionalProperties": False,
    }
    definition = ToolDefinition(name=name, description="Look up group state", parameters=parameters)
    fields = {"name": name, "description": definition.description}
    arguments = {"topic": "tea"}
    encoded_arguments = json.dumps(arguments)

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if api_format == "openai_chat_completions":
            assert body["tools"] == [
                {
                    "type": "function",
                    "function": {**fields, "parameters": parameters, "strict": True},
                }
            ]
            assert body["tool_choice"] == {"type": "function", "function": {"name": name}}
            reply = {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": "status_call",
                                    "function": {"name": name, "arguments": encoded_arguments},
                                }
                            ],
                        }
                    }
                ]
            }
        elif api_format == "openai_responses":
            assert body["tools"] == [
                {"type": "function", **fields, "parameters": parameters, "strict": True}
            ]
            assert body["tool_choice"] == {"type": "function", "name": name}
            reply = {
                "output": [
                    {"type": "reasoning"},
                    {
                        "type": "function_call",
                        "id": "item_id",
                        "call_id": "status_call",
                        "name": name,
                        "arguments": encoded_arguments,
                    },
                ]
            }
        elif api_format == "anthropic_messages":
            assert body["tools"] == [{**fields, "input_schema": parameters}]
            assert body["tool_choice"] == {"type": "tool", "name": name}
            reply = {
                "content": [
                    {"type": "thinking", "thinking": "internal"},
                    {"type": "tool_use", "id": "status_call", "name": name, "input": arguments},
                ]
            }
        else:
            assert body["tools"] == [
                {"functionDeclarations": [{**fields, "parametersJsonSchema": parameters}]}
            ]
            assert body["toolConfig"] == {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}
            }
            reply = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "status_call",
                                        "name": name,
                                        "args": arguments,
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        return httpx.Response(200, json=reply)

    configured = LLMModelConfig(
        provider=provider, api_format=api_format, api_key=_API_KEY, model="test-model"
    )
    client = make_router(respond, {"chat": configured})
    result = await client.chat_completion("chat", _MESSAGES, tools=(definition,), tool_choice=name)

    assert result.content == ""
    assert result.tool_calls == [ToolCall(id="status_call", name=name, arguments=arguments)]


async def test_empty_tool_list_preserves_plain_request_and_response(
    make_router: RouterFactory,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "model": _CHAT.model,
            "messages": [message.model_dump() for message in _MESSAGES],
            "temperature": _CHAT.request_temperature,
        }
        return _completion()

    client = make_router(respond)
    result = await client.chat_completion("chat_model", _MESSAGES, tools=[], tool_choice="auto")
    assert result.content == "在呢" and result.tool_calls == []


async def test_tool_arguments_must_be_json_objects(make_router: RouterFactory) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {"name": "status", "arguments": "[]"},
                                },
                            ]
                        }
                    }
                ]
            },
        )

    with pytest.raises(LLMResponseError):
        await make_router(respond).chat_completion("chat_model", _MESSAGES)
