"""Shared asynchronous client for OpenAI-compatible chat models."""

import random
import time
from asyncio import Semaphore, sleep
from typing import Any

import httpx

from kairos.config import LLMModelConfig
from kairos.models import ChatMessage, LLMResult

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_ALLOWED_OVERRIDES = {"temperature", "max_tokens", "stop", "response_format"}


class LLMError(Exception):
    """Base exception for model routing failures."""


class LLMConfigError(LLMError):
    """The model slot or request override is unsupported."""


class LLMRequestError(LLMError):
    """A provider request failed."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMError):
    """A successful HTTP response did not contain a usable completion."""


class LLMRouter:
    """Route model slots through one client with per-slot concurrency limits."""

    def __init__(
        self,
        models: dict[str, LLMModelConfig],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._models = models.copy()
        self._slots = {key: Semaphore(cfg.max_concurrency) for key, cfg in self._models.items()}
        self._client = httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the shared HTTP client and its connection pool."""
        await self._client.aclose()

    async def chat_completion(
        self, model_key: str, messages: list[ChatMessage], **overrides: Any
    ) -> LLMResult:
        cfg = self._models.get(model_key)
        if cfg is None:
            raise LLMConfigError("Unknown model_key")
        if overrides.keys() - _ALLOWED_OVERRIDES:
            raise LLMConfigError("Unsupported chat completion override")

        payload = {
            "model": cfg.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            **overrides,
        }

        started = time.perf_counter()
        async with self._slots[model_key]:
            response = await self._request_completion(cfg, payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _parse_completion(response, cfg.model, latency_ms)

    async def _request_completion(
        self, cfg: LLMModelConfig, payload: dict[str, Any]
    ) -> httpx.Response:
        url = f"{cfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {cfg.api_key}"}
        timeout = httpx.Timeout(cfg.timeout_read, connect=cfg.timeout_connect)
        retries_remaining = cfg.max_retries
        backoff = 0.5

        while True:
            status_code = None
            retryable = True
            try:
                response = await self._client.post(
                    url, json=payload, headers=headers, timeout=timeout
                )
            except httpx.TimeoutException:
                message = "LLM request timed out"
            except httpx.ConnectError:
                message = "Could not connect to the LLM provider"
            except (httpx.HTTPError, httpx.InvalidURL, TypeError, ValueError):
                retryable = False
                message = "Could not complete the LLM request"
            else:
                if response.is_success:
                    return response
                status_code = response.status_code
                retryable = status_code in _RETRYABLE_STATUS_CODES
                message = f"LLM provider returned HTTP {status_code}"

            # Provider bodies and HTTPX exceptions can contain credentials.
            if not retryable or retries_remaining == 0:
                raise LLMRequestError(message, status_code)
            await sleep(backoff + random.uniform(0.0, 0.1))
            backoff *= 2
            retries_remaining -= 1


def _parse_completion(response: httpx.Response, fallback_model: str, latency_ms: int) -> LLMResult:
    body: dict[str, Any] = {}
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        content = None
    if not isinstance(content, str):
        raise LLMResponseError("LLM response is missing valid choices[0].message.content")

    model = body.get("model")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return LLMResult(
        content=content,
        model=model if isinstance(model, str) and model else fallback_model,
        latency_ms=latency_ms,
        prompt_tokens=_token_count(usage.get("prompt_tokens")),
        completion_tokens=_token_count(usage.get("completion_tokens")),
    )


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None
