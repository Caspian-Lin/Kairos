"""Route text completions through a shared asynchronous HTTP pool."""

import logging
import random
import time
from asyncio import Semaphore, sleep
from typing import Any
from uuid import uuid4

import httpx

from kairos.config import LLMModelConfig
from kairos.llm.formats import CompletionRequest, build_request, parse_response
from kairos.models import ChatMessage, LLMResult

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
_WIRE_LOGGERS = ("hpack.hpack", "httpcore.http11", "httpcore.http2")


def _hide_debug_wire_logs(record: logging.LogRecord) -> bool:
    """Keep protocol traces that can contain headers out of application logs."""
    return record.levelno >= logging.INFO


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
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("Choose either a shared client or a transport")
        for name in _WIRE_LOGGERS:
            logging.getLogger(name).addFilter(_hide_debug_wire_logs)
        self._models = models.copy()
        self._slots = {key: Semaphore(cfg.max_concurrency) for key, cfg in self._models.items()}
        self._go_sessions = {
            key: f"ses_{uuid4().hex}"
            for key, cfg in self._models.items()
            if cfg.provider == "opencode_go"
        }
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(
                http2=True,
                limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
                transport=transport,
            )
        )

    def for_models(self, models: dict[str, LLMModelConfig]) -> "LLMRouter":
        """Use independent model slots and concurrency limits with the same HTTP pool."""
        return LLMRouter(models, client=self._client)

    async def aclose(self) -> None:
        """Close the HTTP pool owned by this router."""
        if self._owns_client:
            await self._client.aclose()

    async def chat_completion(
        self,
        model_key: str,
        messages: list[ChatMessage],
        *,
        session_id: str | None = None,
        **overrides: Any,
    ) -> LLMResult:
        cfg = self._models.get(model_key)
        if cfg is None:
            raise LLMConfigError("Unknown model_key")
        try:
            request = build_request(
                cfg,
                messages,
                overrides,
                session_id=session_id or self._go_sessions.get(model_key),
            )
        except ValueError as exc:
            raise LLMConfigError(str(exc)) from None

        started = time.perf_counter()
        async with self._slots[model_key]:
            response = await self._request_completion(cfg, request)
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            body = response.json()
        except (ValueError, RecursionError):
            body = None
        try:
            return parse_response(body, cfg, latency_ms)
        except (ValueError, TypeError) as exc:
            raise LLMResponseError(str(exc)) from None

    async def _request_completion(
        self, cfg: LLMModelConfig, request: CompletionRequest
    ) -> httpx.Response:
        timeout = httpx.Timeout(cfg.timeout_read, connect=cfg.timeout_connect)
        retries_remaining = cfg.max_retries
        backoff = 0.5

        while True:
            status_code = None
            retryable = True
            try:
                response = await self._client.post(
                    request.url, json=request.body, headers=request.headers, timeout=timeout
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
