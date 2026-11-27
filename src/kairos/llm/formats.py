"""Translate Kairos text conversations to provider HTTP formats."""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from kairos.config import LLMAPIFormat, LLMModelConfig
from kairos.models import ChatMessage, LLMResult, ToolCall, ToolDefinition

# Messages requires a numeric max_tokens even when the Bot has no configured limit.
_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
_TOOL_OVERRIDES = {"tools", "tool_choice"}
_OVERRIDES: dict[LLMAPIFormat, set[str]] = {
    "openai_chat_completions": {"temperature", "max_tokens", "stop", "response_format"}
    | _TOOL_OVERRIDES,
    "openai_responses": {"temperature", "max_tokens", "response_format"} | _TOOL_OVERRIDES,
    "anthropic_messages": {"temperature", "max_tokens", "stop", "response_format"}
    | _TOOL_OVERRIDES,
    "gemini_generate_content": {"temperature", "max_tokens", "stop", "response_format"}
    | _TOOL_OVERRIDES,
}


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


def build_request(
    config: LLMModelConfig,
    messages: list[ChatMessage],
    overrides: dict[str, Any],
    *,
    session_id: str | None,
) -> CompletionRequest:
    api_format = config.request_format
    if overrides.keys() - _OVERRIDES[api_format]:
        raise ValueError(f"Unsupported chat completion override for {api_format}")

    turns = [message.model_dump() for message in messages]
    max_tokens = overrides.get("max_tokens", config.max_tokens)
    temperature = overrides.get("temperature", config.request_temperature)
    sampling = {} if temperature is None else {"temperature": temperature}
    stop = overrides.get("stop")
    stop_sequences = [stop] if isinstance(stop, str) else stop
    response_format = overrides.get("response_format")

    if api_format == "openai_chat_completions":
        path = "chat/completions"
        body: dict[str, Any] = {
            "model": config.model,
            "messages": turns,
            **sampling,
        }
        if max_tokens is not None:
            token_field = "max_completion_tokens" if config.provider == "openai" else "max_tokens"
            body[token_field] = max_tokens
        body.update(
            {key: overrides[key] for key in ("stop", "response_format") if key in overrides}
        )
    elif api_format == "openai_responses":
        path = "responses"
        body = {
            "model": config.model,
            "input": turns,
            "store": False,
            **sampling,
        }
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens
        if response_format is not None:
            body["text"] = {"format": response_format}
    else:
        system = "\n\n".join(turn["content"] for turn in turns if turn["role"] == "system")
        conversation = [turn for turn in turns if turn["role"] != "system"]
        if conversation and conversation[-1]["role"] == "assistant":
            # Some Messages and GenerateContent models require a final user turn.
            conversation.append({"role": "user", "content": "请根据以上对话决定是否回复。"})
        if api_format == "anthropic_messages":
            path = "messages"
            body = {
                "model": config.model,
                "messages": conversation,
                "max_tokens": _ANTHROPIC_DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
                **sampling,
            }
            if system:
                body["system"] = system
            if stop_sequences is not None:
                body["stop_sequences"] = stop_sequences
        else:
            model = quote(config.model.removeprefix("models/"), safe="")
            path = f"models/{model}:generateContent"
            generation: dict[str, Any] = {**sampling}
            if max_tokens is not None:
                generation["maxOutputTokens"] = max_tokens
            if stop_sequences is not None:
                generation["stopSequences"] = stop_sequences
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                generation["responseMimeType"] = "application/json"
            if config.thinking_level is not None:
                generation["thinkingConfig"] = {"thinkingLevel": config.thinking_level.upper()}
            body = {
                "contents": [
                    {
                        "role": "model" if turn["role"] == "assistant" else "user",
                        "parts": [{"text": turn["content"]}],
                    }
                    for turn in conversation
                ],
            }
            if generation:
                body["generationConfig"] = generation
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}

    if tools := overrides.get("tools"):
        body.update(_tool_options(api_format, tools, overrides.get("tool_choice")))

    if config.provider == "anthropic":
        headers = {"x-api-key": config.api_key, "anthropic-version": "2026-10-21"}
    elif config.provider == "google_ai_studio":
        headers = {"x-goog-api-key": config.api_key}
    else:
        headers = {"Authorization": f"Bearer {config.api_key}"}
    if config.provider == "opencode_go" and session_id is not None:
        headers.update({"user-agent": "Kairos", "x-opencode-session": session_id})
    return CompletionRequest(f"{config.endpoint_base_url.rstrip('/')}/{path}", headers, body)


def _tool_options(api_format: LLMAPIFormat, tools: object, choice: object) -> dict[str, Any]:
    if not isinstance(tools, Sequence) or not all(
        isinstance(tool, ToolDefinition) for tool in tools
    ):
        raise ValueError("tools must be a sequence of ToolDefinition")
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        fields = {"name": tool.name, "description": tool.description}
        if api_format == "openai_chat_completions":
            definition = {
                "type": "function",
                "function": {
                    **fields,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                },
            }
        elif api_format == "openai_responses":
            definition = {
                "type": "function",
                **fields,
                "parameters": tool.parameters,
                "strict": tool.strict,
            }
        elif api_format == "anthropic_messages":
            definition = {**fields, "input_schema": tool.parameters}
        else:
            definition = {**fields, "parametersJsonSchema": tool.parameters}
        definitions.append(definition)

    options: dict[str, Any] = {
        "tools": [{"functionDeclarations": definitions}]
        if api_format == "gemini_generate_content"
        else definitions
    }
    if choice is not None:
        options.update(_tool_choice(api_format, choice, {tool.name for tool in tools}))
    return options


def _tool_choice(api_format: LLMAPIFormat, choice: object, names: set[str]) -> dict[str, Any]:
    named = False
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind in ("function", "tool"):
            named = True
            nested = choice.get("function")
            choice = nested.get("name") if isinstance(nested, dict) else choice.get("name")
            if not isinstance(choice, str) or choice not in names:
                raise ValueError("tool_choice names a tool that is not attached")
        else:
            choice = "required" if kind == "any" else kind
    if not isinstance(choice, str):
        raise ValueError("tool_choice must be auto, none, required, or an attached tool name")  # noqa: TRY004 - Invalid overrides use the router configuration error path.
    mode = not named and choice in {"auto", "none", "required"}
    if not mode and choice not in names:
        raise ValueError("tool_choice names a tool that is not attached")

    if api_format == "gemini_generate_content":
        config: dict[str, Any] = {"mode": "ANY"}
        if mode:
            config["mode"] = {"auto": "AUTO", "none": "NONE", "required": "ANY"}[choice]
        if not mode:
            config["allowedFunctionNames"] = [choice]
        return {"toolConfig": {"functionCallingConfig": config}}
    if api_format == "anthropic_messages":
        if mode:
            return {"tool_choice": {"type": "any" if choice == "required" else choice}}
        return {"tool_choice": {"type": "tool", "name": choice}}
    if mode:
        return {"tool_choice": choice}
    if api_format == "openai_chat_completions":
        return {"tool_choice": {"type": "function", "function": {"name": choice}}}
    return {"tool_choice": {"type": "function", "name": choice}}


_TEXT_FIELD: dict[LLMAPIFormat, str] = {
    "openai_chat_completions": "choices[0].message.content",
    "openai_responses": "output messages with output_text",
    "anthropic_messages": "content text blocks",
    "gemini_generate_content": "candidate text parts",
}


def _objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _join_text(values: Iterable[Any]) -> str | None:
    texts = [value for value in values if isinstance(value, str)]
    return "".join(texts) if texts else None


def _text_blocks(blocks: Any, kind: str) -> str | None:
    return _join_text(block.get("text") for block in _objects(blocks) if block.get("type") == kind)


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _parse_tool_call(call_id: Any, name: Any, arguments: Any) -> ToolCall | None:
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            return None
    if not isinstance(arguments, dict):
        return None
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _chat_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls = []
    for item in _objects(message.get("tool_calls")):
        function = item.get("function")
        if item.get("type", "function") != "function" or not isinstance(function, dict):
            continue
        if call := _parse_tool_call(
            item.get("id"), function.get("name"), function.get("arguments")
        ):
            calls.append(call)
    return calls


def _gemini_tool_calls(parts: list[dict[str, Any]]) -> list[ToolCall]:
    calls = []
    for index, part in enumerate(parts):
        function = part.get("functionCall")
        if part.get("thought") is True or not isinstance(function, dict):
            continue
        # Older Gemini versions omit call identifiers. Keep the returned call index stable.
        if call := _parse_tool_call(
            function.get("id") or f"call_{index}",
            function.get("name"),
            function.get("args", {}),
        ):
            calls.append(call)
    return calls


def parse_response(body: Any, config: LLMModelConfig, latency_ms: int) -> LLMResult:
    api_format = config.request_format
    error = f"LLM response is missing valid {_TEXT_FIELD[api_format]}"
    if not isinstance(body, dict):
        raise TypeError(error)

    model = body.get("model")
    usage = body.get("usage")
    usage_keys = ("input_tokens", "output_tokens")
    try:
        if api_format == "openai_chat_completions":
            message = body["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError(error)
            content = message.get("content")
            calls = _chat_tool_calls(message)
            usage_keys = ("prompt_tokens", "completion_tokens")
        elif api_format == "openai_responses":
            items = _objects(body["output"])
            content = _join_text(
                _text_blocks(item.get("content"), "output_text")
                for item in items
                if item.get("type") == "message"
            )
            calls = [
                call
                for item in items
                if item.get("type") == "function_call"
                and (
                    call := _parse_tool_call(
                        item.get("call_id") or item.get("id"),
                        item.get("name"),
                        item.get("arguments"),
                    )
                )
            ]
        elif api_format == "anthropic_messages":
            blocks = _objects(body["content"])
            content = _text_blocks(blocks, "text")
            calls = [
                call
                for block in blocks
                if block.get("type") == "tool_use"
                and (
                    call := _parse_tool_call(block.get("id"), block.get("name"), block.get("input"))
                )
            ]
        else:
            parts = _objects(body["candidates"][0]["content"]["parts"])
            content = _join_text(
                part.get("text") for part in parts if part.get("thought") is not True
            )
            calls = _gemini_tool_calls(parts)
            model = body.get("modelVersion")
            usage = body.get("usageMetadata")
            usage_keys = ("promptTokenCount", "candidatesTokenCount")
    except (KeyError, IndexError, TypeError):
        raise ValueError(error) from None
    if not isinstance(content, str):
        if not calls or content is not None:
            raise TypeError(error)
        content = ""

    if not isinstance(usage, dict):
        usage = {}
    return LLMResult(
        content=content,
        model=model if isinstance(model, str) and model else config.model,
        latency_ms=latency_ms,
        prompt_tokens=_token_count(usage.get(usage_keys[0])),
        completion_tokens=_token_count(usage.get(usage_keys[1])),
        tool_calls=calls,
    )
