"""Typed runtime settings loaded from YAML."""

import os
import re
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_NUMERIC_BOT_ID = re.compile(r"[0-9]+")

type LLMProvider = Literal[
    "openai", "openai_compatible", "anthropic", "opencode_go", "google_ai_studio"
]
type LLMAPIFormat = Literal[
    "openai_chat_completions", "openai_responses", "anthropic_messages", "gemini_generate_content"
]
type LLMThinkingLevel = Literal["minimal", "low", "medium", "high"]

_LLM_PRESETS: dict[LLMProvider, dict[LLMAPIFormat, str]] = {
    "openai": {
        "openai_chat_completions": "https://api.openai.com/v1",
        "openai_responses": "https://api.openai.com/v1",
    },
    "openai_compatible": {"openai_chat_completions": "", "openai_responses": ""},
    "anthropic": {"anthropic_messages": "https://api.anthropic.com/v1"},
    "opencode_go": {
        "openai_chat_completions": "https://opencode.ai/inference/go/openai/v1",
        "openai_responses": "https://opencode.ai/inference/go/openai/v1",
        "anthropic_messages": "https://opencode.ai/inference/go/anthropic/v1",
        "gemini_generate_content": "https://opencode.ai/inference/go/google/v1beta",
    },
    "google_ai_studio": {
        "gemini_generate_content": "https://generativelanguage.googleapis.com/v1beta",
    },
}


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)


class PostgresConfig(_ConfigModel):
    dsn: str = Field(min_length=1, repr=False)
    pool_min_size: int = Field(default=2, ge=0)
    pool_max_size: int = Field(default=10, gt=0)

    @model_validator(mode="after")
    def validate_pool_sizes(self) -> Self:
        if self.pool_min_size > self.pool_max_size:
            raise ValueError("pool_min_size must be less than or equal to pool_max_size")
        return self


class OneBotWSConfig(_ConfigModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=0, le=65535)
    path: str = Field(default="/onebot/v11/ws", pattern=r"^/")
    access_token: str | None = Field(default=None, repr=False)
    send_timeout_seconds: float = Field(default=10.0, gt=0)


class LLMModelConfig(_ConfigModel):
    provider: LLMProvider = "openai_compatible"
    api_format: LLMAPIFormat | None = None
    base_url: str | None = Field(default=None, min_length=1, repr=False)
    api_key: str = Field(min_length=1, repr=False)
    model: str = Field(min_length=1)
    timeout_connect: float = Field(default=5.0, gt=0)
    timeout_read: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=5, gt=0)
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    thinking_level: LLMThinkingLevel | None = None

    @property
    def request_format(self) -> LLMAPIFormat:
        return self.api_format or next(iter(_LLM_PRESETS[self.provider]))

    @property
    def endpoint_base_url(self) -> str:
        return self.base_url or _LLM_PRESETS[self.provider][self.request_format]

    @property
    def request_temperature(self) -> float | None:
        if (
            self.provider == "openai_compatible"
            and self.request_format == "openai_chat_completions"
            and not self.model_fields_set.intersection({"provider", "temperature"})
        ):
            return 0.8
        return self.temperature

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        if self.request_format not in _LLM_PRESETS[self.provider]:
            raise ValueError(
                f"api_format {self.request_format!r} is unsupported by {self.provider!r}"
            )
        if not self.endpoint_base_url:
            raise ValueError("openai_compatible requires base_url")
        try:
            url = urlsplit(self.endpoint_base_url)
        except ValueError:
            raise ValueError("base_url must be an absolute HTTP(S) URL") from None
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an absolute HTTP(S) URL without credentials, query or fragment"
            )
        if not self.api_key.strip():
            raise ValueError("api_key must contain a credential")
        if self.thinking_level is not None and self.request_format != "gemini_generate_content":
            raise ValueError("thinking_level requires gemini_generate_content")
        return self


class LoggingConfig(_ConfigModel):
    level: str = "INFO"
    json_format: bool = False


class GlobalConfig(_ConfigModel):
    postgres: PostgresConfig
    onebot_ws: OneBotWSConfig = Field(default_factory=OneBotWSConfig)
    llm_models: dict[str, LLMModelConfig] = Field(default_factory=dict)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class DebounceConfig(_ConfigModel):
    normal_seconds: float = Field(default=1.8, ge=0)
    max_wait_seconds: float = Field(default=4.5, ge=0)
    mention_seconds: float = Field(default=0.5, ge=0)


class RateLimitConfig(_ConfigModel):
    cooldown_seconds: float = Field(default=3.0, ge=0)
    peer_bot_max_turns: int = Field(default=2, ge=0)


class StaleCheckConfig(_ConfigModel):
    direct_max_delta: int = Field(default=1, ge=0)
    reply_max_delta: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def validate_delta_limits(self) -> Self:
        if self.direct_max_delta > self.reply_max_delta:
            raise ValueError("direct_max_delta must be less than or equal to reply_max_delta")
        return self


class BotProfile(_ConfigModel):
    bot_id: str = Field(min_length=1)
    self_id: int
    nickname: str
    profile_type: Literal["social_companion", "utility_assistant"] = "social_companion"
    persona_prompt: str
    llm: dict[str, LLMModelConfig] = Field(default_factory=dict)
    chat_model_key: str = "chat"
    # Legacy YAMLs can still opt into the original ambient-chat behavior.
    trigger_mode: Literal["mention_only", "all_with_veto"] | None = Field(default=None, repr=False)
    allowed_groups: list[int] = Field(default_factory=list)
    debug_groups: list[int] = Field(default_factory=list)
    debounce: DebounceConfig = Field(default_factory=DebounceConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    stale_check: StaleCheckConfig = Field(default_factory=StaleCheckConfig)
    ring_buffer_size: int = Field(default=40, gt=0)


def _expand_env(value: Any, path: Path) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match[1]
            try:
                return os.environ[name]
            except KeyError:
                raise ValueError(f"{path}: missing environment variable {name}") from None

        return _ENV_PLACEHOLDER.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand_env(item, path) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, path) for item in value]
    return value


def _load_config_file[Config: _ConfigModel](
    path: Path, config_type: type[Config], *, bot_id: str | None = None
) -> Config:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        location = f":{mark.line + 1}:{mark.column + 1}" if mark else ""
        raise ValueError(f"{path}{location}: invalid YAML") from None
    except yaml.YAMLError:
        raise ValueError(f"{path}: invalid YAML") from None

    # Expand after YAML parsing so environment values remain literal scalars.
    expanded = _expand_env(data, path)
    if bot_id is not None and isinstance(expanded, dict):
        if str(expanded.get("bot_id", bot_id)) != bot_id:
            raise ValueError(f"{path}: bot_id must match the numeric filename ({bot_id})")
        expanded["bot_id"] = bot_id
    try:
        return config_type.model_validate(expanded)
    except ValidationError as exc:
        variables: set[str] = set()
        for error in exc.errors(include_context=False, include_input=False, include_url=False):
            value = data
            for part in error["loc"]:
                if isinstance(value, dict):
                    value = value.get(part)
                elif isinstance(value, list) and isinstance(part, int) and 0 <= part < len(value):
                    value = value[part]
                else:
                    value = None
                    break
            if isinstance(value, str):
                variables.update(_ENV_PLACEHOLDER.findall(value))
        hint = f"\nCheck {', '.join(sorted(variables))} in .env." if variables else ""
        raise ValueError(f"{path}: {exc}{hint}") from None


def load_global_config(path: str | Path) -> GlobalConfig:
    """Load the shared PostgreSQL, gateway, LLM and logging settings."""
    return _load_config_file(Path(path), GlobalConfig)


def load_bot_profiles(
    bots_dir: str | Path, global_config: GlobalConfig | None = None
) -> list[BotProfile]:
    """Load numeric Bot YAML files in KAIROS_ENABLED_BOTS order."""
    selection = os.environ.get("KAIROS_ENABLED_BOTS", "").strip()
    if not selection:
        raise ValueError("Set KAIROS_ENABLED_BOTS in .env to the numeric Bot IDs to start")
    bot_ids = [bot_id.strip() for bot_id in selection.split(",")]
    if any(_NUMERIC_BOT_ID.fullmatch(bot_id) is None for bot_id in bot_ids):
        raise ValueError("KAIROS_ENABLED_BOTS must be a comma-separated list of numeric Bot IDs")
    if len(set(bot_ids)) != len(bot_ids):
        raise ValueError("KAIROS_ENABLED_BOTS contains a duplicate Bot ID")

    directory = Path(bots_dir)
    if global_config is None:
        global_config = load_global_config(directory.parent / "global.yaml")
    profiles: list[BotProfile] = []
    self_ids: set[int] = set()
    for bot_id in bot_ids:
        path = directory / f"{bot_id}.yaml"
        if not path.is_file():
            raise ValueError(f"{path}: missing profile selected by KAIROS_ENABLED_BOTS")
        profile = _load_config_file(path, BotProfile, bot_id=bot_id)
        if profile.self_id in self_ids:
            raise ValueError(f"{path}: duplicate self_id {profile.self_id}")
        models = profile.llm or global_config.llm_models
        if profile.chat_model_key not in models:
            location = "llm" if profile.llm else "llm or legacy global llm_models"
            raise ValueError(
                f"{path}: unknown chat_model_key {profile.chat_model_key!r} in {location}"
            )
        profiles.append(profile)
        self_ids.add(profile.self_id)
    return profiles
