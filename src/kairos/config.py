"""Typed runtime settings loaded from YAML."""

import os
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


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
    base_url: str
    api_key: str = Field(repr=False)
    model: str
    timeout_connect: float = Field(default=5.0, gt=0)
    timeout_read: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=5, gt=0)
    temperature: float = 0.8
    max_tokens: int = Field(default=300, gt=0)


class LoggingConfig(_ConfigModel):
    level: str = "INFO"
    json_format: bool = False


class GlobalConfig(_ConfigModel):
    postgres: PostgresConfig
    onebot_ws: OneBotWSConfig = Field(default_factory=OneBotWSConfig)
    llm_models: dict[str, LLMModelConfig]
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
    chat_model_key: str
    trigger_mode: Literal["mention_only", "all_with_veto"] = "all_with_veto"
    allowed_groups: list[int] = Field(default_factory=list)
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


def _load_config_file[Config: _ConfigModel](path: Path, config_type: type[Config]) -> Config:
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
    try:
        return config_type.model_validate(expanded)
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from None


def load_global_config(path: str | Path) -> GlobalConfig:
    """Load the shared PostgreSQL, gateway, LLM and logging settings."""
    return _load_config_file(Path(path), GlobalConfig)


def load_bot_profiles(
    bots_dir: str | Path, global_config: GlobalConfig | None = None
) -> list[BotProfile]:
    """Load active profiles in filename order and validate their global references.

    The sibling global.yaml provides model definitions when no config is passed.
    Files named example_*.yaml are templates. Copy one to an active filename
    before loading it.
    """
    directory = Path(bots_dir)
    if global_config is None:
        global_config = load_global_config(directory.parent / "global.yaml")
    profiles: list[BotProfile] = []
    bot_ids: set[str] = set()
    self_ids: set[int] = set()
    for path in sorted(directory.iterdir()):
        if path.suffix != ".yaml" or path.name.startswith("example_") or not path.is_file():
            continue
        profile = _load_config_file(path, BotProfile)
        if profile.bot_id in bot_ids:
            raise ValueError(f"{path}: duplicate bot_id {profile.bot_id!r}")
        if profile.self_id in self_ids:
            raise ValueError(f"{path}: duplicate self_id {profile.self_id}")
        if profile.chat_model_key not in global_config.llm_models:
            raise ValueError(f"{path}: unknown chat_model_key {profile.chat_model_key!r}")
        profiles.append(profile)
        bot_ids.add(profile.bot_id)
        self_ids.add(profile.self_id)
    return profiles
