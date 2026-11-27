"""Read and update public Bot profile fields while preserving YAML comments."""

import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from pydantic import BaseModel

from kairos.config import (
    _ENV_PLACEHOLDER,
    BotProfile,
    CompactionConfig,
    DebounceConfig,
    GateConfig,
    GlobalConfig,
    LLMModelConfig,
    MemoryConfig,
    RateLimitConfig,
    StaleCheckConfig,
    _expand_env,
)
from kairos.telemetry.log_buffer import redact_payload

_BOT_ID = re.compile(r"[0-9]+")
_MAX_BYTES = 1024 * 1024
_PUBLIC_FIELDS = (
    "nickname",
    "profile_type",
    "prompt_style",
    "persona_prompt",
    "instruction_reminder",
    "allowed_groups",
    "debug_groups",
    "enabled_tools",
)
_SECTIONS: dict[str, type[BaseModel]] = {
    "gate": GateConfig,
    "compaction": CompactionConfig,
    "memory": MemoryConfig,
    "debounce": DebounceConfig,
    "rate_limit": RateLimitConfig,
    "stale_check": StaleCheckConfig,
}
_MODEL_FIELDS = set(LLMModelConfig.model_fields) - {"api_key"}
_COMMENT = re.compile(r"(?<!\S)#[^\r\n]*")


def _dump_inline(value: Any) -> str:
    if isinstance(value, Mapping):
        fields = (f"{_dump_inline(key)}: {_dump_inline(item)}" for key, item in value.items())
        return "{" + ", ".join(fields) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(map(_dump_inline, value)) + "]"
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, float) and "e" in text:
        mantissa, exponent = text.split("e", 1)
        return f"{mantissa}{'.0' if '.' not in mantissa else ''}e{exponent}"
    return text


def _mapping(data: Any) -> dict[str, Any]:
    return data if isinstance(data, dict) else {}


def _load_mapping(source: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(source)
    except (yaml.YAMLError, RecursionError, TypeError):
        raise ValueError("Invalid Bot YAML") from None
    if not isinstance(value, dict):
        raise ValueError("Bot YAML must contain a mapping")  # noqa: TRY004 - Invalid file contents
    return value


def _within_keys(patch: Any, allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(patch, Mapping) or not set(patch) <= allowed:
        raise ValueError("Patch contains an unsupported Bot setting")
    return patch


def _merge(data: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = data.copy()
    for key, value in patch.items():
        if isinstance(value, Mapping) and not value:
            continue
        result[key] = (
            _merge(_mapping(data.get(key)), value) if isinstance(value, Mapping) else value
        )
    return result


def _validate_patch(data: dict[str, Any], patch: Mapping[str, Any]) -> None:
    _within_keys(patch, {*_PUBLIC_FIELDS, *_SECTIONS, "llm"})
    for key, value in patch.items():
        if key in _SECTIONS:
            _within_keys(value, set(_SECTIONS[key].model_fields))
        elif key == "llm":
            for fields in _within_keys(value, set(_mapping(data.get("llm")))).values():
                _within_keys(fields, _MODEL_FIELDS)
        elif isinstance(value, Mapping):
            raise ValueError("Patch contains an unsupported Bot setting")


class _YamlEdits:
    """Apply editable field changes to PyYAML's source spans."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._newline = "\r\n" if "\r\n" in source else "\n"
        self._edits: list[tuple[int, int, str]] = []
        self._tokens = list(yaml.scan(source))
        self._scalars = [token for token in self._tokens if isinstance(token, yaml.ScalarToken)]

    def render(self, current: dict[str, Any], patch: Mapping[str, Any]) -> str:
        root = yaml.compose(self._source)
        if not isinstance(root, yaml.MappingNode):
            raise ValueError("Bot YAML must contain a mapping")  # noqa: TRY004 - Invalid file contents
        self._patch_mapping(root, current, patch)
        result = self._source
        end_limit = len(result)
        self._edits.sort(key=lambda edit: edit[:2])
        for start, end, replacement in reversed(self._edits):
            if end > end_limit:
                raise ValueError("Overlapping definitions in Bot YAML")
            result = result[:start] + replacement + result[end:]
            end_limit = start
        return result

    def _patch_mapping(self, node: Any, current: dict[str, Any], patch: Mapping[str, Any]) -> None:
        entries = {
            key.value: (key, value) for key, value in node.value if isinstance(key, yaml.ScalarNode)
        }
        additions = {}
        for name, new_value in patch.items():
            if isinstance(new_value, Mapping) and not new_value:
                continue
            if name in current and current[name] == new_value:
                continue
            entry = entries.get(name)
            if entry is None:
                additions[name] = new_value
                continue
            key, old_value = entry
            if old_value.start_mark.index < key.end_mark.index:
                raise ValueError("Edit the original value for an aliased Bot field")
            if isinstance(new_value, Mapping) and isinstance(old_value, yaml.MappingNode):
                self._patch_mapping(old_value, _mapping(current.get(name)), new_value)
            elif (
                isinstance(new_value, Mapping)
                and old_value.tag == "tag:yaml.org,2002:null"
                and not node.flow_style
            ):
                self._fill_empty_block(key, old_value, new_value)
            else:
                self._replace_value(key, old_value, new_value, bool(node.flow_style))

        if not additions:
            return
        end = node.end_mark.index
        if node.flow_style:
            last_value_end = max(
                (value.end_mark.index for _, value in entries.values()), default=end
            )
            trailing_comma = any(
                isinstance(token, yaml.FlowEntryToken)
                and last_value_end <= token.start_mark.index < end
                for token in self._tokens
            )
            delimiter = ", " if entries and not trailing_comma else ""
            fields = ", ".join(
                f"{_dump_inline(key)}: {_dump_inline(value)}" for key, value in additions.items()
            )
            self._edits.append((end - 1, end - 1, delimiter + fields))
        else:
            position = self._block_end(end)
            lines = self._new_lines(additions, node.start_mark.column)
            prefix = self._newline if position and self._source[position - 1] != "\n" else ""
            self._edits.append((position, position, prefix + lines))

    def _block_end(self, position: int) -> int:
        if position == len(self._source):
            return position
        return self._source.rfind("\n", 0, position) + 1

    def _new_lines(self, fields: Mapping[str, Any], column: int) -> str:
        indent = " " * column
        return "".join(
            f"{indent}{key}: {_dump_inline(value)}{self._newline}" for key, value in fields.items()
        )

    def _fill_empty_block(self, key: Any, old: Any, value: Mapping[str, Any]) -> None:
        if not value:
            return
        start, end = old.start_mark.index, old.end_mark.index
        if start != end:
            self._edits.append((start, end, ""))
        newline_at = self._source.find("\n", end)
        position = len(self._source) if newline_at == -1 else newline_at + 1
        # Keep the block's commented-out defaults above the new settings.
        for line in self._source[position:].splitlines(keepends=True):
            rest = line.lstrip(" ")
            if rest.startswith("#") and len(line) - len(rest) > key.start_mark.column:
                position += len(line)
            else:
                break
        prefix = self._newline if position and self._source[position - 1] != "\n" else ""
        self._edits.append(
            (position, position, prefix + self._new_lines(value, key.start_mark.column + 2))
        )

    def _replace_value(self, key: Any, old: Any, value: Any, parent_flow: bool) -> None:
        start, end = old.start_mark.index, old.end_mark.index
        rendered = _dump_inline(value)
        if isinstance(old, yaml.ScalarNode) and old.style not in {"|", ">"}:
            # An implicit null has no token and needs a separator after the colon.
            self._edits.append((start, end, (" " if start == end else "") + rendered))
            return

        if parent_flow or getattr(old, "flow_style", False):
            comments = self._comments(start, end)
            preserved = "".join(
                self._newline + " " * key.start_mark.column + text for text in comments
            )
            suffix = self._newline + " " * key.start_mark.column if comments else ""
            self._edits.append((start, end, preserved + suffix + rendered))
            return

        colon = self._source.index(":", key.end_mark.index, start + 1)
        stop = self._block_end(end)
        header_end = self._source.find("\n", colon, stop)
        header_end = stop if header_end < 0 else header_end
        header_comment = _COMMENT.search(self._source[colon + 1 : header_end])
        comments = self._comments(header_end, stop)
        inline = f" {header_comment[0]}" if header_comment else ""
        indent = " " * (key.start_mark.column + 2)
        footer = "".join(f"{indent}{comment}{self._newline}" for comment in comments)
        self._edits.append((colon + 1, stop, f" {rendered}{inline}{self._newline}{footer}"))

    def _comments(self, start: int, stop: int) -> list[str]:
        comments = []
        for match in _COMMENT.finditer(self._source, start, stop):
            if not any(
                token.start_mark.index <= match.start() < token.end_mark.index
                for token in self._scalars
            ):
                comments.append(match[0])
        return comments


class BotConfigEditor:
    def __init__(self, bots_dir: str | Path, global_config: GlobalConfig) -> None:
        self._directory = Path(bots_dir).resolve()
        self._global_config = global_config
        self._lock = Lock()

    def list_bot_ids(self) -> list[str]:
        if not self._directory.is_dir():
            return []
        return sorted(
            path.stem
            for path in self._directory.glob("*.yaml")
            if _BOT_ID.fullmatch(path.stem) and not path.is_symlink() and path.is_file()
        )

    def _path(self, bot_id: str) -> Path:
        if not isinstance(bot_id, str) or not _BOT_ID.fullmatch(bot_id):
            raise ValueError("Bot ID must contain only ASCII digits")
        return self._directory / f"{bot_id}.yaml"

    def _read(self, path: Path) -> tuple[str, os.stat_result]:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_BYTES:
                    raise ValueError("Bot YAML must be a regular configuration file")
                data = stream.read(_MAX_BYTES + 1)
                if len(data) > _MAX_BYTES:
                    raise ValueError("Bot YAML exceeds the size limit")
                return data.decode("utf-8"), metadata
        except (OSError, UnicodeError):
            raise ValueError("Unable to read Bot YAML") from None

    def _global_secrets(self) -> tuple[str, ...]:
        config = self._global_config
        return tuple(
            item
            for item in (
                config.postgres.dsn,
                config.onebot_ws.access_token,
                config.admin_ui.access_token,
                *(model.api_key for model in config.llm_models.values()),
            )
            if item
        )

    @staticmethod
    def _model_secrets(data: dict[str, Any]) -> tuple[str, ...]:
        found = []
        for model in _mapping(data.get("llm")).values():
            key = _mapping(model).get("api_key")
            if isinstance(key, str) and key:
                found.append(key)
                try:
                    found.append(_expand_env(key, Path("bot.yaml")))
                except ValueError:
                    pass
        return tuple(found)

    def _safe_view(self, data: dict[str, Any]) -> dict[str, Any]:
        secrets = (*self._global_secrets(), *self._model_secrets(data))
        safe = {
            key: redact_payload(data[key], secrets=secrets) for key in _PUBLIC_FIELDS if key in data
        }
        for name, schema in _SECTIONS.items():
            if name in data:
                safe[name] = {
                    key: redact_payload(value, secrets=secrets)
                    for key, value in _mapping(data[name]).items()
                    if key in schema.model_fields
                }
        if "llm" in data:
            safe["llm"] = {}
            for name, value in _mapping(data["llm"]).items():
                if not isinstance(name, str) or not isinstance(value, dict):
                    continue
                key = value.get("api_key")
                metadata = {field: value[field] for field in _MODEL_FIELDS if field in value}
                safe_name = redact_payload(name, secrets=secrets)
                safe_model = {
                    field: redact_payload(value, secrets=secrets)
                    for field, value in metadata.items()
                }
                if "api_key" in value:
                    safe_model["api_key"] = (
                        key if isinstance(key, str) and _ENV_PLACEHOLDER.fullmatch(key) else "***"
                    )
                safe["llm"][safe_name] = safe_model
        return safe

    def read_safe(self, bot_id: str) -> dict[str, Any]:
        source, _ = self._read(self._path(bot_id))
        return self._safe_view(_load_mapping(source))

    def _profile(self, path: Path, bot_id: str, data: dict[str, Any]) -> BotProfile:
        try:
            if str(data.get("bot_id", bot_id)) != bot_id:
                raise ValueError("Bot ID does not match filename")
            resolved = _expand_env({**data, "bot_id": bot_id}, path)
            result = BotProfile.model_validate(resolved)
            models = result.llm or self._global_config.llm_models
            if result.gate.enabled and result.gate_model_key is None:
                raise ValueError("Gate model is required")
            if any(
                key is not None and key not in models
                for key in (result.chat_model_key, result.gate_model_key, result.worker_model_key)
            ):
                raise ValueError("Unknown model slot")
        except (ValueError, TypeError, RecursionError):
            raise ValueError("Invalid Bot configuration") from None
        return result

    def build_preview_profile(self, bot_id: str, draft_patch: Mapping[str, Any]) -> BotProfile:
        path = self._path(bot_id)
        source, _ = self._read(path)
        data = _load_mapping(source)
        _validate_patch(data, draft_patch)
        return self._profile(path, bot_id, _merge(data, draft_patch))

    def preview_secrets(self, bot_id: str, draft_patch: Mapping[str, Any]) -> tuple[str, ...]:
        """Collect current credentials and all variables referenced by this preview."""
        source, _ = self._read(self._path(bot_id))
        data = _load_mapping(source)
        _validate_patch(data, draft_patch)
        pending: list[Any] = [data, draft_patch]
        seen: set[int] = set()
        names: set[str] = set()
        while pending:
            item = pending.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            if isinstance(item, str):
                names.update(_ENV_PLACEHOLDER.findall(item))
            elif isinstance(item, Mapping):
                pending.extend(item.values())
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
        expanded = (os.environ.get(name, "") for name in sorted(names))
        return (
            *self._global_secrets(),
            *self._model_secrets(data),
            *(value for value in expanded if value),
        )

    def update_safe(self, bot_id: str, patch: Mapping[str, Any]) -> dict[str, Any]:
        path = self._path(bot_id)
        with self._lock:
            source, original = self._read(path)
            data = _load_mapping(source)
            _validate_patch(data, patch)
            updated = _merge(data, patch)
            self._profile(path, bot_id, updated)
            try:
                rendered = _YamlEdits(source).render(data, patch)
                if len(rendered.encode("utf-8")) > _MAX_BYTES or _load_mapping(rendered) != updated:
                    raise ValueError("Unable to preserve Bot YAML layout")
            except (ValueError, TypeError, yaml.YAMLError, RecursionError):
                raise ValueError("Unable to preserve Bot YAML layout") from None
            if rendered != source:
                self._replace(path, original, rendered)
        return self._safe_view(updated)

    @staticmethod
    def _replace(path: Path, original: os.stat_result, source: str) -> None:
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
                temporary = stream.name
                os.fchmod(stream.fileno(), stat.S_IMODE(original.st_mode))
                stream.write(source.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            current = path.lstat()
            attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(current, field) != getattr(original, field) for field in attributes):
                raise ValueError("Bot YAML changed during the update")
            os.replace(temporary, path)
            temporary = None
        except OSError:
            raise ValueError("Unable to write Bot YAML") from None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
