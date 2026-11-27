from pathlib import Path

import pytest
import yaml

from kairos.config import load_bot_profiles, load_global_config


def test_example_configs_load(config_tree: tuple[Path, Path]) -> None:
    global_path, bots_dir = config_tree
    config = load_global_config(global_path)
    (profile,) = load_bot_profiles(bots_dir)

    assert config.postgres.dsn == "postgresql://test:fake@127.0.0.1/kairos_test"
    assert (config.postgres.pool_min_size, config.postgres.pool_max_size) == (2, 10)
    assert config.onebot_ws.host == "127.0.0.1"
    assert config.onebot_ws.send_timeout_seconds == 10.0
    assert config.onebot_ws.access_token == "test-ws-token"
    assert config.logging.level == "INFO"
    assert not config.llm_models
    model = profile.llm[profile.chat_model_key]
    assert (model.provider, model.request_format, model.model, model.api_key) == (
        "google_ai_studio",
        "gemini_generate_content",
        "gemini-3.5-flash-lite",
        "test-gemini-key",
    )
    assert profile.bot_id == "1"
    assert profile.self_id == 10000000000000001
    assert profile.allowed_groups == []
    assert profile.debounce.normal_seconds == 1.8
    assert profile.rate_limit.peer_bot_max_turns == 2
    assert profile.stale_check.reply_max_delta == 5
    assert profile.ring_buffer_size == 40


def test_env_expansion_preserves_literal_values(
    config_tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    global_path, bots_dir = config_tree
    secret = "abc: # value with YAML punctuation"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("TEST_GROUP_ID", "10000000000000002")
    profile_path = bots_dir / "1.yaml"
    profile_text = profile_path.read_text(encoding="utf-8")
    profile_path.write_text(
        profile_text.replace("${GEMINI_API_KEY}", "prefix/${GEMINI_API_KEY}/suffix").replace(
            "allowed_groups: []", 'allowed_groups: ["${TEST_GROUP_ID}"]'
        ),
        encoding="utf-8",
    )

    config = load_global_config(global_path)
    (profile,) = load_bot_profiles(bots_dir, config)
    assert profile.llm["chat"].api_key == f"prefix/{secret}/suffix"
    assert secret not in repr(profile)
    assert profile.allowed_groups == [10000000000000002]


def test_missing_environment_variable(
    config_tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bots_dir = config_tree
    monkeypatch.delenv("GEMINI_API_KEY")

    with pytest.raises(ValueError, match=r"1\.yaml: missing environment variable GEMINI_API_KEY"):
        load_bot_profiles(bots_dir)


def test_legacy_global_model_remains_loadable(config_tree: tuple[Path, Path]) -> None:
    global_path, bots_dir = config_tree
    settings = yaml.safe_load(global_path.read_text(encoding="utf-8"))
    settings["llm_models"] = {
        "chat_model": {
            "base_url": "https://proxy.invalid/v1",
            "api_key": "legacy-key",
            "model": "chat",
        }
    }
    global_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    profile_path = bots_dir / "1.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    del profile["llm"]
    profile["chat_model_key"] = "chat_model"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    config = load_global_config(global_path)
    (loaded,) = load_bot_profiles(bots_dir, config)
    model = config.llm_models[loaded.chat_model_key]
    assert not loaded.llm
    assert model.request_format == "openai_chat_completions"
    assert model.request_temperature == 0.8


def test_invalid_bot_ids_name_their_env_variables(
    config_tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bots_dir = config_tree
    monkeypatch.setenv("TEST_ACCOUNT_ID", "replace-account-id")
    monkeypatch.setenv("TEST_GROUP_ID", "replace-group-id")
    profile_path = bots_dir / "1.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile.update(self_id="${TEST_ACCOUNT_ID}", allowed_groups=["${TEST_GROUP_ID}"])
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Check TEST_ACCOUNT_ID, TEST_GROUP_ID in \.env"):
        load_bot_profiles(bots_dir)


def test_only_selected_numeric_files_are_loaded(
    config_tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bots_dir = config_tree
    profile = yaml.safe_load((bots_dir / "1.yaml").read_text(encoding="utf-8"))
    profile.update(self_id=10000000000000003, nickname="陈阿呆")
    (bots_dir / "3.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    (bots_dir / "2.yaml").write_text("self_id: ${UNDEFINED_DISABLED_BOT}", encoding="utf-8")
    monkeypatch.setenv("KAIROS_ENABLED_BOTS", "3, 1")

    profiles = load_bot_profiles(bots_dir)

    assert [(item.bot_id, item.nickname, item.self_id) for item in profiles] == [
        ("3", "陈阿呆", 10000000000000003),
        ("1", "林小满", 10000000000000001),
    ]


@pytest.mark.parametrize(
    ("selection", "message"),
    (
        ("", "Set KAIROS_ENABLED_BOTS"),
        ("1,legacy_bot", "comma-separated list of numeric Bot IDs"),
        ("1,,2", "comma-separated list of numeric Bot IDs"),
        ("1,1", "duplicate Bot ID"),
        ("2", r"2\.yaml: missing profile selected by KAIROS_ENABLED_BOTS"),
    ),
)
def test_invalid_bot_selection(
    config_tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, selection: str, message: str
) -> None:
    _, bots_dir = config_tree
    monkeypatch.setenv("KAIROS_ENABLED_BOTS", selection)
    with pytest.raises(ValueError, match=message):
        load_bot_profiles(bots_dir)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("bot_id", 1, "bot_id must match"), ("self_id", 10000000000000001, "duplicate self_id")),
)
def test_invalid_profile_identity(
    config_tree: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
    message: str,
) -> None:
    global_path, bots_dir = config_tree
    second_profile = {
        "self_id": 10000000000000002,
        "nickname": "测试群友",
        "persona_prompt": "我是一位测试群友。",
        "chat_model_key": "chat_model",
        field: value,
    }
    (bots_dir / "2.yaml").write_text(yaml.safe_dump(second_profile), encoding="utf-8")
    monkeypatch.setenv("KAIROS_ENABLED_BOTS", "1,2")

    with pytest.raises(ValueError, match=message):
        load_bot_profiles(bots_dir, load_global_config(global_path))


def test_unknown_chat_model_key(config_tree: tuple[Path, Path]) -> None:
    global_path, bots_dir = config_tree
    profile_path = bots_dir / "1.yaml"
    profile_text = profile_path.read_text(encoding="utf-8")
    profile_path.write_text(profile_text + "\nchat_model_key: unknown_model\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown chat_model_key 'unknown_model'"):
        load_bot_profiles(bots_dir, load_global_config(global_path))
