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
    assert config.llm_models[profile.chat_model_key].api_key == "test-llm-key"
    assert profile.bot_id == "lin_xiaoman"
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
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("TEST_GROUP_ID", "10000000000000002")
    global_text = global_path.read_text(encoding="utf-8")
    global_path.write_text(
        global_text.replace("${OPENAI_API_KEY}", "prefix/${OPENAI_API_KEY}/suffix"),
        encoding="utf-8",
    )
    profile_path = bots_dir / "companion.yaml"
    profile_text = profile_path.read_text(encoding="utf-8")
    profile_path.write_text(
        profile_text.replace("allowed_groups: []", 'allowed_groups: ["${TEST_GROUP_ID}"]'),
        encoding="utf-8",
    )

    config = load_global_config(global_path)
    assert config.llm_models["chat_model"].api_key == f"prefix/{secret}/suffix"
    assert secret not in repr(config)
    (profile,) = load_bot_profiles(bots_dir, config)
    assert profile.allowed_groups == [10000000000000002]


def test_missing_environment_variable(
    config_tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    global_path, _ = config_tree
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(
        ValueError, match=r"global\.yaml: missing environment variable OPENAI_API_KEY"
    ):
        load_global_config(global_path)


@pytest.mark.parametrize(
    ("field", "value"), (("bot_id", "lin_xiaoman"), ("self_id", "${KAIROS_BOT_SELF_ID}"))
)
def test_duplicate_profile_identity(config_tree: tuple[Path, Path], field: str, value: str) -> None:
    global_path, bots_dir = config_tree
    second_profile = {
        "bot_id": "second",
        "self_id": "10000000000000002",
        "nickname": "测试群友",
        "persona_prompt": "我是一位测试群友。",
        "chat_model_key": "chat_model",
        field: value,
    }
    (bots_dir / "second.yaml").write_text(yaml.safe_dump(second_profile), encoding="utf-8")

    with pytest.raises(ValueError, match=f"duplicate {field}"):
        load_bot_profiles(bots_dir, load_global_config(global_path))


def test_unknown_chat_model_key(config_tree: tuple[Path, Path]) -> None:
    global_path, bots_dir = config_tree
    profile_path = bots_dir / "companion.yaml"
    profile_text = profile_path.read_text(encoding="utf-8")
    profile_path.write_text(
        profile_text.replace("chat_model_key: chat_model", "chat_model_key: unknown_model"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown chat_model_key 'unknown_model'"):
        load_bot_profiles(bots_dir, load_global_config(global_path))
