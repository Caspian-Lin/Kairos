from pathlib import Path

import pytest
import yaml

from kairos.admin.config_editor import BotConfigEditor
from kairos.config import GlobalConfig, PostgresConfig
from kairos.telemetry.log_buffer import redact_payload

_YAML = """# profile heading
nickname: original # keep inline
self_id: 10000001
persona_prompt: | # keep persona header
  hello # this belongs to the instruction
  world
llm:
  chat:
    provider: openai
    model: old-model # model note
    api_key: "${KAIROS_TEST_ADMIN_KEY}" # keep placeholder
  worker: {provider: openai, model: worker-model, api_key: literal-api-secret}
gate: # gate heading
  # default gate note
compaction:
memory:
debounce:
rate_limit:
stale_check:
allowed_groups:
- 10001 # group note
- 10002
debug_groups: []
# end of file
"""


@pytest.fixture
def editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[BotConfigEditor, Path]:
    path = tmp_path / "1.yaml"
    monkeypatch.setenv("KAIROS_TEST_ADMIN_KEY", "expanded-api-secret")
    path.write_text(_YAML, encoding="utf-8")
    path.chmod(0o640)
    config = GlobalConfig(postgres=PostgresConfig(dsn="postgresql://test"))
    return BotConfigEditor(tmp_path, config), path


def test_safe_view_returns_only_public_fields_and_placeholder(editor) -> None:
    tool, _ = editor
    view = tool.read_safe("1")
    assert view["llm"]["chat"]["api_key"] == "${KAIROS_TEST_ADMIN_KEY}"
    assert view["llm"]["worker"]["api_key"] == "***"
    assert view["gate"] == {}
    assert "self_id" not in view
    assert "literal-api-secret" not in repr(view)
    assert "expanded-api-secret" not in repr(view)


def test_update_preserves_comments_empty_heads_and_file_permissions(editor) -> None:
    tool, path = editor
    view = tool.update_safe(
        "1",
        {
            "nickname": "林小满",
            "persona_prompt": "new persona\nwith two lines",
            "gate": {"timeout_seconds": 2.0},
            "compaction": {"trigger_tokens": 4096, "target_summary_tokens": 1024},
            "allowed_groups": [10003],
            "llm": {"chat": {"model": "updated-model"}, "worker": {"max_tokens": 300}},
        },
    )
    source = path.read_text(encoding="utf-8")
    for comment in (
        "# profile heading",
        "# keep inline",
        "# keep persona header",
        "# model note",
        "# keep placeholder",
        "# gate heading",
        "# default gate note",
        "# group note",
        "# end of file",
    ):
        assert comment in source
    for block in (
        "gate: # gate heading",
        "compaction:",
        "memory:",
        "debounce:",
        "rate_limit:",
        "stale_check:",
    ):
        assert f"\n{block}\n" in source
    raw = yaml.safe_load(source)
    assert raw["llm"]["worker"]["api_key"] == "literal-api-secret"
    assert raw["llm"]["chat"]["api_key"] == "${KAIROS_TEST_ADMIN_KEY}"
    assert "expanded-api-secret" not in source
    assert view["compaction"]["target_summary_tokens"] == 1024
    assert view["llm"]["worker"]["max_tokens"] == 300
    assert view["allowed_groups"] == [10003]
    assert path.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize(
    "patch",
    [
        {"bot_id": "2"},
        {"self_id": 5},
        {"gate": {"enabled": True}},
        {"debounce": {"mention_seconds": -1}},
        {"llm": {"chat": {"api_key": "overwrite-secret"}}},
        {"llm": {"chat": {"base_url": "https://account:secret@host.example"}}},
        {"llm": {"unknown": {"model": "unused"}}},
        {"persona_prompt": {"api_key": "hidden"}},
    ],
)
def test_invalid_patch_leaves_original_bytes_and_no_temp_files(editor, patch) -> None:
    tool, path = editor
    before = path.read_bytes()
    with pytest.raises(ValueError) as err:
        tool.update_safe("1", patch)
    assert "secret" not in str(err.value)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_path_traversal_and_symlink_are_rejected(editor) -> None:
    tool, path = editor
    (path.parent / "2.yaml").symlink_to(path)
    (path.parent / "3_strict.yaml").write_text(_YAML, encoding="utf-8")
    assert tool.list_bot_ids() == ["1"]
    for bad in ("..", "../1", "1.yaml", "１", "2"):
        with pytest.raises(ValueError):
            tool.read_safe(bad)
        with pytest.raises(ValueError):
            tool.update_safe(bad, {"nickname": "escaped"})
    assert path.read_text(encoding="utf-8") == _YAML


def test_preview_resolves_environment_without_writing_file(editor, monkeypatch) -> None:
    tool, path = editor
    monkeypatch.setenv("KAIROS_TEST_ADMIN_OTHER", "unrelated-credential")
    draft = {"instruction_reminder": "private ${KAIROS_TEST_ADMIN_OTHER}", "prompt_style": "strict"}
    profile = tool.build_preview_profile("1", draft)
    assert profile.prompt_style == "strict"
    assert profile.instruction_reminder == "private unrelated-credential"
    assert (
        redact_payload(profile.instruction_reminder, secrets=tool.preview_secrets("1", draft))
        == "private [REDACTED]"
    )
    assert path.read_text(encoding="utf-8") == _YAML


def test_alias_edit_cannot_accidentally_modify_other_fields(editor) -> None:
    tool, path = editor
    source = _YAML.replace("debug_groups: []", "debug_groups: *groups").replace(
        "allowed_groups:\n", "allowed_groups: &groups\n"
    )
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        tool.update_safe("1", {"debug_groups": [99999]})
    assert path.read_text(encoding="utf-8") == source


def test_empty_patch_is_a_noop_and_additions_keep_indentation(editor) -> None:
    tool, path = editor
    source = _YAML.rstrip() + "\n"
    source = source.replace("\nstale_check:\n", "\n") + "stale_check:"
    path.write_text(source, encoding="utf-8")
    tool.update_safe(
        "1", {"gate": {}, "stale_check": {"reply_max_delta": 7}, "prompt_style": "strict"}
    )
    final = path.read_text(encoding="utf-8")
    assert "\ngate: # gate heading\n  # default gate note\n" in final
    assert "\nstale_check:\n  reply_max_delta: 7\n" in final
    assert yaml.safe_load(final)["prompt_style"] == "strict"


def test_updates_preserve_flow_comments_and_support_small_float(editor) -> None:
    tool, path = editor
    source = _YAML.replace(
        "gate: # gate heading\n  # default gate note",
        "gate: { timeout_seconds: 2, # flow comment\n}",
    )
    path.write_text(source, encoding="utf-8")
    info = tool.update_safe("1", {"gate": {"energy_recover_per_min": 1e-7}})
    assert info["gate"]["energy_recover_per_min"] == 1e-7
    assert "# flow comment" in path.read_text(encoding="utf-8")
