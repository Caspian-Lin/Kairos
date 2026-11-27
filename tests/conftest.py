from pathlib import Path
from shutil import copyfile
from unittest.mock import AsyncMock

import pytest


class FakeStore:
    def __init__(self) -> None:
        self.save_message = AsyncMock(return_value=None)
        self.load_recent_messages = AsyncMock(return_value=[])
        self.save_audit_log = AsyncMock(return_value=None)
        self.health_check = AsyncMock(return_value=True)
        self.save_summary = AsyncMock(return_value=None)
        self.load_recent_summaries = AsyncMock(return_value=[])
        self.upsert_user_impression = AsyncMock(return_value=None)
        self.load_user_impressions = AsyncMock(return_value=[])
        self.save_memory_note = AsyncMock(return_value=None)
        self.recall_memory_notes = AsyncMock(return_value=[])


class FakeSender:
    def __init__(self) -> None:
        self.send_group_msg = AsyncMock(return_value=None)
        self.set_msg_emoji_like = AsyncMock(return_value=None)


@pytest.fixture
def config_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Stage the checked-in examples as an active configuration with test credentials."""
    examples = Path(__file__).resolve().parents[1] / "config"
    global_path = tmp_path / "global.yaml"
    bots_dir = tmp_path / "bots"
    bots_dir.mkdir()
    copyfile(examples / "global.example.yaml", global_path)
    copyfile(examples / "examples" / "bots" / "1.yaml", bots_dir / "1.yaml")

    monkeypatch.setenv("KAIROS_PG_DSN", "postgresql://test:fake@127.0.0.1/kairos_test")
    monkeypatch.setenv("KAIROS_ONEBOT_ACCESS_TOKEN", "test-ws-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("KAIROS_ENABLED_BOTS", "1")
    return global_path, bots_dir
