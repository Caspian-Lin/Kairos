from pathlib import Path
from shutil import copyfile

import pytest


@pytest.fixture
def config_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Stage the checked-in examples as an active configuration with test credentials."""
    examples = Path(__file__).resolve().parents[1] / "config"
    global_path = tmp_path / "global.yaml"
    bots_dir = tmp_path / "bots"
    bots_dir.mkdir()
    copyfile(examples / "global.example.yaml", global_path)
    example_profile = "example_companion.yaml"
    copyfile(examples / "bots" / example_profile, bots_dir / example_profile)
    copyfile(bots_dir / example_profile, bots_dir / "companion.yaml")

    monkeypatch.setenv("KAIROS_PG_DSN", "postgresql://test:fake@127.0.0.1/kairos_test")
    monkeypatch.setenv("KAIROS_ONEBOT_ACCESS_TOKEN", "test-ws-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-llm-key")
    monkeypatch.setenv("KAIROS_BOT_SELF_ID", "10000000000000001")
    return global_path, bots_dir
