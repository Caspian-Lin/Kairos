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
    copyfile(examples / "examples" / "bots" / "1.yaml", bots_dir / "1.yaml")

    monkeypatch.setenv("KAIROS_PG_DSN", "postgresql://test:fake@127.0.0.1/kairos_test")
    monkeypatch.setenv("KAIROS_ONEBOT_ACCESS_TOKEN", "test-ws-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("KAIROS_ENABLED_BOTS", "1")
    return global_path, bots_dir
