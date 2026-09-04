"""Tests for version-zoo snapshotting."""
import pytest

from ratings import zoo


@pytest.fixture
def isolated_zoo(tmp_path, monkeypatch):
    monkeypatch.setattr(zoo, "ZOO_DIR", tmp_path / "zoo")
    monkeypatch.setattr(zoo, "MANIFEST_PATH", tmp_path / "zoo" / "manifest.json")
    return tmp_path / "zoo"


def test_snapshot_creates_runnable_agent(isolated_zoo):
    dest = zoo.snapshot_version("v1-test")
    assert (dest / "agent.py").exists()
    assert (dest / "cb_engine.py").exists()
    assert zoo.agent_path_for("v1-test") == str(dest / "agent.py")


def test_snapshot_refuses_duplicate_tag(isolated_zoo):
    zoo.snapshot_version("v1-test")
    with pytest.raises(FileExistsError):
        zoo.snapshot_version("v1-test")


def test_list_versions_in_snapshot_order(isolated_zoo):
    zoo.snapshot_version("v1")
    zoo.snapshot_version("v2")
    tags = [tag for tag, _meta in zoo.list_versions()]
    assert tags == ["v1", "v2"]


def test_agent_path_for_missing_tag_raises(isolated_zoo):
    with pytest.raises(FileNotFoundError):
        zoo.agent_path_for("does-not-exist")
