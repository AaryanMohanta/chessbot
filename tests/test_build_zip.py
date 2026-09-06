"""tools/build_zip.py's build-tag mechanism (2026-09, item 1 of the
ladder-as-test-harness change): identifies which build/config produced a
shipped zip, and can seed environment-variable feature-flag overrides
into it -- since the real ladder runner gives no way to set our own
environment, this is the only lever for A/B testing a flag on the actual
ladder rather than only in local SPRT.
"""
from __future__ import annotations

import importlib
import sys

import pytest

sys.path.insert(0, "tools")
import build_zip  # noqa: E402


@pytest.fixture
def isolated_build_tag_path(tmp_path, monkeypatch):
    path = tmp_path / "cb_build_tag.py"
    monkeypatch.setattr(build_zip, "BUILD_TAG_PATH", path)
    return path


def test_generate_build_tag_file_writes_the_tag(isolated_build_tag_path):
    build_zip.generate_build_tag_file("my-tag", {}, path=isolated_build_tag_path)
    text = isolated_build_tag_path.read_text(encoding="utf-8")
    assert "BUILD_TAG = 'my-tag'" in text


def test_generate_build_tag_file_is_valid_importable_python(isolated_build_tag_path, monkeypatch):
    build_zip.generate_build_tag_file("import-check", {"SOME_TEST_FLAG_XYZ": "1"}, path=isolated_build_tag_path)
    monkeypatch.setenv("SOME_TEST_FLAG_XYZ", "", prepend=False)
    monkeypatch.delenv("SOME_TEST_FLAG_XYZ", raising=False)
    sys.path.insert(0, str(isolated_build_tag_path.parent))
    try:
        import cb_build_tag  # noqa: F401 -- exercising the generated module, not using it directly here
        importlib.reload(cb_build_tag)
        assert cb_build_tag.BUILD_TAG == "import-check"
        assert cb_build_tag.FLAG_OVERRIDES == {"SOME_TEST_FLAG_XYZ": "1"}
    finally:
        sys.path.remove(str(isolated_build_tag_path.parent))
        sys.modules.pop("cb_build_tag", None)


def test_generate_build_tag_file_seeds_environ_via_setdefault(isolated_build_tag_path, monkeypatch):
    monkeypatch.delenv("SOME_OTHER_TEST_FLAG_XYZ", raising=False)
    build_zip.generate_build_tag_file("env-check", {"SOME_OTHER_TEST_FLAG_XYZ": "1"}, path=isolated_build_tag_path)
    sys.path.insert(0, str(isolated_build_tag_path.parent))
    try:
        sys.modules.pop("cb_build_tag", None)
        import cb_build_tag  # noqa: F401 -- import for its os.environ.setdefault side effect
        import os
        assert os.environ["SOME_OTHER_TEST_FLAG_XYZ"] == "1"
    finally:
        sys.path.remove(str(isolated_build_tag_path.parent))
        sys.modules.pop("cb_build_tag", None)
        monkeypatch.delenv("SOME_OTHER_TEST_FLAG_XYZ", raising=False)


def test_default_build_tag_returns_a_nonempty_string():
    # Either a real git short SHA (+ "-dirty") inside this repo, or the
    # "unknown" fallback outside one -- either way, never blows up.
    tag = build_zip.default_build_tag()
    assert isinstance(tag, str) and len(tag) > 0


def test_build_writes_cb_build_tag_with_the_requested_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(build_zip, "BUILD_TAG_PATH", tmp_path / "cb_build_tag.py")
    out = tmp_path / "submission.zip"
    build_zip.build(out, tag="explicit-tag", flags={"CB_TEST_FLAG": "1"})
    assert out.exists()

    import zipfile
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "cb_build_tag.py" in names
        content = zf.read("cb_build_tag.py").decode("utf-8")
        assert "BUILD_TAG = 'explicit-tag'" in content
        assert "'CB_TEST_FLAG': '1'" in content


def test_parse_flag_rejects_missing_equals():
    with pytest.raises(Exception):
        build_zip._parse_flag("NO_EQUALS_SIGN")


def test_parse_flag_splits_key_value():
    assert build_zip._parse_flag("KEY=VALUE") == ("KEY", "VALUE")
