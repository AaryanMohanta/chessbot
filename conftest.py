"""Registers the `slow` marker and skips slow-marked tests by default.
Run them explicitly with `pytest -m slow` (or `-m ""` to run everything).
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: expensive tests (millions of nodes / many games), opt-in only")


def pytest_collection_modifyitems(config, items):
    if config.option.markexpr:
        return  # user already picked which markers to run (-m ...): don't override
    skip_slow = pytest.mark.skip(reason="slow test: run explicitly with `pytest -m slow`")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
