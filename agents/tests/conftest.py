"""Tests read the shared agents.json only: a developer's agents.local.json must not change them."""

import pytest


@pytest.fixture(autouse=True)
def _no_local_agents_config(monkeypatch):
    monkeypatch.setenv("QUMO_AGENTS_LOCAL_CONFIG", "")
