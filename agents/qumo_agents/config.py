"""Configuration: repository paths, models and back ends (agents/agents.json).

API keys come only from environment variables. Nothing here prints, logs or stores
them; error messages name the variable, never its value.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AGENTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AGENTS_DIR.parent
CONFIG_PATH = AGENTS_DIR / "agents.json"
REPORTS_DIR = AGENTS_DIR / "reports"
BACKENDS = ("openrouter", "claude-code", "codex")


class ConfigError(Exception):
    pass


@dataclass
class Settings:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        p = path or Path(os.environ.get("QUMO_AGENTS_CONFIG", CONFIG_PATH))
        with open(p) as fh:
            raw = json.load(fh)
        if raw.get("schema") != "qumo.agents.v1":
            raise ConfigError(f"{p}: expected schema qumo.agents.v1")
        return cls(raw)

    @property
    def default_backend(self) -> str:
        return os.environ.get("QUMO_AGENTS_BACKEND", self.raw.get("default_backend", "openrouter"))

    def model_for(self, agent: str, override: str | None = None) -> str:
        """--model, then QUMO_AGENTS_MODEL, then the agent's entry, then default_model."""
        return (override or os.environ.get("QUMO_AGENTS_MODEL")
                or self.raw.get("models", {}).get(agent) or self.raw["default_model"])

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))


def openrouter_chat_model(settings: Settings, model: str):
    """A LangChain chat model that talks to OpenRouter's OpenAI-compatible API."""
    from langchain_openai import ChatOpenAI

    cfg = settings.section("openrouter")
    env = cfg.get("api_key_env", "OPENROUTER_API_KEY")
    key = os.environ.get(env)
    if not key:
        raise ConfigError(f"set {env} to use the openrouter back end (or pick --backend claude-code / codex)")
    return ChatOpenAI(
        model=model,
        base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"),
        api_key=key,
        default_headers=cfg.get("headers", {}),
        timeout=cfg.get("timeout", 300),
        max_retries=cfg.get("max_retries", 2),
    )
