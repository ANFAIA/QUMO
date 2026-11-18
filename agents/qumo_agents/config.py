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
BACKENDS = ("openrouter", "local", "claude-code", "codex")
LOCAL_KEY_ENV = "QUMO_AGENTS_LOCAL_API_KEY"
API_BACKENDS = ("openrouter", "local")  # a deepagents graph on an OpenAI-compatible endpoint
_SECTION = {"openrouter": "openrouter", "local": "local", "claude-code": "claude_code", "codex": "codex"}


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
        return self._checked(os.environ.get("QUMO_AGENTS_BACKEND") or self.raw.get("default_backend", "openrouter"),
                             "QUMO_AGENTS_BACKEND / default_backend")

    @staticmethod
    def _checked(backend: str, where: str) -> str:
        if backend not in BACKENDS:
            raise ConfigError(f"{where}: unknown back end {backend!r}; choose one of {', '.join(BACKENDS)}")
        return backend

    def backend_for(self, agent: str, override: str | None = None) -> str:
        """The harness that runs an agent: --backend, then QUMO_AGENTS_BACKEND, then the
        agent's entry in "backends", then default_backend."""
        if override:
            return self._checked(override, "--backend")
        if os.environ.get("QUMO_AGENTS_BACKEND"):
            return self.default_backend
        per_agent = self.raw.get("backends", {}).get(agent)
        if per_agent:
            return self._checked(per_agent, f'agents.json "backends" for {agent}')
        return self.default_backend

    def model_for(self, agent: str, override: str | None = None, backend: str = "openrouter",
                  fallback: str | None = None) -> str:
        """The model an agent runs on with one back end.

        --model wins everywhere. QUMO_AGENTS_MODEL comes next for the API back ends
        (openrouter, local); the CLI back ends ignore it, because their model names are
        not OpenRouter ids. Then the back end's own "models" entry for the agent, then
        its "model". openrouter ends at the top-level "models" and "default_model". A CLI
        back end may return "": the CLI then uses its own default model. `fallback` is
        the model of the run that delegates to this agent: a subagent with no local model
        of its own takes it, so `--backend local --model X` covers the whole tree.
        """
        if override:
            return override
        sec = self.section(_SECTION[backend])
        if backend in API_BACKENDS and os.environ.get("QUMO_AGENTS_MODEL"):
            return os.environ["QUMO_AGENTS_MODEL"]
        found = sec.get("models", {}).get(agent) or sec.get("model")
        if found:
            return found
        if backend == "openrouter":
            return self.raw.get("models", {}).get(agent) or self.raw["default_model"]
        if backend == "local" and fallback:
            return fallback
        if backend == "local":
            raise ConfigError('the local back end has no model: pass --model, set QUMO_AGENTS_MODEL, '
                              'or fill "local": {"model": ...} in agents.json')
        return ""

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


def local_chat_model(settings: Settings, model: str):
    """A LangChain chat model on a local OpenAI-compatible server (Ollama, LM Studio,
    llama.cpp, vLLM). QUMO_AGENTS_LOCAL_URL overrides the base URL of agents.json.

    The key, if the server wants one, comes only from QUMO_AGENTS_LOCAL_API_KEY. The file
    cannot name the variable: base_url is in the same file, so a changed agents.json could
    otherwise send another service's key to a host of its choosing. An explicit placeholder
    also keeps the client from reading OPENAI_API_KEY on its own.
    """
    from langchain_openai import ChatOpenAI

    cfg = settings.section("local")
    return ChatOpenAI(
        model=model,
        base_url=local_base_url(settings),
        api_key=os.environ.get(LOCAL_KEY_ENV) or "not-needed",
        timeout=cfg.get("timeout", 600),
        max_retries=cfg.get("max_retries", 1),
    )


def local_base_url(settings: Settings) -> str:
    return os.environ.get("QUMO_AGENTS_LOCAL_URL") or settings.section("local").get("base_url", "http://localhost:11434/v1")


def chat_model(settings: Settings, backend: str, model: str):
    if backend == "local":
        return local_chat_model(settings, model)
    return openrouter_chat_model(settings, model)
