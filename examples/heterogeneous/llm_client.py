"""Minimal OpenAI-compatible chat-completions client (standard library only).

One client reaches many models: OpenRouter (hosted models from many vendors,
one API key) and local servers that speak the same protocol (Ollama, LM
Studio, llama.cpp `llama-server`, vLLM). The heterogeneous example uses it to
ask a model for a *plan*; nothing here knows about quantum computing.

Design rules
------------
* Standard library only (`urllib`). The `openai` package is used only when it
  is installed *and* explicitly requested (`transport="openai"`).
* API keys come only from environment variables named in the configuration
  (`OPENROUTER_API_KEY` by default). They are never logged, printed, stored in
  reports or echoed in error messages: every error text passes through
  `redact()`.
* Structured output: `response_format = {"type": "json_schema", ...}` when the
  provider/model accepts it; otherwise the schema goes into the prompt and the
  JSON is extracted from the text (`extract_json`). Callers must still
  validate the result: a model can return valid JSON that violates the schema.
* Bounded retries with backoff for 429 / 5xx / timeouts / connection errors;
  401 / 403 / 400 are not retried.
* Every response records provider, model, latency, token usage and (OpenRouter)
  the billed cost when the API returns it.

Profiles (examples/heterogeneous/llm_profiles.json) name providers and presets:
`openrouter` (default), `ollama`, `lmstudio`, `llamacpp`, `vllm`, and the
offline `rules` selector, which never calls a model.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILES = HERE / "llm_profiles.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
USER_AGENT = "qumo-heterogeneous-example/1.0"

# --------------------------------------------------------------------------- errors
RETRYABLE_KINDS = {"rate_limit", "server", "timeout", "connection"}


class LLMError(Exception):
    """A failed model call. `kind` is one of: config, auth, rate_limit, server,
    timeout, connection, bad_request, bad_response, refusal, length."""

    def __init__(self, kind: str, message: str, status: int | None = None, retry_after: float | None = None):
        self.kind = kind
        self.status = status
        self.retry_after = retry_after
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS


_SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)((?:api[_-]?key|authorization|x-api-key|token)[\"'\s:=]+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
]


def redact(text: str, secrets: list[str] | tuple[str, ...] = ()) -> str:
    """Remove secrets (exact values, then anything that looks like a key or bearer token)."""
    out = str(text)
    for s in secrets:
        if s and len(s) >= 4:
            out = out.replace(s, "<redacted>")
    for pat in _SECRET_PATTERNS:
        if pat.groups:
            out = pat.sub(lambda m: m.group(1) + "<redacted>", out)
        else:
            out = pat.sub("<redacted>", out)
    return out


# --------------------------------------------------------------------------- config
@dataclass
class LLMConfig:
    provider: str = "openrouter"  # openrouter | local | rules
    preset: str = "openrouter"  # profile name that produced this config
    base_url: str = OPENROUTER_BASE_URL
    model: str = ""
    api_key_env: str = ""  # name of the environment variable holding the key ("" = no key)
    headers: dict[str, str] = field(default_factory=dict)  # non-secret extra headers
    timeout: float = 60.0
    max_retries: int = 2
    backoff: float = 1.0  # seconds; doubled per retry, Retry-After honoured up to max_backoff
    max_backoff: float = 8.0
    structured: str = "auto"  # auto | json_schema | prompt
    temperature: float = 0.0
    max_tokens: int = 2000
    transport: str = "urllib"  # urllib | openai (optional package)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict:
        """JSON-safe description for logs and reports (never contains a key)."""
        d = asdict(self)
        d["api_key_present"] = bool(self.api_key_env and os.environ.get(self.api_key_env))
        return d

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


def load_profiles(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_PROFILES
    with open(p, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or "profiles" not in doc:
        raise LLMError("config", f"{p}: expected an object with 'profiles'")
    return doc


def resolve_config(provider: str | None = None, model: str | None = None, base_url: str | None = None,
                   profiles_path: str | Path | None = None, env: dict | None = None, **overrides: Any) -> LLMConfig:
    """Build an LLMConfig from a profile name (or provider alias) plus CLI overrides.

    `provider` may be a profile name (openrouter, ollama, lmstudio, llamacpp,
    vllm, rules) or the generic 'local' (= the profile named by
    `default_local`, or a custom server when `base_url` is given). Environment
    overrides: QUMO_LLM_PROVIDER, QUMO_LLM_MODEL, QUMO_LLM_BASE_URL."""
    env = os.environ if env is None else env
    doc = load_profiles(profiles_path)
    profiles = doc["profiles"]
    name = provider or env.get("QUMO_LLM_PROVIDER") or doc.get("default", "openrouter")
    if name == "local":
        name = doc.get("default_local", "ollama")
    if name not in profiles:
        raise LLMError("config", f"unknown LLM provider/profile '{name}' (known: {', '.join(sorted(profiles))}, local)")
    prof = dict(profiles[name])
    cfg = LLMConfig(
        provider=prof.get("provider", name),
        preset=name,
        base_url=str(prof.get("base_url", "")),
        model=str(prof.get("model", "")),
        api_key_env=str(prof.get("api_key_env", "")),
        headers={str(k): str(v) for k, v in (prof.get("headers") or {}).items()},
        timeout=float(prof.get("timeout", 60.0)),
        max_retries=int(prof.get("max_retries", 2)),
        structured=str(prof.get("structured", "auto")),
        max_tokens=int(prof.get("max_tokens", 2000)),
        extra_body=dict(prof.get("extra_body") or {}),
    )
    # non-secret header values may come from the environment (e.g. OPENROUTER_HTTP_REFERER)
    for hdr, var in (prof.get("header_env") or {}).items():
        if env.get(var):
            cfg.headers[str(hdr)] = str(env[var])
    m = model or env.get("QUMO_LLM_MODEL")
    if m:
        cfg.model = m
    b = base_url or env.get("QUMO_LLM_BASE_URL")
    if b:
        cfg.base_url = b
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.base_url = cfg.base_url.rstrip("/")
    if cfg.provider not in ("openrouter", "local", "rules"):
        raise LLMError("config", f"profile '{name}': provider must be openrouter, local or rules")
    if cfg.provider != "rules" and not re.match(r"^https?://", cfg.base_url):
        raise LLMError("config", f"profile '{name}': base_url must start with http:// or https://")
    return cfg


# --------------------------------------------------------------------------- response
@dataclass
class LLMResponse:
    text: str
    provider: str
    preset: str
    base_url: str
    model_requested: str
    model: str  # as reported by the server (OpenRouter: the routed model)
    id: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    latency_s: float = 0.0
    structured_mode: str = ""  # json_schema | prompt
    attempts: int = 1
    refusal: str = ""

    def log(self) -> dict:
        d = asdict(self)
        d["text_chars"] = len(self.text)
        return d


def extract_json(text: str) -> Any:
    """First JSON object in `text`: the whole text, a ```json fence, or the first
    balanced {...} block. Raises ValueError when there is none."""
    s = (text or "").strip()
    if not s:
        raise ValueError("empty model output")
    try:
        return json.loads(s)
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    start = s.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except ValueError:
                        break
        start = s.find("{", start + 1)
    raise ValueError("no JSON object found in model output")


# --------------------------------------------------------------------------- transport
Transport = Callable[[str, str, dict, bytes | None, float], tuple[int, dict, bytes]]


def urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float):
    """(status, headers, body) for one HTTP request; network errors raise LLMError."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - URL from the user's profile
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read() or b""
    except (socket.timeout, TimeoutError) as exc:
        raise LLMError("timeout", f"request timed out after {timeout:g} s ({type(exc).__name__})") from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise LLMError("timeout", f"request timed out after {timeout:g} s") from None
        raise LLMError("connection", f"cannot reach {url.split('?')[0]} ({type(reason).__name__}: {reason})") from None
    except OSError as exc:
        raise LLMError("connection", f"cannot reach {url.split('?')[0]} ({type(exc).__name__})") from None


def _status_error(status: int, body: bytes, retry_after: str | None) -> LLMError:
    msg = ""
    try:
        doc = json.loads(body.decode("utf-8", "replace"))
        err = doc.get("error", doc) if isinstance(doc, dict) else doc
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
    except ValueError:
        msg = body[:300].decode("utf-8", "replace")
    msg = (msg or "").strip().splitlines()[0][:300] if msg else ""
    ra = None
    if retry_after:
        try:
            ra = float(retry_after)
        except ValueError:
            ra = None
    if status in (401, 403):
        return LLMError("auth", f"HTTP {status}: authentication failed ({msg or 'check the API key variable'})", status)
    if status == 402:
        return LLMError("auth", f"HTTP 402: insufficient credits ({msg})", status)
    if status == 429:
        return LLMError("rate_limit", f"HTTP 429: rate limited ({msg})", status, ra)
    if status >= 500:
        return LLMError("server", f"HTTP {status}: server error ({msg})", status, ra)
    return LLMError("bad_request", f"HTTP {status}: {msg or 'request rejected'}", status)


class ChatClient:
    """Chat completions against one OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig, transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep):
        if config.provider == "rules":
            raise LLMError("config", "the 'rules' provider has no model to call")
        self.config = config
        self._transport = transport or urllib_transport
        self._sleep = sleep
        self._supports_schema: bool | None = None if config.structured == "auto" else (config.structured == "json_schema")

    # -- helpers
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
        h.update(self.config.headers)
        key = self.config.api_key()
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _secrets(self) -> list[str]:
        k = self.config.api_key()
        return [k] if k else []

    def _request(self, method: str, path: str, payload: dict | None, timeout: float | None = None) -> dict:
        url = self.config.base_url + path
        body = json.dumps(payload).encode() if payload is not None else None
        try:
            status, headers, raw = self._transport(method, url, self._headers(), body, timeout or self.config.timeout)
        except LLMError as exc:
            raise LLMError(exc.kind, redact(str(exc), self._secrets()), exc.status, exc.retry_after) from None
        if status >= 400:
            ra = None
            for k, v in (headers or {}).items():
                if k.lower() == "retry-after":
                    ra = v
            err = _status_error(status, raw, ra)
            raise LLMError(err.kind, redact(str(err), self._secrets()), err.status, err.retry_after)
        try:
            doc = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            raise LLMError("bad_response", f"HTTP {status}: response is not JSON") from None
        if not isinstance(doc, dict):
            raise LLMError("bad_response", "response is not a JSON object")
        if "error" in doc and "choices" not in doc:
            e = doc["error"]
            msg = e.get("message", "") if isinstance(e, dict) else str(e)
            code = e.get("code") if isinstance(e, dict) else None
            kind = "rate_limit" if code == 429 else ("server" if isinstance(code, int) and code >= 500 else "bad_response")
            raise LLMError(kind, redact(f"provider error: {msg[:300]}", self._secrets()), code if isinstance(code, int) else None)
        return doc

    # -- public API
    def list_models(self, timeout: float = 3.0) -> list[str]:
        doc = self._request("GET", "/models", None, timeout)
        return [str(m.get("id")) for m in doc.get("data", []) if isinstance(m, dict) and m.get("id")]

    def reachable(self, timeout: float = 2.0) -> tuple[bool, str]:
        try:
            models = self.list_models(timeout)
        except LLMError as exc:
            return False, f"{self.config.base_url} not reachable ({exc.kind}: {exc})"
        return True, f"{len(models)} model(s) listed"

    def chat(self, messages: list[dict], schema: dict | None = None, schema_name: str = "response") -> LLMResponse:
        """One chat completion with bounded retries; returns the raw text plus metadata.

        With a schema, `response_format` json_schema is requested when supported;
        if the server rejects it (HTTP 400/404/415/422/501), the call is repeated
        once with the schema in the prompt instead (structured_mode='prompt')."""
        cfg = self.config
        if not cfg.model:
            raise LLMError("config", f"no model configured for provider '{cfg.preset}' (use --model)")
        use_schema = schema is not None and self._supports_schema is not False
        attempts = 0
        delay = cfg.backoff
        last: LLMError | None = None
        while attempts <= cfg.max_retries:
            attempts += 1
            msgs = messages if (schema is None or use_schema) else _with_schema_prompt(messages, schema)
            payload: dict[str, Any] = {"model": cfg.model, "messages": msgs, "temperature": cfg.temperature,
                                       "max_tokens": cfg.max_tokens, "stream": False}
            if use_schema:
                payload["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
            if cfg.provider == "openrouter":
                payload["usage"] = {"include": True}
                if use_schema:
                    payload["provider"] = {"require_parameters": True}
            payload.update(cfg.extra_body)
            t0 = time.perf_counter()
            try:
                doc = self._call(payload)
            except LLMError as exc:
                last = exc
                if use_schema and exc.kind == "bad_request" and exc.status in (400, 404, 415, 422, 501):
                    use_schema = False  # provider/model cannot do structured outputs: schema in the prompt
                    self._supports_schema = False
                    attempts -= 1  # the downgrade is not a retry
                    continue
                if exc.retryable and attempts <= cfg.max_retries:
                    wait = min(cfg.max_backoff, exc.retry_after if exc.retry_after is not None else delay)
                    self._sleep(max(0.0, wait))
                    delay *= 2
                    continue
                raise
            latency = time.perf_counter() - t0
            resp = _parse_completion(doc, cfg, latency, "json_schema" if use_schema else ("prompt" if schema else "text"))
            resp.attempts = attempts
            if use_schema:
                self._supports_schema = True
            return resp
        assert last is not None
        raise last

    def _call(self, payload: dict) -> dict:
        if self.config.transport == "openai":
            return self._call_openai(payload)
        return self._request("POST", "/chat/completions", payload)

    def _call_openai(self, payload: dict) -> dict:
        """Optional path through the `openai` package (same endpoint, same payload)."""
        try:
            import openai  # type: ignore
        except ImportError:
            raise LLMError("config", "transport 'openai' requested but the openai package is not installed") from None
        cfg = self.config
        client = openai.OpenAI(base_url=cfg.base_url, api_key=cfg.api_key() or "not-needed",
                               timeout=cfg.timeout, max_retries=0, default_headers=cfg.headers or None)
        body = dict(payload)
        extra = {k: body.pop(k) for k in ("usage", "provider") if k in body}
        extra.update({k: body.pop(k) for k in list(body) if k not in (
            "model", "messages", "temperature", "max_tokens", "stream", "response_format")})
        try:
            r = client.chat.completions.create(**body, extra_body=extra or None)
        except Exception as exc:  # map SDK errors onto LLMError kinds
            status = getattr(exc, "status_code", None)
            text = redact(f"{type(exc).__name__}: {exc}", self._secrets())
            if status is not None:
                err = _status_error(int(status), text.encode(), None)
                raise LLMError(err.kind, text, err.status) from None
            kind = "timeout" if "Timeout" in type(exc).__name__ else "connection"
            raise LLMError(kind, text) from None
        return r.model_dump() if hasattr(r, "model_dump") else json.loads(r.json())


def _with_schema_prompt(messages: list[dict], schema: dict) -> list[dict]:
    note = ("Return ONLY one JSON object, no prose and no code fences, that validates against this JSON "
            "Schema (every property required, no extra properties):\n" + json.dumps(schema, sort_keys=True))
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = str(out[0]["content"]) + "\n\n" + note
    else:
        out.insert(0, {"role": "system", "content": note})
    return out


def _content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):  # content parts
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _parse_completion(doc: dict, cfg: LLMConfig, latency: float, mode: str) -> LLMResponse:
    choices = doc.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMError("bad_response", "response has no choices")
    ch = choices[0]
    msg = ch.get("message") or {}
    usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
    cost = usage.get("cost")
    resp = LLMResponse(
        text=_content_text(msg),
        provider=cfg.provider,
        preset=cfg.preset,
        base_url=cfg.base_url,
        model_requested=cfg.model,
        model=str(doc.get("model") or cfg.model),
        id=str(doc.get("id") or ""),
        finish_reason=str(ch.get("finish_reason") or ch.get("native_finish_reason") or ""),
        usage={k: v for k, v in usage.items() if isinstance(v, (int, float, dict))},
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        latency_s=round(latency, 4),
        structured_mode=mode,
        refusal=str(msg.get("refusal") or ""),
    )
    return resp
