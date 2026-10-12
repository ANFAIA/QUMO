"""Provider registry.

Built-in providers: `ibm`, `quantinuum`, `local`. Third-party packages can add
providers without touching QUMO by declaring an entry point:

    [project.entry-points."qumo_hardware.providers"]
    braket = "qumo_braket:BraketProvider"
"""

from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any, Callable

from .errors import E_UNKNOWN_BACKEND, HardwareError


def _ibm():
    from .ibm import IBMProvider

    return IBMProvider


def _quantinuum():
    from .quantinuum import QuantinuumProvider

    return QuantinuumProvider


def _local():
    from .local import LocalEmulatorProvider

    return LocalEmulatorProvider


BUILTIN: dict[str, Callable[[], type]] = {"ibm": _ibm, "quantinuum": _quantinuum, "local": _local}


def _entry_points() -> dict[str, Any]:
    try:
        eps = metadata.entry_points(group="qumo_hardware.providers")
    except Exception:
        return {}
    return {ep.name: ep for ep in eps}


def provider_names() -> list[str]:
    return sorted(set(BUILTIN) | set(_entry_points()))


def get_provider(name: str, **options: Any):
    """Instantiate provider `name`; options the constructor does not accept are ignored."""
    if name in BUILTIN:
        cls = BUILTIN[name]()
    else:
        eps = _entry_points()
        if name not in eps:
            raise HardwareError(E_UNKNOWN_BACKEND, f"unknown provider '{name}'",
                                f"available providers: {', '.join(provider_names())}")
        cls = eps[name].load()
    params = inspect.signature(cls.__init__).parameters
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    kwargs = {k: v for k, v in options.items() if v is not None and (accepts_kwargs or k in params)}
    return cls(**kwargs)
