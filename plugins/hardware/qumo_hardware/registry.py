"""Provider registry.

Built-in providers: `ibm`, `quantinuum`, `ionq`, `local`, `speqtrum`. Third-party packages can add
providers without touching QUMO by declaring an entry point:

    [project.entry-points."qumo_hardware.providers"]
    braket = "qumo_braket:BraketProvider"
"""

from __future__ import annotations

import inspect
import os
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


def _speqtrum():
    from .speqtrum import SpeQtrumProvider

    return SpeQtrumProvider


def _ionq():
    from .ionq import IonQProvider

    return IonQProvider


def _ibm_native():
    from .ibm_native import IBMNativeProvider

    return IBMNativeProvider


def _speqtrum_native():
    from .speqtrum_native import SpeQtrumNativeProvider

    return SpeQtrumNativeProvider


def _ionq_python():
    from .ionq_sdk import IonQSdkProvider

    return IonQSdkProvider


# Client implementations per built-in provider. "native" runs the provider protocol in
# Mojo (the qumo_py extension); "python" uses the vendor's Python SDK. Native is the
# default wherever it exists; Python stays available as an explicit fallback.
NATIVE: dict[str, Callable[[], type]] = {"ionq": _ionq, "local": _local, "speqtrum": _speqtrum_native,
                                         "ibm": _ibm_native}
PYTHON: dict[str, Callable[[], type]] = {"ionq": _ionq_python, "ibm": _ibm, "quantinuum": _quantinuum,
                                         "speqtrum": _speqtrum}
BUILTIN: dict[str, Callable[[], type]] = {"ibm": _ibm, "quantinuum": _quantinuum, "ionq": _ionq,
                                          "local": _local, "speqtrum": _speqtrum}

CLIENT_VAR = "QUMO_HW_CLIENT"  # native | python; QUMO_HW_CLIENT_<PROVIDER> overrides it per provider
CLIENTS = ("native", "python")


def requested_client(name: str, client: str | None = None) -> str:
    """The client the caller asked for: argument, then QUMO_HW_CLIENT_<NAME>, then QUMO_HW_CLIENT, else native."""
    value = (client or os.environ.get(f"{CLIENT_VAR}_{name.upper()}") or os.environ.get(CLIENT_VAR)
             or "native").strip().lower()
    if value not in CLIENTS:
        raise HardwareError(E_UNKNOWN_BACKEND, f"unknown client '{value}'",
                            f"use --client native (default) or --client python; {CLIENT_VAR}={'|'.join(CLIENTS)}")
    return value


def resolve_client(name: str, client: str | None = None) -> tuple[Callable[[], type], str]:
    """(class loader, client actually used) for built-in provider `name`.

    native requested, provider not ported yet -> its Python client (the only one).
    python requested, provider has no Python client (local) -> refused."""
    want = requested_client(name, client)
    if want == "native" and name in NATIVE:
        return NATIVE[name], "native"
    if want == "python":
        if name not in PYTHON:
            explicit = bool(client) or bool(os.environ.get(f"{CLIENT_VAR}_{name.upper()}"))
            if explicit:
                raise HardwareError(E_UNKNOWN_BACKEND, f"provider '{name}' has no Python fallback client",
                                    f"drop --client python / {CLIENT_VAR}_{name.upper()}=python for this provider")
            return NATIVE[name], "native"  # a global QUMO_HW_CLIENT=python does not apply to native-only providers
        return PYTHON[name], "python"
    return PYTHON[name], "python"  # not ported to Mojo yet


def _entry_points() -> dict[str, Any]:
    try:
        eps = metadata.entry_points(group="qumo_hardware.providers")
    except Exception:
        return {}
    return {ep.name: ep for ep in eps}


def provider_names() -> list[str]:
    return sorted(set(BUILTIN) | set(_entry_points()))


def get_provider(name: str, **options: Any):
    """Instantiate provider `name`; options the constructor does not accept are ignored.

    `client` ("native" | "python", default from QUMO_HW_CLIENT) picks the implementation of a
    built-in provider; the instance records the one used in its `client` attribute."""
    client = options.pop("client", None)
    used = None
    if name in BUILTIN:
        loader, used = resolve_client(name, client)
        cls = loader()
    else:
        eps = _entry_points()
        if name not in eps:
            raise HardwareError(E_UNKNOWN_BACKEND, f"unknown provider '{name}'",
                                f"available providers: {', '.join(provider_names())}")
        cls = eps[name].load()
    params = inspect.signature(cls.__init__).parameters
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    kwargs = {k: v for k, v in options.items() if v is not None and (accepts_kwargs or k in params)}
    inst = cls(**kwargs)
    if used is not None:
        try:
            inst.client = used
        except AttributeError:
            pass
    return inst
