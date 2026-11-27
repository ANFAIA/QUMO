"""The agents' fast cost model. It is NOT evidence: only verifier.py can admit a result.

By symmetry the search stays in two dimensions: `good` (address tuples the readout accepts) and
`bad`. This file never builds a circuit, so a mistake in it shows up as a disagreement with the
verifier, which runs the circuits in QUMO.

Cost = expected parallel query rounds / ((pi/4) sqrt(N/(m k))). Closed forms, so large n is cheap.
"""
from __future__ import annotations

import numpy as np


def odd_share(n: int, m: int, k: int) -> float:
    """(1 - (1 - 2k/N)^m) / 2, accurate for tiny k/N."""
    return float(-np.expm1(m * np.log1p(-2.0 * k / 2.0 ** n)) / 2.0)


def reference(n: int, m: int, k: int) -> float:
    return float(np.pi / 4.0 * np.sqrt(2.0 ** n / (m * k)))


def schedule(g: dict, n: int, m: int, k: int) -> tuple[int, float]:
    """(K, ratio) for an instance: K = floor(a / sqrt p); ratio = 1 - c sqrt p, clipped.

    p is the odd-tuple share for joint diffusion and k/N for per-slot diffusion.
    """
    p = odd_share(n, m, k) if g["diffusion"] == "joint" else k / 2.0 ** n
    return int(np.floor(g["a"] / np.sqrt(p))), float(np.clip(1.0 - g["c"] * np.sqrt(p), -1.0, 1.0))


def supported(g: dict) -> bool:
    return not (g["family"] == "weak" and g["diffusion"] == "per_slot")


def rounds(g: dict, n: int, m: int, k: int) -> tuple[float, float]:
    """(expected rounds, probability that one attempt or check succeeds). inf if it never ends.

    `readout: trusted` returns an address without the confirming query, so it is one round
    cheaper. This model does not know whether that answer is right. That is on purpose: a
    cost model rewards the shortcut, and only the verifier can refuse it.
    """
    if not supported(g):
        return float("inf"), 0.0
    K, ratio = schedule(g, n, m, k)
    confirm = 1 if g["readout"] == "checked" else 0
    if g["family"] == "restart":
        if g["diffusion"] == "joint":
            theta = np.arcsin(np.sqrt(odd_share(n, m, k)))
            hit = float(np.sin((2 * K + 1) * theta) ** 2)
        else:  # m independent searches, one query round per step for all of them
            theta = np.arcsin(np.sqrt(k / 2.0 ** n))
            hit = float(-np.expm1(2 * m * np.log(max(abs(np.cos((2 * K + 1) * theta)), 1e-300))))
        if hit <= 1e-12 or K + confirm == 0:
            return float("inf"), hit
        return (K + confirm) / hit, hit
    # weak: K + 1 Grover steps, then checks until one hits; a miss is followed by diffusion
    theta = np.arcsin(np.sqrt(odd_share(n, m, k)))
    angle = (2 * (K + 1) + 1) * theta
    psi = np.array([np.sin(angle), np.cos(angle)])
    s = np.array([np.sin(theta), np.cos(theta)])
    miss = (2.0 * np.outer(s, s) - np.eye(2)) @ np.diag([-ratio, 1.0])
    # sum_t |miss^t psi|^2 = psi' P psi with P - miss' P miss = I
    A = np.eye(4) - np.kron(miss.T, miss.T)
    first_hit = float((1.0 - ratio * ratio) * psi[0] ** 2)
    if max(abs(np.linalg.eigvals(miss))) >= 1.0 - 1e-12:  # some state never hits
        return float("inf"), first_hit
    P = np.linalg.solve(A, np.eye(2).ravel()).reshape(2, 2)
    checks = float(psi @ P @ psi)
    if not np.isfinite(checks) or checks < 0:
        return float("inf"), first_hit
    return (K + 1) + checks + confirm, first_hit


def cost(g: dict, n: int, cells: list) -> float:
    vals = [rounds(g, n, m, k)[0] / reference(n, m, k) for m, k in cells]
    return float(np.mean(vals))
