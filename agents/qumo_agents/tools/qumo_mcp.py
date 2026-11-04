"""QUMO's own MCP server (build/qumo mcp) as LangChain tools.

The server's descriptions say what each tool computes. Agents also need to know when
to pick it, what to call first, and the mistakes that cost a turn, so each description
gets a short guide in front of the server text. Tools without a guide keep the server
text unchanged, and a server tool the guide does not know still loads.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from qumo_agents.config import REPO_ROOT

QUMO_BIN = Path(os.environ.get("QUMO_BIN", REPO_ROOT / "build" / "qumo"))

# name -> (use when, prefer instead / call first, pitfalls)
TOOL_GUIDE: dict[str, tuple[str, str, str]] = {
    "qumo_describe": (
        "you need facts about QUMO itself: gate list and syntax, an error code's meaning and fix, capabilities, compiler passes or AI models.",
        "call it before guessing a gate name, a pass name or what an error code means.",
        "topic is required in practice: 'gates', 'gate'+name, 'errors', 'error'+name (e.g. QM-E506), 'capabilities', 'passes', 'models'."),
    "qumo_build_circuit": (
        "a standard algorithm is wanted (ghz, qft, grover, bernstein_vazirani, deutsch_jozsa, qaoa_maxcut, hea, uccsd, trotter, kernels).",
        "use it instead of writing gates by hand; pass its OpenQASM to the simulators.",
        "unknown kinds or arguments are refused (QM-E801/QM-E802); parameterised output needs 'params' when simulated."),
    "qumo_validate_qasm": (
        "you wrote or received OpenQASM and want parse errors and lint before running anything.",
        "always run it first on hand-written or user-supplied QASM.",
        "it does not simulate; errors carry QM-E5xx codes with line and column."),
    "qumo_circuit_info": (
        "you need size facts without simulating: qubits, depth, gate counts, 2-qubit gates, lint.",
        "use before choosing a simulator, or to report a circuit's cost.", ""),
    "qumo_estimate_resources": (
        "you must decide how or where to run: recommended simulator, memory, NISQ success estimate, fault-tolerant cost.",
        "call before simulating anything above ~20 qubits or before hardware.",
        "profile is 'superconducting' | 'trapped_ion' | 'neutral_atom'."),
    "qumo_run_qasm": (
        "you need measurement counts of a circuit or a program with mid-circuit measurement or while loops.",
        "qumo_statevector for exact amplitudes, qumo_expectation for <H>, qumo_emulate for a device's noise.",
        "counts are bitstrings with the leftmost character as the highest qubit; set seed for reproducible runs."),
    "qumo_statevector": (
        "you need exact amplitudes (small circuits, debugging, checking an algorithm).",
        "qumo_run_qasm when only counts matter.", "memory grows as 2^n; respect max_qubits."),
    "qumo_expectation": (
        "you need an exact expectation value of a Pauli-sum observable, e.g. '0.5*Z0 Z1 - 1.2*X0'.",
        "", "observable syntax is coefficient*Pauli+index terms separated by + or -."),
    "qumo_density": (
        "noise must be exact (density matrix), up to 15 qubits.", "qumo_emulate for larger noisy circuits (sampled).", ""),
    "qumo_emulate": (
        "you want counts under a target device's noise model (T1/T2, gate and readout errors).",
        "compile first with qumo_transpile or set compile=true.", ""),
    "qumo_run_device": (
        "a specific simulator device or precision is required (CPU, NVIDIA, AMD, Apple GPU).",
        "call qumo_gpu_info first to see if a GPU exists.", ""),
    "qumo_gpu_info": ("you need to know whether this build has a usable GPU.", "", ""),
    "qumo_transpile": (
        "a circuit must run on a target topology and basis, or you want it optimised.",
        "give target as a one-line spec ('grid:4x4/sc_cz', 'heavy_hex:5/rz,sx,x,ecr', 'line:20/u,cx') or edges+basis.",
        "level 0-3 picks a preset; 'ai': true is refused for programs with runtime control flow (QM-E630)."),
    "qumo_draw": ("you need to show a circuit to a person.", "", ""),
    "qumo_convert": ("you need canonical OpenQASM 3 or circuit JSON from another form.", "", ""),
    "qumo_to_pbc": ("you study Pauli-based computation or T-count after commuting Cliffords out.", "", ""),
    "qumo_vqe": (
        "you need a ground-state energy of a Pauli-sum Hamiltonian.",
        "set seed; compare with qumo_expectation on the returned circuit if you need a check.", ""),
    "qumo_qaoa_maxcut": ("you solve MaxCut on a small graph given as '0-1 1-2 2-3:0.5'.", "", ""),
    "qumo_qec_memory": (
        "you need a logical error rate of a memory experiment for a code, distance, rounds and p.",
        "qumo_qec_dem to inspect the detector error model.", "set max_errors and shots to bound run time."),
    "qumo_qec_dem": ("you need the detector error model (Stim DEM text) of a memory circuit.", "", ""),
    "qumo_stim_import": ("you have Stim circuit text to analyse or convert.", "", ""),
    "qumo_ft_codes": ("you need QUMO's stabilizer codes for fault-tolerant programs.", "", ""),
    "qumo_ft_encode": (
        "you want to encode a logical circuit on a code and sample it with noise.",
        "qumo_ft_codes first to choose the code.", ""),
    "qumo_ft_verify": ("you must prove a physical gadget implements a logical operation exactly.", "", ""),
}


def guided_description(name: str, server_text: str) -> str:
    g = TOOL_GUIDE.get(name)
    if not g:
        return server_text
    use, prefer, pitfalls = g
    lines = [f"Use when {use}"]
    if prefer:
        lines.append(f"Instead or first: {prefer}")
    if pitfalls:
        lines.append(f"Watch out: {pitfalls}")
    return "\n".join(lines) + "\n\n" + server_text


def server_config() -> dict:
    if not QUMO_BIN.exists():
        raise FileNotFoundError(f"{QUMO_BIN} not found: build it with `make cli`, or set QUMO_BIN")
    return {"qumo": {"command": str(QUMO_BIN), "args": ["mcp"], "transport": "stdio"}}


async def load_qumo_tools_async(names: tuple[str, ...] | None = None):
    """QUMO MCP tools as LangChain tools, with guided descriptions. `names` keeps a subset."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(server_config())
    tools = await client.get_tools()
    out = []
    for t in tools:
        if names and t.name not in names:
            continue
        t.description = guided_description(t.name, t.description or "")
        out.append(t)
    return out


def load_qumo_tools(names: tuple[str, ...] | None = None):
    return asyncio.run(load_qumo_tools_async(names))
