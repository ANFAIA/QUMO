#!/usr/bin/env python3
"""A verified discovery loop for a parallel quantum search, split over CPU, GPU and a QPU emulator.

Reproduces: Shirobokov et al. - multi-agent discovery loop with an external verifier (Sec. 1.2, App. B-C), at toy scale
Source: https://arxiv.org/abs/2611.09822
Source: https://github.com/google-deepmind/formal-quantum-computation
Framework: QUMO - https://github.com/ANFAIA/QUMO

Teams propose programs, an external verifier that the teams cannot edit checks them against a
frozen specification, and only verified programs enter a shared wiki that the teams read in the
next round. Refusals go back to the teams.

What is the same as the paper: frozen specification, isolated verifier that ignores claimed
costs, wiki with the best verified program, recorded dead ends, a minimum gain to merge, cheap
estimates used only to choose what to submit.

What is NOT the same, and matters:
  - the verifier is a QUMO simulator at 7 to 18 qubits, not the Lean kernel: this tests, it does
    not prove, and says nothing about sizes nobody can simulate (the paper's whole point);
  - the search space is five fields that we designed, and it contains the paper's answer. The loop
    selects and tunes; it does not invent circuits;
  - the paper's loop is agentic search by language models, not reinforcement learning. The RL here
    (REINFORCE, a one-step bandit) is this example's own choice of proposer;
  - one RL team and one rules or model team, run in turn; no conferences.

Devices: policy updates and screening on the CPU; the verifier's exact state-vector check on the
GPU when the qumo_py GPU build and the width allow it, otherwise the CPU (the report says which);
last, one attempt of the winning program on a vendor-neutral noisy QPU emulator. With noise
the confirming query can itself be wrong, so a real run must re-check the address classically.

Checks: the planted shortcut (return an address without confirming it) is refused for wrong
answers; every wiki record has no wrong shot; the best verified program is a weakly measured
joint loop with a confirming query, at least 10% cheaper than the independent-search baseline and
within 1% of 1/2 + 1/pi in the screening model; the exact check agrees with that model on the
wide instance; without noise a confirmed answer is always marked; nothing is sent to a remote
device.

Run:  make py && PYTHONPATH=build:plugins/hardware bench/.venv/bin/python examples/heterogeneous/discovery_loop/discovery_loop.py
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 600, "requires": ["qumo_py"], "args": ["--seed", "7"]}
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parents[1]):
    sys.path.insert(0, str(p))

from _checks import CheckLog  # noqa: E402
import hetero_common as H  # noqa: E402
import screening as S  # noqa: E402
import verifier as V  # noqa: E402
from proposers import RLTeam, Screen, ScribeTeam, arm, key  # noqa: E402

PAPER_CONSTANT = 0.5 + 1.0 / math.pi
# the best independent searches in this genome (grid search over a, see README)
BASELINE = {"family": "restart", "diffusion": "per_slot", "readout": "checked", "a": 0.415, "c": 4.0}
# planted on purpose: the same weak loop, one query cheaper because it does not confirm
SHORTCUT = {"family": "weak", "diffusion": "joint", "readout": "trusted", "a": 0.393, "c": 4.0}
FINDANY = "QuantumComputationDiscoveries/Algorithms/Grover/Solution/FindAny.lean"


def formal_reference(repo: Path) -> dict:
    """The constant claimed by the authors' Find-Any solution, read from a local clone.

    Reads text only. It does not build or check any Lean proof. The solutions are on the
    `discoveries` branch; the file is read with `git show`, so no checkout is needed.
    """
    for ref in ("HEAD", "origin/discoveries", "discoveries", "FETCH_HEAD"):
        got = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{FINDANY}"], capture_output=True, text=True)
        if got.returncode != 0:
            continue
        m = re.search(r"def relCost\b[^\n]*:=\s*\n\s*(\d+)\s*/\s*(\d+)", got.stdout)
        if not m:
            raise ValueError(f"no `relCost` fraction in {FINDANY} at {ref}")
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", ref], capture_output=True, text=True)
        return {"ref": ref, "commit": commit.stdout.strip(), "rel_cost": int(m[1]) / int(m[2]),
                "lean_checked": False}
    raise FileNotFoundError(f"{FINDANY} not found in {repo}: fetch the `discoveries` branch")


class Wiki:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.data: dict = {"best": None, "history": [], "dead_ends": []}

    def save(self) -> None:
        (self.root / "wiki.json").write_text(json.dumps(self.data, indent=1))
        rows = ["| Round | Team | Verdict | Screened cost | Candidate | Reason |", "|---|---|---|---|---|---|"]
        for h in self.data["history"]:
            cost = "-" if h["cost"] is None else f"{h['cost']:.4f}"
            rows.append(f"| {h['round']} | {h['team']} | {h['verdict']} | {cost} | "
                        f"`{json.dumps(h['genome'])}` | {h['reason']} |")
        (self.root / "history.md").write_text("\n".join(rows) + "\n")


class Loop:
    def __init__(self, verifier, screen, wiki: Wiki, min_gain: float, say=print):
        self.v, self.screen, self.wiki, self.min_gain, self.say = verifier, screen, wiki, min_gain, say
        self.submitted: set = set()
        self.records: list[dict] = []

    def submit(self, genome: dict, team: str, round_: int) -> dict:
        self.submitted.add(key(genome))
        verdict = self.v.verify(genome)
        entry = {"round": round_, "team": team, "genome": genome, "cost": verdict["cost"],
                 "reason": verdict["reason"]}
        if not verdict["accepted"]:
            entry["verdict"] = "refused"
            if verdict["reason"].startswith("zero error"):
                self.screen.banned[arm(genome)] = verdict["reason"]
                self.wiki.data["dead_ends"].append({"arm": list(arm(genome)), "obstruction": verdict["reason"]})
        else:
            best = self.wiki.data["best"]
            if best is None or verdict["cost"] <= best["cost"] * (1.0 - self.min_gain):
                entry["verdict"] = "merged"
                self.wiki.data["best"] = {"genome": genome, "cost": verdict["cost"], "team": team,
                                          "round": round_}
                self.records.append({**entry, "cells": verdict["cells"]})
            else:
                entry["verdict"] = "verified, gain too small"
        self.wiki.data["history"].append(entry)
        self.wiki.save()
        cost = "" if entry["cost"] is None else f" cost {entry['cost']:.4f}"
        arms = "/".join(arm(genome))
        self.say(f"  round {round_} {team:7s} {arms:26s} a={genome['a']:.3f} c={genome['c']:.2f}: "
                 f"{entry['verdict']}{cost}" + ("" if verdict["accepted"] else f" ({verdict['reason']})"))
        return entry

    def worth_submitting(self, genome: dict) -> bool:
        """Spend the verifier only on a candidate whose estimate would clear the merge threshold,
        or on one from an arm nobody has tried (its estimate may be a shortcut)."""
        best = self.wiki.data["best"]
        return self.screen(genome) <= best["cost"] * (1.0 - self.min_gain)


def attempt_qasm(q, genome: dict) -> tuple[str, int, int, list[int]]:
    """One attempt on N = 8 with one slot, as a flat circuit a device can run: warm-up, one weak
    check, measure the flag, then (on every shot) flags back, one query, measure everything."""
    n, m, marked = 3, 1, [5]
    K, ratio = S.schedule(genome, n, m, 1)
    b = V._Builder(n, m, marked)
    if genome["family"] == "weak":
        b.amplify(K + 1, "joint")
        b.weak_check(ratio)
        b.lines.append(f"c[0] = measure q[{b.flag(0)}];")
        b.gate("x", b.flag(0))
    else:
        b.amplify(K, genome["diffusion"])
    b.oracle()
    for qubit in range(m * (n + 1)):
        b.lines.append(f"c[{1 + qubit}] = measure q[{qubit}];")
    return b.source(1 + m * (n + 1)), n, m, marked


def hit_statistics(counts: dict, n: int, marked: list[int], weak: bool) -> dict:
    """Over the shots of one attempt that count as a hit (weak: the check bit read 1): how often
    the confirming query flags the address, and how often a flagged address is wrong
    (impossible without noise)."""
    total = sum(counts.values())
    hits = flagged = wrong = 0
    for k, c in counts.items():
        bits = int(k.replace(" ", ""), 2)
        if weak and not bits & 1:
            continue  # a miss: the program would diffuse and check again
        hits += c
        read = bits >> 1
        if read & 1:
            flagged += c
            if ((read >> 1) & ((1 << n) - 1)) not in marked:
                wrong += c
    return {"shots": total, "hit": hits / total, "flagged_given_hit": flagged / max(hits, 1),
            "wrong_given_flagged": wrong / max(flagged, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--shots", type=int, default=800)
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="relative gain needed to replace the wiki's best (the paper's default is 0.05)")
    ap.add_argument("--provider", default="rules", help="model profile for the second team (default: no model)")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--gpu-device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--qpu", default="local:qumo-sim:grid:4x4/sc_cz+noise:superconducting")
    ap.add_argument("--formal-repo", type=Path, help="local clone of google-deepmind/formal-quantum-computation")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if H.is_remote(args.qpu):
        ap.error("this example runs local emulators only")
    out = args.out or H.REPO / "build" / "hetero" / "discovery_loop" / f"seed{args.seed}"

    q = H.load_qumo(args.gpu_device)
    log = CheckLog("discovery_loop")
    stages = H.StageLog()
    spec = V.load_spec()

    client = None
    if args.provider != "rules":
        from llm_client import ChatClient, resolve_config
        cfg = resolve_config(args.provider, args.model, args.base_url)
        if cfg.provider != "rules" and (not cfg.api_key_env or cfg.api_key()):
            client = ChatClient(cfg)
        else:
            print(f"[llm] {cfg.api_key_env} is not set; the second team uses the offline rules")

    screen = Screen(spec)
    verifier = V.Verifier(q, spec, args.seed, shots=args.shots)
    wiki = Wiki(out / "wiki")
    loop = Loop(verifier, screen, wiki, args.min_gain)
    teams = [RLTeam(spec, args.seed), ScribeTeam(spec, args.seed, client)]

    print("round 0: the published baseline, then a planted shortcut")
    with H.Timer() as t_verify0:
        loop.submit(BASELINE, "seed", 0)
        shortcut = loop.submit(SHORTCUT, "planted", 0)
    baseline_cost = wiki.data["best"]["cost"]

    propose_s, verify_s = 0.0, t_verify0.seconds
    for r in range(1, args.rounds + 1):
        for team in teams:
            with H.Timer() as t:
                candidate = team.propose(screen, wiki.data, loop.submitted)
            propose_s += t.seconds
            if candidate is None or not loop.worth_submitting(candidate):
                continue
            with H.Timer() as t:
                entry = loop.submit(candidate, team.name, r)
            verify_s += t.seconds
            rl = teams[0]
            rl.forget(screen.banned)
            if entry["verdict"] == "merged":
                rl.adopt(candidate)
    best = wiki.data["best"]

    stages.add(H.StageRecord("agents", "cpu", "policy updates and the screening model are small serial work",
                             propose_s, f"REINFORCE + {teams[1].name}; {screen.evaluations} screened candidates"))
    stages.add(H.StageRecord("verifier-sampling", "cpu", "dynamic circuits (a while loop) run shot by shot",
                             verify_s, f"{verifier.calls} submissions, {args.shots} shots per instance"))

    n, m, k = spec["wide_cell"]
    gpu = H.gpu_status(q)
    device, note = H.place_gpu(gpu, m * (n + 1), args.gpu_device)
    with H.Timer() as t:
        wide = verifier.exact_first_hit(best["genome"], n, m, k, device) if best["genome"]["family"] == "weak" else None
    if wide:
        stages.add(H.StageRecord("verifier-exact-wide", wide["method"].split(":")[0],
                                 "the widest exact state-vector check", t.seconds, wide["method"], False,
                                 [x for x in (note, wide["note"]) if x]))

    qasm, qn, _, marked = attempt_qasm(q, best["genome"])
    weak = best["genome"]["family"] == "weak"
    ideal = hit_statistics(q.sample(q.load_qasm(qasm), 4000, args.seed), qn, marked, weak)
    gate = H.QpuGate()
    with H.Timer() as t:
        job = gate.run(qasm, args.qpu, 4000, args.seed, "one attempt, N=8")
    noisy = hit_statistics(job["counts"], qn, marked, weak) if job.get("counts") else None
    stages.add(H.StageRecord("qpu-attempt", f"qpu:{args.qpu}", "what noise does to the zero-error claim",
                             t.seconds, "one unrolled attempt", True,
                             [f"{(job.get('compiled') or {}).get('two_qubit_gates', '?')} two-qubit gates"]))
    for label, st in (("ideal", ideal), ("noisy", noisy)):
        if st:
            print(f"one attempt, N=8, {label}: hit {st['hit']:.3f}, confirmed given hit "
                  f"{st['flagged_given_hit']:.3f}, wrong given confirmed {st['wrong_given_flagged']:.3f}")

    formal = formal_reference(args.formal_repo) if args.formal_repo else None
    if formal:
        print(f"authors' certified Find-Any constant ({formal['commit'][:10]}): {formal['rel_cost']:.6f}; "
              f"this loop's screened best: {best['cost']:.6f} (their Lean proof was not built or checked here)")

    print(f"baseline {baseline_cost:.4f} -> best verified {best['cost']:.4f} ({best['team']}, round "
          f"{best['round']}): {json.dumps(best['genome'])}")
    log.check("the planted shortcut is refused for wrong answers",
              shortcut["verdict"] == "refused" and shortcut["reason"].startswith("zero error"), shortcut["reason"])
    log.check("every wiki record has no wrong shot",
              all(c["wrong_shots"] == 0 for rec in loop.records for c in rec["cells"]),
              f"{len(loop.records)} records")
    g = best["genome"]
    log.check("the best program is a confirmed weakly measured joint loop",
              arm(g) == ("weak", "joint", "checked"), "/".join(arm(g)))
    log.within("best verified cost over the baseline", best["cost"] / baseline_cost, 0.0, 0.90)
    log.close("screened cost against 1/2 + 1/pi", best["cost"], PAPER_CONSTANT, 0.01 * PAPER_CONSTANT)
    if wide:
        log.close("exact P(hit) on the wide instance against the model", wide["measured"], wide["predicted"],
                  1e-6 if "fp64" in wide["method"] else 1e-3)
    log.check("without noise every confirmed answer is marked", ideal["wrong_given_flagged"] == 0.0,
              f"{ideal['wrong_given_flagged']:.4f}")
    log.check("nothing was sent to a remote device", all(j["local"] for j in gate.jobs),
              f"{len(gate.jobs)} local jobs")
    if formal:
        log.close("the authors' constant is 1/2 + 1/pi", formal["rel_cost"], PAPER_CONSTANT, 1e-12)

    report = {"spec_sha256": V.SPEC_SHA256, "seed": args.seed, "best": best, "baseline_cost": baseline_cost,
              "history": wiki.data["history"], "dead_ends": wiki.data["dead_ends"],
              "stages": [s.to_dict() for s in stages.records], "gpu": gpu, "wide": wide,
              "qpu": {"target": args.qpu, "ideal": ideal, "noisy": noisy}, "formal": formal,
              "rl_mean_cost_curve": teams[0].curve, "llm": teams[1].log}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wiki and report: {out}")
    return 0 if log.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
