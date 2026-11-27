"""Two kinds of proposer. Both return DATA (genomes); neither can touch the verifier.

RLTeam     a REINFORCE policy over the genome. It is reinforcement learning in its simplest form,
           a one-step problem (contextual bandit): sample a program, get a reward, move the policy.
           The reward comes from the cheap screening model; the verifier's refusals are fed back.
ScribeTeam reads the wiki and edits the best verified program: with offline rules, or by asking a
           language model (any OpenAI-compatible server, ../llm_client.py) for one genome.
"""
from __future__ import annotations

import json
import math

import numpy as np

import screening as S

DISCRETE = ("family", "diffusion", "readout")


def arm(g: dict) -> tuple:
    return tuple(g[k] for k in DISCRETE)


def key(g: dict) -> tuple:
    return arm(g) + (round(g["a"], 3), round(g["c"], 2))


class Screen:
    """The agents' view: cheap, unverified costs, plus what the verifier has refused."""

    def __init__(self, spec: dict):
        self.spec, self.evaluations = spec, 0
        self.banned: dict[tuple, str] = {}

    def __call__(self, g: dict) -> float:
        self.evaluations += 1
        if arm(g) in self.banned:
            return float("inf")
        return S.cost(g, self.spec["screening_n"], self.spec["screening_cells"])


class RLTeam:
    name = "rl"

    def __init__(self, spec: dict, seed: int, steps: int = 60, batch: int = 64):
        self.spec, self.steps, self.batch = spec, steps, batch
        self.rng = np.random.default_rng(seed)
        self.choices = {k: spec["genome"][k] for k in DISCRETE}
        self.logits = {k: np.zeros(len(v)) for k, v in self.choices.items()}
        self.mean = np.log([0.8, 6.0])  # log a, log c
        self.std = np.array([0.5, 0.6])
        self.best: dict[tuple, tuple[float, dict]] = {}  # arm -> (screened cost, genome)
        self.curve: list[float] = []

    def probabilities(self, name: str) -> np.ndarray:
        e = np.exp(self.logits[name] - self.logits[name].max())
        return e / e.sum()

    def _clip(self, name: str, x: float) -> float:
        lo, hi = self.spec["genome"][name]
        return float(min(max(x, lo), hi))

    def train(self, screen: Screen) -> None:
        for _ in range(self.steps):
            picks, noise, genomes, rewards = [], [], [], []
            for _ in range(self.batch):
                idx = {k: int(self.rng.choice(len(v), p=self.probabilities(k))) for k, v in self.choices.items()}
                z = self.rng.standard_normal(2)
                x = np.exp(self.mean + self.std * z)
                g = {k: self.choices[k][i] for k, i in idx.items()}
                g["a"], g["c"] = self._clip("a", x[0]), self._clip("c", x[1])
                cost = screen(g)
                reward = -min(cost, 3.0) if math.isfinite(cost) else -3.0
                if math.isfinite(cost) and cost < self.best.get(arm(g), (math.inf, None))[0]:
                    self.best[arm(g)] = (cost, g)
                picks.append(idx)
                noise.append(z)
                genomes.append(g)
                rewards.append(reward)
            r = np.array(rewards)
            self.curve.append(float(-r.mean()))
            adv = (r - r.mean()) / (r.std() + 1e-9)
            for k in self.choices:
                grad = np.zeros_like(self.logits[k])
                for a, idx in zip(adv, picks):
                    onehot = np.eye(len(grad))[idx[k]]
                    grad += a * (onehot - self.probabilities(k))
                # keep every choice possible (probability floor through a logit clamp)
                self.logits[k] = np.clip(self.logits[k] + 0.25 * grad / self.batch, -2.0, 2.0)
            z = np.array(noise)
            # the c coordinate only matters for the weak family: learn it from those samples
            used = np.array([[1.0, 1.0 if g["family"] == "weak" else 0.0] for g in genomes])
            count = np.maximum(used.sum(axis=0), 1.0)
            self.mean += 0.35 * self.std * (adv[:, None] * z * used).sum(axis=0) / count
            self.std *= np.exp(0.08 * (adv[:, None] * (z * z - 1.0) * used).sum(axis=0) / count)
            self.std = np.clip(self.std, 0.02, 0.8)

    def forget(self, banned: dict) -> None:
        for a in list(self.best):
            if a in banned:
                del self.best[a]

    def adopt(self, g: dict) -> None:
        """Start the next round from the wiki's best program, with fresh exploration."""
        for k in DISCRETE:
            self.logits[k][self.choices[k].index(g[k])] += 0.5
        self.std = np.maximum(self.std, 0.15)

    def propose(self, screen: Screen, wiki: dict, submitted: set) -> dict | None:
        self.train(screen)
        ranked = sorted(self.best.values(), key=lambda t: t[0])
        for _, g in ranked:
            if key(g) not in submitted:
                return g
        return None


def genome_schema(spec: dict) -> dict:
    rules = spec["genome"]
    props = {k: {"type": "string", "enum": rules[k]} for k in DISCRETE}
    for k in ("a", "c"):
        props[k] = {"type": "number", "minimum": rules[k][0], "maximum": rules[k][1]}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


class ScribeTeam:
    def __init__(self, spec: dict, seed: int, client=None, say=print):
        self.spec, self.client, self.say = spec, client, say
        self.name = "llm" if client else "rules"
        self.rng = np.random.default_rng(seed + 1)
        self.log: list[dict] = []

    def _mutations(self, g: dict) -> list[dict]:
        out = []
        for k in DISCRETE:
            for v in self.spec["genome"][k]:
                if v != g[k]:
                    out.append({**g, k: v})
        for k in ("a", "c"):
            lo, hi = self.spec["genome"][k]
            for f in (0.7, 0.9, 1.1, 1.4):
                out.append({**g, k: float(min(max(g[k] * f, lo), hi))})
        return out

    def propose(self, screen: Screen, wiki: dict, submitted: set) -> dict | None:
        if self.client is not None:
            g = self._ask(wiki)
            if g is not None:
                screen(g)  # counted; the model never sees a cost it did not earn
                return g if key(g) not in submitted else None
        scored = [(screen(m), i, m) for i, m in enumerate(self._mutations(wiki["best"]["genome"]))
                  if key(m) not in submitted]
        scored = [t for t in scored if math.isfinite(t[0])]
        return min(scored)[2] if scored else None

    def _ask(self, wiki: dict) -> dict | None:
        from llm_client import LLMError, extract_json
        from verifier import validate_genome

        notes = {"specification": self.spec, "wiki_best": wiki["best"], "dead_ends": wiki["dead_ends"],
                 "history": wiki["history"][-8:]}
        messages = [
            {"role": "system", "content":
                "You improve a parallel Grover search. Answer with ONE candidate as JSON. family: restart "
                "or weak (weakly measured loop). diffusion: per_slot (independent searches) or joint (one "
                "reflection over all slots). readout: checked (confirm with one more query) or trusted. a "
                "sets the warm-up length floor(a/sqrt(p)); c sets the weak check, ratio = 1 - c sqrt(p). "
                "An external verifier runs your candidate and refuses wrong answers."},
            {"role": "user", "content": json.dumps(notes)},
        ]
        try:
            reply = self.client.chat(messages, schema=genome_schema(self.spec), schema_name="candidate")
            g = validate_genome(extract_json(reply.text), self.spec)
            self.log.append({"model": reply.model, "usage": reply.usage, "cost_usd": reply.cost_usd, "genome": g})
            return g
        except (LLMError, ValueError) as exc:
            self.log.append({"error": str(exc)[:300]})
            self.say(f"[llm] no usable candidate ({str(exc)[:120]}); using the offline rules")
            return None
