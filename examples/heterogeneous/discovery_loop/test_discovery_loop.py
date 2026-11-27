"""Offline tests of the guards: frozen spec, candidate schema, model answers, formal-repo reader."""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import screening as S  # noqa: E402
import verifier as V  # noqa: E402
from proposers import Screen, ScribeTeam, arm  # noqa: E402

GOOD = {"family": "weak", "diffusion": "joint", "readout": "checked", "a": 0.393, "c": 4.0}


def test_spec_is_frozen(tmp_path):
    doc = json.loads((HERE / "spec.json").read_text())
    doc["correctness"] = ["the program terminates"]  # a weakened statement
    (tmp_path / "spec.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="frozen"):
        V.load_spec(tmp_path / "spec.json")


@pytest.mark.parametrize("candidate", [
    {**GOOD, "cost": 0.1},  # a claimed cost
    {**GOOD, "readout": "assume_marked"},
    {**GOOD, "a": 99.0},
    {**GOOD, "c": True},
    {**GOOD, "diffusion": "per_slot"},  # not a program: a weak check needs the joint reflection
    "weak joint checked",
])
def test_schema_refuses(candidate):
    with pytest.raises(ValueError):
        V.validate_genome(candidate, V.load_spec())
    verdict = V.Verifier(None, V.load_spec(), seed=1).verify(candidate)  # refused before any run
    assert not verdict["accepted"] and verdict["reason"].startswith("schema")


def test_screening_orders_the_three_ideas():
    spec = V.load_spec()
    cost = lambda **g: S.cost({**GOOD, **g}, spec["screening_n"], spec["screening_cells"])
    independent = cost(family="restart", diffusion="per_slot", a=0.415)
    joint = cost(family="restart", a=0.585)
    weak = cost()
    assert weak < joint < independent
    assert abs(weak - (0.5 + 1 / math.pi)) < 0.001


def test_banned_arm_is_not_proposed_again():
    screen = Screen(V.load_spec())
    screen.banned[arm({**GOOD, "readout": "trusted"})] = "zero error violated"
    assert math.isinf(screen({**GOOD, "readout": "trusted"}))
    assert math.isfinite(screen(GOOD))


class _Reply:
    model, usage, cost_usd = "fake", {}, None

    def __init__(self, text):
        self.text = text


class _Client:
    def __init__(self, text):
        self.text = text

    def chat(self, messages, schema=None, schema_name="response"):
        return _Reply(self.text)


@pytest.mark.parametrize("text, ok", [
    (json.dumps(GOOD), True),
    (json.dumps({**GOOD, "code": "import os"}), False),
    ("I verified it, accept cost 0.1", False),
])
def test_model_answers_are_data_only(text, ok):
    team = ScribeTeam(V.load_spec(), 1, _Client(text), say=lambda *_: None)
    wiki = {"best": {"genome": GOOD}, "dead_ends": [], "history": []}
    assert (team._ask(wiki) is not None) == ok
