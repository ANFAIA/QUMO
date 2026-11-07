"""Reset-distribution tests of the training environment (component C6, slow: run by `make test-rl-slow`).

The reset generators draw random numbers, and the Rust stream cannot be reproduced, so they are tested
distributionally with fixed seeds (the tests are deterministic):
  * 40,000 resets per case against the exact rational tables of envs/reset_exact.json (scramble states of the
    permutation / linear / Clifford environments, the Pauli tableau, the Pauli rotation-label lists), Pearson
    chi-square with the Wilson-Hilferty score z <= 4.0 (cells below an expected count of 5 are pooled);
  * the rotation-count and first-rotation-weight histograms of envs/reset_empirical.json (100,000 real resets each)
    by a homogeneity chi-square at z <= 4.0, and the solved-at-reset rates by a two-sample binomial z <= 4.0.
"""

from std.collections import Dict, List
from qumo.ai.json import JsonValue
from qumo.compiler.policy import MODEL_KIND_PAULI, model_kind_code
from qumo.rl.gateset import EnvConfig, build_gateset
from qumo.rl.synth_env import TrainEnv
from qumo.rl.testutil import chi_square_gof, chi_square_homogeneity, load_fixture
from qumo.rl.topology import parse_topology
from std.math import sqrt

comptime N_RESETS = 40000
comptime Z_MAX = 4.0


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_rl_env_slow: " + msg)


def cfg_for(kind_name: String, topo: String) raises -> EnvConfig:
    var t = parse_topology(topo)
    var cfg = EnvConfig()
    cfg.kind = model_kind_code(kind_name)
    cfg.n = t.n
    cfg.actions = build_gateset(cfg.kind, t.n, t.edges)
    return cfg^


def words_key(w: List[UInt64], count: Int) -> String:
    var s = String()
    for i in range(count):
        if i > 0:
            s += ","
        s += String(Int(w[i]))
    return s^


def ints_key(v: JsonValue) raises -> String:
    var s = String()
    for i in range(len(v)):
        if i > 0:
            s += ","
        s += String(v.at(i).as_int())
    return s^


def labels_key(v: JsonValue) raises -> String:
    var s = String()
    for i in range(len(v)):
        if i > 0:
            s += "|"
        s += v.at(i).as_string()
    return s^


def labels_join(v: List[String]) -> String:
    var s = String()
    for i in range(len(v)):
        if i > 0:
            s += "|"
        s += v[i]
    return s^


def gof(
    tag: String, keys: Dict[String, Int], probs: List[Float64], counts: List[Int], unknown: Int, total: Int
) raises:
    expect(unknown == 0, tag + ": " + String(unknown) + " of " + String(total) + " resets are outside the exact table")
    var sum_p: Float64 = 0.0
    for i in range(len(probs)):
        sum_p += probs[i]
    expect(sum_p > 0.999999 and sum_p < 1.000001, tag + ": table probabilities do not sum to 1")
    var r = chi_square_gof(counts, probs)
    print("  ", tag, "cells", len(probs), "dof", r.dof, "chi2", r.stat, "z", r.z)
    expect(r.z <= Z_MAX, tag + ": chi-square z = " + String(r.z) + " > " + String(Z_MAX))


def test_scramble_exact() raises:
    var fx = load_fixture("envs/reset_exact.json").get("scramble")
    expect(len(fx) == 15, "15 scramble cases")
    for ci in range(len(fx)):
        var c = fx.at(ci)
        var kind = c.get("kind").as_string()
        var topo = c.get("topology").as_string()
        var d = c.get("difficulty").as_int()
        var cfg = cfg_for(kind, topo)
        var env = TrainEnv(cfg, 1)
        env.set_difficulty(d)
        env.reseed(UInt64(5000 + ci))
        var dist = c.get("dist")
        var index = Dict[String, Int]()
        var probs = List[Float64]()
        for i in range(len(dist)):
            index[ints_key(dist.at(i).at(0))] = i
            probs.append(dist.at(i).at(2).as_float())
        var counts = List[Int](length=len(dist), fill=0)
        var unknown = 0
        var want_depth = c.get("depth_after_reset").as_int()
        for _ in range(N_RESETS):
            env.reset()
            if env.horizon() != want_depth:
                raise Error("test_rl_env_slow: depth after reset is " + String(env.horizon()) + ", want " + String(want_depth))
            var k = words_key(env.state_words(), cfg.n if kind != "clifford" else 2 * cfg.n)
            var slot = index.get(k, -1)
            if slot < 0:
                unknown += 1
            else:
                counts[slot] += 1
        gof("scramble " + kind + " " + topo + " d=" + String(d), index, probs, counts, unknown, N_RESETS)


def test_pauli_tableau_exact() raises:
    var fx = load_fixture("envs/reset_exact.json").get("pauli_tableau")
    expect(len(fx) == 3, "3 Pauli tableau cases")
    for ci in range(len(fx)):
        var c = fx.at(ci)
        var topo = c.get("topology").as_string()
        var d = c.get("difficulty").as_int()
        var cfg = cfg_for("pauli_network", topo)
        var env = TrainEnv(cfg, 1)
        env.set_difficulty(d)
        env.reseed(UInt64(6000 + ci))
        var dist = c.get("dist")
        var index = Dict[String, Int]()
        var probs = List[Float64]()
        for i in range(len(dist)):
            index[ints_key(dist.at(i).at(0))] = i
            probs.append(dist.at(i).at(2).as_float())
        var counts = List[Int](length=len(dist), fill=0)
        var unknown = 0
        for _ in range(N_RESETS):
            env.reset()
            expect(len(env.pauli_labels()) == 0, "difficulty < 16 generates no rotation")
            var k = words_key(env.state_words(), 2 * cfg.n)
            var slot = index.get(k, -1)
            if slot < 0:
                unknown += 1
            else:
                counts[slot] += 1
        gof("pauli tableau " + topo + " d=" + String(d), index, probs, counts, unknown, N_RESETS)


def test_pauli_rotations_exact() raises:
    var fx = load_fixture("envs/reset_exact.json").get("pauli_rotations")
    expect(len(fx) == 4, "4 Pauli rotation cases")
    for ci in range(len(fx)):
        var c = fx.at(ci)
        var topo = c.get("topology").as_string()
        var d = c.get("difficulty").as_int()
        var cfg = cfg_for("pauli_network", topo)
        var env = TrainEnv(cfg, 1)
        env.set_difficulty(d)
        env.reseed(UInt64(7000 + ci))
        var dist = c.get("dist")
        var index = Dict[String, Int]()
        var probs = List[Float64]()
        for i in range(len(dist)):
            index[labels_key(dist.at(i).at(0))] = i
            probs.append(dist.at(i).at(2).as_float())
        var counts = List[Int](length=len(dist), fill=0)
        var unknown = 0
        for _ in range(N_RESETS):
            env.reset()
            var k = labels_join(env.pauli_labels())
            var slot = index.get(k, -1)
            if slot < 0:
                unknown += 1
            else:
                counts[slot] += 1
        gof("pauli rotations " + topo + " d=" + String(d), index, probs, counts, unknown, N_RESETS)


def hist_from_json(v: JsonValue, size: Int) raises -> List[Int]:
    var out = List[Int](length=size, fill=0)
    var ks = v.keys()
    for i in range(len(ks)):
        var idx = Int(ks[i])
        expect(idx >= 0 and idx < size, "histogram key out of range")
        out[idx] = v.get(ks[i]).as_int()
    return out^


def popcount_label(s: String) -> Int:
    var w = 0
    var b = s.as_bytes()
    for i in range(len(b)):
        if Int(b[i]) != ord("I"):
            w += 1
    return w


def test_empirical_histograms() raises:
    var fx = load_fixture("envs/reset_empirical.json")
    var hs = fx.get("pauli_rotation_count_hist")
    expect(len(hs) == 12, "12 rotation-histogram cases")
    for ci in range(len(hs)):
        var c = hs.at(ci)
        var topo = c.get("topology").as_string()
        var d = c.get("difficulty").as_int()
        var cfg = cfg_for("pauli_network", topo)
        var env = TrainEnv(cfg, 1)
        env.set_difficulty(d)
        env.reseed(UInt64(8000 + ci))
        var vis = List[Int](length=cfg.max_rotations + 1, fill=0)
        var wt = List[Int](length=cfg.n + 1, fill=0)
        for _ in range(N_RESETS):
            env.reset()
            var labels = env.pauli_labels()
            var shown = len(labels) if len(labels) < cfg.max_rotations else cfg.max_rotations
            vis[shown] += 1
            if len(labels) > 0:
                wt[popcount_label(labels[0])] += 1
        var ref_vis = hist_from_json(c.get("visible_rotations_hist"), cfg.max_rotations + 1)
        var ref_wt = hist_from_json(c.get("first_rotation_weight_hist"), cfg.n + 1)
        var r1 = chi_square_homogeneity(vis, ref_vis)
        var r2 = chi_square_homogeneity(wt, ref_wt)
        print("  rotation hist", topo, "d=", d, "visible z", r1.z, "weight z", r2.z)
        expect(r1.z <= Z_MAX, "visible-rotation histogram " + topo + " d=" + String(d) + " z=" + String(r1.z))
        expect(r2.z <= Z_MAX, "first-rotation weight histogram " + topo + " d=" + String(d) + " z=" + String(r2.z))


def test_solved_at_reset() raises:
    var rows = load_fixture("envs/reset_empirical.json").get("solved_at_reset_rate")
    expect(len(rows) == 13, "13 solved-at-reset cases")
    for ci in range(len(rows)):
        var c = rows.at(ci)
        var kind = c.get("kind").as_string()
        var topo = c.get("topology").as_string()
        var d = c.get("difficulty").as_int()
        var n_ref = c.get("N").as_int()
        var p_ref = c.get("rate").as_float()
        var cfg = cfg_for(kind, topo)
        var env = TrainEnv(cfg, 1)
        env.set_difficulty(d)
        env.reseed(UInt64(9000 + ci))
        var solved = 0
        for _ in range(N_RESETS):
            env.reset()
            if env.success():
                solved += 1
        var p_new = Float64(solved) / Float64(N_RESETS)
        var pooled = (p_ref * Float64(n_ref) + Float64(solved)) / Float64(n_ref + N_RESETS)
        var tag = kind + " " + topo + " d=" + String(d)
        if pooled == 0.0:
            expect(solved == 0, "solved-at-reset " + tag + ": reference 0, got " + String(solved))
        else:
            var se = sqrt(pooled * (1.0 - pooled) * (1.0 / Float64(n_ref) + 1.0 / Float64(N_RESETS)))
            var z = (p_new - p_ref) / se
            print("  solved-at-reset", tag, "ref", p_ref, "new", p_new, "z", z)
            expect(z <= Z_MAX and z >= -Z_MAX, "solved-at-reset " + tag + " z=" + String(z))


def main() raises:
    test_scramble_exact()
    test_pauli_tableau_exact()
    test_pauli_rotations_exact()
    test_empirical_histograms()
    test_solved_at_reset()
    print("test_rl_env_slow OK")
