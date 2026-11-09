"""Bridge tests of the routing job files: QUMO circuits / targets from a job set, and the benchmark evaluation set.

The halves of the former `test_rl_jobs` that need QUMO: `circuit_from_json` / `target_from_json`
(qumo_bridge.circuits) on every generated circuit and target, and `eval_bench_set` / `eval_sets` over QUMO's
benchmark circuits (`bench/neutral/circuits`, the CRC-32 placement rule, the QASM parser's qubit counts). Also
pins qrl's copy of the heavy-hex lattice against `Target.heavy_hex_edges`. Run from the QUMO root with `-I . -I rl` (rl/Makefile `test-bridge`).
"""

from std.collections import List
from std.os import makedirs
from std.os.path import exists
from qrl.json import JsonValue, parse_json
from qrl.digest import crc32_string
from qrl.targets import eval_topologies, heavy_hex_lattice_edges
from qrl.jobs import JobSet, build_jobs, eval_synth_set, eval_bench_set, eval_sets
from qrl.targets import RTopo
from qumo.compiler.target import Target
from qumo.ir.qasm import load_qasm
from qumo_bridge.circuits import circuit_from_json, target_from_json

comptime OUT_DIR = "build/test_bridge_jobs"
comptime BENCH = "bench/neutral/circuits"


def check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_bridge_jobs failed: " + msg)


def read_json_file(path: String) raises -> JsonValue:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return parse_json(text)


def test_json_roundtrip() raises:
    var js = build_jobs(11, 60, List[RTopo](), List[String](), 20, 0.5, 0.4)
    var doc = parse_json(js.to_json())
    check(len(doc.get("targets")) == 25, "json targets")
    check(len(doc.get("circuits")) == len(js.circuits), "json circuits")
    check(len(doc.get("jobs")) == js.num_jobs(), "json jobs")
    for i in range(len(js.targets)):
        var t = target_from_json(doc.get("targets").at(i))
        check(t.num_qubits == js.targets[i].n, "target_from_json qubits")
        check(doc.get("targets").at(i).get_string("family", "") == js.targets[i].family, "json target family")
    for i in range(len(js.circuits)):
        var cj = doc.get("circuits").at(i)
        var cc = circuit_from_json(cj)
        check(cc.num_qubits == js.circuits[i].n, "circuit_from_json qubits: " + js.circuits[i].name)
        check(len(cc.ops) == js.circuits[i].num_ops(), "circuit_from_json op count: " + js.circuits[i].name)
        check(cj.get("family").as_string() == js.circuits[i].family, "json circuit family")
    for j in range(js.num_jobs()):
        var jj = doc.get("jobs").at(j)
        check(jj.at(0).as_int() == js.jobs[3 * j] and jj.at(1).as_int() == js.jobs[3 * j + 1] and jj.at(2).as_int() == js.jobs[3 * j + 2], "json job row")


def test_eval_bench_set() raises:
    var s = eval_synth_set()
    var bset = eval_bench_set(BENCH)
    check(len(bset.circuits) > 40, "bench: circuits found (" + String(len(bset.circuits)) + ")")
    check(bset.num_jobs() == 3 * len(bset.circuits), "bench: 3 jobs per circuit")
    var known_names: List[String] = ["feynman__W-state.qasm", "feynman__adder.qasm", "feynman__barenco_tof_3.qasm"]
    var known_crc: List[UInt32] = [UInt32(0xB45C9159), UInt32(0xEAEFBDC0), UInt32(0x07B6122A)]
    var found = 0
    for ci in range(len(bset.circuits)):
        var c = bset.circuits[ci].copy()
        check(c.n >= 3 and c.n <= 14, "bench: 3 <= n <= 14 for " + c.name)
        check(c.qasm == String(BENCH) + "/" + c.name + ".qasm", "bench: qasm path " + c.qasm)
        check(exists(c.qasm), "bench: file exists " + c.qasm)
        var h = crc32_string(c.name + ".qasm")
        for k in range(3):
            var row = 3 * (3 * ci + k)
            check(bset.jobs[row] == ci, "bench: circuit index")
            check(bset.jobs[row + 1] == (Int(h) + 3 * k) % 10, "bench: topology (crc32 + 3k) % 10 for " + c.name)
            check(bset.jobs[row + 2] == 7, "bench: seed 7")
        for i in range(3):
            if c.name + ".qasm" == known_names[i]:
                found += 1
                check(h == known_crc[i], "bench: known crc")
    check(found == 3, "bench: the three reference files are in the set (" + String(found) + ")")
    # the qubit counts agree with the QASM parser for the circuits that are in (and a few that are out)
    for ci in range(0, len(bset.circuits), 6):
        var c = bset.circuits[ci].copy()
        check(load_qasm(c.qasm).num_qubits == c.n, "bench: parser qubit count of " + c.name)
    # eval_sets writes both files, loadable as jobs files
    eval_sets(OUT_DIR + "/sets", BENCH)
    var ds = read_json_file(OUT_DIR + "/sets/eval_synth.json")
    var db = read_json_file(OUT_DIR + "/sets/eval_bench.json")
    check(len(ds.get("jobs")) == s.num_jobs() and len(ds.get("targets")) == 10, "eval_synth.json content")
    check(len(db.get("jobs")) == bset.num_jobs() and len(db.get("circuits")) == len(bset.circuits), "eval_bench.json content")
    var c0 = circuit_from_json(db.get("circuits").at(0))
    check(c0.num_qubits == bset.circuits[0].n, "bench circuit loads through circuit_from_json")


def test_heavy_hex_copy() raises:
    for d in range(1, 12, 2):
        var a = heavy_hex_lattice_edges(d)
        var b = Target.heavy_hex_edges(d)
        check(len(a) == len(b), "heavy-hex d=" + String(d) + ": edge count")
        for k in range(len(a)):
            check(a[k] == b[k], "heavy-hex d=" + String(d) + ": entry " + String(k))
    for d in range(0, 7, 2):
        var ra = False
        var rb = False
        try:
            _ = heavy_hex_lattice_edges(d)
        except e:
            ra = True
        try:
            _ = Target.heavy_hex_edges(d)
        except e:
            rb = True
        check(ra and rb, "heavy-hex even d=" + String(d) + " raises in both")


def main() raises:
    makedirs(OUT_DIR, exist_ok=True)
    test_json_roundtrip()
    test_eval_bench_set()
    test_heavy_hex_copy()
    print("test_bridge_jobs OK")
