"""Golden corpus: parse_qasm + to_qasm output and seeded sampling must not drift.

`tests/fixtures/golden/manifest.txt` pins an FNV-1a hash of
  * to_qasm(parse_qasm(x)) for every tracked .qasm file and for the inline
    sources below (flat `if`, `switch`, `box`, `for`, modifiers, pragmas), and
  * the counts of seeded trajectory, tableau, branching and static runs, which
    fails on any change in the order of the RNG draws.

A difference here means a straight-line or flat-parser path changed. The file is
regenerated with QUMO_UPDATE_GOLDEN=1 .venv/bin/mojo run -I . tests/test_golden.mojo;
do that only for a change you can justify entry by entry.
"""

from std.os import getenv, listdir
from std.os.path import exists, isdir
from std.testing import assert_true

from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.sim.runner import sample
from qumo.sim.stabilizer import sample_stabilizer
from qumo.sim.branching import sample_branching
from qumo.exec.dispatch import sample_frame

comptime MANIFEST = "tests/fixtures/golden/manifest.txt"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _walk(dir: String, ext: String, mut files: List[String]) raises:
    if not exists(dir):
        return
    for name in listdir(dir):
        var path = dir + "/" + name
        if isdir(path):
            _walk(path, ext, files)  # examples/ is organised in category folders
        elif path.endswith(ext):
            files.append(path)


def _fnv(text: String) -> String:
    """FNV-1a 64, in hex: fixed across runs and platforms, unlike hash()."""
    var h: UInt64 = 0xcbf29ce484222325
    for b in text.as_bytes():
        h = (h ^ UInt64(Int(b))) * 0x100000001b3
    var digits = String("0123456789abcdef")
    var out = String()
    var s = 60
    while s >= 0:
        out += String(digits[byte=Int((h >> UInt64(s)) & 15)])
        s -= 4
    return out^


def _sources() -> List[String]:
    """Flat-parser features; every entry parses and answers correctly today."""
    var s = List[String]()
    s.append("qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0]; if (c[0] == 1) x q[1]; c[1] = measure q[1];")
    s.append("qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0]; if (c[0]) { x q[1]; z q[1]; } else { h q[1]; } c[1] = measure q[1];")
    s.append("qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0]; if (!c[0]) x q[1]; c[1] = measure q[1];")
    s.append("qubit[2] q; bit[2] c; x q[0]; c[0] = measure q[0]; if (c[0] == 1) c[0] = measure q[1]; c[1] = measure q[1];")
    s.append("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[2]; h q[0]; measure q[0] -> c[0]; measure q[1] -> c[1]; if(c==1) x q[2]; if(c==3) x q[1]; measure q[2] -> c[1];")
    s.append("qubit[3] q; bit[2] c; h q[0]; h q[1]; c = measure q[0:1]; switch (c) { case 0 { x q[2]; } case 1, 2 { z q[2]; } default { h q[2]; } } c[0] = measure q[2];")
    s.append("qubit[3] q; bit[2] c; h q[0]; h q[1]; c[0] = measure q[0]; c[1] = measure q[1]; int d; d = c; switch (d) { case 3 { x q[2]; } } c[0] = measure q[2];")
    s.append("qubit[3] q; bit[3] c; box { h q[0]; cx q[0], q[1]; } box [10ns] { x q[2]; } c = measure q;")
    s.append("qubit[4] q; bit[4] c; for uint i in [0:3] { h q[i]; } for int i in {0, 2} { rz(pi/4) q[i]; } c = measure q;")
    s.append("input float theta; const float a = pi/3; int k = 2; qubit[3] q; bit[3] c; rz(2*theta + a) q[k]; ry(theta / 2) q[0]; c = measure q;")
    s.append("gate g(t) a, b { cx a, b; rz(t) b; cx a, b; } qubit[3] q; bit[3] c; g(0.3) q[0], q[1]; inv @ g(0.3) q[1], q[2]; ctrl @ g(0.1) q[0], q[1], q[2]; pow(2) @ h q[0]; negctrl @ x q[0], q[1]; c = measure q;")
    s.append("qubit[2] q; bit[2] c; h q[0]; #pragma qumo depolarize1(0.1) q[0];\n#pragma qumo tick\nc[0] = measure q[0]; c[1] = measure q[1];\n#pragma qumo detector 0\n#pragma qumo observable(0) 1\n")
    s.append("qubit[2] q; bit[2] c; #pragma qumo pauli_channel_2(0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) q[0], q[1];\n#pragma qumo correlated_error(0.2) x q[0], z q[1];\n#pragma qumo else_correlated_error(0.3) y q[0];\n#pragma qumo kraus1 \"bf\" (0, 0, 1, 0, 1, 0, 0, 0) q[1];\nc = measure q;")
    s.append("qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0]; #pragma qumo if (c[0] == 1) x_error(0.5) q[1];\nc[1] = measure q[1];")
    s.append("qubit[2] q; bit[2] c; h q[0]; reset q[0]; x q[0]; barrier q; c = measure q; gphase(0.3); delay[10ns] q[0];")
    s.append("qubit[3] q; bit[3] c; u3(0.1, 0.2, 0.3) q[0]; u2(0.1, 0.2) q[1]; cu1(0.5) q[0], q[1]; cu(0.1, 0.2, 0.3, 0.4) q[0], q[2]; csx q[1], q[2]; ecr q[0], q[1]; iswap q[1], q[2]; rzz(0.4) q[0], q[2]; ccz q[0], q[1], q[2]; sxdg q[0]; c = measure q;")
    return s^


def _dynamic() -> List[String]:
    """Mid-circuit measurement, reset, feed-forward and noise: per-shot paths."""
    var s = List[String]()
    s.append("qubit[3] q; bit[3] c; h q[0]; cx q[0], q[1]; c[0] = measure q[0]; if (c[0] == 1) x q[1]; h q[2]; t q[2]; cx q[2], q[1]; c[1] = measure q[1]; reset q[1]; rx(0.7) q[1]; c[2] = measure q[1];")
    s.append("qubit[2] q; bit[2] c; ry(0.4) q[0]; #pragma qumo depolarize1(0.05) q[0];\n#pragma qumo x_error(0.1) q[1];\n#pragma qumo depolarize2(0.03) q[0], q[1];\nc[0] = measure q[0]; ry(1.1) q[1]; c[1] = measure q[1];")
    s.append("qubit[2] q; bit[2] c; h q[0]; #pragma qumo pauli_channel_2(0.1, 0.02, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1) q[0], q[1];\n#pragma qumo correlated_error(0.2) x q[0], z q[1];\n#pragma qumo else_correlated_error(0.3) y q[0];\n#pragma qumo kraus1 \"ad\" (1, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0.4358898943540674, 0, 0, 0, 0, 0) q[1];\nt q[0]; c = measure q;")
    s.append("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[2]; h q[0]; h q[1]; measure q[0] -> c[0]; measure q[1] -> c[1]; if(c==3) x q[2]; measure q[2] -> c[0];")
    return s^


def _clifford() -> List[String]:
    var s = List[String]()
    s.append("qubit[3] q; bit[3] c; h q[0]; cx q[0], q[1]; c[0] = measure q[0]; if (c[0] == 1) x q[2]; h q[1]; c[1] = measure q[1]; reset q[0]; h q[0]; c[2] = measure q[0];")
    s.append("qubit[2] q; bit[2] c; h q[0]; #pragma qumo depolarize1(0.2) q[0];\n#pragma qumo x_error(0.1) q[1];\ncx q[0], q[1]; c = measure q;")
    return s^


def _collect() raises -> List[String]:
    """`name hash` lines, in a fixed order."""
    var files = List[String]()
    _walk("bench/neutral/circuits", ".qasm", files)
    _walk("examples", ".qasm", files)
    sort(files)
    var out = List[String]()
    for f in files:
        out.append("file:" + f + " " + _fnv(to_qasm(parse_qasm(_read(f)))))
    var src = _sources()
    for i in range(len(src)):
        out.append("source:" + String(i) + " " + _fnv(to_qasm(parse_qasm(src[i]))))
    var dyn = _dynamic()
    for i in range(len(dyn)):
        for seed in range(3):
            var r = sample(parse_qasm(dyn[i]), 400, UInt64(11 + seed))
            out.append("traj:" + String(i) + ":" + String(seed) + " " + _fnv(r.method + String(r)))
    var cl = _clifford()
    for i in range(len(cl)):
        var c = parse_qasm(cl[i])
        var a = sample_stabilizer(c, 400, 7)
        out.append("tableau:" + String(i) + " " + _fnv(a.method + String(a)))
        var b = sample_branching(c, 200, 7)
        out.append("branching:" + String(i) + " " + _fnv(b.method + String(b)))
        var d = sample(c, 400, 7)
        out.append("clifford_sv:" + String(i) + " " + _fnv(d.method + String(d)))
    var fr = parse_qasm(cl[1])
    var f = sample_frame(fr, 2048, 5)
    out.append("frame:0 " + _fnv(f.method + String(f)))
    var st = parse_qasm("qubit[6] q; bit[6] c; h q; cx q[0], q[1]; t q[2]; cx q[2], q[3]; rx(0.3) q[4]; c = measure q;")
    var r = sample(st, 4000, 21)
    out.append("static:0 " + _fnv(r.method + String(r)))
    return out^


def main() raises:
    var got = _collect()
    var text = String()
    for line in got:
        text += line + "\n"
    if getenv("QUMO_UPDATE_GOLDEN") == "1":
        with open(MANIFEST, "w") as f:
            f.write(text)
        print("manifest regenerated:", len(got), "entries")
    assert_true(exists(MANIFEST), "missing " + String(MANIFEST))
    var want = _read(MANIFEST).split("\n")
    var bad = String()
    var n = 0
    for i in range(len(got)):
        if i >= len(want) or String(want[i]) != got[i]:
            n += 1
            if n <= 10:
                bad += "\n  now " + got[i] + "\n  was " + (String(want[i]) if i < len(want) else String("<missing>"))
    assert_true(n == 0, String(n) + " golden entries changed:" + bad)
    assert_true(len(want) == len(got) + 1, "manifest has entries the corpus no longer produces")
    print("test_golden OK")
