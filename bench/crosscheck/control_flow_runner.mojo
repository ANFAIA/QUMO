"""Runs every .qasm file of a directory on QUMO for bench/crosscheck/control_flow.py.

For each `x.qasm` it writes `x.qasm.out`: one line per distinct joint outcome,
`count bitstring value value ...`, in the order of the program's outputs, or a
line starting with `error` when the program raises. With a target and a level,
each program is compiled first, so the oracle judges the compiled program, and
`verify.txt` gets the counts of blocks `verify_program` checked and did not check.
"""

from std.collections import Dict
from std.os import listdir
from std.sys import argv

from qumo.compiler.program_compile import compact_qubits, compile_program
from qumo.compiler.target import parse_target
from qumo.ir.qasm import load_any
from qumo.sim.program_runner import sample_program
from qumo.sim.runner import bitstring


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var shots = Int(String(args[2]))
    var seed = UInt64(Int(String(args[3])))
    var checked = 0
    var unchecked = 0
    for name in listdir(dir):
        if not name.endswith(".qasm"):
            continue
        var path = dir + "/" + name
        var out = String()
        try:
            var text: String
            with open(path, "r") as f:
                text = f.read()
            var p = load_any(text)
            if len(args) > 4:
                var r = compile_program(p, parse_target(String(args[4])), Int(String(args[5])))
                checked += r.blocks_checked
                unchecked += r.blocks_unchecked
                p = compact_qubits(r.program)
            var r = sample_program(p, shots, seed)
            var n = len(r.out_names)
            var rows = Dict[String, Int]()
            for s in range(shots):
                var key = bitstring(r.sample.memory[s], r.sample.num_clbits)
                if key.byte_length() == 0:
                    key = "-"
                for j in range(n):
                    var v = r.out_int[s * n + j]
                    if r.out_is_real[j]:
                        key += " " + String(r.out_real[s * n + j])
                    elif p.vars[p.var_index(r.out_names[j])].ty.is_signed():
                        key += " " + String(v)
                    else:
                        key += " " + String(UInt64(v))
                rows[key] = rows.get(key, 0) + 1
            var head = String("outputs")
            for nm in r.out_names:
                head += " " + nm
            out = head + "\n"
            for kv in rows.items():
                out += String(kv.value) + " " + kv.key + "\n"
        except e:
            out = "error " + String(e).split("\n")[0] + "\n"
        with open(path + ".out", "w") as f:
            f.write(out)
    if len(args) > 4:
        with open(dir + "/verify.txt", "w") as f:
            f.write("verify_program: blocks checked " + String(checked) + ", not checked " + String(unchecked) + "\n")
    print("control_flow_runner OK")
