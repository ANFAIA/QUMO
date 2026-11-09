"""QUMO glue of the routing job files: a jobs-file circuit / target (qrl.json) to a QUMO `Circuit` / `Target`.

Moved out of `qrl.jobs` (which only deals in flat op lists and edge lists) because these two build QUMO IR objects;
`route_tool`, the router export and the router PPO use them.
"""

from std.collections import List
from qrl.json import JsonValue
from qumo.ir.circuit import Circuit
from qumo.ir.qasm import load_qasm
from qumo.compiler.target import Target


def circuit_from_json(cj: JsonValue) raises -> Circuit:
    if cj.has("qasm"):
        return load_qasm(cj.get("qasm").as_string())
    var n = cj.get("n").as_int()
    var nm = 0
    var ops = cj.get("ops")
    for i in range(len(ops)):
        if ops.at(i).at(0).as_string() == "measure":
            nm += 1
    var c = Circuit(n, nm)
    var m = 0
    for i in range(len(ops)):
        var o = ops.at(i)
        var k = o.at(0).as_string()
        var a = o.at(1).as_int()
        var b = o.at(2).as_int()
        if k == "cx":
            c.cx(a, b)
        elif k == "cz":
            c.cz(a, b)
        elif k == "swap":
            c.swap(a, b)
        elif k == "u2":
            c.rzz(0.3, a, b)
        elif k == "u1":
            c.h(a)
        elif k == "measure":
            c.measure(a, m)
            m += 1
        elif k == "barrier":
            c.barrier()
        else:
            raise Error("QM-E602: unknown op kind '" + k + "' in jobs file. Hint: use cx cz swap u2 u1 measure barrier.")
    return c^


def target_from_json(tj: JsonValue) raises -> Target:
    var n = tj.get("n").as_int()
    var e = List[Int]()
    var ej = tj.get("edges")
    for i in range(len(ej)):
        e.append(ej.at(i).at(0).as_int())
        e.append(ej.at(i).at(1).as_int())
    var t = Target.from_edges(n, e, "any", tj.get_string("name", "custom"))
    return t^
