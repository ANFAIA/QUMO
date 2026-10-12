"""Pass manager with an LLM-friendly text pipeline language.

A pipeline is a comma separated list of passes, each optionally with
arguments in parentheses (commas inside parentheses belong to the pass):

    remove_identities,elide_swaps,fixpoint(cancel,commute,fuse1q),
    route(sabre,layout=auto,trials=20,swap_trials=20,seed=7),kak,
    basis(rz,sx,x,cz),fixpoint(cancel,commute,eject_z,fuse1q),check

Grammar:   pipeline := item ((","|";") item)*
           item     := name [ "(" arg ("," arg)* ")" ]
           arg      := word | key "=" value | item        (fixpoint nests)

Design (best-of synthesis): Qiskit's PassManager + preset levels O0-O3,
TKET's predicates (the `check` pass verifies basis + connectivity and fails
loudly), Cirq's "optimizers as plain functions". Dynamic dispatch is a switch
over pass ids (no trait objects), each pass reports gates / 2q / depth before
and after plus wall time, and errors are stable `QM-E6xx` codes with a
`Hint:` listing valid choices (with a did-you-mean suggestion) so agents can
self-correct. `CompileResult.to_json()` is the machine-readable summary.
"""

from std.time import perf_counter_ns
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.common import GateSet, fmt_num, basis_str, wrap_angle, TWO_PI
from qumo.compiler.target import Target, parse_basis
from qumo.compiler.optimize import (
    remove_identities, cancel_inverses, merge_rotations, commute_rotations,
    fuse_1q, eject_z, elide_swaps,
)
from qumo.compiler.kak import consolidate_2q
from qumo.compiler.pauli_simp import pauli_resynthesize
from qumo.compiler.phase_fold import phase_fold
from qumo.compiler.basis import translate_to_basis, unroll_3q
from qumo.compiler.routing import (
    RouteOptions, RoutePlan, route_circuit, star_preroute, plan_route, finish_route, has_exact_layout,
)


struct PassId:
    comptime REMOVE_IDENTITIES = 0
    comptime CANCEL = 1
    comptime MERGE = 2
    comptime COMMUTE = 3
    comptime FUSE1Q = 4
    comptime PHASE_FOLD = 5
    comptime EJECT_Z = 6
    comptime UNROLL3Q = 7
    comptime ROUTE = 8
    comptime BASIS = 9
    comptime FIXPOINT = 10
    comptime CHECK = 11
    comptime KAK = 12
    comptime ELIDE_SWAPS = 13
    comptime PAULI_SIMP = 14
    comptime NUM = 15


def pass_name(id: Int) -> String:
    if id == PassId.REMOVE_IDENTITIES: return "remove_identities"
    if id == PassId.CANCEL: return "cancel"
    if id == PassId.MERGE: return "merge"
    if id == PassId.COMMUTE: return "commute"
    if id == PassId.FUSE1Q: return "fuse1q"
    if id == PassId.PHASE_FOLD: return "phase_fold"
    if id == PassId.EJECT_Z: return "eject_z"
    if id == PassId.UNROLL3Q: return "unroll3q"
    if id == PassId.ROUTE: return "route"
    if id == PassId.BASIS: return "basis"
    if id == PassId.FIXPOINT: return "fixpoint"
    if id == PassId.CHECK: return "check"
    if id == PassId.KAK: return "kak"
    if id == PassId.ELIDE_SWAPS: return "elide_swaps"
    if id == PassId.PAULI_SIMP: return "pauli_simp"
    return "unknown"


def pass_description(id: Int) -> String:
    if id == PassId.REMOVE_IDENTITIES: return "drop gates within 1e-12 process infidelity of the identity (phase tracked)"
    if id == PassId.CANCEL: return "commutation-aware inverse-pair cancellation; arg: window (default 32)"
    if id == PassId.MERGE: return "merge adjacent same-axis rotations (rz rz, t t -> s, rzz rzz, cp cp)"
    if id == PassId.COMMUTE: return "merge/cancel rotations through commuting gates; arg: window (default 32)"
    if id == PassId.FUSE1Q: return "fuse 1q runs and resynthesise over the basis; arg: basis (default: target basis)"
    if id == PassId.PHASE_FOLD: return "phase-polynomial folding of diagonal rotations on equal parities (T-count)"
    if id == PassId.EJECT_Z: return "push Z rotations through commuting gates to the end (virtual Z)"
    if id == PassId.UNROLL3Q: return "decompose ccx/ccz/cswap into 1q + cx"
    if id == PassId.ROUTE: return "VF2 exact layout, else SabreLayout trials + LightSABRE routing (star pre-routing portfolio); args: sabre, layout=auto|sabre|dense|trivial|random|vf2, seed, trials, iterations, swap_trials, lookahead, weight, decay, heuristic=auto|extended|layer, vf2_steps, valve, star=on|off"
    if id == PassId.BASIS: return "translate into a basis; args: gate list or preset (default: target basis)"
    if id == PassId.FIXPOINT: return "repeat the inner pipeline until gates/2q/depth stop improving; args: passes..., max=N"
    if id == PassId.CHECK: return "predicate: fail with QM-E615 unless every gate is in the target basis and on a coupling edge"
    if id == PassId.KAK: return "consolidate 2q blocks and resynthesise with KAK (<= 3 cx) when cheaper"
    if id == PassId.ELIDE_SWAPS: return "remove swap gates by relabelling (folded into final_layout); only trailing swaps once routed"
    if id == PassId.PAULI_SIMP: return "Pauli-gadget resynthesis (Clifford tableau extraction, identical-Pauli merging, ladder-sharing chains); kept only when it saves 2q gates"
    return ""


def pass_id_from_name(name: String) -> Int:
    var s = String(name.strip()).lower()
    if s == "remove_identities" or s == "remove_identity" or s == "rmid" or s == "identities": return PassId.REMOVE_IDENTITIES
    if s == "cancel" or s == "cancellation" or s == "inverse_cancel" or s == "cancel_inverses": return PassId.CANCEL
    if s == "merge" or s == "merge_rotations": return PassId.MERGE
    if s == "commute" or s == "commutation" or s == "commute_rotations": return PassId.COMMUTE
    if s == "fuse1q" or s == "fuse" or s == "fuse_1q" or s == "optimize1q": return PassId.FUSE1Q
    if s == "phase_fold" or s == "phase_folding" or s == "phasefold": return PassId.PHASE_FOLD
    if s == "eject_z" or s == "ejectz" or s == "virtual_z": return PassId.EJECT_Z
    if s == "unroll3q" or s == "unroll_3q" or s == "decompose3q": return PassId.UNROLL3Q
    if s == "route" or s == "routing" or s == "sabre": return PassId.ROUTE
    if s == "basis" or s == "translate" or s == "basis_translate" or s == "rebase": return PassId.BASIS
    if s == "fixpoint" or s == "fix" or s == "repeat" or s == "loop": return PassId.FIXPOINT
    if s == "check" or s == "verify" or s == "predicate": return PassId.CHECK
    if s == "kak" or s == "consolidate" or s == "consolidate_blocks" or s == "2q_synth": return PassId.KAK
    if s == "elide_swaps" or s == "elide" or s == "elide_permutations": return PassId.ELIDE_SWAPS
    if s == "pauli_simp" or s == "paulisimp" or s == "pauli" or s == "gadgets": return PassId.PAULI_SIMP
    return -1


def valid_pass_names() -> String:
    var s = String()
    for i in range(PassId.NUM):
        if i > 0:
            s += ", "
        s += pass_name(i)
    return s


def describe_passes() -> String:
    """JSON catalog of passes for agents."""
    var s = String("[")
    for i in range(PassId.NUM):
        if i > 0:
            s += ","
        s += "{\"name\":\"" + pass_name(i) + "\",\"description\":\"" + pass_description(i) + "\"}"
    return s + "]"


def _edit_distance(a: String, b: String) -> Int:
    var x = a.as_bytes()
    var y = b.as_bytes()
    var m = len(y)
    var prev = List[Int](length=m + 1, fill=0)
    var cur = List[Int](length=m + 1, fill=0)
    for j in range(m + 1):
        prev[j] = j
    for i in range(1, len(x) + 1):
        cur[0] = i
        for j in range(1, m + 1):
            var cost = 0 if x[i - 1] == y[j - 1] else 1
            cur[j] = min(min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost)
        for j in range(m + 1):
            prev[j] = cur[j]
    return prev[m]


def _suggest(name: String) -> String:
    var best = String()
    var bd = 1000
    for i in range(PassId.NUM):
        var d = _edit_distance(name, pass_name(i))
        if d < bd:
            bd = d
            best = pass_name(i)
    if bd <= 3:
        return " Did you mean '" + best + "'?"
    return ""


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
@fieldwise_init
struct PassSpec(Copyable, Writable):
    var id: Int
    var args: List[String]

    def arg_string(self) -> String:
        var s = String()
        for i in range(len(self.args)):
            if i > 0:
                s += ","
            s += self.args[i]
        return s

    def write_to(self, mut w: Some[Writer]):
        w.write(pass_name(self.id))
        if len(self.args) > 0:
            w.write("(", self.arg_string(), ")")


def split_top_level(s: String) raises -> List[String]:
    """Split at ',' / ';' outside parentheses (whitespace trimmed, empties dropped)."""
    var out = List[String]()
    var bs = s.as_bytes()
    var depth = 0
    var start = 0
    for i in range(len(bs)):
        var ch = Int(bs[i])
        if ch == ord("("):
            depth += 1
        elif ch == ord(")"):
            depth -= 1
            if depth < 0:
                raise Error("QM-E601: unbalanced ')' at byte " + String(i) + " in pipeline '" + s + "'. Hint: every '(' needs a matching ')'.")
        elif (ch == ord(",") or ch == ord(";")) and depth == 0:
            var part = String(String(unsafe_from_utf8=bs[start:i]).strip())
            if part.byte_length() > 0:
                out.append(part)
            start = i + 1
    if depth != 0:
        raise Error("QM-E601: unbalanced '(' in pipeline '" + s + "'. Hint: close every '(' with ')'.")
    var last = String(String(unsafe_from_utf8=bs[start:len(bs)]).strip())
    if last.byte_length() > 0:
        out.append(last)
    return out^


def _is_int(s: String) -> Bool:
    try:
        _ = Int(s)
        return True
    except:
        return False


def _is_float(s: String) -> Bool:
    try:
        _ = Float64(s)
        return True
    except:
        return False


def _validate_args(id: Int, args: List[String]) raises:
    var pn = pass_name(id)
    if id == PassId.CANCEL or id == PassId.COMMUTE:
        for a in args:
            var v = a
            if a.startswith("window="):
                v = String(a[byte=7:])
            if not _is_int(v):
                raise Error("QM-E602: bad argument '" + a + "' for pass '" + pn + "'. Hint: use an integer window, e.g. " + pn + "(64).")
        return
    if id == PassId.FUSE1Q or id == PassId.BASIS:
        if len(args) > 0:
            var s = String()
            for i in range(len(args)):
                if i > 0:
                    s += ","
                s += args[i]
            _ = parse_basis(s)
        return
    if id == PassId.ROUTE:
        for a in args:
            var eq = a.find("=")
            if eq < 0:
                if a != "sabre" and a != "lightsabre":
                    raise Error("QM-E602: unknown routing method '" + a + "'. Hint: use route(sabre) with optional key=value args: layout, seed, trials, iterations, lookahead, weight, decay, vf2_steps.")
                continue
            var key = String(a[byte=0:eq].strip())
            var val = String(a[byte=eq + 1:].strip())
            if key == "layout":
                if val != "auto" and val != "sabre" and val != "dense" and val != "trivial" and val != "random" and val != "vf2":
                    raise Error("QM-E602: unknown layout '" + val + "'. Hint: layout=auto|sabre|dense|trivial|random|vf2.")
            elif key == "heuristic":
                if val != "layer" and val != "extended" and val != "auto":
                    raise Error("QM-E602: route heuristic must be 'auto', 'layer' or 'extended', got '" + val + "'. Hint: route(sabre,heuristic=auto).")
            elif key == "star" or key == "mirror" or key == "commute":
                if val != "0" and val != "1" and val != "true" and val != "false" and val != "on" and val != "off" and not (key == "mirror" and (val == "auto" or val == "free")) and not (key == "commute" and val == "auto"):
                    raise Error("QM-E602: route argument '" + key + "' must be on/off" + ("/free/auto" if key == "mirror" else ("/auto" if key == "commute" else "")) + ", got '" + val + "'. Hint: route(sabre," + key + "=off).")
            elif key == "seed" or key == "trials" or key == "iterations" or key == "lookahead" or key == "vf2_steps" or key == "valve" or key == "swap_trials":
                if not _is_int(val):
                    raise Error("QM-E602: route argument '" + key + "' needs an integer, got '" + val + "'. Hint: e.g. route(sabre," + key + "=4).")
            elif key == "weight" or key == "decay" or key == "absorb" or key == "absorb_alt" or key == "mirror_gain":
                if not _is_float(val):
                    raise Error("QM-E602: route argument '" + key + "' needs a number, got '" + val + "'. Hint: e.g. route(sabre," + key + "=0.5).")
            else:
                raise Error("QM-E602: unknown route argument '" + key + "'. Hint: valid keys are layout, seed, trials, iterations, lookahead, weight, decay, vf2_steps, valve, star, heuristic, swap_trials, absorb, absorb_alt, mirror, mirror_gain, commute.")
        return
    if id == PassId.FIXPOINT:
        var inner = String()
        var cnt = 0
        for a in args:
            if a.startswith("max="):
                if not _is_int(String(a[byte=4:])):
                    raise Error("QM-E602: fixpoint max needs an integer, got '" + a + "'. Hint: fixpoint(cancel,merge,max=10).")
                continue
            if cnt > 0:
                inner += ","
            inner += a
            cnt += 1
        if cnt == 0:
            raise Error("QM-E602: fixpoint() needs at least one inner pass. Hint: fixpoint(cancel,commute,fuse1q).")
        _ = parse_pipeline(inner)
        return
    if len(args) > 0:
        raise Error("QM-E602: pass '" + pn + "' takes no arguments, got '" + args[0] + "'. Hint: write just '" + pn + "'.")


def parse_pipeline(spec: String) raises -> List[PassSpec]:
    """Parse a text pipeline into pass specs (QM-E600..E602 / E610 on error)."""
    var out = List[PassSpec]()
    for item in split_top_level(spec):
        var bs = item.as_bytes()
        var open = item.find("(")
        var name: String
        var args = List[String]()
        if open >= 0:
            if Int(bs[len(bs) - 1]) != ord(")"):
                raise Error("QM-E601: malformed pass '" + item + "' (text after ')'). Hint: write name(arg1,arg2).")
            name = String(String(unsafe_from_utf8=bs[0:open]).strip())
            var inner = String(unsafe_from_utf8=bs[open + 1:len(bs) - 1])
            args = split_top_level(inner)
        else:
            name = item
        var id = pass_id_from_name(name)
        if id < 0:
            raise Error("QM-E600: unknown pass '" + name + "' in pipeline." + _suggest(String(name.lower())) + " Hint: valid passes are " + valid_pass_names() + ".")
        _validate_args(id, args)
        out.append(PassSpec(id, args^))
    return out^


def pipeline_to_string(passes: List[PassSpec]) -> String:
    var s = String()
    for i in range(len(passes)):
        if i > 0:
            s += ","
        s += String(passes[i])
    return s


# ----------------------------------------------------------------------------
# Reports and results
# ----------------------------------------------------------------------------
@fieldwise_init
struct PassReport(Copyable, Writable):
    var name: String
    var gates_before: Int
    var gates_after: Int
    var twoq_before: Int
    var twoq_after: Int
    var depth_before: Int
    var depth_after: Int
    var elapsed_us: Float64
    var note: String

    def to_json(self) -> String:
        return (String("{\"pass\":\"") + self.name + "\",\"gates\":[" + String(self.gates_before) + "," + String(self.gates_after)
                + "],\"2q\":[" + String(self.twoq_before) + "," + String(self.twoq_after)
                + "],\"depth\":[" + String(self.depth_before) + "," + String(self.depth_after)
                + "],\"us\":" + fmt_num(self.elapsed_us, 1) + ",\"note\":\"" + self.note + "\"}")

    def write_to(self, mut w: Some[Writer]):
        w.write(self.to_json())


def _pad(s: String, n: Int) -> String:
    var out = s
    var k = n - s.byte_length()
    for _ in range(k):
        out += " "
    return out


def _lpad(s: String, n: Int) -> String:
    var out = String()
    var k = n - s.byte_length()
    for _ in range(k):
        out += " "
    return out + s


struct CompileResult(Copyable, Writable):
    var circuit: Circuit
    var reports: List[PassReport]
    var initial_layout: List[Int]
    var final_layout: List[Int]
    var pipeline: String
    var target_name: String
    var basis: String
    var total_us: Float64
    var gates_before: Int
    var twoq_before: Int
    var depth_before: Int
    var t_before: Int
    var est_fidelity: Float64
    var symbolic_phase_dropped: Bool

    def __init__(out self, circuit: Circuit):
        self.circuit = circuit.copy()
        self.reports = List[PassReport]()
        self.initial_layout = List[Int]()
        self.final_layout = List[Int]()
        self.pipeline = String()
        self.target_name = String()
        self.basis = String()
        self.total_us = 0.0
        self.gates_before = circuit.num_unitary_gates()
        self.twoq_before = circuit.num_2q_gates()
        self.depth_before = circuit.depth()
        self.t_before = circuit.t_count()
        self.est_fidelity = 1.0
        self.symbolic_phase_dropped = False

    def num_swaps(self) -> Int:
        var n = 0
        for r in self.reports:
            if r.name.startswith("route"):
                n += r.twoq_after - r.twoq_before
        return n

    def summary(self) -> String:
        var c = self.circuit.copy()
        var s = String("compiled '") + c.name + "' for " + self.target_name + " [" + self.basis + "]: "
        s += "gates " + String(self.gates_before) + " -> " + String(c.num_unitary_gates())
        s += ", 2q " + String(self.twoq_before) + " -> " + String(c.num_2q_gates())
        s += ", depth " + String(self.depth_before) + " -> " + String(c.depth())
        s += ", T " + String(self.t_before) + " -> " + String(c.t_count())
        s += ", " + String(len(self.reports)) + " passes in " + fmt_num(self.total_us / 1000.0, 3) + " ms"
        if self.est_fidelity < 1.0:
            s += ", est. fidelity " + fmt_num(self.est_fidelity, 4)
        if self.symbolic_phase_dropped:
            s += " (symbolic global phase dropped)"
        return s

    def table(self) -> String:
        var s = _pad("pass", 34) + _lpad("gates", 21) + _lpad("2q", 19) + _lpad("depth", 17) + _lpad("time(us)", 12) + "  note\n"
        for r in self.reports:
            var nm = r.name
            if nm.byte_length() > 33:
                nm = String(nm[byte=0:30]) + "..."
            s += _pad(nm, 34)
            s += _lpad(String(r.gates_before) + " -> " + String(r.gates_after), 21)
            s += _lpad(String(r.twoq_before) + " -> " + String(r.twoq_after), 19)
            s += _lpad(String(r.depth_before) + " -> " + String(r.depth_after), 17)
            s += _lpad(fmt_num(r.elapsed_us, 1), 12)
            s += "  " + r.note + "\n"
        return s

    def to_json(self) -> String:
        var c = self.circuit.copy()
        var s = String("{\"target\":\"") + self.target_name + "\",\"basis\":\"" + self.basis + "\",\"pipeline\":\"" + self.pipeline + "\""
        s += ",\"gates\":[" + String(self.gates_before) + "," + String(c.num_unitary_gates()) + "]"
        s += ",\"2q\":[" + String(self.twoq_before) + "," + String(c.num_2q_gates()) + "]"
        s += ",\"depth\":[" + String(self.depth_before) + "," + String(c.depth()) + "]"
        s += ",\"t_count\":[" + String(self.t_before) + "," + String(c.t_count()) + "]"
        s += ",\"total_us\":" + fmt_num(self.total_us, 1) + ",\"est_fidelity\":" + fmt_num(self.est_fidelity, 6)
        s += ",\"initial_layout\":["
        for i in range(len(self.initial_layout)):
            if i > 0:
                s += ","
            s += String(self.initial_layout[i])
        s += "],\"final_layout\":["
        for i in range(len(self.final_layout)):
            if i > 0:
                s += ","
            s += String(self.final_layout[i])
        s += "],\"passes\":["
        for i in range(len(self.reports)):
            if i > 0:
                s += ","
            s += self.reports[i].to_json()
        s += "]}"
        return s

    def write_to(self, mut w: Some[Writer]):
        w.write(self.summary())


def conformance_error(c: Circuit, target: Target) -> String:
    """First violation of basis / connectivity, or "" when the circuit conforms."""
    for i in range(len(c.ops)):
        var g = c.ops[i]
        if not op_is_unitary(g.kind):
            continue
        if not target.supports(g.kind):
            return "instruction #" + String(i) + " '" + String(g) + "' is not in the basis [" + basis_str(target.basis) + "]"
        var a = op_arity(g.kind)
        if a >= 2 and not target.full:
            if a > 2:
                return "instruction #" + String(i) + " '" + String(g) + "' acts on 3 qubits (no 3q coupling)"
            if not target.are_connected(g.q0, g.q1):
                return "instruction #" + String(i) + " '" + String(g) + "' is not on a coupling edge of " + target.name
        for s in range(a):
            if g.qubit(s) >= target.num_qubits:
                return "instruction #" + String(i) + " uses qubit " + String(g.qubit(s)) + " beyond the target size"
    return ""


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------
struct _State(Movable):
    var init: List[Int]
    var final: List[Int]
    var dropped: Bool
    var seed: Int
    var routed: Bool

    def __init__(out self, n: Int, seed: Int):
        self.init = List[Int](length=n, fill=0)
        for i in range(n):
            self.init[i] = i
        self.final = self.init.copy()
        self.dropped = False
        self.seed = seed
        self.routed = False


def _compose(prev: List[Int], route: List[Int]) -> List[Int]:
    """Layout composition: logical -> wire (prev) then wire -> physical (route)."""
    var out = List[Int](length=len(route), fill=0)
    var nprev = len(prev)
    for l in range(len(route)):
        var w = prev[l] if l < nprev else l
        out[l] = route[w]
    return out^


def _gs_from_args(args: List[String], c: Circuit, target: Target) raises -> GateSet:
    if len(args) == 0:
        return working_gate_set(c, target)
    var s = String()
    for i in range(len(args)):
        if i > 0:
            s += ","
        s += args[i]
    return GateSet(parse_basis(s))


def _int_arg(args: List[String], default: Int) raises -> Int:
    for a in args:
        if a.startswith("window="):
            return Int(String(a[byte=7:]))
        return Int(a)
    return default


def working_gate_set(c: Circuit, target: Target) -> GateSet:
    """Gate set optimisation passes may emit: the target basis once the circuit
    is expressed in it (so passes after `basis` keep it), otherwise any gate
    (pre-translation passes should not expand gates into a basis early)."""
    if len(target.basis) == 0:
        return GateSet()
    for g in c.ops:
        if op_is_unitary(g.kind) and not target.supports(g.kind):
            return GateSet()
    return GateSet(target.basis)


def _apply(p: PassSpec, mut c: Circuit, target: Target, mut st: _State) raises -> String:
    """Run one pass in place; returns a short note."""
    var id = p.id
    var gs = working_gate_set(c, target)
    if id == PassId.REMOVE_IDENTITIES:
        return "removed " + String(remove_identities(c))
    if id == PassId.CANCEL:
        return "cancelled " + String(cancel_inverses(c, gs, _int_arg(p.args, 32))) + " pairs"
    if id == PassId.MERGE:
        return "merged " + String(merge_rotations(c, gs))
    if id == PassId.COMMUTE:
        return "merged " + String(commute_rotations(c, gs, _int_arg(p.args, 32)))
    if id == PassId.FUSE1Q:
        var g2 = _gs_from_args(p.args, c, target)
        return "saved " + String(fuse_1q(c, g2)) + " 1q gates " + String(g2)
    if id == PassId.PHASE_FOLD:
        var t0 = c.t_count()
        var r = phase_fold(c, gs)
        return "folded " + String(r) + " rotations, T " + String(t0) + "->" + String(c.t_count())
    if id == PassId.EJECT_Z:
        return "removed " + String(eject_z(c, gs)) + " z rotations"
    if id == PassId.UNROLL3Q:
        return "unrolled " + String(unroll_3q(c)) + " 3q gates"
    if id == PassId.BASIS:
        var g2 = GateSet(target.basis)
        if len(p.args) > 0:
            g2 = _gs_from_args(p.args, c, target)
        var dropped = False
        var r = translate_to_basis(c, g2, dropped)
        if dropped:
            st.dropped = True
        return "translated " + String(r) + " gates to " + String(g2) + (" (symbolic phase dropped)" if dropped else "")
    if id == PassId.ROUTE:
        var opts = RouteOptions()
        opts.seed = st.seed
        var star = True
        for a in p.args:
            var eq = a.find("=")
            if eq < 0:
                continue
            var key = String(a[byte=0:eq].strip())
            var val = String(a[byte=eq + 1:].strip())
            if key == "layout": opts.layout = val
            elif key == "seed": opts.seed = Int(val)
            elif key == "trials": opts.trials = Int(val)
            elif key == "iterations": opts.iterations = Int(val)
            elif key == "lookahead": opts.lookahead = Int(val)
            elif key == "weight": opts.weight = Float64(val)
            elif key == "decay": opts.decay = Float64(val)
            elif key == "vf2_steps": opts.vf2_steps = Int(val)
            elif key == "valve": opts.valve = Int(val)
            elif key == "star": star = val != "0" and val != "false" and val != "off"
            elif key == "heuristic": opts.heuristic = val
            elif key == "swap_trials": opts.swap_trials = Int(val)
            elif key == "absorb": opts.absorb = Float64(val)
            elif key == "absorb_alt": opts.absorb_alt = Float64(val)
            elif key == "mirror":
                opts.mirror_auto = val == "auto"
                opts.mirror = val != "0" and val != "false" and val != "off"
                if val == "free":
                    opts.mirror_gain = 1e9
            elif key == "mirror_gain": opts.mirror_gain = Float64(val)
            elif key == "commute":
                opts.commute_auto = val == "auto"
                opts.commute = val != "0" and val != "false" and val != "off"
        _ = unroll_3q(c)
        var star_map = List[Int]()
        var star_note = String()
        var sc = Circuit(1)
        var sm = List[Int]()
        var cov = 0.0
        var ins = 0
        var use_star = star and not target.full
        if use_star:
            ins = star_preroute(c, 3, sc, sm, cov)
        # Variant gating on large circuits (planning a variant costs a full
        # set of SABRE trials): star-shaped circuits (long runs of blocks
        # sharing a centre: QFT, synthesised Cliffords) only plan the
        # star-prerouted variant, circuits without long runs (QV, QAOA,
        # lattice Trotter steps) only the original; small circuits and the
        # middle ground keep the full portfolio.
        var big = c.num_2q_gates() >= 150
        var star_only = use_star and ins > 0 and big and cov >= 0.5 and not has_exact_layout(c, target, opts.vf2_steps)
        var plan: RoutePlan
        if star_only:
            plan = plan_route(sc, target, opts)
            c = sc^
            star_map = sm^
            star_note = ", star pre-routing +" + String(ins) + " absorbed swaps (long-run coverage " + fmt_num(cov, 2) + ")"
        else:
            plan = plan_route(c, target, opts)
            if use_star and ins > 0 and not plan.vf2 and not (big and cov < 0.1):
                # portfolio (Qiskit StarPreRouting as a candidate): plan the
                # star-prerouted variant too and finish only the cheaper one
                var o2 = opts.copy()
                o2.trials = max(opts.trials // 2, min(opts.trials, 4))
                var plan2 = plan_route(sc, target, o2)
                if plan2.est_swaps + ins < plan.est_swaps:
                    c = sc^
                    plan = plan2^
                    star_map = sm^
                    star_note = ", star pre-routing +" + String(ins) + " absorbed swaps"
        var il = List[Int]()
        var fl = List[Int]()
        var note = String()
        var swaps = finish_route(c, target, opts, plan, il, fl, note)
        note += star_note
        _ = swaps
        st.init = _compose(st.init, il)
        if len(star_map) > 0:
            st.final = _compose(st.final, star_map)
        st.final = _compose(st.final, fl)
        st.routed = not target.full
        return note
    if id == PassId.KAK:
        # repeat: removing an identity block can make two blocks on the same
        # pair adjacent (maximal consolidation, as Qiskit's DAG-based collection)
        var blocks = 0
        var saved = 0
        var tgs = GateSet(target.basis)
        var rounds = 4 if len(c.ops) < 20000 else 1
        for _ in range(rounds):
            var b2 = 0
            var s2 = consolidate_2q(c, tgs, b2)
            blocks += b2
            saved += s2
            if s2 <= 0:
                break
            _ = cancel_inverses(c, GateSet(), 32)
        if not gs.any and saved > 0:
            var dropped = False
            _ = translate_to_basis(c, gs, dropped)
        return "resynthesised " + String(blocks) + " blocks, 2q cost -" + String(saved)
    if id == PassId.PAULI_SIMP:
        var saved = pauli_resynthesize(c, 64, GateSet(target.basis))
        return ("saved " + String(saved) + " 2q gates") if saved > 0 else "kept original"
    if id == PassId.ELIDE_SWAPS:
        var wm = List[Int]()
        var trailing = st.routed and not target.full
        var k = elide_swaps(c, trailing, wm)
        if k > 0:
            st.final = _compose(st.final, wm)
        return "elided " + String(k) + " swaps" + (" (trailing only)" if trailing else "")
    if id == PassId.CHECK:
        var err = conformance_error(c, target)
        if err.byte_length() > 0:
            raise Error("QM-E615: compiled circuit does not conform to target '" + target.name + "': " + err + ". Hint: add 'route' and 'basis' passes before 'check' (or use a preset level).")
        return "conforms to " + target.name
    return ""


def _stats(c: Circuit, mut g: Int, mut t: Int, mut d: Int):
    """Unitary gate count, 2q count and depth in one sweep (same values as
    `num_unitary_gates`, `num_2q_gates`, `depth`)."""
    var nq = c.num_qubits
    var level = List[Int](length=nq, fill=0)
    var lp = level.unsafe_ptr()
    g = 0
    t = 0
    d = 0
    for i in range(len(c.ops)):
        var k = c.ops[i].kind
        if k == Op.BARRIER:
            var m = 0
            for q in range(nq):
                m = max(m, lp[unsafe_offset=q])
            for q in range(nq):
                lp[unsafe_offset=q] = m
            continue
        if not op_is_unitary(k):
            continue
        var a = op_arity(k)
        if k != Op.I:
            g += 1
        if a == 2:
            t += 1
        var q0 = c.ops[i].q0
        if a == 1:
            var l = lp[unsafe_offset=q0] + 1
            lp[unsafe_offset=q0] = l
            d = max(d, l)
        elif a == 2:
            var q1 = c.ops[i].q1
            var l = max(lp[unsafe_offset=q0], lp[unsafe_offset=q1]) + 1
            lp[unsafe_offset=q0] = l
            lp[unsafe_offset=q1] = l
            d = max(d, l)
        elif a == 3:
            var q1 = c.ops[i].q1
            var q2 = c.ops[i].q2
            var l = max(max(lp[unsafe_offset=q0], lp[unsafe_offset=q1]), lp[unsafe_offset=q2]) + 1
            lp[unsafe_offset=q0] = l
            lp[unsafe_offset=q1] = l
            lp[unsafe_offset=q2] = l
            d = max(d, l)
    _ = len(level)


def _run_list(passes: List[PassSpec], mut c: Circuit, target: Target, mut st: _State,
              mut reports: List[PassReport]) raises:
    # the "after" statistics of a pass are the "before" statistics of the
    # next one: each pass boundary is measured once (one fused sweep)
    var g0 = 0
    var t0 = 0
    var d0 = 0
    _stats(c, g0, t0, d0)
    for p in passes:
        var start = perf_counter_ns()
        var note: String
        if p.id == PassId.FIXPOINT:
            var inner_txt = String()
            var max_it = 10
            var cnt = 0
            for a in p.args:
                if a.startswith("max="):
                    max_it = Int(String(a[byte=4:]))
                    continue
                if cnt > 0:
                    inner_txt += ","
                inner_txt += a
                cnt += 1
            var inner = parse_pipeline(inner_txt)
            var deltas = List[Int](length=len(inner), fill=0)
            var it = 0
            var gb = g0
            var tb = t0
            var db = d0
            while it < max_it:
                for k in range(len(inner)):
                    var before = c.num_unitary_gates()
                    if inner[k].id == PassId.FIXPOINT:
                        var sub = List[PassReport]()
                        var one = List[PassSpec]()
                        one.append(inner[k].copy())
                        _run_list(one, c, target, st, sub)
                    else:
                        _ = _apply(inner[k], c, target, st)
                    deltas[k] += c.num_unitary_gates() - before
                it += 1
                var ga = 0
                var ta = 0
                var da = 0
                _stats(c, ga, ta, da)
                var improved = ta < tb or (ta == tb and ga < gb) or (ta == tb and ga == gb and da < db)
                gb = ga
                tb = ta
                db = da
                if not improved:
                    break
            note = String(it) + " iters:"
            for k in range(len(inner)):
                note += " " + pass_name(inner[k].id) + " " + String(deltas[k])
            var us = Float64(perf_counter_ns() - start) / 1000.0
            reports.append(PassReport(String(p), g0, gb, t0, tb, d0, db, us, note))
            g0 = gb
            t0 = tb
            d0 = db
            continue
        note = _apply(p, c, target, st)
        var us = Float64(perf_counter_ns() - start) / 1000.0
        var g1 = 0
        var t1 = 0
        var d1 = 0
        _stats(c, g1, t1, d1)
        reports.append(PassReport(String(p), g0, g1, t0, t1, d0, d1, us, note))
        g0 = g1
        t0 = t1
        d0 = d1


struct PassManager(Copyable, Writable):
    var spec: String
    var passes: List[PassSpec]
    var seed: Int

    def __init__(out self, spec: String, seed: Int = 7) raises:
        self.spec = spec
        self.passes = parse_pipeline(spec)
        self.seed = seed

    @staticmethod
    def preset(level: Int, target: Target, seed: Int = 7) raises -> PassManager:
        return PassManager(preset_pipeline(level, target), seed)

    def canonical(self) -> String:
        return pipeline_to_string(self.passes)

    def run(self, circuit: Circuit, target: Target) raises -> CompileResult:
        circuit.validate()
        var res = CompileResult(circuit)
        res.pipeline = self.canonical()
        res.target_name = target.name
        res.basis = basis_str(target.basis)
        var c = circuit.copy()
        var st = _State(c.num_qubits, self.seed)
        var t0 = perf_counter_ns()
        _run_list(self.passes, c, target, st, res.reports)
        res.total_us = Float64(perf_counter_ns() - t0) / 1000.0
        res.initial_layout = st.init.copy()
        res.final_layout = st.final.copy()
        res.symbolic_phase_dropped = st.dropped
        res.est_fidelity = target.estimate_fidelity(c)
        c.global_phase = wrap_angle(c.global_phase, TWO_PI)
        res.circuit = c^
        return res^

    def write_to(self, mut w: Some[Writer]):
        w.write(self.canonical())


def preset_pipeline(level: Int, target: Target) raises -> String:
    """Text pipeline for optimisation level 0..3 adapted to the target."""
    if level < 0 or level > 3:
        raise Error("QM-E609: optimization level must be 0..3, got " + String(level) + ". Hint: 0 = translate only, 1 = light, 2 = default, 3 = heavy (phase folding, more layout trials).")
    var route = not target.full
    var basis = len(target.basis) > 0
    var parts = List[String]()
    if level == 0:
        if route:
            parts.append("route(sabre,layout=trivial,trials=1,iterations=0,star=off)")
        if basis:
            parts.append("basis")
    elif level == 1:
        parts.append("remove_identities")
        parts.append("cancel")
        parts.append("merge")
        if route:
            parts.append("route(sabre,layout=auto,trials=5,iterations=1,swap_trials=5)")
        if basis:
            parts.append("basis")
        parts.append("cancel")
        parts.append("merge")
        parts.append("fuse1q")
        parts.append("check")
    elif level == 2:
        parts.append("remove_identities")
        parts.append("elide_swaps")
        parts.append("pauli_simp")
        parts.append("fixpoint(cancel,commute,fuse1q,max=4)")
        if route:
            parts.append("route(sabre,layout=auto,trials=20,iterations=2,swap_trials=20)")
        elif basis and not target.supports(Op.CCX):
            parts.append("unroll3q")  # let KAK see the cx of ccx / cswap / ccz
        parts.append("kak")
        if basis:
            parts.append("basis")
        parts.append("cancel")
        parts.append("commute")
        parts.append("fuse1q")
        parts.append("check")
    else:
        parts.append("remove_identities")
        parts.append("elide_swaps")
        parts.append("pauli_simp")
        parts.append("fixpoint(cancel,commute,phase_fold,fuse1q,max=4)")
        if not route and basis and not target.supports(Op.CCX):
            parts.append("unroll3q")
        parts.append("kak")
        if route:
            parts.append("route(sabre,layout=auto,trials=40,iterations=3,swap_trials=40)")
            parts.append("kak")
        if basis:
            parts.append("basis")
        parts.append("fixpoint(cancel,commute,phase_fold,eject_z,fuse1q,max=10)")
        parts.append("check")
    var s = String()
    for i in range(len(parts)):
        if i > 0:
            s += ","
        s += parts[i]
    return s


def _better(a: CompileResult, b: CompileResult) -> Bool:
    """a strictly better than b: fewer 2q gates, then lower depth, then fewer gates."""
    var a2 = a.circuit.num_2q_gates()
    var b2 = b.circuit.num_2q_gates()
    if a2 != b2:
        return a2 < b2
    var ad = a.circuit.depth()
    var bd = b.circuit.depth()
    if ad != bd:
        return ad < bd
    return a.circuit.num_unitary_gates() < b.circuit.num_unitary_gates()


def transpile(circuit: Circuit, target: Target, level: Int = 2, seed: Int = 7) raises -> CompileResult:
    """Compile `circuit` for `target` with preset level 0..3.

    Level 3 is a portfolio: the O3 pipeline and the O2 pipeline both run and
    the better result (2q count, then depth) is returned, so O3 is never worse
    than O2.
    """
    var pm = PassManager(preset_pipeline(level, target), seed)
    var r = pm.run(circuit, target)
    if level == 3:
        var pm2 = PassManager(preset_pipeline(2, target), seed)
        var r2 = pm2.run(circuit, target)
        if _better(r2, r):
            r2.total_us += r.total_us
            return r2^
        r.total_us += r2.total_us
    return r^


def transpile(circuit: Circuit, level: Int = 2, seed: Int = 7) raises -> CompileResult:
    """Compile for the default, vendor-neutral target: all-to-all, any gate."""
    return transpile(circuit, Target.all_to_all(circuit.num_qubits), level, seed)


def compile_spec(circuit: Circuit, target: Target, spec: String, seed: Int = 7) raises -> CompileResult:
    """Compile with an explicit text pipeline."""
    var pm = PassManager(spec, seed)
    return pm.run(circuit, target)
