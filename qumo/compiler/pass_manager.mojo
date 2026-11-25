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
loudly), Cirq's "optimizers as plain functions". A pass is declared once, in
`pass_decl` (names, aliases, description, arguments); dynamic dispatch is a
switch over pass ids in `_apply` (no trait objects), which reads arguments
only through `PassArgs`. Each pass reports gates / 2q / depth before
and after plus wall time, and errors are stable `QM-E6xx` codes with a
`Hint:` listing valid choices (with a did-you-mean suggestion) so agents can
self-correct. `CompileResult.to_json()` is the machine-readable summary.
"""

from std.time import perf_counter_ns
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.common import fmt_num, basis_str, reduce_large_angles, wrap_angle, TWO_PI, BlockKind
from qumo.ir.gateset import GateSet
from qumo.compiler.target import Target, parse_basis, parse_angle_bound
from qumo.compiler.optimize import (
    remove_identities, cancel_inverses, merge_rotations, commute_rotations,
    fuse_1q, eject_z, elide_swaps,
)
from qumo.compiler.kak import consolidate_2q
from qumo.compiler.pauli_simp import pauli_resynthesize
from qumo.compiler.phase_fold import phase_fold
from qumo.compiler.basis import translate_to_basis, unroll_3q, first_inexpressible_1q, can_synthesize_1q
from qumo.compiler.routing import (
    RouteOptions, RoutePlan, route_circuit, star_preroute, plan_route, finish_route, has_exact_layout,
)
from qumo.compiler.resynth import SynthOptions, SynthStats, resynthesize_blocks
from qumo.compiler.ai_route import AIRouteOptions, ai_route, routed_cost
from qumo.compiler.wrap import wrap_angles
from qumo.compiler.layout_fidelity import post_layout


struct PassId:
    """Ids of the passes. A pass is declared once, in `pass_decl`: its names,
    description and arguments. `_apply` holds the branch that runs it."""
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
    # AI-transpiler block resynthesis (docs/AI_TRANSPILER.md); the block kind
    # of each is `PassDecl.block_kind`, not a difference of ids
    comptime CLIFFORD_SYNTH = 15
    comptime LINEAR_SYNTH = 16
    comptime PERM_SYNTH = 17
    comptime PAULI_NETWORK = 18
    comptime WRAP_ANGLES = 19
    comptime POST_LAYOUT = 20
    comptime NUM = 21


# ----------------------------------------------------------------------------
# The pass table: one declaration per pass (`pass_decl`). Name and alias
# lookup, the descriptions, the argument check of `parse_pipeline`, its hints
# and the values `_apply` reads (`PassArgs`) all come from it.
# ----------------------------------------------------------------------------
struct ArgKind:
    comptime INT = 0
    comptime FLOAT = 1
    comptime FLAG = 2  # 0|1|true|false|on|off, plus the words in `choices`
    comptime CHOICE = 3  # one of `choices`
    comptime TEXT = 4  # free text: a model id, a directory
    # list kinds: every argument that is not a declared key, in order
    comptime GATES = 5  # gate list or basis preset
    comptime BOUNDS = 6  # gate[min:max]
    comptime PASSES = 7  # nested pipeline items


struct ArgForm:
    comptime KEY = 0  # written key=value
    comptime KEY_OR_BARE = 1  # key=value, or the value alone
    comptime BARE = 2  # the value alone


comptime _NO_LO = -(1 << 62)
comptime _NO_HI = 1 << 62
comptime _FLAG_WORDS = "0|1|true|false|on|off"


def arg_kind_name(kind: Int) -> String:
    if kind == ArgKind.INT: return "int"
    if kind == ArgKind.FLOAT: return "float"
    if kind == ArgKind.FLAG: return "flag"
    if kind == ArgKind.CHOICE: return "choice"
    if kind == ArgKind.TEXT: return "text"
    if kind == ArgKind.GATES: return "gates"
    if kind == ArgKind.BOUNDS: return "bounds"
    if kind == ArgKind.PASSES: return "passes"
    return "unknown"


struct ArgDecl(Copyable):
    """One declared argument of a pass."""

    var name: String
    var kind: Int  # ArgKind
    var form: Int  # ArgForm
    var default: String  # the value when it is not written; "" = none, or set at run time (see `help`)
    var choices: String  # "a|b|c": the values of a CHOICE, the extra words of a FLAG
    var lo: Int  # INT: smallest value (_NO_LO = no limit)
    var hi: Int  # INT: largest value (_NO_HI = no limit)
    var required: Bool  # TEXT: not empty; list kinds: at least one item
    var code: String  # error code of a bad value
    var over_code: String  # INT: error code of a value above `hi` ("" = `code`)
    var over_hint: String  # and its hint
    var help: String

    def __init__(out self, name: String, kind: Int, form: Int, default: String, choices: String, help: String):
        self.name = name
        self.kind = kind
        self.form = form
        self.default = default
        self.choices = choices
        self.lo = _NO_LO
        self.hi = _NO_HI
        self.required = False
        self.code = "QM-E602"
        self.over_code = ""
        self.over_hint = ""
        self.help = help

    def is_list(self) -> Bool:
        return self.kind >= ArgKind.GATES

    def values(self) -> String:
        """What may be written for it: "a|b|c", "N", "X", "<text>"."""
        if self.kind == ArgKind.INT:
            if self.lo != _NO_LO and self.hi != _NO_HI:
                return String(self.lo) + ".." + String(self.hi)
            return "N"
        if self.kind == ArgKind.FLOAT:
            return "X"
        if self.kind == ArgKind.FLAG:
            return ("on|off|" + self.choices) if self.choices.byte_length() > 0 else String("on|off")
        if self.kind == ArgKind.CHOICE:
            return self.choices
        if self.kind == ArgKind.TEXT:
            return "<" + self.name + ">"
        if self.kind == ArgKind.GATES:
            return "gate list or basis preset"
        if self.kind == ArgKind.BOUNDS:
            return "gate[min:max]"
        return "pass"

    def usage(self) -> String:
        """How it is written, with its default: "trials=N (default 4)"."""
        var s: String
        if self.is_list():
            s = self.values() + ("..." if self.kind != ArgKind.GATES else "")
            if self.help.byte_length() > 0:
                s += " (" + self.help + ")"
        elif self.form == ArgForm.BARE:
            s = self.values()
        elif self.form == ArgForm.KEY_OR_BARE:
            s = "[" + self.name + "=]" + self.values()
        else:
            s = self.name + "=" + self.values()
        if self.default.byte_length() > 0:
            s += " (default " + self.default + ")"
        return s

    def example(self, pn: String) -> String:
        """A valid use, for hints: "route(trials=4)"."""
        var v = self.default
        if v.byte_length() == 0:
            if self.kind == ArgKind.INT:
                v = String(self.lo) if self.lo != _NO_LO else String("4")
            elif self.kind == ArgKind.FLOAT:
                v = "0.5"
            elif self.kind == ArgKind.FLAG:
                v = "off"
            elif self.kind == ArgKind.CHOICE:
                v = _choice_list(self.choices)[0]
            elif self.kind == ArgKind.GATES:
                v = "rz,sx,x,cz"
            elif self.kind == ArgKind.BOUNDS:
                v = "rzz[0:pi/2]"
            elif self.kind == ArgKind.PASSES:
                v = "cancel,commute,fuse1q"
            else:
                v = "<" + self.name + ">"
        if self.form == ArgForm.KEY:
            return pn + "(" + self.name + "=" + v + ")"
        return pn + "(" + v + ")"


def _choice_list(choices: String) -> List[String]:
    var out = List[String]()
    for part in choices.split("|"):
        out.append(String(part))
    return out^


def _in_choices(choices: String, v: String) -> Bool:
    """Is `v` one of the words of "a|b|c"?"""
    var n = v.byte_length()
    if n == 0 or v.find("|") >= 0:
        return False
    var total = choices.byte_length()
    var at = choices.find(v)
    while at >= 0:
        var cb = choices.as_bytes()
        var left = at == 0 or Int(cb[at - 1]) == ord("|")
        var right = at + n == total or Int(cb[at + n]) == ord("|")
        if left and right:
            return True
        at = choices.find(v, at + 1)
    return False


struct PassDecl(Copyable):
    """The declaration of a pass: what `pass_decl` returns."""

    var id: Int
    var name: String  # canonical name ("" = no pass has this id)
    var aliases: String  # "a|b|c": other names accepted in a pipeline
    var summary: String  # one line, without the arguments
    var args: List[ArgDecl]
    var block_kind: Int  # BlockKind of a collect-and-resynthesise pass, else -1
    var relabels: Bool  # renames qubits for the passes after it (it changes the layouts)
    var names_only: Bool  # `args` was not filled (name lookups skip that work)

    def __init__(out self, id: Int, names_only: Bool):
        self.id = id
        self.name = String()
        self.aliases = String()
        self.summary = String()
        self.args = List[ArgDecl]()
        self.block_kind = -1
        self.relabels = False
        self.names_only = names_only

    def named(mut self, name: String, aliases: String, summary: String):
        self.name = name
        self.aliases = aliases
        self.summary = summary

    # --- arguments (each call declares one) ---
    def _add(mut self, var a: ArgDecl):
        if len(self.args) == 0:
            self.args.reserve(8)
        self.args.append(a^)

    def int_arg(mut self, name: String, default: String, help: String, form: Int = ArgForm.KEY,
                lo: Int = _NO_LO, hi: Int = _NO_HI, over_code: String = "", over_hint: String = ""):
        if self.names_only:
            return
        var a = ArgDecl(name, ArgKind.INT, form, default, "", help)
        a.lo = lo
        a.hi = hi
        a.over_code = over_code
        a.over_hint = over_hint
        self._add(a^)

    def float_arg(mut self, name: String, default: String, help: String):
        if self.names_only:
            return
        self._add(ArgDecl(name, ArgKind.FLOAT, ArgForm.KEY, default, "", help))

    def flag_arg(mut self, name: String, default: String, help: String, extra: String = ""):
        if self.names_only:
            return
        self._add(ArgDecl(name, ArgKind.FLAG, ArgForm.KEY, default, extra, help))

    def choice_arg(mut self, name: String, choices: String, default: String, help: String,
                   form: Int = ArgForm.KEY, code: String = "QM-E602"):
        if self.names_only:
            return
        var a = ArgDecl(name, ArgKind.CHOICE, form, default, choices, help)
        a.code = code
        self._add(a^)

    def text_arg(mut self, name: String, default: String, help: String, required: Bool = False,
                 code: String = "QM-E602"):
        if self.names_only:
            return
        var a = ArgDecl(name, ArgKind.TEXT, ArgForm.KEY, default, "", help)
        a.required = required
        a.code = code
        self._add(a^)

    def list_arg(mut self, name: String, kind: Int, help: String, required: Bool = False):
        if self.names_only:
            return
        var a = ArgDecl(name, kind, ArgForm.BARE, "", "", help)
        a.required = required
        self._add(a^)

    # --- lookups ---
    def arg_index(self, name: String) -> Int:
        for i in range(len(self.args)):
            if self.args[i].name == name:
                return i
        return -1

    def keyed_index(self, key: String) -> Int:
        """The argument written `key=value`, or -1."""
        for i in range(len(self.args)):
            if self.args[i].form != ArgForm.BARE and self.args[i].name == key:
                return i
        return -1

    def bare_index(self) -> Int:
        """The argument a word without a key belongs to, or -1."""
        for i in range(len(self.args)):
            if self.args[i].form != ArgForm.KEY:
                return i
        return -1

    def list_index(self) -> Int:
        for i in range(len(self.args)):
            if self.args[i].is_list():
                return i
        return -1

    def usage(self) -> String:
        """Every argument as it is written, comma separated ("" = takes none)."""
        var s = String()
        for i in range(len(self.args)):
            if i > 0:
                s += ", "
            s += self.args[i].usage()
        return s

    def alias_list(self) -> List[String]:
        if self.aliases.byte_length() == 0:
            return List[String]()
        return _choice_list(self.aliases)


def _synth_args(mut d: PassDecl, max_qubits: String, min_block: String):
    """Arguments of clifford_synth / linear_synth / perm_synth / pauli_network
    (the defaults are the IBM collector defaults: `SynthOptions.default`)."""
    d.choice_arg("method", "auto|exact|astar|greedy|policy", "auto", "how a block is resynthesised; list policy models with describe", ArgForm.KEY_OR_BARE, "QM-E616")
    d.int_arg("max_qubits", max_qubits, "widest block collected", lo=1, hi=64, over_code="QM-E618",
              over_hint="lower max_qubits or use method=greedy")
    d.int_arg("min_block", min_block, "fewest gates of a collected block", lo=0)
    d.choice_arg("direction", "forward|backward|both", "both", "collection sweeps")
    d.flag_arg("commute", "1", "commutation-aware collection")
    d.int_arg("reps", "10", "sweeps at most", lo=1)
    d.text_arg("model", "auto", "auto|off|<model id>; list models with describe", required=True, code="QM-E616")
    d.int_arg("beams", "8", "beam width of the policy search", lo=1)
    d.int_arg("searches", "4", "policy searches per block", lo=1)
    d.choice_arg("metric", "cost_depth|cost", "cost_depth", "what makes a replacement cheaper: 2q cost then 2q depth, or 2q cost only")
    d.flag_arg("strict", "0", "raise QM-E619 instead of keeping a block whose proposal fails verification")
    d.int_arg("budget", "0", "search node cap (0 = the method's own)", lo=0)
    d.text_arg("model_dir", "", "model directory, searched first")
    d.int_arg("seed", "", "default: the compile seed")


def pass_decl(id: Int, names_only: Bool = False) -> PassDecl:
    """THE pass table: the declaration of pass `id` (`name == ""` when no pass
    has that id). To add a pass: a `PassId`, a block here, a branch in `_apply`."""
    var d = PassDecl(id, names_only)
    if id == PassId.REMOVE_IDENTITIES:
        d.named("remove_identities", "remove_identity|rmid|identities",
                "drop gates within 1e-12 process infidelity of the identity (phase tracked)")
    elif id == PassId.CANCEL:
        d.named("cancel", "cancellation|inverse_cancel|cancel_inverses",
                "commutation-aware inverse-pair cancellation")
        d.int_arg("window", "32", "gates looked back along a wire", ArgForm.KEY_OR_BARE)
    elif id == PassId.MERGE:
        d.named("merge", "merge_rotations",
                "merge adjacent same-axis rotations (rz rz, t t -> s, rzz rzz, cp cp)")
    elif id == PassId.COMMUTE:
        d.named("commute", "commutation|commute_rotations",
                "merge/cancel rotations through commuting gates")
        d.int_arg("window", "32", "gates looked back along a wire", ArgForm.KEY_OR_BARE)
    elif id == PassId.FUSE1Q:
        d.named("fuse1q", "fuse|fuse_1q|optimize1q",
                "fuse 1q runs and resynthesise over the basis")
        d.list_arg("gates", ArgKind.GATES, "default: the target basis")
    elif id == PassId.PHASE_FOLD:
        d.named("phase_fold", "phase_folding|phasefold",
                "phase-polynomial folding of diagonal rotations on equal parities (T-count)")
    elif id == PassId.EJECT_Z:
        d.named("eject_z", "ejectz|virtual_z",
                "push Z rotations through commuting gates to the end (virtual Z)")
    elif id == PassId.UNROLL3Q:
        d.named("unroll3q", "unroll_3q|decompose3q",
                "decompose ccx/ccz/cswap into 1q + cx")
    elif id == PassId.ROUTE:
        d.named("route", "routing|sabre",
                "VF2 exact layout, else SabreLayout trials + LightSABRE routing (star pre-routing portfolio)")
        d.relabels = True
        d.choice_arg("method", "sabre|lightsabre", "", "the routing method (one, under two names)", ArgForm.BARE)
        d.choice_arg("layout", "auto|sabre|dense|trivial|random|vf2", "auto", "initial layout")
        d.int_arg("seed", "", "default: the compile seed")
        d.int_arg("trials", "4", "layout trials")
        d.int_arg("iterations", "2", "forward/backward refinements of a layout trial")
        d.int_arg("lookahead", "20", "extended-set size")
        d.float_arg("weight", "0.5", "weight of the extended set")
        d.float_arg("decay", "0.001", "decay added to a qubit each time it is swapped")
        d.int_arg("vf2_steps", "20000", "step budget of the exact layout search")
        d.int_arg("valve", "0", "swaps without progress before the release valve (0 = 10 * num_qubits)")
        d.flag_arg("star", "on", "star pre-routing as a candidate")
        d.choice_arg("heuristic", "auto|extended|layer", "auto", "swap score (auto: trials alternate)")
        d.int_arg("swap_trials", "1", "routing seeds tried on the chosen layout (best kept)")
        d.float_arg("absorb", "1.0", "score factor for swaps KAK can absorb (< 1 favours them)")
        d.float_arg("absorb_alt", "0.9", "that factor on every other swap trial")
        d.flag_arg("mirror", "auto", "mirror-gate routing: apply swaps absorbed by the gate just executed on the pair (free: also the ones that cost nothing; auto: decided per circuit)", "free|auto")
        d.float_arg("mirror_gain", "1.0", "heuristic gain a 1-CX mirror must reach")
        d.flag_arg("commute", "auto", "commutation-aware routing DAG (auto: decided per circuit)", "auto")
        d.text_arg("policy", "off", "off|auto|<model id>: learned-policy layout + routing after SABRE, kept only when verified and cheaper", required=True, code="QM-E616")
        d.int_arg("ai_level", "2", "effort of the learned router", lo=1, hi=3)
        d.choice_arg("layout_mode", "optimize|improve|keep", "optimize", "what the learned router does with the SABRE layout")
        d.text_arg("model_dir", "", "model directory, searched first")
    elif id == PassId.BASIS:
        d.named("basis", "translate|basis_translate|rebase",
                "translate into a basis")
        d.list_arg("gates", ArgKind.GATES, "default: the target basis")
    elif id == PassId.FIXPOINT:
        d.named("fixpoint", "fix|repeat|loop",
                "repeat the inner pipeline until gates/2q/depth stop improving")
        d.list_arg("passes", ArgKind.PASSES, "the inner pipeline", required=True)
        d.int_arg("max", "10", "iterations at most")
    elif id == PassId.CHECK:
        d.named("check", "verify|predicate",
                "predicate: fail with QM-E615 unless every gate is in the target basis and on a coupling edge")
    elif id == PassId.KAK:
        d.named("kak", "consolidate|consolidate_blocks|2q_synth",
                "consolidate 2q blocks and resynthesise with KAK (<= 3 cx) when cheaper")
    elif id == PassId.ELIDE_SWAPS:
        d.named("elide_swaps", "elide|elide_permutations",
                "remove swap gates by relabelling (folded into final_layout); only trailing swaps once routed")
        d.relabels = True
    elif id == PassId.PAULI_SIMP:
        d.named("pauli_simp", "paulisimp|pauli|gadgets",
                "Pauli-gadget resynthesis (Clifford tableau extraction, identical-Pauli merging, ladder-sharing chains); kept only when it saves 2q gates")
        d.choice_arg("objective", "2q|t", "2q", "t keeps the result on fewer T / non-Clifford rotations instead (FT track, may cost 2q gates, not in O0-O3)")
    elif id == PassId.CLIFFORD_SYNTH:
        d.named("clifford_synth", "ai_clifford|clifford_synthesis",
                "collect Clifford blocks (h,s,sdg,sx,x,y,z,cx,cy,cz,swap; <= 9 qubits) and resynthesise them topology-aware (exact search, greedy or learned policy); a block is replaced only when verified equal by tableau and cheaper (2q cost, then 2q depth)")
        d.block_kind = BlockKind.CLIFFORD
        _synth_args(d, "9", "2")
    elif id == PassId.LINEAR_SYNTH:
        d.named("linear_synth", "ai_linear|ai_linear_function|linear_function_synth",
                "collect linear-function blocks (cx,swap; <= 9 qubits) and resynthesise them on the coupling subgraph; replaced only when verified equal by GF(2) matrix and cheaper")
        d.block_kind = BlockKind.LINEAR
        _synth_args(d, "9", "4")
    elif id == PassId.PERM_SYNTH:
        d.named("perm_synth", "ai_permutation|permutation_synth",
                "collect SWAP networks (<= 12 qubits) and resynthesise the permutation by token swapping (exact A* for small blocks, greedy or learned policy); replaced only when verified equal and cheaper")
        d.block_kind = BlockKind.PERM
        _synth_args(d, "12", "4")
    elif id == PassId.PAULI_NETWORK:
        d.named("pauli_network", "ai_pauli_network|pauli_network_synth",
                "collect Pauli-rotation networks (Clifford + rx/ry/rz/rzz/rxx/ryy; <= 6 qubits) and resynthesise rotations + Clifford tail topology-aware; replaced only when verified equal (unitary) and cheaper")
        d.block_kind = BlockKind.PAULI
        _synth_args(d, "6", "4")
    elif id == PassId.WRAP_ANGLES:
        d.named("wrap_angles", "wrap|fold_angles",
                "bring constant angles into the target's angle bounds (spec gate[min:max]) with Pauli corrections, exact up to the tracked global phase")
        d.list_arg("bounds", ArgKind.BOUNDS, "a bound added for this pass")
    elif id == PassId.POST_LAYOUT:
        d.named("post_layout", "vf2_post_layout|noise_layout",
                "rename physical qubits to the embedding with the best estimated fidelity under the target calibration (never lowers it; no effect on uniform targets)")
        d.relabels = True
        d.int_arg("call_limit", "200000", "search steps at most", ArgForm.KEY_OR_BARE)
    return d^


def pass_table() -> List[PassDecl]:
    var out = List[PassDecl](capacity=PassId.NUM)
    for i in range(PassId.NUM):
        out.append(pass_decl(i))
    return out^


# The lookups below run for every pass of every compilation, so they read the
# table when the module is compiled (`comptime for` unrolls over the ids and
# each declaration folds to the one field asked for).
def pass_name(id: Int) -> String:
    comptime for i in range(PassId.NUM):
        comptime name = pass_decl(i, True).name
        if id == i:
            return name
    return "unknown"


def pass_description(id: Int) -> String:
    """One line: what the pass does, then its arguments as they are written."""
    var d = pass_decl(id)
    if len(d.args) == 0:
        return d.summary
    return d.summary + "; args: " + d.usage()


def pass_aliases(id: Int) -> List[String]:
    return pass_decl(id, True).alias_list()


def pass_block_kind(id: Int) -> Int:
    """BlockKind a collect-and-resynthesise pass works on, -1 for the others."""
    comptime for i in range(PassId.NUM):
        comptime kind = pass_decl(i, True).block_kind
        if id == i:
            return kind
    return -1


def is_synth_pass(id: Int) -> Bool:
    """True for the collect-and-resynthesise passes."""
    return pass_block_kind(id) >= 0


def synth_pass_of_kind(kind: Int) -> Int:
    """Id of the pass that resynthesises blocks of `kind`, or -1."""
    comptime for i in range(PassId.NUM):
        comptime k = pass_decl(i, True).block_kind
        if kind == k:
            return i
    return -1


def pass_relabels(id: Int) -> Bool:
    """Does the pass rename qubits for what follows it (route, elide_swaps, post_layout)?"""
    comptime for i in range(PassId.NUM):
        comptime relabels = pass_decl(i, True).relabels
        if id == i:
            return relabels
    return False


def pass_id_from_name(name: String) -> Int:
    """Id of a canonical name or an alias (case-insensitive), or -1."""
    var s = String(name.strip()).lower()
    # canonical names first
    comptime for i in range(PassId.NUM):
        comptime canonical = pass_decl(i, True).name
        if s == canonical:
            return i
    comptime for i in range(PassId.NUM):
        comptime aliases = pass_decl(i, True).aliases
        if _in_choices(aliases, s):
            return i
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


def check_pass_table() raises:
    """Raise unless the table is whole: every id has a declaration, no name or
    alias is taken twice, every argument can be told from the others when it
    is written, every default is a value its own check accepts, and the
    synthesis passes cover the block kinds one to one. (`_apply` raises for an
    id without a branch; tests/test_pass_table.mojo runs every pass once.)"""
    var seen = List[String]()
    for i in range(PassId.NUM):
        var d = pass_decl(i)
        var where = "pass id " + String(i)
        if d.id != i or d.name.byte_length() == 0 or d.summary.byte_length() == 0:
            raise Error("QM-E600: " + where + " has no declaration in pass_decl. Hint: add its block to pass_decl.")
        if pass_decl(i, True).name != d.name:
            raise Error("QM-E600: " + where + " is named differently with and without its arguments.")
        var names = d.alias_list()
        names.append(d.name)
        for n in names:
            if n != String(n.strip()).lower() or n.byte_length() == 0:
                raise Error("QM-E600: name '" + n + "' of " + where + " must be lower case, without spaces.")
            for s in seen:
                if s == n:
                    raise Error("QM-E600: name '" + n + "' is declared twice (" + where + "). Hint: a name or alias belongs to one pass.")
            seen.append(n)
        var bare = 0
        var lists = 0
        for k in range(len(d.args)):
            ref a = d.args[k]
            var arg = "argument '" + a.name + "' of pass '" + d.name + "'"
            if d.arg_index(a.name) != k:
                raise Error("QM-E602: " + arg + " is declared twice.")
            if a.form != ArgForm.KEY:
                bare += 1
            if a.is_list():
                lists += 1
            elif a.default.byte_length() > 0:
                _check_value(d.name, a, a.default)
            if a.kind == ArgKind.CHOICE and a.choices.byte_length() == 0:
                raise Error("QM-E602: " + arg + " has no choices.")
        if bare > 1 or lists > 1:
            raise Error("QM-E602: pass '" + d.name + "' declares more than one argument written without a key. Hint: a bare word must belong to one argument.")
    for kind in range(BlockKind.NUM):
        var n = 0
        for i in range(PassId.NUM):
            if pass_block_kind(i) == kind:
                n += 1
        if n != 1:
            raise Error("QM-E600: block kind " + String(kind) + " has " + String(n) + " passes. Hint: one synthesis pass per BlockKind.")
    for i in range(PassId.NUM):
        var k = pass_block_kind(i)
        if k >= BlockKind.NUM:
            raise Error("QM-E600: pass '" + pass_name(i) + "' has block kind " + String(k) + ", past BlockKind.NUM.")


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


def _is_bool_word(v: String) -> Bool:
    return _in_choices(_FLAG_WORDS, v)


def _arg_key_end(a: String) -> Int:
    """Byte index of the '=' that ends the key of `key=value`, or -1 for an
    argument without a key. An '=' after '(' or '[' is inside a nested item."""
    var bs = a.as_bytes()
    for i in range(len(bs)):
        var ch = Int(bs[i])
        if ch == ord("="):
            return i
        if ch == ord("(") or ch == ord("["):
            return -1
    return -1


def _join_args(items: List[String]) -> String:
    var s = String()
    for i in range(len(items)):
        if i > 0:
            s += ","
        s += items[i]
    return s


def _check_value(pn: String, a: ArgDecl, v: String) raises:
    """Raise unless `v` is a value of argument `a` of pass `pn` (not for list kinds)."""
    var what = pn + " argument '" + a.name + "'"
    if a.kind == ArgKind.INT:
        if not _is_int(v):
            raise Error(a.code + ": " + what + " needs an integer, got '" + v + "'. Hint: e.g. " + a.example(pn) + ".")
        var iv = Int(v)
        if iv > a.hi and a.over_code.byte_length() > 0:
            raise Error(a.over_code + ": " + pn + " " + a.name + "=" + v + " is above the limit of " + String(a.hi) + ". Hint: " + a.over_hint + ".")
        if iv < a.lo or iv > a.hi:
            var bound = (">= " + String(a.lo)) if a.hi == _NO_HI else (("<= " + String(a.hi)) if a.lo == _NO_LO else (String(a.lo) + ".." + String(a.hi)))
            raise Error(a.code + ": " + what + " must be " + bound + ", got " + v + ". Hint: e.g. " + a.example(pn) + ".")
    elif a.kind == ArgKind.FLOAT:
        if not _is_float(v):
            raise Error(a.code + ": " + what + " needs a number, got '" + v + "'. Hint: e.g. " + a.example(pn) + ".")
    elif a.kind == ArgKind.FLAG:
        if not _is_bool_word(v) and not _in_choices(a.choices, v):
            raise Error(a.code + ": " + what + " must be one of " + _FLAG_WORDS + ("|" + a.choices if a.choices.byte_length() > 0 else "") + ", got '" + v + "'. Hint: e.g. " + a.example(pn) + ".")
    elif a.kind == ArgKind.CHOICE:
        if not _in_choices(a.choices, v):
            raise Error(a.code + ": unknown " + a.name + " '" + v + "' for pass '" + pn + "'. Hint: " + a.usage() + "; " + a.help + ".")
    elif a.kind == ArgKind.TEXT:
        if a.required and v.byte_length() == 0:
            raise Error(a.code + ": empty " + a.name + " for pass '" + pn + "'. Hint: " + a.name + "=" + a.help + ".")


struct PassArgs(Movable):
    """The arguments of one pass, bound to its declaration. This is the only
    reader of argument text: `parse_pipeline` checks with `validate`, and
    `_apply` takes every value from `get_int` / `get_float` / `get_flag` /
    `get_text` / `get_list`, so what is accepted is what runs. When an
    argument is written twice the last one counts."""

    var decl: PassDecl
    var raw: List[String]
    var slot: List[Int]  # per written argument: its index in decl.args, -1 = none
    var value: List[String]  # per written argument: its value (the text after "key=")

    def __init__(out self, id: Int, args: List[String]):
        self.decl = pass_decl(id)
        self.raw = args.copy()
        self.slot = List[Int](capacity=len(args))
        self.value = List[String](capacity=len(args))
        for a in args:
            var s: Int
            var v = a
            var eq = _arg_key_end(a)
            if eq < 0:
                s = self.decl.bare_index()
            else:
                s = self.decl.keyed_index(String(a[byte=0:eq].strip()))
                if s >= 0:
                    v = String(a[byte=eq + 1:].strip())
                else:
                    # not a key of this pass: an item of its list, which has its own check
                    s = self.decl.list_index()
            self.slot.append(s)
            self.value.append(v)

    def validate(self) raises:
        """Raise QM-E602 (or the code the argument declares) for an argument
        the pass does not have or a value it does not take."""
        var pn = self.decl.name
        for i in range(len(self.raw)):
            var s = self.slot[i]
            if s < 0:
                if len(self.decl.args) == 0:
                    raise Error("QM-E602: pass '" + pn + "' takes no arguments, got '" + self.raw[i] + "'. Hint: write just '" + pn + "'.")
                var eq = _arg_key_end(self.raw[i])
                if eq >= 0:
                    raise Error("QM-E602: unknown " + pn + " argument '" + String(self.raw[i][byte=0:eq].strip()) + "'. Hint: the arguments of " + pn + " are " + self.decl.usage() + ".")
                raise Error("QM-E602: bad argument '" + self.raw[i] + "' for pass '" + pn + "'. Hint: the arguments of " + pn + " are " + self.decl.usage() + ".")
            if not self.decl.args[s].is_list():
                _check_value(pn, self.decl.args[s], self.value[i])
        for s in range(len(self.decl.args)):
            ref a = self.decl.args[s]
            if not a.is_list():
                continue
            var items = self._items(s)
            if len(items) == 0:
                if a.required:
                    raise Error("QM-E602: " + pn + "() needs at least one " + a.values() + ". Hint: e.g. " + a.example(pn) + ".")
                continue
            if a.kind == ArgKind.GATES:
                _ = parse_basis(_join_args(items))
            elif a.kind == ArgKind.BOUNDS:
                var scratch = Target.all_to_all(1)
                for it in items:
                    if it.find("[") < 0 or it.find("]") < 0:
                        raise Error("QM-E602: bad argument '" + it + "' for pass '" + pn + "'. Hint: use gate[min:max], e.g. " + a.example(pn) + ".")
                    parse_angle_bound(it, scratch)
            elif a.kind == ArgKind.PASSES:
                _ = parse_pipeline(_join_args(items))

    def _index(self, name: String, kind_a: Int, kind_b: Int = -1, kind_c: Int = -1) raises -> Int:
        var s = self.decl.arg_index(name)
        if s < 0:
            raise Error("QM-E602: pass '" + self.decl.name + "' reads an argument '" + name + "' it does not declare. Hint: declare it in pass_decl.")
        var k = self.decl.args[s].kind
        if k != kind_a and k != kind_b and k != kind_c:
            raise Error("QM-E602: argument '" + name + "' of pass '" + self.decl.name + "' is declared as " + arg_kind_name(k) + " and read as " + arg_kind_name(kind_a) + ". Hint: read it with the accessor of its kind.")
        return s

    def _last(self, s: Int) -> Int:
        var at = -1
        for i in range(len(self.slot)):
            if self.slot[i] == s:
                at = i
        return at

    def _scalar(self, s: Int) raises -> String:
        var at = self._last(s)
        if at >= 0:
            return self.value[at]
        ref a = self.decl.args[s]
        if a.default.byte_length() == 0 and a.kind != ArgKind.TEXT:
            raise Error("QM-E602: argument '" + a.name + "' of pass '" + self.decl.name + "' has no default. Hint: test given() before reading it.")
        return a.default

    def _items(self, s: Int) -> List[String]:
        var out = List[String]()
        for i in range(len(self.slot)):
            if self.slot[i] == s:
                out.append(self.value[i])
        return out^

    def position(self, name: String) raises -> Int:
        """Where the argument was written last (index among the arguments), -1 if it was not."""
        var s = self.decl.arg_index(name)
        if s < 0:
            raise Error("QM-E602: pass '" + self.decl.name + "' reads an argument '" + name + "' it does not declare. Hint: declare it in pass_decl.")
        return self._last(s)

    def given(self, name: String) raises -> Bool:
        return self.position(name) >= 0

    def get_int(self, name: String) raises -> Int:
        return Int(self._scalar(self._index(name, ArgKind.INT)))

    def get_float(self, name: String) raises -> Float64:
        return Float64(self._scalar(self._index(name, ArgKind.FLOAT)))

    def get_flag(self, name: String) raises -> Bool:
        """False for 0 / false / off; true for the other words the flag takes."""
        var v = self._scalar(self._index(name, ArgKind.FLAG))
        return v != "0" and v != "false" and v != "off"

    def get_text(self, name: String) raises -> String:
        """The word of a choice or a text argument (or of a flag, for its extra words)."""
        return self._scalar(self._index(name, ArgKind.CHOICE, ArgKind.TEXT, ArgKind.FLAG))

    def get_list(self, name: String) raises -> List[String]:
        """The items of a list argument, in the order written."""
        return self._items(self._index(name, ArgKind.GATES, ArgKind.BOUNDS, ArgKind.PASSES))


def nested_pipeline(p: PassSpec) raises -> String:
    """The inner pipeline of a pass that holds one (fixpoint), else ""."""
    var d = pass_decl(p.id)
    var s = d.list_index()
    if s < 0 or d.args[s].kind != ArgKind.PASSES:
        return ""
    return _join_args(PassArgs(p.id, p.args).get_list(d.args[s].name))


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
        PassArgs(id, args).validate()
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
        if target.angle_bound[g.kind] and not g.p0.is_symbolic():
            var v = g.p0.value()
            if v < target.angle_lo[g.kind] - 1e-9 or v > target.angle_hi[g.kind] + 1e-9:
                return "instruction #" + String(i) + " '" + String(g) + "' has its angle outside [" + String(target.angle_lo[g.kind]) + ", " + String(target.angle_hi[g.kind]) + "]"
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
    var exact: Bool  # a block of a Program: see GateSet.exact
    # running totals of the AI synthesis passes, per BlockKind:
    # blocks, proposed, verified, replaced, saved 2q cost, rejected
    var synth_totals: List[Int]

    def __init__(out self, n: Int, seed: Int):
        self.exact = False
        self.init = List[Int](length=n, fill=0)
        for i in range(n):
            self.init[i] = i
        self.final = self.init.copy()
        self.dropped = False
        self.seed = seed
        self.routed = False
        self.synth_totals = List[Int](length=BlockKind.NUM * 6, fill=0)


def _compose(prev: List[Int], route: List[Int]) -> List[Int]:
    """Layout composition: logical -> wire (prev) then wire -> physical (route)."""
    var out = List[Int](length=len(route), fill=0)
    var nprev = len(prev)
    for l in range(len(route)):
        var w = prev[l] if l < nprev else l
        out[l] = route[w]
    return out^


def _gs_from_args(gates: List[String], c: Circuit, target: Target) raises -> GateSet:
    if len(gates) == 0:
        return working_gate_set(c, target)
    return GateSet(parse_basis(_join_args(gates)))


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


def route_options(args: List[String], seed: Int, mut aopts: AIRouteOptions, mut star: Bool) raises -> RouteOptions:
    """Options of a `route(...)` pass: the routing ones, and what is not routing
    proper, the learned router in `aopts` and star pre-routing in `star`.
    Every value, written or default, comes from the pass table."""
    var a = PassArgs(PassId.ROUTE, args)
    var opts = RouteOptions()
    opts.layout = a.get_text("layout")
    opts.seed = a.get_int("seed") if a.given("seed") else seed
    opts.trials = a.get_int("trials")
    opts.iterations = a.get_int("iterations")
    opts.lookahead = a.get_int("lookahead")
    opts.weight = a.get_float("weight")
    opts.decay = a.get_float("decay")
    opts.vf2_steps = a.get_int("vf2_steps")
    opts.valve = a.get_int("valve")
    star = a.get_flag("star")
    opts.heuristic = a.get_text("heuristic")
    opts.swap_trials = a.get_int("swap_trials")
    opts.absorb = a.get_float("absorb")
    opts.absorb_alt = a.get_float("absorb_alt")
    opts.mirror_auto = a.get_text("mirror") == "auto"
    opts.mirror = a.get_flag("mirror")
    opts.mirror_gain = a.get_float("mirror_gain")
    if a.get_text("mirror") == "free" and a.position("mirror") > a.position("mirror_gain"):
        opts.mirror_gain = 1e9  # unless a mirror_gain is written after it
    opts.commute_auto = a.get_text("commute") == "auto"
    opts.commute = a.get_flag("commute")
    aopts.policy = a.get_text("policy")
    aopts.level = a.get_int("ai_level")
    aopts.layout_mode = a.get_text("layout_mode")
    aopts.model_dir = a.get_text("model_dir")
    aopts.seed = opts.seed
    return opts^


def _apply(p: PassSpec, mut c: Circuit, target: Target, mut st: _State) raises -> String:
    """Run one pass in place; returns a short note."""
    var id = p.id
    var a = PassArgs(id, p.args)
    var gs = working_gate_set(c, target)
    gs.exact = st.exact
    if id == PassId.REMOVE_IDENTITIES:
        return "removed " + String(remove_identities(c, st.exact))
    if id == PassId.CANCEL:
        return "cancelled " + String(cancel_inverses(c, gs, a.get_int("window"))) + " pairs"
    if id == PassId.MERGE:
        return "merged " + String(merge_rotations(c, gs))
    if id == PassId.COMMUTE:
        return "merged " + String(commute_rotations(c, gs, a.get_int("window")))
    if id == PassId.FUSE1Q:
        var g2 = _gs_from_args(a.get_list("gates"), c, target)
        g2.exact = st.exact
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
        var gates = a.get_list("gates")
        if len(gates) > 0:
            g2 = _gs_from_args(gates, c, target)
        g2.exact = st.exact
        var dropped = False
        var r = translate_to_basis(c, g2, dropped)
        if dropped:
            st.dropped = True
        return "translated " + String(r) + " gates to " + String(g2) + (" (symbolic phase dropped)" if dropped else "")
    if id == PassId.ROUTE:
        var aopts = AIRouteOptions()
        var star = True
        var opts = route_options(p.args, st.seed, aopts, star)
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
        var use_ai = aopts.policy != "off" and not plan.trivial and not plan.vf2
        var pre = Circuit(1)
        if use_ai:
            pre = c.copy()
        var swaps = finish_route(c, target, opts, plan, il, fl, note)
        note += star_note
        if use_ai and swaps > 0:
            note += _ai_route_portfolio(pre, c, target, opts, aopts, plan, il, fl)
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
        tgs.exact = st.exact
        var anyset = GateSet()
        anyset.exact = st.exact
        var rounds = 4 if len(c.ops) < 20000 else 1
        for _ in range(rounds):
            var b2 = 0
            var s2 = consolidate_2q(c, tgs, b2)
            blocks += b2
            saved += s2
            if s2 <= 0:
                break
            _ = cancel_inverses(c, anyset, 32)
        if not gs.any and saved > 0:
            var dropped = False
            _ = translate_to_basis(c, gs, dropped)
        return "resynthesised " + String(blocks) + " blocks, 2q cost -" + String(saved)
    if a.decl.block_kind >= 0:
        var opts = synth_options_from_args(id, p.args, st.seed)
        var stats = SynthStats()
        var physical = st.routed and not target.full
        var saved = resynthesize_blocks(c, target, physical, opts, stats)
        var k = a.decl.block_kind * 6
        st.synth_totals[k] += stats.blocks
        st.synth_totals[k + 1] += stats.proposed
        st.synth_totals[k + 2] += stats.verified
        st.synth_totals[k + 3] += stats.replaced
        st.synth_totals[k + 4] += stats.saved_cost
        st.synth_totals[k + 5] += stats.rejected_not_equal
        return _synth_note(stats, saved, physical)
    if id == PassId.PAULI_SIMP:
        var objective_t = a.get_text("objective") == "t"
        var pgs = GateSet(target.basis)
        pgs.exact = st.exact
        var saved = pauli_resynthesize(c, 64, pgs, objective_t)
        if objective_t:
            return ("saved " + String(saved) + " T gates") if saved > 0 else "kept original"
        return ("saved " + String(saved) + " 2q gates") if saved > 0 else "kept original"
    if id == PassId.ELIDE_SWAPS:
        var wm = List[Int]()
        var trailing = st.routed and not target.full
        var k = elide_swaps(c, trailing, wm)
        if k > 0:
            st.final = _compose(st.final, wm)
        return "elided " + String(k) + " swaps" + (" (trailing only)" if trailing else "")
    if id == PassId.WRAP_ANGLES:
        var bounded = target.copy()
        for b in a.get_list("bounds"):
            parse_angle_bound(b, bounded)
        var changed = wrap_angles(c, bounded)
        var g2 = GateSet(target.basis)
        g2.exact = st.exact
        if changed > 0 and len(target.basis) > 0:
            # Pauli corrections may be outside the basis
            var dropped = False
            _ = translate_to_basis(c, g2, dropped)
        return "wrapped " + String(changed) + " angles"
    if id == PassId.POST_LAYOUT:
        var wm = List[Int]()
        var gain = post_layout(c, target, wm, a.get_int("call_limit"))
        if gain > 0.0:
            st.init = _compose(st.init, wm)
            st.final = _compose(st.final, wm)
        return "log fidelity gain " + fmt_num(gain)
    if id == PassId.CHECK:
        var err = conformance_error(c, target)
        if err.byte_length() > 0:
            raise Error("QM-E615: compiled circuit does not conform to target '" + target.name + "': " + err + ". Hint: add 'route' and 'basis' passes before 'check' (or use a preset level).")
        return "conforms to " + target.name
    raise Error("QM-E600: pass '" + pass_name(id) + "' (id " + String(id) + ") has no branch in _apply. Hint: add the branch that runs it.")


def synth_options_from_args(id: Int, args: List[String], seed: Int) raises -> SynthOptions:
    """SynthOptions of a synthesis pass. Every value, written or default,
    comes from the pass table (the defaults are the IBM collector defaults,
    as `SynthOptions.default`)."""
    var a = PassArgs(id, args)
    var opts = SynthOptions(a.decl.block_kind)
    opts.method = a.get_text("method")
    opts.max_qubits = a.get_int("max_qubits")
    opts.min_block = a.get_int("min_block")
    opts.direction = a.get_text("direction")
    opts.commute = a.get_flag("commute")
    opts.reps = a.get_int("reps")
    opts.model = a.get_text("model")
    opts.beams = a.get_int("beams")
    opts.searches = a.get_int("searches")
    opts.metric = a.get_text("metric")
    opts.strict = a.get_flag("strict")
    opts.budget = a.get_int("budget")
    opts.model_dir = a.get_text("model_dir")
    opts.seed = a.get_int("seed") if a.given("seed") else seed
    return opts^


def _json_safe(s: String) -> String:
    """Notes are embedded verbatim in PassReport JSON: drop quotes / backslashes / newlines."""
    return s.replace("\\", "/").replace("\"", "'").replace("\n", " ")


def _synth_note(stats: SynthStats, saved: Int, physical: Bool) -> String:
    var s = String("blocks=") + String(stats.blocks) + " proposed=" + String(stats.proposed)
    s += " verified=" + String(stats.verified) + " replaced=" + String(stats.replaced)
    s += " saved_cost=" + String(stats.saved_cost) + " saved_depth=" + String(stats.saved_depth)
    s += " rejected=" + String(stats.rejected_not_equal)
    s += " (" + ("topology-aware" if physical else "all-to-all") + ")"
    if stats.note.byte_length() > 0:
        s += " " + _json_safe(stats.note)
    _ = saved
    return s


def _synth_totals_note(inner: List[PassSpec], before: List[Int], after: List[Int]) -> String:
    """Synthesis statistics accumulated inside a fixpoint ("" when it holds
    no synthesis pass, so heuristic presets keep their notes)."""
    var s = String()
    for kind in range(BlockKind.NUM):
        var sid = synth_pass_of_kind(kind)
        var nm = pass_name(sid)
        var present = False
        for p in inner:
            if p.id == sid:
                present = True
        if not present:
            continue
        var k = kind * 6
        s += " | " + nm + " blocks=" + String(after[k] - before[k]) + " proposed=" + String(after[k + 1] - before[k + 1])
        s += " verified=" + String(after[k + 2] - before[k + 2]) + " replaced=" + String(after[k + 3] - before[k + 3])
        s += " saved_cost=" + String(after[k + 4] - before[k + 4]) + " rejected=" + String(after[k + 5] - before[k + 5])
    return s


def _ai_route_portfolio(pre: Circuit, mut c: Circuit, target: Target, opts: RouteOptions,
                        aopts: AIRouteOptions, plan: RoutePlan,
                        mut il: List[Int], mut fl: List[Int]) raises -> String:
    """Learned-policy routing of the pre-route circuit `pre`, seeded with the
    SABRE layout; replaces the SABRE result `c` (+ its layouts `il` / `fl`)
    only when the policy result is verified by `ai_route` and strictly cheaper
    under `routed_cost` (2q cost in CX units, then 2q depth)."""
    var a2 = aopts.copy()
    a2.seed_layout = plan.layout.copy()
    a2.portfolio = False  # SABRE already ran: the portfolio is decided here
    if len(opts.initial_layout) > 0 and a2.layout_mode == "optimize":
        a2.layout_mode = "improve"  # IBM rule: a user layout is improved, not replaced
    var out = Circuit(1)
    var ail = List[Int]()
    var afl = List[Int]()
    var anote = String()
    var ok: Bool
    try:
        ok = ai_route(pre, target, opts, a2, out, ail, afl, anote)
    except e:
        var msg = String(e)
        if msg.startswith("QM-E616") or msg.startswith("QM-E617") or msg.startswith("QM-E620"):
            if a2.policy != "auto":
                raise e^  # an explicitly requested model must exist and match
        return ", ai-route: failed (" + _json_safe(msg) + "), kept sabre"
    if not ok:
        return ", ai-route: no proposal (" + _json_safe(anote) + "), kept sabre"
    var cs = routed_cost(c, target)
    var ca = routed_cost(out, target)
    var tag = ", ai-route cost " + String(ca[0]) + "/" + String(ca[1]) + " vs sabre " + String(cs[0]) + "/" + String(cs[1])
    if ca[0] < cs[0] or (ca[0] == cs[0] and ca[1] < cs[1]):
        c = out^
        il = ail^
        fl = afl^
        return tag + ": kept ai-route (" + _json_safe(anote) + ")"
    return tag + ": kept sabre"


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
            var fa = PassArgs(p.id, p.args)
            var max_it = fa.get_int("max")
            var inner = parse_pipeline(_join_args(fa.get_list("passes")))
            var deltas = List[Int](length=len(inner), fill=0)
            var totals0 = st.synth_totals.copy()
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
            note += _synth_totals_note(inner, totals0, st.synth_totals)
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


def _name_inexpressible_gate(err: String, circuit: Circuit, target: Target) raises -> Error:
    """QM-E604 points at whatever gate the pipeline reached: after fuse1q and
    routing that is a fused `u` on a physical qubit, which the user never wrote.
    Name the first gate of the input that has no exact form in the basis."""
    if not err.startswith("QM-E604"):
        return Error(err)
    var gs = GateSet(target.basis)
    var i = first_inexpressible_1q(circuit, gs)
    if i < 0:
        return Error(err)
    var hint = String("include rz+sx, rz+ry, rz+rx, rx+ry or u in the basis")
    if not can_synthesize_1q(gs):
        hint = "this basis is not universal for exact 1q synthesis: use angles it can express exactly (for Clifford+T, multiples of pi/4), or approximate the rotation first (qumo.ft.rz)"
    return Error("QM-E604: cannot express 1-qubit gate '" + String(circuit.ops[i]) + "' (instruction #" + String(i) + " of the input circuit) in basis " + String(gs) + ". Hint: " + hint + ".")


struct PassManager(Copyable, Writable):
    var spec: String
    var passes: List[PassSpec]
    var seed: Int
    var exact: Bool  # no pass rounds off more than EXACT_TOL per gate: blocks of a Program

    def __init__(out self, spec: String, seed: Int = 7) raises:
        self.spec = spec
        self.passes = parse_pipeline(spec)
        self.seed = seed
        self.exact = False

    @staticmethod
    def preset(level: Int, target: Target, seed: Int = 7) raises -> PassManager:
        return PassManager(preset_pipeline(level, target), seed)

    def canonical(self) -> String:
        return pipeline_to_string(self.passes)

    def run(self, circuit: Circuit, target: Target) raises -> CompileResult:
        circuit.validate()
        if circuit.count(Op.CORRELATED_ERROR) > 0:
            raise Error("QM-E624: E / correlated_error reaches the compiler, which cannot see the qubits it touches. Hint: compile first, then add noise (qumo.exec.emulate does that order).")
        var res = CompileResult(circuit)
        res.pipeline = self.canonical()
        res.target_name = target.name
        res.basis = basis_str(target.basis)
        var c = circuit.copy()
        reduce_large_angles(c)
        var st = _State(c.num_qubits, self.seed)
        st.exact = self.exact
        var t0 = perf_counter_ns()
        try:
            _run_list(self.passes, c, target, st, res.reports)
        except e:
            var named = _name_inexpressible_gate(String(e), circuit, target)
            raise named^
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
    return _with_device_tail(parts^, level, target)


def _with_device_tail(var parts: List[String], level: Int, target: Target) -> String:
    """Join a preset, adding what only some targets need before its `check`:
    `wrap_angles` for angle bounds, `post_layout` for real calibration data."""
    var tail = List[String]()
    if target.has_angle_bounds():
        tail.append("wrap_angles")
        tail.append("fuse1q")
    if target.is_calibrated() and level >= 1:
        tail.append("post_layout")
    if len(tail) > 0:
        var check = len(parts) > 0 and parts[len(parts) - 1] == "check"
        if check:
            _ = parts.pop()
        for t in tail:
            parts.append(t)
        if check:
            parts.append("check")
    var s = String()
    for i in range(len(parts)):
        if i > 0:
            s += ","
        s += parts[i]
    return s


def _better(a: CompileResult, b: CompileResult) -> Bool:
    """a strictly better than b: fewer 2q gates, then lower depth, then fewer gates."""
    return _better_on(a, b, False)


def _better_on(a: CompileResult, b: CompileResult, calibrated: Bool) -> Bool:
    """`_better`, or higher estimated fidelity when the target is calibrated."""
    if calibrated and a.est_fidelity != b.est_fidelity:
        return a.est_fidelity > b.est_fidelity
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
        if _better_on(r2, r, target.is_calibrated()):
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


# ----------------------------------------------------------------------------
# Hybrid heuristic + AI presets (docs/AI_TRANSPILER.md)
# ----------------------------------------------------------------------------
def preset_pipeline_ai(level: Int, target: Target) raises -> String:
    """Hybrid heuristic + AI pipeline for AI level 1..3 (the vendor-neutral
    counterpart of qiskit-ibm-transpiler's `generate_ai_pass_manager`).

    * ai1: O1 with learned-policy routing (`route(...,policy=auto,ai_level=1)`).
    * ai2: O2 with policy routing (ai_level=2), then the IBM post-routing
      linear-function resynthesis (forward, then backward collection, no
      commutation analysis) and permutation resynthesis of residual SWAP
      networks, then kak, basis and cleanup.
    * ai3: O3 with all-to-all Clifford resynthesis before routing, policy
      routing (ai_level=3, layout_mode=optimize), kak, then the post-routing
      synthesis list [linear_synth (commutation-aware, both directions),
      clifford_synth, perm_synth, pauli_network] run twice inside a fixpoint,
      then basis and the O3 cleanup fixpoint. `transpile_ai(..., 3)` runs it
      as a portfolio with `transpile(..., 3)`.

    Every synthesis pass keeps a block only when its replacement is verified
    exactly equal and cheaper, so without trained models (or on all-to-all
    targets) the exact / greedy synthesizers still run and routing falls back
    to SABRE.
    """
    if level < 1 or level > 3:
        raise Error("QM-E609: AI optimization level must be 1..3, got " + String(level) + ". Hint: 1 = O1 + policy routing, 2 = default (+ linear / permutation resynthesis), 3 = heavy (+ Clifford / Pauli-network resynthesis, portfolio with O3).")
    var route = not target.full
    var basis = len(target.basis) > 0
    var parts = List[String]()
    if level == 1:
        parts.append("remove_identities")
        parts.append("cancel")
        parts.append("merge")
        if route:
            parts.append("route(sabre,layout=auto,trials=5,iterations=1,swap_trials=5,policy=auto,ai_level=1)")
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
            parts.append("route(sabre,layout=auto,trials=20,iterations=2,swap_trials=20,policy=auto,ai_level=2)")
        elif basis and not target.supports(Op.CCX):
            parts.append("unroll3q")
        # IBM's level-2 post-routing linear-function stage, placed before
        # kak: KAK then consolidates the resynthesised CX networks into the
        # target entangler (measured on the neutral set: 31 wins / 1 loss vs
        # O2 instead of 19 / 0 when run after kak, same compile time)
        parts.append("linear_synth(direction=forward,commute=0)")
        parts.append("linear_synth(direction=backward,commute=0)")
        parts.append("perm_synth")
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
        parts.append("clifford_synth")
        parts.append("fixpoint(cancel,commute,phase_fold,fuse1q,max=4)")
        if not route and basis and not target.supports(Op.CCX):
            parts.append("unroll3q")
        parts.append("kak")
        if route:
            parts.append("route(sabre,layout=auto,trials=40,iterations=3,swap_trials=40,policy=auto,ai_level=3,layout_mode=optimize)")
            parts.append("kak")
        parts.append("fixpoint(linear_synth(direction=both,commute=1),clifford_synth,perm_synth,pauli_network,max=2)")
        if basis:
            parts.append("basis")
        parts.append("fixpoint(cancel,commute,phase_fold,eject_z,fuse1q,max=10)")
        parts.append("check")
    return _with_device_tail(parts^, level, target)


def transpile_ai(circuit: Circuit, target: Target, level: Int = 2, seed: Int = 7) raises -> CompileResult:
    """Compile `circuit` for `target` with the hybrid heuristic + AI preset
    `preset_pipeline_ai(level)` (level 1..3).

    Level 3 is a portfolio with `transpile(circuit, target, 3, seed)` (itself
    O3 + O2): the better result under `_better` (2q count, then depth, then
    gate count) is returned, so AI level 3 is never worse than O3. Results are
    exact (every learned proposal is verified) and deterministic for a seed.
    """
    var pm = PassManager(preset_pipeline_ai(level, target), seed)
    var r = pm.run(circuit, target)
    if level == 3:
        var r2 = transpile(circuit, target, 3, seed)
        if _better_on(r2, r, target.is_calibrated()):
            r2.total_us += r.total_us
            return r2^
        r.total_us += r2.total_us
    return r^


def transpile_ai(circuit: Circuit, level: Int = 2, seed: Int = 7) raises -> CompileResult:
    """`transpile_ai` for the default, vendor-neutral target: all-to-all, any gate."""
    return transpile_ai(circuit, Target.all_to_all(circuit.num_qubits), level, seed)
