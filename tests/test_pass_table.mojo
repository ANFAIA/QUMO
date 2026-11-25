"""The pass table (`pass_decl` in qumo/compiler/pass_manager.mojo) and what is
derived from it: names, aliases, descriptions, the argument check of
`parse_pipeline` and the values `_apply` reads through `PassArgs`.

Regression: `post_layout(call_limit=N)` passed the check and failed when run."""

from std.testing import assert_equal, assert_true, assert_false
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve
from qumo.compiler.common import BlockKind
from qumo.compiler.collect import KIND_CLIFFORD, KIND_LINEAR, KIND_PERM, KIND_PAULI
from qumo.compiler.policy import MODEL_KIND_CLIFFORD, MODEL_KIND_LINEAR, MODEL_KIND_PERM, MODEL_KIND_PAULI, MODEL_KIND_ROUTING
from qumo.compiler.target import Target
from qumo.compiler.routing import RouteOptions
from qumo.compiler.resynth import SynthOptions
from qumo.compiler.ai_route import AIRouteOptions
from qumo.compiler.pass_manager import (
    PassId, PassDecl, ArgDecl, ArgKind, ArgForm, PassArgs, pass_decl, pass_table, check_pass_table,
    pass_name, pass_description, pass_aliases, pass_id_from_name, pass_block_kind, is_synth_pass,
    synth_pass_of_kind, pass_relabels, parse_pipeline, pipeline_to_string, compile_spec, describe_passes,
    route_options, synth_options_from_args, valid_pass_names,
)
from qumo.compiler.program_compile import compile_spec_program
from qumo.ir.program import Program
from qumo.ai.describe import describe_passes_json
from qumo.core.json import parse_json


def error_of(spec: String) -> String:
    try:
        _ = parse_pipeline(spec)
    except e:
        return String(e)
    return ""


def calibrated_line() raises -> Target:
    """A 6-qubit line whose edges 3-4 and 4-5 are ten times better than the rest."""
    var t = Target.line(6, "sc_cz")
    t.set_edge_error(0, 1, 0.05)
    t.set_edge_error(1, 2, 0.05)
    t.set_edge_error(2, 3, 0.05)
    t.set_edge_error(3, 4, 0.005)
    t.set_edge_error(4, 5, 0.005)
    return t^


def chain() -> Circuit:
    var c = Circuit(6)
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.rz(0.3, 2)
    return c^


def sample_value(a: ArgDecl) -> String:
    """A value the argument takes that is not its default where one exists."""
    if a.kind == ArgKind.INT:
        return "3"
    if a.kind == ArgKind.FLOAT:
        return "0.25"
    if a.kind == ArgKind.FLAG:
        return "off"
    if a.kind == ArgKind.CHOICE:
        var parts = a.choices.split("|")
        return String(parts[len(parts) - 1])
    if a.kind == ArgKind.TEXT:
        return "some_dir"
    if a.kind == ArgKind.GATES:
        return "rz"
    if a.kind == ArgKind.BOUNDS:
        return "rzz[0:pi/2]"
    return "cancel"


def test_post_layout_call_limit() raises:
    """The form the check accepts and its hint prints used to raise
    "String is not convertible to integer" when the pass ran."""
    var t = calibrated_line()
    assert_true(t.is_calibrated())
    var c = chain()
    var base = compile_spec(c, t, "route(sabre),basis,post_layout,check")
    for spec in ["post_layout(call_limit=1000)", "post_layout(1000)", "post_layout(call_limit = 1000)", "noise_layout(call_limit=1000)"]:
        var r = compile_spec(c, t, "route(sabre),basis," + spec + ",check")
        assert_equal(r.circuit.num_2q_gates(), base.circuit.num_2q_gates(), spec)
        assert_true(r.est_fidelity >= base.est_fidelity - 1e-12, spec)
    # a limit of one step still returns a valid circuit
    var tiny = compile_spec(c, t, "route(sabre),basis,post_layout(call_limit=1),check")
    assert_true(tiny.est_fidelity <= base.est_fidelity + 1e-12)
    # the other passes with a number written both ways
    var a = compile_spec(c, t, "cancel(window=4),commute(window=4),route(sabre),basis,check")
    var b = compile_spec(c, t, "cancel(4),commute(4),route(sabre),basis,check")
    assert_equal(a.circuit.num_unitary_gates(), b.circuit.num_unitary_gates())
    assert_equal(pipeline_to_string(parse_pipeline("post_layout(call_limit=1000)")), "post_layout(call_limit=1000)")
    print("  post_layout(call_limit=N), cancel(window=N), commute(window=N) run")


def test_table_is_whole() raises:
    check_pass_table()
    assert_equal(len(pass_table()), PassId.NUM)
    var names = List[String]()
    for i in range(PassId.NUM):
        var d = pass_decl(i)
        assert_equal(d.id, i)
        assert_true(d.name.byte_length() > 0, "pass id " + String(i) + " has a declaration")
        assert_equal(pass_name(i), d.name)
        assert_equal(pass_decl(i, True).name, d.name)
        assert_equal(pass_id_from_name(d.name), i, d.name)
        assert_equal(pass_id_from_name("  " + d.name.upper() + " "), i, "case and spaces: " + d.name)
        assert_true(valid_pass_names().find(d.name) >= 0)
        assert_true(describe_passes().find("\"" + d.name + "\"") >= 0)
        for a in pass_aliases(i):
            assert_equal(pass_id_from_name(a), i, "alias " + a)
            names.append(a)
        names.append(d.name)
        assert_equal(pass_block_kind(i), d.block_kind)
        assert_equal(is_synth_pass(i), d.block_kind >= 0)
        assert_equal(pass_relabels(i), d.relabels)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert_true(names[i] != names[j], "name declared twice: " + names[i])
    assert_equal(pass_name(PassId.NUM), "unknown")
    assert_equal(pass_name(-1), "unknown")
    assert_equal(pass_id_from_name("no_such_pass"), -1)
    assert_equal(pass_id_from_name(""), -1)
    assert_equal(pass_id_from_name("cancel|merge"), -1)
    # every name the six hand-written tables took before the table (public: specs in docs and tools use them)
    var old: List[String] = [
        "remove_identities", "remove_identity", "rmid", "identities", "cancel", "cancellation", "inverse_cancel",
        "cancel_inverses", "merge", "merge_rotations", "commute", "commutation", "commute_rotations", "fuse1q", "fuse",
        "fuse_1q", "optimize1q", "phase_fold", "phase_folding", "phasefold", "eject_z", "ejectz", "virtual_z",
        "unroll3q", "unroll_3q", "decompose3q", "route", "routing", "sabre", "basis", "translate", "basis_translate",
        "rebase", "fixpoint", "fix", "repeat", "loop", "check", "verify", "predicate", "kak", "consolidate",
        "consolidate_blocks", "2q_synth", "elide_swaps", "elide", "elide_permutations", "pauli_simp", "paulisimp",
        "pauli", "gadgets", "clifford_synth", "ai_clifford", "clifford_synthesis", "linear_synth", "ai_linear",
        "ai_linear_function", "linear_function_synth", "perm_synth", "ai_permutation", "permutation_synth",
        "pauli_network", "ai_pauli_network", "pauli_network_synth", "wrap_angles", "wrap", "fold_angles",
        "post_layout", "vf2_post_layout", "noise_layout",
    ]
    var old_ids: List[Int] = [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10,
        10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 18, 18, 18,
        19, 19, 19, 20, 20, 20,
    ]
    assert_equal(len(old), len(old_ids))
    assert_equal(len(old), len(names), "no name was added or lost")
    for i in range(len(old)):
        assert_equal(pass_id_from_name(old[i]), old_ids[i], old[i])
    print("  every id has a declaration,", len(names), "names and aliases resolve")


def test_block_kinds() raises:
    """One definition of the block kinds; the pass of each kind is named by the table."""
    assert_equal(KIND_CLIFFORD, BlockKind.CLIFFORD)
    assert_equal(KIND_LINEAR, BlockKind.LINEAR)
    assert_equal(KIND_PERM, BlockKind.PERM)
    assert_equal(KIND_PAULI, BlockKind.PAULI)
    assert_equal(MODEL_KIND_CLIFFORD, BlockKind.CLIFFORD)
    assert_equal(MODEL_KIND_LINEAR, BlockKind.LINEAR)
    assert_equal(MODEL_KIND_PERM, BlockKind.PERM)
    assert_equal(MODEL_KIND_PAULI, BlockKind.PAULI)
    assert_equal(MODEL_KIND_ROUTING, BlockKind.NUM)
    assert_equal(BlockKind.NUM, 4)
    assert_equal(synth_pass_of_kind(BlockKind.CLIFFORD), PassId.CLIFFORD_SYNTH)
    assert_equal(synth_pass_of_kind(BlockKind.LINEAR), PassId.LINEAR_SYNTH)
    assert_equal(synth_pass_of_kind(BlockKind.PERM), PassId.PERM_SYNTH)
    assert_equal(synth_pass_of_kind(BlockKind.PAULI), PassId.PAULI_NETWORK)
    assert_equal(synth_pass_of_kind(BlockKind.NUM), -1)
    var n = 0
    for i in range(PassId.NUM):
        if is_synth_pass(i):
            n += 1
            assert_equal(synth_pass_of_kind(pass_block_kind(i)), i)
            assert_equal(synth_options_from_args(i, List[String](), 7).kind, pass_block_kind(i))
    assert_equal(n, BlockKind.NUM)
    print("  block kinds: one definition, passes mapped by the table")


def test_every_argument() raises:
    """Each declared argument is accepted by the check and read back by the
    accessor of its kind, written with its key and, where declared, bare."""
    var count = 0
    for i in range(PassId.NUM):
        var d = pass_decl(i)
        for a in d.args:
            var v = sample_value(a)
            var forms = List[String]()
            if a.form != ArgForm.BARE:
                forms.append(a.name + "=" + v)
                forms.append(a.name + " = " + v)  # spaces around '=' are the same argument
            if a.form != ArgForm.KEY:
                forms.append(v)
            for f in forms:
                var args = List[String]()
                args.append(f)
                if d.id == PassId.FIXPOINT and a.kind != ArgKind.PASSES:
                    args.append("cancel")  # fixpoint needs an inner pass
                var pa = PassArgs(i, args)
                pa.validate()
                assert_true(pa.given(a.name), d.name + "(" + f + ")")
                assert_equal(pa.position(a.name), 0)
                if a.kind == ArgKind.INT:
                    assert_equal(pa.get_int(a.name), 3, d.name + "(" + f + ")")
                elif a.kind == ArgKind.FLOAT:
                    assert_equal(pa.get_float(a.name), 0.25)
                elif a.kind == ArgKind.FLAG:
                    assert_false(pa.get_flag(a.name), d.name + "(" + f + ")")
                    assert_equal(pa.get_text(a.name), "off")
                elif a.kind == ArgKind.CHOICE or a.kind == ArgKind.TEXT:
                    assert_equal(pa.get_text(a.name), v, d.name + "(" + f + ")")
                else:
                    var items = pa.get_list(a.name)
                    assert_equal(len(items), 1)
                    assert_equal(items[0], v)
                # and through the parser, as text
                var spec = d.name + "(" + f + (",cancel" if len(args) == 2 else "") + ")"
                assert_equal(error_of(spec), "", spec)
                count += 1
            # not written: the default, when the table has one
            var none = PassArgs(i, List[String]())
            assert_false(none.given(a.name))
            assert_equal(none.position(a.name), -1)
            if a.default.byte_length() > 0:
                if a.kind == ArgKind.INT:
                    assert_equal(none.get_int(a.name), Int(a.default))
                elif a.kind == ArgKind.FLOAT:
                    assert_equal(none.get_float(a.name), Float64(a.default))
                elif a.kind == ArgKind.FLAG:
                    assert_equal(none.get_flag(a.name), a.default != "0" and a.default != "off" and a.default != "false")
                else:
                    assert_equal(none.get_text(a.name), a.default)
            # every extra word of a flag and every choice
            if a.kind == ArgKind.FLAG or a.kind == ArgKind.CHOICE:
                var words = String("0|1|true|false|on|off") if a.kind == ArgKind.FLAG else a.choices
                if a.kind == ArgKind.FLAG and a.choices.byte_length() > 0:
                    words += "|" + a.choices
                for w in words.split("|"):
                    var one = List[String]()
                    one.append((a.name + "=" + String(w)) if a.form != ArgForm.BARE else String(w))
                    var pw = PassArgs(i, one)
                    pw.validate()
                    assert_equal(pw.get_text(a.name), String(w))
            # the description and the unknown-argument hint name it
            if not a.is_list() and a.form != ArgForm.BARE:
                assert_true(pass_description(i).find(a.name + "=") >= 0 or pass_description(i).find(a.name + "=]") >= 0,
                            d.name + " description lists " + a.name)
                var hint = error_of(d.name + "(no_such_key=1)")
                assert_true(hint.startswith("QM-E602") or d.list_index() >= 0, hint)
                if d.list_index() < 0:
                    assert_true(hint.find(a.name + "=") >= 0, "hint of " + d.name + " lists " + a.name)
    # reading what is not declared, or with the wrong accessor, is an error and not a silent default
    var pa = PassArgs(PassId.CANCEL, List[String]())
    var raised = 0
    try:
        _ = pa.get_int("windows")
    except e:
        raised += 1 if String(e).startswith("QM-E602") else 0
    try:
        _ = pa.get_text("window")
    except e:
        raised += 1 if String(e).startswith("QM-E602") else 0
    try:
        _ = PassArgs(PassId.ROUTE, List[String]()).get_int("seed")  # no default: the compile seed
    except e:
        raised += 1 if String(e).startswith("QM-E602") else 0
    assert_equal(raised, 3)
    print("  ", count, "argument forms accepted by the check and read by the accessor")


def test_rejections() raises:
    """Unknown or bad arguments keep their error codes."""
    var bad602: List[String] = [
        "merge(3)", "check(x)", "kak(a=1)", "cancel(x)", "cancel(window=x)", "cancel(width=3)", "commute(1.5)",
        "post_layout(x)", "post_layout(call_limit=x)", "post_layout(window=5)", "pauli_simp(objective=x)",
        "pauli_simp(t)", "wrap_angles(rzz)", "fixpoint()", "fixpoint(max=3)", "fixpoint(cancel,max=x)",
        "route(magic)", "route(sabre,layout=magic)", "route(sabre,speed=9)", "route(method=sabre)",
        "route(sabre,trials=x)", "route(sabre,weight=x)", "route(sabre,star=maybe)", "route(sabre,star=auto)",
        "route(sabre,mirror=maybe)", "route(sabre,commute=free)", "route(sabre,heuristic=x)",
        "route(sabre,ai_level=0)", "route(sabre,ai_level=4)", "route(sabre,ai_level=x)",
        "route(sabre,layout_mode=random)", "linear_synth(speed=3)", "perm_synth(max_qubits=x)",
        "perm_synth(max_qubits=0)", "perm_synth(reps=0)", "perm_synth(beams=0)", "perm_synth(searches=0)",
        "perm_synth(min_block=-1)", "perm_synth(budget=-1)", "pauli_network(direction=sideways)",
        "pauli_network(commute=maybe)", "pauli_network(strict=2)", "pauli_network(metric=fidelity)",
    ]
    for s in bad602:
        assert_true(error_of(s).startswith("QM-E602"), s + " -> " + error_of(s))
        assert_true(error_of(s).find("Hint:") > 0, s)
    for s in ["linear_synth(method=magic)", "clifford_synth(quantum)", "perm_synth(model=)", "route(sabre,policy=)"]:
        assert_true(error_of(s).startswith("QM-E616"), s + " -> " + error_of(s))
    assert_true(error_of("perm_synth(max_qubits=99)").startswith("QM-E618"))
    assert_true(error_of("basis(rz,zz_top)").startswith("QM-E610"))
    assert_true(error_of("fixpoint(cancle)").startswith("QM-E600"))
    assert_true(error_of("fixpoint(route(sabre,speed=9))").startswith("QM-E602"))
    # what was accepted stays accepted
    var ok: List[String] = [
        "cancel(64)", "cancel(window=64)", "cancel()", "fuse1q(rz,sx,x)", "basis(sc_cz)", "basis(rz,sx,x,cx)",
        "fixpoint(cancel,commute,max=3)", "fixpoint(fixpoint(cancel,max=2),linear_synth(direction=both,commute=1),max=2)",
        "route(sabre)", "route(lightsabre)", "route", "sabre", "route(layout=dense)", "route(sabre, seed = 3)",
        "route(sabre,mirror=free,mirror_gain=2,commute=auto,absorb=0.8,absorb_alt=0.7,valve=5,vf2_steps=10,star=0)",
        "route(sabre,policy=auto,ai_level=3,layout_mode=improve,model_dir=models/ai_transpiler)",
        "route(sabre,trials=-1)", "wrap_angles(rzz[0:pi/2])", "wrap_angles", "post_layout", "post_layout(1000)",
        "pauli_simp(objective=t)", "pauli_simp(objective=2q)", "linear_synth(greedy,direction=backward,commute=0)",
        "ai_pauli_network(model=off,beams=4,searches=2,metric=cost,strict=1,reps=3,model_dir=/tmp/models,seed=3)",
        "perm_synth(max_qubits=64)", "clifford_synth(model_dir=)",
    ]
    for s in ok:
        assert_equal(error_of(s), "", s)
    # an unknown key is named, with every key the pass takes
    var m = error_of("route(sabre,speed=9)")
    assert_true(m.find("'speed'") > 0, m)
    print("  bad arguments keep QM-E602 / E616 / E618 / E610,", len(ok), "accepted forms still parse")


def test_route_help_is_complete() raises:
    """The route description and hint list every argument the check accepts
    (the hand-written description had lost nine of them)."""
    var d = pass_decl(PassId.ROUTE)
    var desc = pass_description(PassId.ROUTE)
    var hint = error_of("route(sabre,speed=9)")
    var keys: List[String] = [
        "layout", "seed", "trials", "iterations", "lookahead", "weight", "decay", "vf2_steps", "valve", "star",
        "heuristic", "swap_trials", "absorb", "absorb_alt", "mirror", "mirror_gain", "commute", "policy", "ai_level",
        "layout_mode", "model_dir",
    ]
    assert_equal(len(d.args), len(keys) + 1, "the keys the check took before the table, and the bare method")
    for k in keys:
        assert_true(d.keyed_index(k) >= 0, "route declares " + k)
        assert_true(desc.find(k + "=") >= 0, "route description lists " + k)
        assert_true(hint.find(k + "=") >= 0, "route hint lists " + k)
    assert_true(desc.find("sabre|lightsabre") >= 0 and hint.find("sabre|lightsabre") >= 0)
    for a in d.args:
        if a.form != ArgForm.BARE:
            var found = False
            for k in keys:
                found = found or k == a.name
            assert_true(found, "route accepts " + a.name + ", which the list above does not have")
    # the JSON catalog (qumo describe passes, MCP qumo_describe) carries the same arguments
    var j = parse_json(describe_passes_json())
    var seen = 0
    for i in range(len(j.get("passes"))):
        var p = j.get("passes").at(i)
        var pd = pass_decl(pass_id_from_name(p.get("name").as_string()))
        assert_equal(len(p.get("args")), len(pd.args), pd.name)
        assert_equal(len(p.get("aliases")), len(pd.alias_list()), pd.name)
        for k in range(len(pd.args)):
            assert_equal(p.get("args").at(k).get("name").as_string(), pd.args[k].name)
        seen += 1
    assert_equal(seen, PassId.NUM)
    for k in ["policy", "ai_level", "layout_mode", "model_dir"]:
        assert_true(j.get("route_ai_keys").has(k) and d.keyed_index(k) >= 0, k)
    print("  route description, hint and JSON catalog list its", len(keys), "keys")


def test_defaults_are_the_option_defaults() raises:
    """The table's defaults are the ones of RouteOptions / AIRouteOptions /
    SynthOptions.default, so a pass without arguments runs as before."""
    var none = List[String]()
    var aopts = AIRouteOptions()
    var star = False
    var o = route_options(none, 11, aopts, star)
    var r = RouteOptions()
    assert_equal(o.layout, r.layout)
    assert_equal(o.seed, 11)
    assert_equal(o.trials, r.trials)
    assert_equal(o.iterations, r.iterations)
    assert_equal(o.lookahead, r.lookahead)
    assert_equal(o.weight, r.weight)
    assert_equal(o.decay, r.decay)
    assert_equal(o.vf2_steps, r.vf2_steps)
    assert_equal(o.valve, r.valve)
    assert_equal(o.heuristic, r.heuristic)
    assert_equal(o.swap_trials, r.swap_trials)
    assert_equal(o.absorb, r.absorb)
    assert_equal(o.absorb_alt, r.absorb_alt)
    assert_equal(o.mirror, r.mirror)
    assert_equal(o.mirror_auto, r.mirror_auto)
    assert_equal(o.mirror_gain, r.mirror_gain)
    assert_equal(o.commute, r.commute)
    assert_equal(o.commute_auto, r.commute_auto)
    assert_equal(len(o.initial_layout), 0)
    assert_true(star, "star pre-routing is on by default")
    var a0 = AIRouteOptions()
    assert_equal(aopts.policy, "off")  # the pass does not use the learned router unless asked
    assert_equal(aopts.level, a0.level)
    assert_equal(aopts.layout_mode, a0.layout_mode)
    assert_equal(aopts.model_dir, a0.model_dir)
    assert_equal(aopts.seed, 11)
    for i in range(PassId.NUM):
        if not is_synth_pass(i):
            continue
        var s = synth_options_from_args(i, none, 5)
        var w = SynthOptions.default(pass_block_kind(i))
        assert_equal(s.kind, w.kind)
        assert_equal(s.method, w.method)
        assert_equal(s.max_qubits, w.max_qubits, pass_name(i))
        assert_equal(s.min_block, w.min_block, pass_name(i))
        assert_equal(s.direction, w.direction)
        assert_equal(s.commute, w.commute)
        assert_equal(s.reps, w.reps)
        assert_equal(s.model, w.model)
        assert_equal(s.beams, w.beams)
        assert_equal(s.searches, w.searches)
        assert_equal(s.metric, w.metric)
        assert_equal(s.strict, w.strict)
        assert_equal(s.budget, w.budget)
        assert_equal(s.model_dir, w.model_dir)
        assert_equal(s.seed, 5)
    # written values, the order rule of mirror / mirror_gain, and the last one counting
    var args: List[String] = ["sabre", "layout=dense", "seed=3", "trials=9", "weight=0.25", "star=off", "mirror=free", "commute=0", "policy=auto", "ai_level=3"]
    var o2 = route_options(args, 11, aopts, star)
    assert_equal(o2.layout, "dense")
    assert_equal(o2.seed, 3)
    assert_equal(o2.trials, 9)
    assert_equal(o2.weight, 0.25)
    assert_false(star)
    assert_true(o2.mirror and not o2.mirror_auto)
    assert_equal(o2.mirror_gain, 1e9)
    assert_true(not o2.commute and not o2.commute_auto)
    assert_equal(aopts.policy, "auto")
    assert_equal(aopts.level, 3)
    assert_equal(aopts.seed, 3)
    var g1: List[String] = ["mirror=free", "mirror_gain=2"]
    assert_equal(route_options(g1, 1, aopts, star).mirror_gain, 2.0)
    var g2: List[String] = ["mirror_gain=2", "mirror=free"]
    assert_equal(route_options(g2, 1, aopts, star).mirror_gain, 1e9)
    var g3: List[String] = ["trials=2", "trials=6"]
    assert_equal(route_options(g3, 1, aopts, star).trials, 6)
    print("  table defaults equal RouteOptions / AIRouteOptions / SynthOptions.default")


def test_every_pass_runs() raises:
    """Every pass id has a branch that runs it (`_apply` raises QM-E600 for one without)."""
    var t = Target.line(6, "any")  # the circuit conforms to it, so `check` passes alone
    t.set_edge_error(3, 4, 0.005)
    var c = chain()
    var want = evolve(c)
    for i in range(PassId.NUM):
        var d = pass_decl(i)
        var spec = d.name
        if d.list_index() >= 0 and d.args[d.list_index()].required:
            spec += "(cancel)"
        var r = compile_spec(c, t, spec)
        assert_equal(len(r.reports), 1, spec)
        if not d.relabels and i != PassId.BASIS:
            assert_true(want.fidelity(evolve(r.circuit)) > 1 - 1e-9, spec + " keeps the state")
        # and inside fixpoint
        _ = compile_spec(c, t, "fixpoint(" + spec + ",max=1)")
    print("  all", PassId.NUM, "passes run from their name")


def test_program_policy_spacing() raises:
    """A Program refuses the learned router however the argument is spaced
    (the check looked for the text "policy=" and missed "policy = auto")."""
    var body = Circuit(2)
    body.h(0)
    body.cx(0, 1)
    var p = Program(body^)
    var t = Target.line(3, "sc_cz")
    _ = compile_spec_program(p, t, "cancel,route(sabre,policy=off),basis")
    _ = compile_spec_program(p, t, "cancel,route(sabre,policy = off),basis")
    for spec in ["cancel,route(sabre,policy=auto),basis", "cancel,route(sabre,policy = auto),basis", "cancel,route(sabre,policy =auto),basis"]:
        var code = String()
        try:
            _ = compile_spec_program(p, t, spec)
        except e:
            code = String(String(e)[byte=0:7])
        assert_equal(code, "QM-E630", spec)
    print("  Program: learned router refused for every spelling of policy")


def main() raises:
    test_table_is_whole()
    test_block_kinds()
    test_every_argument()
    test_rejections()
    test_route_help_is_complete()
    test_defaults_are_the_option_defaults()
    test_every_pass_runs()
    test_post_layout_call_limit()
    test_program_policy_spacing()
    print("test_pass_table OK")
