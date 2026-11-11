"""Front-end helpers shared by the CLI (`cli/app.mojo`) and its tests.

* `did_you_mean` - nearest name for a mistyped option, command or tool.
* `tool_help_text` - a tool's description and JSON input schema as readable text
  (`qumo tool <name> --help`): arguments, types, required, defaults, one example.
* `tool_example` - one argument object per registered tool that runs as written
  from the repository root (tests/test_cli_frontdoor.mojo runs every one).
* `with_file_as_qasm` - the CLI-only `"file": "path"` argument of tools that take
  `qasm`. The MCP server never reads files; only the CLI resolves `file`.

Nothing here is on a simulation or compilation path.
"""

from qumo.ai.json import JsonWriter, JsonValue, parse_json
from qumo.ai.tools import ToolSpec


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------
def edit_distance(a: String, b: String) -> Int:
    """Edit distance over bytes; swapping two neighbours counts as one edit
    (optimal string alignment), so 'rnu' is one edit from 'run'."""
    var x = a.as_bytes()
    var y = b.as_bytes()
    var m = len(y)
    var prev2 = List[Int](length=m + 1, fill=0)
    var prev = List[Int](length=m + 1, fill=0)
    var cur = List[Int](length=m + 1, fill=0)
    for j in range(m + 1):
        prev[j] = j
    for i in range(1, len(x) + 1):
        cur[0] = i
        for j in range(1, m + 1):
            var cost = 0 if x[i - 1] == y[j - 1] else 1
            cur[j] = min(min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost)
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        for j in range(m + 1):
            prev2[j] = prev[j]
            prev[j] = cur[j]
    return prev[m]


def did_you_mean(name: String, candidates: List[String]) -> String:
    """The candidate closest to `name`, or "" when none is close: a candidate
    that `name` is a prefix of (or the reverse) wins, then edit distance <= 2
    (<= 1 for names of three bytes or fewer). Ties go to the first candidate."""
    var best = String()
    var bd = 1000
    for c in candidates:
        var d = edit_distance(name, c)
        if name.byte_length() >= 2 and (c.startswith(name) or name.startswith(c)):
            d = min(d, 1)
        if d < bd:
            bd = d
            best = c
    var limit = 1 if name.byte_length() <= 3 else 2
    if bd <= limit:
        return best^
    return String()


# ---------------------------------------------------------------------------
# Text layout
# ---------------------------------------------------------------------------
def pad_right(s: String, w: Int) -> String:
    var out = s
    for _ in range(w - s.byte_length()):
        out += " "
    return out^


def wrap_text(text: String, width: Int, indent: Int) -> String:
    """Greedy word wrap; every line (the first included) starts with `indent` spaces."""
    var pad = String()
    for _ in range(indent):
        pad += " "
    var out = String()
    var line = String()
    for word in text.split(" "):
        var wd = String(word)
        if wd.byte_length() == 0:
            continue
        if line.byte_length() > 0 and line.byte_length() + 1 + wd.byte_length() > width - indent:
            out += pad + line + "\n"
            line = String()
        if line.byte_length() > 0:
            line += " "
        line += wd
    out += pad + line
    return out^


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
def tool_takes(spec: ToolSpec, arg: String) -> Bool:
    """True when the tool's input schema has a property called `arg`."""
    try:
        return parse_json(spec.input_schema).get("properties").has(arg)
    except:
        return False


def _type_text(p: JsonValue) raises -> String:
    if p.has("type"):
        var t = p.get("type").as_string()
        if t == "array" and p.get("items").has("type"):
            return "array of " + p.get("items").get("type").as_string()
        return t^
    if p.has("oneOf"):
        var alts = p.get("oneOf")
        var out = String()
        for i in range(len(alts)):
            if i > 0:
                out += " | "
            out += alts.at(i).get_string("type", "any")
        return out^
    return "any"


def _scalar_text(v: JsonValue) raises -> String:
    if v.is_string():
        return "'" + v.as_string() + "'"
    return v.to_json()


def _is_required(schema: JsonValue, name: String) raises -> Bool:
    var req = schema.get("required")
    if req.is_defined() and req.is_array():
        for i in range(len(req)):
            if req.at(i).as_string() == name:
                return True
    return False


def _arg_details(schema: JsonValue, name: String, p: JsonValue) raises -> String:
    var parts = List[String]()
    if _is_required(schema, name):
        parts.append("required")
    if p.has("default"):
        parts.append("default " + _scalar_text(p.get("default")))
    if p.has("enum"):
        var e = p.get("enum")
        var vals = List[String]()
        for i in range(len(e)):
            vals.append(_scalar_text(e.at(i)))
        parts.append("one of " + String(", ").join(vals))
    if p.has("minimum") and p.has("maximum"):
        parts.append("range " + p.get("minimum").to_json() + " to " + p.get("maximum").to_json())
    elif p.has("minimum"):
        parts.append("minimum " + p.get("minimum").to_json())
    elif p.has("maximum"):
        parts.append("maximum " + p.get("maximum").to_json())
    var s = String("; ").join(parts)
    var d = p.get_string("description", "")
    if d.byte_length() > 0:
        if s.byte_length() > 0:
            s += ". "
        s += d
    return s^


comptime _FILE_HELP = "CLI only: path of an OpenQASM 2/3 file read into 'qasm' ('-' = stdin)"


def tool_help_text(spec: ToolSpec) -> String:
    """Readable help for one tool: title, description, one line per argument
    (name, type, required / default / allowed values, description) and an
    example command. Never raises: a schema it cannot read is printed raw."""
    var out = spec.name + " - " + spec.title + "\n\n" + wrap_text(spec.description, 92, 0) + "\n"
    try:
        var schema = parse_json(spec.input_schema)
        var props = schema.get("properties")
        var names = props.keys() if props.is_defined() and props.is_object() else List[String]()
        var takes_qasm = False
        var wn = 4
        var wt = 4
        for i in range(len(names)):
            wn = max(wn, names[i].byte_length())
            wt = max(wt, _type_text(props.at(i)).byte_length())
            if names[i] == "qasm":
                takes_qasm = True
        if len(names) == 0:
            out += "\narguments: none\n"
        else:
            out += "\narguments (one JSON object):\n"
            var indent = 2 + wn + 2 + wt + 2
            for i in range(len(names)):
                var p = props.at(i)
                var head = "  " + pad_right(names[i], wn + 2) + pad_right(_type_text(p), wt + 2)
                var det = wrap_text(_arg_details(schema, names[i], p), 100, indent)
                out += head + String(det[byte=min(indent, det.byte_length()):]) + "\n"
                if names[i] == "qasm":
                    var fdet = wrap_text(_FILE_HELP, 100, indent)
                    out += "  " + pad_right("file", wn + 2) + pad_right("string", wt + 2) + String(fdet[byte=indent:]) + "\n"
        if takes_qasm and not _is_required(schema, "qasm"):
            out += "\nGive the circuit as exactly one of: qasm, file" + (", circuit" if props.has("circuit") else "") + (", program" if props.has("program") else "") + ".\n"
    except:
        out += "\ninput schema (JSON Schema):\n  " + spec.input_schema + "\n"
    var ex = tool_example(spec.name)
    if ex.byte_length() > 0:
        out += "\nexample:\n  qumo tool " + spec.name + " '" + ex + "'\n"
    out += "\nOutput is one JSON object on stdout (add --pretty to indent it). Arguments may also come from stdin: qumo tool " + spec.name + " -"
    return out^


comptime _BELL = "examples/foundations/bell.qasm"


def tool_example(name: String) -> String:
    """One argument object per tool that runs as written from the repository
    root; "" for a tool this table does not know."""
    if name == "qumo_run_qasm":
        return "{\"file\": \"" + _BELL + "\", \"shots\": 1000, \"seed\": 7}"
    if name == "qumo_statevector":
        return "{\"file\": \"" + _BELL + "\"}"
    if name == "qumo_expectation":
        return "{\"file\": \"" + _BELL + "\", \"observable\": [\"Z0 Z1\", \"X0 X1\"]}"
    if name == "qumo_validate_qasm":
        return "{\"file\": \"" + _BELL + "\"}"
    if name == "qumo_describe":
        return "{\"topic\": \"gate\", \"name\": \"cx\"}"
    if name == "qumo_circuit_info":
        return "{\"file\": \"" + _BELL + "\"}"
    if name == "qumo_draw":
        return "{\"file\": \"" + _BELL + "\", \"style\": \"unicode\"}"
    if name == "qumo_convert":
        return "{\"file\": \"" + _BELL + "\", \"to\": \"qasm2\"}"
    if name == "qumo_transpile":
        return "{\"file\": \"" + _BELL + "\", \"target\": \"grid:2x2/sc_cz\", \"level\": 2}"
    if name == "qumo_estimate_resources":
        return "{\"file\": \"" + _BELL + "\", \"profile\": \"trapped_ion\"}"
    if name == "qumo_vqe":
        return "{\"hamiltonian\": \"-1.05*II + 0.39*ZI - 0.39*IZ - 0.01*ZZ + 0.18*XX\", \"layers\": 1}"
    if name == "qumo_qaoa_maxcut":
        return "{\"graph\": \"0-1 1-2 2-3 3-0\", \"p\": 1}"
    if name == "qumo_qec_memory":
        return "{\"code\": \"repetition\", \"distance\": 3, \"p\": 0.01, \"shots\": 10000}"
    if name == "qumo_emulate":
        return "{\"file\": \"" + _BELL + "\", \"target\": \"grid:2x2/sc_cz\", \"compile\": true, \"shots\": 1000}"
    if name == "qumo_density":
        return "{\"file\": \"" + _BELL + "\", \"observable\": \"Z0 Z1\"}"
    if name == "qumo_to_pbc":
        return "{\"qasm\": \"OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[2] q; h q[0]; t q[0]; cx q[0], q[1];\"}"
    if name == "qumo_stim_import":
        return "{\"stim\": \"H 0\\nCX 0 1\\nM 0 1\\nDETECTOR rec[-1] rec[-2]\"}"
    if name == "qumo_qec_dem":
        return "{\"code\": \"repetition\", \"distance\": 3, \"p\": 0.01}"
    if name == "qumo_gpu_info":
        return "{}"
    if name == "qumo_run_device":
        return "{\"file\": \"" + _BELL + "\", \"device\": \"cpu\", \"shots\": 1000}"
    if name == "qumo_build_circuit":
        return "{\"kind\": \"grover\", \"args\": {\"marked\": [\"101\"]}, \"measure\": true}"
    if name == "qumo_ft_codes":
        return "{\"code\": \"steane\"}"
    if name == "qumo_ft_encode":
        return "{\"file\": \"" + _BELL + "\", \"architecture\": \"steane\"}"
    if name == "qumo_ft_verify":
        return "{\"code\": \"steane\", \"kind\": \"clifford\", \"semantic_qasm\": \"OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[1] q; x q[0];\", \"impl_qasm\": \"OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[7] q; x q[0]; x q[1]; x q[2]; x q[3]; x q[4]; x q[5]; x q[6];\"}"
    return String()


def with_file_as_qasm(args: JsonValue, source: String) -> String:
    """`args` (a JSON object) with its `file` member replaced by `"qasm": source`.
    The caller has read `source` from the path `file` names."""
    var w = JsonWriter()
    w.begin_object()
    for i in range(len(args)):
        var k = args.key_at(i)
        if k != "file":
            w.kv_raw(k, args.at(i).to_json())
    w.kv_string("qasm", source)
    w.end_object()
    return w.finish()
