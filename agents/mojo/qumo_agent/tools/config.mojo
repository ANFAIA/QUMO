"""Run-time configuration, read from the files the Python agents export (DESIGN.md 2.8.1).

Single source of truth for the nine Python agents (never copied into Mojo):
* agents/prompts/specs.json    qumo.agent_specs.v1: tool groups, MCP tools, subagents, generated tools
* agents/prompts/<agent>.md    system prompts
* agents/prompts/preamble.md   the shared preamble (Mojo-only agents start with it)
* agents/prompts/modules.json  qumo.modules.v1: the QUMO modules table
* agents/prompts/tools.json    qumo.local_tools.v1: local tool descriptions, schemas and limits
* agents/agents.json           models and OpenRouter settings ($QUMO_AGENTS_CONFIG overrides it)
`agents/.venv/bin/python -m qumo_agents export-prompts` regenerates the agents/prompts files.

Mojo-only agents (the toolsmith) live in agents/mojo/specs/*.json, with their role prompt in
agents/mojo/prompts/<name>.role.md.
"""

from std.os import getenv, listdir
from std.os.path import exists, dirname, isdir

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.core.endpoint import ModelEndpoint
from qumo_agent.core.errors import E_CONFIG, qa_error
from qumo_agent.core.toolspec import ToolSpec, TOOL_LOCAL
from qumo_agent.harness.env import AgentDef
from qumo_agent.util.pyfmt import py_str_repr

comptime SPECS_SCHEMA = "qumo.agent_specs.v1"
comptime MODULES_SCHEMA = "qumo.modules.v1"
comptime TOOLS_SCHEMA = "qumo.local_tools.v1"
comptime AGENTS_SCHEMA = "qumo.agents.v1"
comptime EXPORT_HINT = "run agents/.venv/bin/python -m qumo_agents export-prompts"


def _stale(rel_path: String) -> Error:
    """QA-E140 for a file `export-prompts` writes that is missing or out of date."""
    return qa_error(E_CONFIG, rel_path + " is stale", EXPORT_HINT)


def _read_text(path: String) raises -> String:
    if not exists(path):
        raise qa_error(E_CONFIG, path + " not found", "")
    with open(path, "r") as f:
        return f.read()


def _read_export(root: String, rel_path: String) raises -> String:
    """A file written by export-prompts; a missing one means the export is stale."""
    var path = root + "/" + rel_path
    if not exists(path):
        raise _stale(rel_path)
    with open(path, "r") as f:
        return f.read()


def _string_list(v: JsonValue) raises -> List[String]:
    if not v.is_array():
        raise Error("expected a list of strings, got " + v.kind_name())
    var out = List[String]()
    for i in range(len(v)):
        out.append(v.at(i).as_string())
    return out^


def _contains(items: List[String], x: String) -> Bool:
    for s in items:
        if s == x:
            return True
    return False


def _join(items: List[String], sep: String) -> String:
    var out = String()
    for i in range(len(items)):
        if i > 0:
            out += sep
        out += items[i]
    return out^


@fieldwise_init
struct AgentSpecData(Copyable):
    var name: String
    var description: String
    var tool_groups: List[String]     # research, read, check, report, sandbox, toolsmith
    var mcp_all: Bool
    var mcp_tools: List[String]
    var subagents: List[String]
    var writes_code: Bool
    var generated_all: Bool
    var generated_tools: List[String]
    var prompt: String
    var model_hint: String            # Mojo-only specs may name a model
    var mojo_only: Bool


def _all_or_list(v: JsonValue) raises -> Tuple[Bool, List[String]]:
    """`"all"` or a list of names (the specs.json form of `None` / a tuple)."""
    if v.is_string():
        if v.as_string() != "all":
            raise Error("expected \"all\" or a list, got " + py_str_repr(v.as_string()))
        return (True, List[String]())
    return (False, _string_list(v))


def _spec_from_json(name: String, v: JsonValue, prompt: String, mojo_only: Bool) raises -> AgentSpecData:
    """One agent entry of a qumo.agent_specs.v1 file. Raises a plain message (the caller
    adds the file and the code)."""
    if not v.is_object():
        raise Error("agent " + py_str_repr(name) + " is not an object")
    var fields: List[String] = ["description", "tool_groups", "mcp_tools", "subagents", "writes_code", "generated_tools"]
    for f in fields:
        if not v.has(f):
            raise Error("agent " + py_str_repr(name) + " has no field " + py_str_repr(f))
    var mcp = _all_or_list(v.get("mcp_tools"))
    var gen = _all_or_list(v.get("generated_tools"))
    var model = String()
    if v.has("model") and not v.get("model").is_null():
        model = v.get("model").as_string()
    return AgentSpecData(
        name,
        v.get("description").as_string(),
        _string_list(v.get("tool_groups")),
        mcp[0],
        mcp[1].copy(),
        _string_list(v.get("subagents")),
        v.get("writes_code").as_bool(),
        gen[0],
        gen[1].copy(),
        prompt,
        model,
        mojo_only,
    )


struct Registry(Copyable):
    """The agents: the nine of specs.json (Python is their source) followed by the Mojo-only
    agents of agents/mojo/specs/, plus the models and endpoint of agents.json."""

    var root: String
    var specs: List[AgentSpecData]
    var config: JsonValue             # agents.json
    var config_path: String

    def __init__(out self, root: String) raises:
        self.root = root
        self.specs = List[AgentSpecData]()
        var doc: JsonValue
        try:
            doc = parse_json(_read_export(root, "agents/prompts/specs.json"))
        except e:
            if String(e).startswith("QA-E140"):
                raise Error(String(e))
            raise _stale("agents/prompts/specs.json")
        if doc.get_string("schema", "") != SPECS_SCHEMA:
            raise _stale("agents/prompts/specs.json")
        var agents = doc.get("agents")
        if not agents.is_object():
            raise _stale("agents/prompts/specs.json")
        for i in range(len(agents)):
            var name = agents.key_at(i)
            var prompt = _read_export(root, "agents/prompts/" + name + ".md")
            try:
                self.specs.append(_spec_from_json(name, agents.at(i), prompt, False))
            except:
                raise _stale("agents/prompts/specs.json")
        self.config_path = getenv("QUMO_AGENTS_CONFIG", root + "/agents/agents.json")
        self.config = parse_json(_read_text(self.config_path))
        if self.config.get_string("schema", "") != AGENTS_SCHEMA:
            raise qa_error(E_CONFIG, self.config_path + ": expected schema " + AGENTS_SCHEMA, "")
        self._load_mojo_only()

    def _load_mojo_only(mut self) raises:
        var dir = self.root + "/agents/mojo/specs"
        if not isdir(dir):
            return
        var files = List[String]()
        for f in listdir(dir):
            if f.endswith(".json"):
                files.append(f)
        sort(files)
        var preamble = String()
        if len(files) > 0:
            preamble = _read_export(self.root, "agents/prompts/preamble.md")
        for f in files:
            var rel = "agents/mojo/specs/" + f
            var doc = parse_json(_read_text(self.root + "/" + rel))
            if doc.get_string("schema", "") != SPECS_SCHEMA:
                raise qa_error(E_CONFIG, rel + ": expected schema " + SPECS_SCHEMA, "")
            var agents = doc.get("agents")
            for i in range(len(agents)):
                var name = agents.key_at(i)
                if self._index(name) >= 0:
                    raise qa_error(E_CONFIG, rel + ": agent " + py_str_repr(name) + " is defined twice", "")
                var role = _read_text(self.root + "/agents/mojo/prompts/" + name + ".role.md")
                try:
                    self.specs.append(_spec_from_json(name, agents.at(i), preamble + role, True))
                except e:
                    raise qa_error(E_CONFIG, rel + ": " + String(e), "")

    def _index(self, name: String) -> Int:
        for i in range(len(self.specs)):
            if self.specs[i].name == name:
                return i
        return -1

    def names(self) -> List[String]:
        """specs.json order, then the Mojo-only agents."""
        var out = List[String]()
        for s in self.specs:
            out.append(s.name)
        return out^

    def get(self, name: String) raises -> AgentSpecData:
        var i = self._index(name)
        if i < 0:
            raise qa_error(E_CONFIG, "unknown agent " + py_str_repr(name) + "; choose one of " + _join(self.names(), ", "), "")
        return self.specs[i].copy()

    def model_for(self, name: String, override: String) -> String:
        """--model, then $QUMO_AGENTS_MODEL, then agents.json models[name], then the spec's
        model hint, then default_model."""
        if override.byte_length() > 0:
            return override
        var env = getenv("QUMO_AGENTS_MODEL", "")
        if env.byte_length() > 0:
            return env
        try:
            var m = self.config.get("models").get_string(name, "")
            if m.byte_length() > 0:
                return m
        except:
            pass
        var i = self._index(name)
        if i >= 0 and self.specs[i].model_hint.byte_length() > 0:
            return self.specs[i].model_hint
        try:
            return self.config.get_string("default_model", "")
        except:
            return String()

    def endpoint(self) raises -> ModelEndpoint:
        """agents.json "openrouter" with Python's defaults (config.openrouter_chat_model)."""
        var orc = self.config.get("openrouter")
        var names = List[String]()
        var values = List[String]()
        var hs = orc.get("headers")
        for i in range(len(hs)):
            names.append(hs.key_at(i))
            values.append(hs.at(i).as_string())
        return ModelEndpoint(
            orc.get_string("base_url", "https://openrouter.ai/api/v1"),
            orc.get_string("api_key_env", "OPENROUTER_API_KEY"),
            names^,
            values^,
            orc.get_int("timeout", 300),
            orc.get_int("max_retries", 2),
        )

    def agent_def(self, name: String, ancestors: List[String], model_override: String, recursion_limit: Int) raises -> AgentDef:
        """The harness's definition of `name`: subagents without itself and every ancestor
        on the delegation path (graph.py build_agent)."""
        var spec = self.get(name)
        var subs = List[String]()
        for s in spec.subagents:
            if s != name and not _contains(ancestors, s):
                subs.append(s)
        return AgentDef(name, spec.description, spec.prompt, self.model_for(name, model_override), subs^, recursion_limit)


@fieldwise_init
struct ModuleInfo(Copyable):
    var name: String
    var summary: String
    var paths: List[String]
    var docs: List[String]
    var tests: List[String]
    var sdks: List[String]
    var arxiv_categories: List[String]
    var arxiv_queries: List[String]
    var mcp_tools: List[String]


struct Modules(Copyable):
    """agents/prompts/modules.json (qumo_agents.modules.MODULES)."""

    var modules: List[ModuleInfo]

    def __init__(out self, root: String) raises:
        self.modules = List[ModuleInfo]()
        var rel = String("agents/prompts/modules.json")
        try:
            var doc = parse_json(_read_export(root, rel))
            if doc.get_string("schema", "") != MODULES_SCHEMA:
                raise Error("schema")
            var ms = doc.get("modules")
            if not ms.is_object():
                raise Error("modules")
            for i in range(len(ms)):
                var m = ms.at(i)
                self.modules.append(ModuleInfo(
                    ms.key_at(i),
                    m.get("summary").as_string(),
                    _string_list(m.get("paths")),
                    _string_list(m.get("docs")),
                    _string_list(m.get("tests")),
                    _string_list(m.get("sdks")),
                    _string_list(m.get("arxiv_categories")),
                    _string_list(m.get("arxiv_queries")),
                    _string_list(m.get("mcp_tools")),
                ))
        except e:
            if String(e).startswith("QA-E140"):
                raise Error(String(e))
            raise _stale(rel)

    def names(self) -> List[String]:
        var out = List[String]()
        for m in self.modules:
            out.append(m.name)
        return out^

    def get(self, name: String) raises -> ModuleInfo:
        """Unknown: Python's ValueError text, `unknown module 'x'; choose one of ir, sim, ...`
        (no code: tools put it in their result)."""
        for m in self.modules:
            if m.name == name:
                return m.copy()
        raise Error("unknown module " + py_str_repr(name) + "; choose one of " + _join(self.names(), ", "))

    def task_prefix(self, name: String) raises -> String:
        """cli.py _task: `Module: <name> (<summary>) Code: <paths>.` and a blank line."""
        var m = self.get(name)
        return "Module: " + m.name + " (" + m.summary + ") Code: " + _join(m.paths, ", ") + ".\n\n"


struct LocalToolsCatalog(Copyable):
    """agents/prompts/tools.json: the QUMO local tools as LangChain sends them, the tool
    groups and the limits of the repository, research and alphaXiv tools."""

    var doc: JsonValue

    def __init__(out self, root: String) raises:
        var rel = String("agents/prompts/tools.json")
        self.doc = parse_json(_read_export(root, rel))
        if self.doc.get_string("schema", "") != TOOLS_SCHEMA:
            raise _stale(rel)
        var keys: List[String] = ["tools", "groups", "repo", "research", "alphaxiv"]
        for key in keys:
            if not self.doc.get(key).is_object():
                raise _stale(rel)

    def spec(self, name: String) raises -> ToolSpec:
        """TOOL_LOCAL with the exact wire texts; count_json is the form the token counter
        measures (DESIGN.md 2.7.6)."""
        var t = self.doc.get("tools").get(name)
        if not t.is_object() or not t.get("description").is_string() or not t.get("parameters").is_object() or not t.get("count_parameters").is_object():
            raise _stale("agents/prompts/tools.json")
        var description = t.get("description").as_string()
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("name", name)
        w.kv_string("description", description)
        w.kv_raw("parameters", t.get("count_parameters").to_json())
        w.end_object()
        return ToolSpec(name, description, t.get("parameters").to_json(), TOOL_LOCAL, True, w.finish())

    def group(self, group: String) raises -> List[String]:
        var g = self.doc.get("groups").get(group)
        if not g.is_array():
            raise qa_error(E_CONFIG, "unknown tool group " + py_str_repr(group) + " in agents/prompts/tools.json", EXPORT_HINT)
        return _string_list(g)

    def run_check_allowed(self) -> List[List[String]]:
        """The argv prefixes of run_check's allow-list, in order."""
        var out = List[List[String]]()
        var a = self.doc.get("repo").get("run_check_allowed")
        for i in range(len(a)):
            try:
                out.append(_string_list(a.at(i).at(1)))
            except:
                pass
        return out^

    def skip_dirs(self) -> List[String]:
        try:
            return _string_list(self.doc.get("repo").get("skip_dirs"))
        except:
            return List[String]()

    def max_read(self) -> Int:
        try:
            return self.doc.get("repo").get_int("max_read", 60000)
        except:
            return 60000

    def max_output(self) -> Int:
        try:
            return self.doc.get("repo").get_int("max_output", 12000)
        except:
            return 12000

    def research_user_agent(self) -> String:
        try:
            return self.doc.get("research").get_string("user_agent", "qumo-agents/0.1 (+https://github.com/ANFAIA/QUMO)")
        except:
            return "qumo-agents/0.1 (+https://github.com/ANFAIA/QUMO)"

    def research_max_bytes(self) -> Int:
        try:
            return self.doc.get("research").get_int("max_bytes", 400000)
        except:
            return 400000

    def research_timeout_s(self) -> Int:
        try:
            return self.doc.get("research").get_int("timeout", 30)
        except:
            return 30

    def alphaxiv_json(self) -> String:
        """The "alphaxiv" object: url_env, url_default, key_env, read_tools, guide, description_suffix."""
        return self.doc.get("alphaxiv").to_json()


def find_root(argv0: String) raises -> String:
    """The QUMO repository root: $QUMO_ROOT, else the parent of the executable's directory
    (build/qumo-agent), else the working directory or one of its parents."""
    var marker = "/agents/prompts/specs.json"
    var env = getenv("QUMO_ROOT", "")
    if env.byte_length() > 0:
        if not exists(env + marker):
            raise qa_error(E_CONFIG, "QUMO_ROOT=" + env + " has no agents/prompts/specs.json", "")
        return env
    var candidates = List[String]()
    var d = dirname(argv0)
    if d.byte_length() > 0:
        candidates.append(dirname(d) if dirname(d).byte_length() > 0 else String("."))
    candidates.append(".")
    candidates.append("..")
    candidates.append("../..")
    for c in candidates:
        if exists(c + marker):
            return c
    raise qa_error(E_CONFIG, "cannot find the QUMO repository (agents/prompts/specs.json)", "run from the repository or set QUMO_ROOT")
