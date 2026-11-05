"""Run-time configuration, read from the files the Python agents generate.

Single source of truth (never copied into Mojo):
* agents/prompts/<agent>.md   system prompts
* agents/prompts/specs.json   per-agent MCP tool subset ("all" or a list) and tool groups
* agents/prompts/tool_guide.json  the MCP tool guides
* agents/agents.json          models and OpenRouter settings
`agents/.venv/bin/python -m qumo_agents export-prompts` regenerates the first three.
"""

from std.os import getenv
from std.os.path import exists, dirname

from qumo.ai.json import JsonValue, parse_json

# The agents ported to Mojo. Every other agent needs tools that exist only in Python
# (repository reading, research, sandboxes, delegation).
def is_ported(name: String) -> Bool:
    return name == "qumo-operator" or name == "execution"


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def find_root(argv0: String) raises -> String:
    """The QUMO repository root: $QUMO_ROOT, else the parent of the executable's
    directory (build/qumo-agent), else the working directory or one of its parents."""
    var marker = "/agents/prompts/specs.json"
    var env = getenv("QUMO_ROOT", "")
    if env.byte_length() > 0:
        if not exists(env + marker):
            raise Error("QUMO_ROOT=" + env + " has no agents/prompts/specs.json")
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
    raise Error("cannot find the QUMO repository (agents/prompts/specs.json): run from the repository or set QUMO_ROOT")


struct ToolGuide(Copyable):
    var names: List[String]
    var use_when: List[String]
    var instead_or_first: List[String]
    var watch_out: List[String]

    def __init__(out self):
        self.names = List[String]()
        self.use_when = List[String]()
        self.instead_or_first = List[String]()
        self.watch_out = List[String]()

    def load(mut self, text: String) raises:
        var doc = parse_json(text)
        if doc.get_string("schema", "") != "qumo.tool_guide.v1":
            raise Error("tool_guide.json: expected schema qumo.tool_guide.v1")
        var tools = doc.get("tools")
        for i in range(len(tools)):
            var g = tools.at(i)
            self.names.append(tools.key_at(i))
            self.use_when.append(g.get_string("use_when", ""))
            self.instead_or_first.append(g.get_string("instead_or_first", ""))
            self.watch_out.append(g.get_string("watch_out", ""))

    def guided_description(self, name: String, server_text: String) -> String:
        """Byte-identical to qumo_agents.tools.qumo_mcp.guided_description."""
        var k = -1
        for i in range(len(self.names)):
            if self.names[i] == name:
                k = i
        if k < 0:
            return server_text
        var out = "Use when " + self.use_when[k]
        if self.instead_or_first[k].byte_length() > 0:
            out += "\nInstead or first: " + self.instead_or_first[k]
        if self.watch_out[k].byte_length() > 0:
            out += "\nWatch out: " + self.watch_out[k]
        return out + "\n\n" + server_text


struct AgentConfig(Copyable):
    var name: String
    var description: String
    var prompt: String
    var mcp_all: Bool
    var mcp_tools: List[String]
    var model: String
    var base_url: String
    var api_key_env: String
    var header_names: List[String]
    var header_values: List[String]
    var timeout_s: Int
    var max_retries: Int
    var guide: ToolGuide
    var root: String
    var qumo_bin: String

    def __init__(out self, root: String, name: String, model_override: String) raises:
        if not is_ported(name):
            raise Error("agent '" + name + "' is Python-only (its tools exist only in agents/qumo_agents); run it with bin/qumo-agents run " + name + ". Mojo agents: qumo-operator, execution")
        self.root = root
        self.name = name
        var specs = parse_json(read_text(root + "/agents/prompts/specs.json"))
        if specs.get_string("schema", "") != "qumo.agent_specs.v1":
            raise Error("specs.json: expected schema qumo.agent_specs.v1")
        var spec = specs.get("agents").get(name)
        if not spec.is_object():
            raise Error("specs.json has no agent " + name + ": run `agents/.venv/bin/python -m qumo_agents export-prompts`")
        self.description = spec.get_string("description", "")
        var groups = spec.get("tool_groups")
        for i in range(len(groups)):
            var g = groups.at(i).as_string()
            if g != "report":
                raise Error("agent " + name + " needs tool group '" + g + "', which the Mojo agents do not have")
        self.mcp_tools = List[String]()
        var mt = spec.get("mcp_tools")
        self.mcp_all = mt.is_string() and mt.as_string() == "all"
        if mt.is_array():
            for i in range(len(mt)):
                self.mcp_tools.append(mt.at(i).as_string())
        self.prompt = read_text(root + "/agents/prompts/" + name + ".md")
        self.guide = ToolGuide()
        self.guide.load(read_text(root + "/agents/prompts/tool_guide.json"))

        var cfg_path = getenv("QUMO_AGENTS_CONFIG", root + "/agents/agents.json")
        var cfg = parse_json(read_text(cfg_path))
        if cfg.get_string("schema", "") != "qumo.agents.v1":
            raise Error(cfg_path + ": expected schema qumo.agents.v1")
        # --model, then QUMO_AGENTS_MODEL, then the agent's entry, then default_model.
        var model = model_override
        if model.byte_length() == 0:
            model = getenv("QUMO_AGENTS_MODEL", "")
        if model.byte_length() == 0:
            model = cfg.get("models").get_string(name, "")
        if model.byte_length() == 0:
            model = cfg.get_string("default_model", "")
        self.model = model
        var orc = cfg.get("openrouter")
        self.base_url = orc.get_string("base_url", "https://openrouter.ai/api/v1")
        self.api_key_env = orc.get_string("api_key_env", "OPENROUTER_API_KEY")
        self.timeout_s = orc.get_int("timeout", 300)
        self.max_retries = orc.get_int("max_retries", 2)
        self.header_names = List[String]()
        self.header_values = List[String]()
        var hs = orc.get("headers")
        for i in range(len(hs)):
            self.header_names.append(hs.key_at(i))
            self.header_values.append(hs.at(i).as_string())
        self.qumo_bin = getenv("QUMO_BIN", root + "/build/qumo")

    def wants_tool(self, tool: String) -> Bool:
        if self.mcp_all:
            return True
        for t in self.mcp_tools:
            if t == tool:
                return True
        return False


def describe_agents(root: String) raises -> String:
    """One line per agent: name, model, Mojo or Python-only, description."""
    var specs = parse_json(read_text(root + "/agents/prompts/specs.json")).get("agents")
    var cfg = parse_json(read_text(getenv("QUMO_AGENTS_CONFIG", root + "/agents/agents.json")))
    var out = String()
    for i in range(len(specs)):
        var name = specs.key_at(i)
        var model = cfg.get("models").get_string(name, cfg.get_string("default_model", ""))
        var where = String("mojo  ") if is_ported(name) else String("python")
        var pad = String()
        for _ in range(max(0, 20 - name.byte_length())):
            pad += " "
        var mpad = String()
        for _ in range(max(0, 30 - model.byte_length())):
            mpad += " "
        out += name + pad + " " + where + "  " + model + mpad + " " + specs.at(i).get_string("description", "") + "\n"
    return out^
