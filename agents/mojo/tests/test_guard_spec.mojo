"""Every vector of agents/spec/guard_rules.json through the real functions of the Mojo harness.
agents/tests/test_guard_spec.py runs the same vectors through the Python harness: the two
tests reading one file is what keeps the two sets of rules from drifting (agents/spec/README.md).

Also here: core/guard_rules.mojo (generated from the file) is compared with the file value
by value, so a stale generated module fails without Python; and the lists the harness reads
from agents/prompts/tools.json at start-up are the ones of the file.

Pure: no name is resolved and no socket is opened. A vector lists the addresses its host
names resolve to; they are judged by the functions `tcp_connect(public_only=True)` uses.
The hop limit of a redirect chain is enforced inside `HttpClient` (tests/test_net_redirects.mojo
drives it with a fixture server); here each hop passes the real URL rules and the count is
compared with the same constant.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, assert_false

from qumo.core.json import JsonValue, parse_json
from qumo_agent.core.endpoint import ModelEndpoint
from qumo_agent.core.guard_rules import (
    ARG_OTHER_CHARS, GIT_FLAGS_TEXT, GUARD_RULES_SCHEMA, MAX_ANSWER_CHARS, MAX_REDIRECTS, QUMO_SUBCOMMANDS_TEXT,
    RESEARCH_PORT, RESEARCH_SCHEME, UNTRUSTED_MARKER, api_hosts, arg_ranges, check_env_extra_substrings, exec_groups,
    git_flags, git_injected_exact, git_injected_numbered, minimal_child_env_names, qumo_options, qumo_output_options,
    qumo_subcommands, run_check_allowed, secret_env_substrings, secret_name_contains, secret_name_prefixes,
    secret_name_suffixes, skip_dirs, web_groups,
)
from qumo_agent.llm.client import LlmClient
from qumo_agent.native.fs import make_dirs, make_temp_dir, realpath_strict, remove_tree, symlink, write_text
from qumo_agent.native.spawn import check_env_keeps, is_secret_env_name
from qumo_agent.net.guard import UrlGuard, literal_is_global
from qumo_agent.net.url import Url, parse_url, resolve_reference
from qumo_agent.harness.untrusted import untrusted_wrap
from qumo_agent.tools.config import AgentSpecData, LocalToolsCatalog, Registry, check_web_exec
from qumo_agent.tools.repo import resolve_repo_path, run_check_refusal

comptime SPEC_PATH = "agents/spec/guard_rules.json"
comptime HARNESS = "mojo"
comptime KEY_ENV = "QUMO_GUARD_SPEC_TEST_KEY"


def strings(v: JsonValue) raises -> List[String]:
    var out = List[String]()
    for i in range(len(v)):
        out.append(v.at(i).as_string())
    return out^


def same(got: List[String], want: JsonValue, what: String) raises:
    var w = strings(want)
    assert_equal(len(got), len(w), what + " is stale: run python3 agents/spec/gen_mojo_rules.py")
    for i in range(len(w)):
        assert_equal(got[i], w[i], what + " is stale: run python3 agents/spec/gen_mojo_rules.py")


def sorted_text(v: JsonValue) raises -> String:
    var items = strings(v)
    sort(items)
    return String(" ").join(items)


def reason_of(spec: JsonValue, rule: String, text: String) raises -> String:
    """The reason class of a refusal text: the first class of the rule with a marker in it."""
    var classes = spec.get("reasons").get(rule)
    for i in range(len(classes)):
        var markers = classes.at(i)
        for k in range(len(markers)):
            if text.find(markers.at(k).as_string()) >= 0:
                return classes.key_at(i)
    return "unclassified: " + text


# --------------------------------------------------------------------------- the rules
def run_path(inp: JsonValue, root: String) raises -> String:
    try:
        _ = resolve_repo_path(inp.get("path").as_string(), root, skip_dirs())
    except e:
        return String(e)
    return String()


def run_command(inp: JsonValue, root: String) raises -> String:
    var sandbox = String()
    if inp.get("sandbox_open").as_bool():
        sandbox = root
    return run_check_refusal(strings(inp.get("argv")), run_check_allowed(), root, sandbox, skip_dirs())


def _addresses_refusal(inp: JsonValue, host: String) raises -> String:
    """What `tcp_connect(host, 443, public_only=True)` decides for the addresses the vector
    gives `host`: refused when any of them is not global."""
    if literal_is_global(host) >= 0:
        return String()   # an IP literal: UrlGuard.check has judged it
    var table = inp.get("resolve")
    if not table.is_object() or not table.has(host):
        raise Error("the vector must list the addresses of " + host + " in `resolve`")
    var addrs = strings(table.get(host))
    for a in addrs:
        if literal_is_global(a) != 1:
            return host + " is not a public address"
    return String()


def run_url(inp: JsonValue) raises -> String:
    var hosts = List[String]()
    if inp.get("hosts").as_string() == "api":
        hosts = api_hosts()
    var guard = UrlGuard(True, hosts^, String())
    var hops = strings(inp.get("hops"))
    var current: Url
    try:
        current = parse_url(hops[0])
        _ = guard.check(current)
    except e:
        return String(e)
    var why = _addresses_refusal(inp, current.host)
    if why.byte_length() > 0:
        return why
    for i in range(1, len(hops)):
        if i > MAX_REDIRECTS:
            return "refused URL: too many redirects"
        try:
            var target = resolve_reference(current, hops[i])
            _ = guard.check(target)
            current = target^
        except e:
            return String(e)
        why = _addresses_refusal(inp, current.host)
        if why.byte_length() > 0:
            return why
    return String()


def run_model_key(inp: JsonValue) raises -> String:
    try:
        _ = LlmClient(ModelEndpoint(inp.get("base_url").as_string(), KEY_ENV, List[String](), List[String](), 5, 0), None)
    except e:
        return String(e)
    return String()


def run_delegation(inp: JsonValue) raises -> String:
    var agents = inp.get("agents")
    var specs = List[AgentSpecData]()
    for i in range(len(agents)):
        var a = agents.at(i)
        specs.append(AgentSpecData(
            agents.key_at(i), String(), strings(a.get("tool_groups")), False, List[String](),
            strings(a.get("subagents")), False, False, List[String](), String(), String(), False,
            strings(a.get("untrusted_subagents"))))
    try:
        check_web_exec(specs)
    except e:
        return String(e)
    return String()


def run_delegation_shipped() raises -> String:
    """The agents this harness really loads: agents/prompts/specs.json and agents/mojo/specs/."""
    try:
        _ = Registry(".")
    except e:
        return String(e)
    return String()


def run_untrusted_wrap(vec: JsonValue) raises -> String:
    """"" when the wrapped text is the vector's `output` (its random token written TOKEN)."""
    var inp = vec.get("input")
    var got = untrusted_wrap(inp.get("text").as_string(), inp.get("source").as_string(), inp.get_int("limit", 0), "TOKEN")
    if got == vec.get("output").as_string():
        return String()
    return "output differs: " + got


def run_vector(spec: JsonValue, vec: JsonValue, root: String) raises -> String:
    """"" when the harness allows the input, else the reason class of its refusal."""
    var rule = vec.get("rule").as_string()
    var inp = vec.get("input")
    var text: String
    if rule == "path":
        text = run_path(inp, root)
    elif rule == "command":
        text = run_command(inp, root)
    elif rule == "url":
        text = run_url(inp)
    elif rule == "model_key":
        text = run_model_key(inp)
    elif rule == "delegation":
        text = run_delegation(inp)
    elif rule == "delegation_shipped":
        return run_delegation_shipped()
    elif rule == "untrusted_wrap":
        return run_untrusted_wrap(vec)
    elif rule == "check_env":
        return String() if check_env_keeps(inp.get("name").as_string()) else "dropped"
    elif rule == "secret_env_name":
        return "secret" if is_secret_env_name(inp.get("name").as_string()) else String()
    else:
        raise Error("no runner for rule " + rule + " (vector " + vec.get("id").as_string() + ")")
    if text.byte_length() == 0:
        return text
    return reason_of(spec, rule, text)


def _for_this_harness(vec: JsonValue) raises -> Bool:
    if not vec.has("harnesses"):
        return True
    for h in strings(vec.get("harnesses")):
        if h == HARNESS:
            return True
    return False


def test_vectors(spec: JsonValue) raises:
    # the tree of `path_fixture`
    var base = realpath_strict(make_temp_dir("qumo-guard-spec-"))
    var fx = spec.get("path_fixture")
    for rel in strings(fx.get("files")):
        var p = base + "/" + rel
        make_dirs(String(p[byte=0:p.rfind("/")]))
        write_text(p, "x\n")
    var links = fx.get("symlinks")
    for i in range(len(links)):
        symlink(links.at(i).at(1).as_string(), base + "/" + links.at(i).at(0).as_string())
    var root = base + "/root"
    _ = setenv(KEY_ENV, "not-a-real-key-0000")

    var vectors = spec.get("vectors")
    var ran = 0
    var failures = List[String]()
    for i in range(len(vectors)):
        var vec = vectors.at(i)
        var id = vec.get("id").as_string()
        if not _for_this_harness(vec):
            continue
        var expect = vec.get("expect").as_string()
        var reason = vec.get_string("reason", "")
        if vec.has("by_harness") and vec.get("by_harness").has(HARNESS):
            var over = vec.get("by_harness").get(HARNESS)
            if over.has("expect"):
                expect = over.get("expect").as_string()
                reason = String()
            reason = over.get_string("reason", reason)
        var got: String
        try:
            got = run_vector(spec, vec, root)
        except e:
            got = "error: " + String(e)
        ran += 1
        if expect == "allowed" or expect == "wrapped":
            if got.byte_length() > 0:
                failures.append(id + ": expected allowed, refused with " + got)
        elif got.byte_length() == 0:
            failures.append(id + ": expected a refusal, was allowed")
        elif reason.byte_length() > 0 and got != reason:
            failures.append(id + ": expected " + reason + ", got " + got)
    try:
        remove_tree(base)
    except:
        pass
    for f in failures:
        print("FAIL " + f)
    assert_equal(len(failures), 0, "vectors of " + SPEC_PATH + " that fail")
    assert_true(ran > 150, "the vectors ran")


def test_generated_module_is_current(spec: JsonValue) raises:
    var lists = spec.get("lists")
    var repo = lists.get("repo")
    var env = lists.get("env")
    var research = lists.get("research")
    assert_equal(spec.get("schema").as_string(), GUARD_RULES_SCHEMA)
    same(skip_dirs(), repo.get("skip_dirs"), "skip_dirs")
    same(secret_name_prefixes(), repo.get("secret_names").get("prefix"), "secret_name_prefixes")
    same(secret_name_contains(), repo.get("secret_names").get("contains"), "secret_name_contains")
    same(secret_name_suffixes(), repo.get("secret_names").get("suffix"), "secret_name_suffixes")
    same(arg_ranges(), repo.get("arg_chars").get("ranges"), "arg_ranges")
    assert_equal(ARG_OTHER_CHARS, repo.get("arg_chars").get("others").as_string())
    same(git_flags(), repo.get("git_flags"), "git_flags")
    same(qumo_subcommands(), repo.get("qumo_subcommands"), "qumo_subcommands")
    same(qumo_output_options(), repo.get("qumo_output_options"), "qumo_output_options")
    same(qumo_options(), repo.get("qumo_options"), "qumo_options")
    assert_equal(GIT_FLAGS_TEXT, sorted_text(repo.get("git_flags")))
    assert_equal(QUMO_SUBCOMMANDS_TEXT, sorted_text(repo.get("qumo_subcommands")))
    var allowed = run_check_allowed()
    var want = repo.get("run_check_allowed")
    assert_equal(len(allowed), len(want), "run_check_allowed is stale")
    for i in range(len(want)):
        same(allowed[i], want.at(i).at(1), "run_check_allowed")
    same(secret_env_substrings(), env.get("secret_name_substrings"), "secret_env_substrings")
    same(check_env_extra_substrings(), env.get("check_env_extra_substrings"), "check_env_extra_substrings")
    same(git_injected_exact(), env.get("git_injected").get("exact"), "git_injected_exact")
    same(git_injected_numbered(), env.get("git_injected").get("numbered"), "git_injected_numbered")
    same(minimal_child_env_names(), env.get("minimal_child_env"), "minimal_child_env_names")
    same(api_hosts(), research.get("api_hosts"), "api_hosts")
    assert_equal(MAX_REDIRECTS, research.get_int("max_redirects", -1))
    assert_equal(RESEARCH_PORT, research.get_int("port", -1))
    assert_equal(RESEARCH_SCHEME, research.get("scheme").as_string())
    same(web_groups(), lists.get("delegation").get("web_groups"), "web_groups")
    same(exec_groups(), lists.get("delegation").get("exec_groups"), "exec_groups")
    assert_equal(UNTRUSTED_MARKER, lists.get("untrusted").get("marker").as_string())
    assert_equal(MAX_ANSWER_CHARS, lists.get("untrusted").get_int("max_answer_chars", -1))


def test_start_up_lists(spec: JsonValue) raises:
    """What the harness reads from agents/prompts/tools.json at start-up (exported by the
    Python harness, which loads the file) is what the file says."""
    var catalog = LocalToolsCatalog(".")
    var repo = spec.get("lists").get("repo")
    same(catalog.skip_dirs(), repo.get("skip_dirs"), "agents/prompts/tools.json skip_dirs")
    var allowed = catalog.run_check_allowed()
    var want = repo.get("run_check_allowed")
    assert_equal(len(allowed), len(want))
    for i in range(len(want)):
        same(allowed[i], want.at(i).at(1), "agents/prompts/tools.json run_check_allowed")


def main() raises:
    var spec: JsonValue
    with open(SPEC_PATH, "r") as f:
        spec = parse_json(f.read())
    test_generated_module_is_current(spec)
    test_start_up_lists(spec)
    test_vectors(spec)
    print("test_guard_spec OK")
