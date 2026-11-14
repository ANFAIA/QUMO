"""`qumo-agent doctor`: what a run needs, one line per check (DESIGN.md 7.2).

Each line is `[ok] <check>` or `[--] <check>: <hint>` (the Python `doctor` style). Names of
variables are printed, never their values. Nothing here calls a model; unless `--offline`
the network check is one TLS handshake and an unauthenticated
`GET <base_url>/models/count` (no key is sent).

Exit code 0 when a run can work: the repository, `build/qumo`, and either OpenSSL with a
trust store and the model key, or `--offline` with a replay file. Everything else is
advice.
"""

from std.memory import ArcPointer
from std.os import getenv

from qumo_agent.core.errors import error_code
from qumo_agent.mcp.protocol import MCP_CLIENT_NAME, MCP_CLIENT_VERSION
from qumo_agent.mcp.stdio_client import McpStdioSession, StdioServerParams
from qumo_agent.native.fs import exists, is_file, join_path, read_text_lossy
from qumo_agent.native.libc import has_symbol
from qumo_agent.native.sandbox import Confinement, sandbox_kind
from qumo_agent.native.spawn import STDIO_INHERIT, mcp_server_env, which
from qumo_agent.net.http import HttpClient, HttpOptions, HttpRequest, REDIRECT_NONE
from qumo_agent.net.tls import TlsContext
from qumo_agent.tools.config import Registry
from qumo_agent.tools.qumo_mcp import default_qumo_bin
from qumo_agent.toolsmith.manifest import (
    ToolEntry, check_entry_shape, load_manifest, tool_store_dir, verify_entry,
)
from qumo_agent.util.pyfmt import cp_head

from qumo_agent.app.options import AppOptions


struct _Report(Movable):
    var lines: List[String]
    var blocking: Bool

    def __init__(out self):
        self.lines = List[String]()
        self.blocking = False

    def check(mut self, ok: Bool, what: String, hint: String):
        if ok:
            self.lines.append("[ok] " + what)
        else:
            self.lines.append("[--] " + what + ": " + hint)

    def note(mut self, line: String):
        self.lines.append(line)


def _one_line(s: String) -> String:
    return s.replace("\n", " ").replace("\r", " ")


def _qumo_answers(qumo_bin: String) -> String:
    """"" when `qumo_bin mcp` answers `initialize`, else the reason."""
    if not exists(qumo_bin):
        return qumo_bin + " not found (build it with `make cli`, or set QUMO_BIN)"
    try:
        var args: List[String] = ["mcp"]
        var params = StdioServerParams(qumo_bin, args^, mcp_server_env(), String(), 20000, 1 << 20,
                                       STDIO_INHERIT, Optional[Confinement](None), String())
        var s = McpStdioSession(params)
        try:
            s.initialize(MCP_CLIENT_NAME, MCP_CLIENT_VERSION)
        except e:
            try:
                _ = s.close()
            except:
                pass
            return qumo_bin + " did not answer initialize: " + _one_line(String(e))
        _ = s.close()
        return String()
    except e:
        return qumo_bin + " cannot start: " + _one_line(String(e))


def _exports_fresh(root: String) -> String:
    """"" when the exported files the Mojo agents read are present and current."""
    var files: List[String] = ["agents/prompts/specs.json", "agents/prompts/tools.json",
                               "agents/prompts/modules.json", "agents/prompts/preamble.md",
                               "agents/prompts/tool_guide.json"]
    for f in files:
        if not is_file(join_path(root, f)):
            return f + " is missing"
    try:
        var specs = read_text_lossy(join_path(root, "agents/prompts/specs.json"))
        if specs.find("generated_tools") < 0:
            return "agents/prompts/specs.json has no generated_tools"
        _ = Registry(root)
    except e:
        return _one_line(String(e))
    return String()


def _models_count(base_url: String) -> String:
    """"" when a TLS handshake and `GET <base_url>/models/count` (no key sent) work."""
    var url = base_url
    while url.endswith("/"):
        var cut = String(url[byte=0:url.byte_length() - 1])
        url = cut^
    url += "/models/count"
    try:
        var tls: Optional[ArcPointer[TlsContext]] = None
        var http = HttpClient(tls)
        var names: List[String] = ["Accept", "User-Agent", "Accept-Encoding"]
        var values: List[String] = ["application/json", "qumo-agent/0.2 (Mojo) doctor", "identity"]
        var plain = base_url.lower().startswith("http://")
        var resp = http.request(HttpRequest("GET", url, names^, values^, String()),
                                HttpOptions(15000, 15000, 1 << 20, True, REDIRECT_NONE, 0, plain, False))
        if resp.status >= 200 and resp.status < 500:
            return String()
        return "GET " + url + " answered HTTP " + String(resp.status)
    except e:
        return _one_line(String(e))


def run_doctor(opts: AppOptions, offline: Bool) -> Int:
    var rep = _Report()
    var root = opts.root
    var has_root = root.byte_length() > 0 and exists(join_path(root, "agents/prompts/specs.json"))
    rep.check(has_root, "repository root" + ((" " + root) if has_root else String()),
              "run from the repository or set QUMO_ROOT")
    if not has_root:
        rep.blocking = True
    var qumo_bin = default_qumo_bin(root)
    var qumo_err = _qumo_answers(qumo_bin) if has_root else String("no repository")
    rep.check(qumo_err.byte_length() == 0, "QUMO MCP server " + qumo_bin + " answers initialize", qumo_err)
    if qumo_err.byte_length() > 0:
        rep.blocking = True
    var export_err = _exports_fresh(root) if has_root else String("no repository")
    rep.check(export_err.byte_length() == 0, "exported prompts and specs are current (generated_tools, tools, modules, preamble)",
              export_err + ". Hint: run agents/.venv/bin/python -m qumo_agents export-prompts")
    var base_url = String("https://openrouter.ai/api/v1")
    var key_env = String("OPENROUTER_API_KEY")
    if has_root and export_err.byte_length() == 0:
        try:
            var ep = Registry(root).endpoint()
            base_url = ep.base_url
            key_env = ep.api_key_env
        except:
            pass
    # TLS: OpenSSL and the trust store.
    var tls_ok = False
    var tls_what: String
    try:
        var t = TlsContext()
        tls_ok = True
        tls_what = "OpenSSL " + t.lib.version_text + " (" + t.lib.libssl_path + "), trust store " + t.trust_source
    except e:
        tls_what = _one_line(String(e))
    var plain_model = base_url.lower().startswith("http://")
    rep.check(tls_ok or plain_model, "TLS: " + tls_what if tls_ok else "TLS",
              tls_what if not tls_ok else String())
    var key_set = getenv(key_env, "").byte_length() > 0
    rep.check(key_set, "model key " + key_env + " is set", "export the key to use the default back end")
    var net_ok = True
    if not offline:
        var net_err = String("skipped: no TLS") if (not tls_ok and not plain_model) else _models_count(base_url)
        net_ok = net_err.byte_length() == 0
        rep.check(net_ok, "model endpoint " + base_url + " reachable (TLS handshake, GET /models/count without a key)", net_err)
    rep.check(getenv("ALPHAXIV_API_KEY", "").byte_length() > 0, "alphaXiv papers (ALPHAXIV_API_KEY)",
              "optional; without it agents use the arXiv API")
    rep.check(getenv("GITHUB_TOKEN", "").byte_length() > 0, "GitHub token (GITHUB_TOKEN)",
              "optional; raises the release-notes rate limit")
    rep.check(which("git", getenv("PATH", "")).byte_length() > 0, "git in PATH", "install git (sandboxes are git worktrees)")
    var mojo_bin = join_path(root, ".venv/bin/mojo")
    rep.check(has_root and exists(mojo_bin), ".venv/bin/mojo (the toolsmith builds tools with it)", "run `make setup`")
    rep.check(has_symbol("posix_spawn_file_actions_addchdir_np"), "posix_spawn has a chdir action",
              "use glibc 2.29 or newer; the chdir trampoline __exec-in is used instead")
    # OS sandbox.
    var kind = sandbox_kind(opts.self_binary) if opts.self_binary.byte_length() > 0 else String("none")
    if kind == "none":
        rep.check(False, "OS sandbox for model-written code: none",
                  "the toolsmith cannot build or test tools, unreviewed tools do not run, and run_check inside a sandbox is refused, unless a human passes --allow-unsandboxed-tool-tests")
    else:
        rep.check(True, "OS sandbox for model-written code: " + kind, String())
    # Generated tools.
    var store = opts.tools_store if opts.tools_store.byte_length() > 0 else tool_store_dir(root)
    var reviewed = 0
    var unreviewed = 0
    var problems = List[String]()
    try:
        var m = load_manifest(store)
        for e in m.entries:
            var check = verify_entry(store, e)
            if not check.ok:
                for p in check.problems:
                    problems.append(p)
            elif e.reviewed:
                reviewed += 1
            else:
                unreviewed += 1
    except e:
        problems.append(_one_line(String(e)))
    rep.check(len(problems) == 0, "generated tools in " + store + ": " + String(reviewed) + " reviewed, "
              + String(unreviewed) + " unreviewed (run `qumo-agent tools review NAME`), 0 refused",
              String(len(problems)) + " refused")
    for p in problems:
        rep.note("     refused: " + cp_head(_one_line(p), 300))
    if kind == "none" and reviewed > 0 and not opts.allow_unsandboxed_tool_tests:
        rep.note("     note: with no OS sandbox, reviewed tools run unconfined")
    # Verdict.
    var model_ok = (tls_ok or plain_model) and key_set and net_ok
    var offline_ok = offline and opts.replay_file.byte_length() > 0 and exists(opts.replay_file)
    var usable = not rep.blocking and (model_ok or offline_ok)
    for l in rep.lines:
        print(l)
    print("")
    print("a run can work: " + ("yes" if usable else "no"))
    return 0 if usable else 1
