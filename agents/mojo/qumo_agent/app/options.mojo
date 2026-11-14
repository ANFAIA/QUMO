"""Command-line options of the application as one value (DESIGN.md 2.10 `AppOptions`).

Kept in its own module so `app/subproc.mojo` and `app/env.mojo` can both use it without
importing each other; `app/env.mojo` re-exports the names of the design (`AppOptions`).
"""

from std.os import getenv

comptime DEFAULT_RECURSION_LIMIT = 200
comptime DEFAULT_PARALLEL = 4


@fieldwise_init
struct AppOptions(Copyable):
    var root: String
    var model_override: String
    var replay_file: String
    var recursion_limit: Int          # 200
    var parallel: Int                 # 4; 1 = inline
    var allow_unreviewed_tools: Bool
    var allow_unsandboxed_tool_tests: Bool   # human-only flag: run model-written code without an OS sandbox (6.8)
    var tools_store: String
    var verbose: Bool
    var self_binary: String           # for subagent children and `call`
    var deterministic: Bool           # $QUMO_AGENT_DETERMINISTIC=1: fixed clock and ids (tests)
    var api_key_env: String           # the model endpoint's key variable ("" until the registry is read)

    def child_flags(self) -> List[String]:
        """The options a child process (`subagent`, `serve`) gets from its parent. The
        model override is not passed: it applies to the top agent only."""
        var out = List[String]()
        if self.replay_file.byte_length() > 0:
            out.append("--replay")
            out.append(self.replay_file)
        if self.tools_store.byte_length() > 0:
            out.append("--tools-dir")
            out.append(self.tools_store)
        if self.allow_unreviewed_tools:
            out.append("--allow-unreviewed-tools")
        if self.allow_unsandboxed_tool_tests:
            out.append("--allow-unsandboxed-tool-tests")
        out.append("--parallel")
        out.append(String(self.parallel))
        if not self.verbose:
            out.append("--quiet")
        return out^


def default_options(root: String, self_binary: String) -> AppOptions:
    return AppOptions(root, String(), String(), DEFAULT_RECURSION_LIMIT, DEFAULT_PARALLEL, False, False,
                      String(), True, self_binary, getenv("QUMO_AGENT_DETERMINISTIC", "") == "1", String())
