"""Lint of agents/mojo/qumo_agent (DESIGN.md 4.1 rule 1, 8.2). Run from the
repository root.

1. `external_call` appears only in qumo_agent/native/libc.mojo, except in native/shutdown.mojo,
   which passes C-ABI handler functions to `signal` (a function cannot go through an Int).
2. No `std.subprocess`, `std.os.process`, `/bin/sh`, `curl`, `mktemp` or a string
   starting with `date ` anywhere in qumo_agent/ (the runtime never starts a shell,
   curl, mktemp or date).
3. Every file that carries ported deepagents/langchain text (one of the marker
   phrases below, or "Portions ported from") starts with the MIT header of
   DESIGN.md 0.5.

While the pre-port modules still exist (qumo_agent/{agent,config,llm,mcp_client,
proc,server}.mojo, deleted by Integration, DESIGN.md 1.3), those six files and the
package docstring that describes them (qumo_agent/__init__.mojo, rewritten by
Integration) are skipped; once proc.mojo is gone they are checked like any file.
"""

from std.testing import assert_true

from qumo_agent.native.fs import walk, read_text_lossy, exists, is_file

comptime _ROOT = "agents/mojo/qumo_agent"
comptime _HEADER_1 = "Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4"
comptime _HEADER_2 = "(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt)."


def _is_ident(c: Int) -> Bool:
    return (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F


def _has_word(text: String, word: String) -> Bool:
    """`word` as a whole identifier-like token (so "curly" is not "curl")."""
    var bs = text.as_bytes()
    var start = 0
    while True:
        var i = text.find(word, start)
        if i < 0:
            return False
        var j = i + word.byte_length()
        var before_ok = i == 0 or not _is_ident(Int(bs[i - 1]))
        var after_ok = j >= len(bs) or not _is_ident(Int(bs[j]))
        if before_ok and after_ok:
            return True
        start = i + 1


def _markers() -> List[String]:
    var m: List[String] = [
        "Launch an ephemeral subagent",
        "Tool result too large, the result of this tool call",
        "Context Extraction Assistant",
        "is not a valid tool, try one of",
        "Lists all files in a directory",
        "Path traversal not allowed",
        "Recursion limit of",
        "Message content too large and was saved",
        "In order to complete the objective that the user asks of you",
        "could not be executed - arguments",
        "Here is a summary of the conversation to date",
        "Portions ported from",
    ]
    return m^


def main() raises:
    assert_true(exists(_ROOT), "run from the repository root")
    var legacy = exists(_ROOT + "/proc.mojo")
    var skipped: List[String] = [
        _ROOT + "/agent.mojo", _ROOT + "/config.mojo", _ROOT + "/llm.mojo", _ROOT + "/mcp_client.mojo",
        _ROOT + "/proc.mojo", _ROOT + "/server.mojo", _ROOT + "/__init__.mojo",
    ]
    var problems = List[String]()
    var checked = 0
    for path in walk(_ROOT, False):
        if not path.endswith(".mojo") or not is_file(path):
            continue
        var rel = String(path[byte = path.find(_ROOT) :])
        if legacy:
            var skip = False
            for s in skipped:
                if rel == s:
                    skip = True
            if skip:
                continue
        checked += 1
        var text = read_text_lossy(path)
        if text.find("external_call") >= 0 and rel != _ROOT + "/native/libc.mojo" and rel != _ROOT + "/native/shutdown.mojo":
            problems.append(rel + ": external_call outside native/libc.mojo")
        if text.find("std.subprocess") >= 0:
            problems.append(rel + ": uses std.subprocess")
        if text.find("std.os.process") >= 0:
            problems.append(rel + ": uses std.os.process")
        if text.find("/bin/sh") >= 0:
            problems.append(rel + ": mentions /bin/sh")
        if _has_word(text, "curl"):
            problems.append(rel + ": mentions curl")
        if _has_word(text, "mktemp"):
            problems.append(rel + ": mentions mktemp")
        if text.find('"date ') >= 0 or text.find("'date ") >= 0:
            problems.append(rel + ": runs date")
        var ported = False
        for m in _markers():
            if text.find(m) >= 0:
                ported = True
                break
        if ported:
            var head = String()
            var lines = text.split("\n")
            for i in range(min(len(lines), 20)):
                head += String(lines[i]) + "\n"
            if head.find(_HEADER_1) < 0 or head.find(_HEADER_2) < 0:
                problems.append(rel + ": carries ported deepagents/langchain text without the MIT header (DESIGN.md 0.5)")
    assert_true(checked > 0, "no Mojo files found under " + _ROOT)
    if len(problems) > 0:
        for p in problems:
            print("lint: " + p)
    assert_true(len(problems) == 0, String(len(problems)) + " lint problem(s)")
    if legacy:
        print("note: pre-port modules still present; they and qumo_agent/__init__.mojo were skipped")
    print("test_native_lint OK")
