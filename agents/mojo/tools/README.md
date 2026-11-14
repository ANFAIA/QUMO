# Generated Mojo tools

This directory holds the generated tools that the `mojo-toolsmith` agent writes, and
`gen_bitstring_stats/`, the reference tool to copy from. A generated tool is a small
native program with exactly one tool, which other QUMO agents call over MCP once a
human has reviewed it. The design is in `agents/mojo/DESIGN.md`, section 6.

## Layout

```
agents/mojo/tools/<name>/
  tool.mojo          the program (written by toolsmith_new, edited with edit_file)
  cases.json         its test cases (schema qumo.tool_cases.v1)
  build/<name>       the binary (toolsmith_build; ignored by git)
  build/build.json   sha256 of the compiled source, the SDK and the binary
  build/last_test.json   the last toolsmith_test run
```

Names are `gen_` followed by 2 to 40 characters from `a-z`, `0-9` and `_`.

## Writing a tool

A tool is built on `qumo_agent/tool_sdk.mojo`:

```mojo
from qumo.ai.json import JsonValue
from qumo_agent.tool_sdk import ToolInfo, ToolOutput, ok, fail, ok_json, arg_string, serve_tool

def handle(args: JsonValue) raises -> ToolOutput:
    var text = arg_string(args, "text", "")
    if text.byte_length() == 0:
        return fail("text is required")
    return ok(text)

def main() raises:
    serve_tool(ToolInfo("gen_echo", "Echo", "Returns its input text.", SCHEMA, "0.1.0"), handle)
```

- `ok(text)` is a result, `fail(text)` an error result for bad input, and
  `ok_json(object_json)` a result whose text and `structuredContent` are the same JSON
  object (build it with `JsonWriter` from `qumo.ai.json`).
- `arg_string`, `arg_int`, `arg_float` and `arg_bool` return the default for a
  missing or null argument and raise for a value of the wrong type. An exception
  raised by `handle` becomes the result `error: <message>`, so check arguments and
  return `fail(...)` with a clear message instead.
- `SCHEMA` is the JSON Schema of the arguments: an object with `properties`,
  `required` and `"additionalProperties": false`.
- Results are cut at 1,000,000 bytes; keep them well under 100 KB.
- Only `std`, `qumo.ai.json` and `qumo_agent.tool_sdk` are available. Put everything
  in `tool.mojo`: it is compiled alone, against the main checkout's `qumo/` and
  `qumo_agent/`, so other files you change in the sandbox are not part of the build.
- The dialect is Mojo 1.2 (`docs/MOJO_GUIDE.md`). One trap found while writing the
  reference tool: `std.math.log2` is off by about 1e-10 for float64 in Mojo 1.2; use
  `log(x) / log(2.0)`.

## What a tool can do at run time

Every call starts a private copy of the binary in a fresh temporary directory, with
an empty environment, a timeout (30 s by default, at most 300 s) and capped output.
With an OS sandbox (`sandbox-exec` on macOS, user namespaces and Landlock on Linux)
the tool cannot open network connections, start other processes (macOS), read the
user's home directory (except the Mojo runtime libraries) or write outside its
temporary directory. Nothing is kept between calls.

## Test cases

```json
{"schema": "qumo.tool_cases.v1", "tool": "gen_bitstring_stats", "cases": [
  {"name": "bell", "arguments": {"counts": "{\"00\": 500, \"11\": 500}"},
   "expect": {"is_error": false, "json_equals": {"shots": 1000, "entropy_bits": 1.0}}},
  {"name": "bad input", "arguments": {"counts": "nope"},
   "expect": {"is_error": true, "text_contains": ["error"]}}]}
```

`expect` keys: `is_error` (default false), `text_equals`, `text_contains` (a string or
a list; every one must occur) and `json_equals` (the result text parsed as JSON must
equal this value; object key order does not matter, numbers are compared exactly).
Write at least three cases: a normal input, an edge case, and an invalid input that
must return an error.

## The flow

1. `open_sandbox(topic)`: tools are written in a sandbox branch, never in the main
   checkout.
2. `toolsmith_new(name, description, source, cases)`: writes `tool.mojo` and
   `cases.json` (templates when `source` or `cases` is empty).
3. `toolsmith_build(name)`: compiles `tool.mojo` hermetically under the OS sandbox and
   returns the exit code, the diagnostics and the binary's sha256.
4. `toolsmith_test(name)`: runs every case against the binary under the OS sandbox and
   returns `PASS p/t` or `FAIL p/t` with one line per case.
5. `toolsmith_publish(name, version)`: copies the tested source, cases and binary to
   the tool store (`$QUMO_AGENT_TOOLS_DIR`, default `build/agent-tools/`) and adds a
   manifest entry with their sha256, marked `reviewed: false`. It refuses when
   `tool.mojo`, `cases.json` or the binary changed after the last passing test.

## Review

No agent can call a tool until a human reviews it:

```
qumo-agent tools review <name>
```

The command checks the stored hashes, prints the provenance, the source, the cases
and the test summary, and asks for the tool name. It then rebuilds the binary from
exactly the stored source the human just read, re-runs the stored cases against it,
replaces the agent-built binary, and marks the entry reviewed. `tools revoke`,
`tools remove`, `tools list`, `tools show` and `tools verify` manage the store.

Agents that may use generated tools declare them in their spec (`generated_tools`:
`"all"` or a list of names).
