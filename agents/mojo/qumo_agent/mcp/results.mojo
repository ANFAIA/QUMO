"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).
The error placeholder text follows langchain-mcp-adapters 0.3.2 (same copyright and
license).

MCP tool results as the harness gets them (critic B6).

Python's adapter turns an MCP `CallToolResult` into a list of content blocks, so the
tool message carries an array of text parts on the wire (never joined text), with
the same shape for `isError` (`status="error"` is not sent). `structuredContent` is
an artifact the model never sees, so it is dropped here.
"""

from qumo.ai.json import JsonValue

from qumo_agent.core.toolspec import ToolOutcome, parts_outcome
from qumo_agent.mcp.protocol import McpCallResult, block_text

# SP/langchain_mcp_adapters/tools.py:95-96 (_summarize_tool_error, no content at all).
comptime MCP_EMPTY_ERROR_TEXT = "MCP tool returned an error with empty content."


def call_result_to_outcome(r: McpCallResult) -> ToolOutcome:
    """`parts_outcome(parts, is_error)`; an error with no content gets the adapter's
    single placeholder part; a success with no content stays an empty part list."""
    var parts = r.parts.copy()
    if r.is_error and len(parts) == 0:
        parts.append(String(MCP_EMPTY_ERROR_TEXT))
    return parts_outcome(parts^, r.is_error)


def content_block_text(block: JsonValue) raises -> String:
    """`text` blocks give their text; `resource` blocks with `resource.text` give that
    text; every other block gives `[<type> content omitted: qumo-agent passes text only]`
    (X9)."""
    return block_text(block)
