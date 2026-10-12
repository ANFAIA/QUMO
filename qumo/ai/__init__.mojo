"""QUMO's AI-native interface layer.

* `json`        - streaming JSON writer + arena JSON parser.
* `diagnostics` - stable error codes -> structured `Diagnostic`s, lint, catalog.
* `describe`    - machine-readable gates / errors / capabilities, circuit JSON.
* `tools`       - `ToolRegistry`: one implementation per agent-facing operation.
* `mcp`         - Model Context Protocol server (JSON-RPC 2.0 over stdio).

See docs/AI_NATIVE.md.
"""
