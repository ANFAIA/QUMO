"""QUMO's AI-native interface layer.

* `diagnostics` - stable error codes -> structured `Diagnostic`s, lint, catalog.
* `describe`    - machine-readable gates / errors / capabilities, circuit JSON.
* `tools`       - `ToolRegistry`: one implementation per agent-facing operation.
* `mcp`         - Model Context Protocol server (JSON-RPC 2.0 over stdio).
* `yaml_io`     - flow-style YAML reader and writer for Pauli sums: no tags, anchors or
                   block style, so no document can name a class (docs/QILISDK_MAPPING.md).

See docs/AI_NATIVE.md.
"""
