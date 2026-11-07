"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

The MCP `inputSchema` transform Python applies before a tool reaches the model
(critic B7): langchain-core `convert_to_openai_tool` on the adapter's tool, that is
`_convert_json_schema_to_openai_function` (SP/langchain_core/utils/function_calling.py:121-153):

1. `dereference_refs` (SP/langchain_core/utils/json_schema.py) with its defaults:
   `$ref` objects are replaced by the referenced value, extra keys next to a `$ref`
   are merged over it, a reference already being resolved on the current path is
   cut (only the extra keys stay), and values under a `$defs` key are copied without
   resolving.
2. Top-level `definitions` and `$defs` are dropped, `title` is popped, and
   `description` is popped and used only when the tool's own description is empty.
3. `_rm_titles`: every `title` key is removed through nested objects, except a
   `title` whose value is an object directly under `properties` (a property named
   `title`); arrays are not entered.

Objects keep Python dict semantics: key order is insertion order, a repeated key keeps
its first position and its last value.
"""

from qumo.ai.json import JsonValue, JsonKind, JsonWriter, parse_json

from qumo_agent.core.errors import E_MCP, qa_error


struct _Node(Copyable):
    var kind: Int              # JsonKind.*
    var scalar: String         # JSON text of a scalar
    var text: String           # decoded value of a string
    var keys: List[String]     # object member names, in order
    var kids: List[Int]        # object member values / array items (node indices)

    def __init__(out self, kind: Int):
        self.kind = kind
        self.scalar = String()
        self.text = String()
        self.keys = List[String]()
        self.kids = List[Int]()


struct _Tree(Movable):
    """Immutable nodes in one arena: transforms build new nodes and share unchanged ones."""

    var nodes: List[_Node]

    def __init__(out self):
        self.nodes = List[_Node]()

    def add(mut self, var n: _Node) -> Int:
        self.nodes.append(n^)
        return len(self.nodes) - 1

    def kind(self, i: Int) -> Int:
        return self.nodes[i].kind

    def find(self, obj: Int, key: String) -> Int:
        """Position of `key` among the members of object `obj`, or -1."""
        ref n = self.nodes[obj]
        for j in range(len(n.keys)):
            if n.keys[j] == key:
                return j
        return -1


def _set_member(mut obj: _Node, key: String, value: Int):
    """dict assignment: replace in place when present, else append."""
    for j in range(len(obj.keys)):
        if obj.keys[j] == key:
            obj.kids[j] = value
            return
    obj.keys.append(key)
    obj.kids.append(value)


def _import(mut tree: _Tree, v: JsonValue) raises -> Int:
    var k = v.kind()
    if k == JsonKind.OBJECT:
        var n = _Node(JsonKind.OBJECT)
        for i in range(len(v)):
            var child = _import(tree, v.at(i))
            _set_member(n, v.key_at(i), child)
        return tree.add(n^)
    if k == JsonKind.ARRAY:
        var n = _Node(JsonKind.ARRAY)
        for i in range(len(v)):
            n.kids.append(_import(tree, v.at(i)))
        return tree.add(n^)
    var n = _Node(k)
    n.scalar = v.to_json()
    if k == JsonKind.STRING:
        n.text = v.as_string()
    return tree.add(n^)


def _is_ascii_digits(s: String) -> Bool:
    var bs = s.as_bytes()
    if len(bs) == 0:
        return False
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c < 0x30 or c > 0x39:
            return False
    return True


def _not_found(path: String) -> Error:
    return qa_error(E_MCP, "invalid inputSchema: Reference '" + path + "' not found.", "")


def _retrieve_ref(tree: _Tree, path: String, full: Int) raises -> Int:
    """`_retrieve_ref`: walk a `#/a/b/0` fragment from the schema root."""
    var comps = path.split("/")
    if len(comps) == 0 or String(comps[0]) != "#":
        raise qa_error(E_MCP, "invalid inputSchema: ref paths are expected to be URI fragments, meaning they should start with #.", "")
    var cur = full
    for ci in range(1, len(comps)):
        var comp = String(comps[ci])
        var kind = tree.kind(cur)
        if kind == JsonKind.OBJECT:
            var pos = tree.find(cur, comp)
            if pos < 0:
                raise _not_found(path)
            cur = tree.nodes[cur].kids[pos]
        elif kind == JsonKind.ARRAY:
            # Python checks `component in list` first: a string item equal to the
            # component makes the lookup fail.
            ref items = tree.nodes[cur].kids
            for j in range(len(items)):
                if tree.kind(items[j]) == JsonKind.STRING and tree.nodes[items[j]].text == comp:
                    raise _not_found(path)
            if not _is_ascii_digits(comp) or comp.byte_length() > 9:
                raise _not_found(path)
            var index = Int(comp)
            if index >= len(items):
                raise _not_found(path)
            cur = items[index]
        else:
            raise _not_found(path)
    return cur


def _process_members(mut tree: _Tree, keys: List[String], kids: List[Int], full: Int,
                     mut processing: List[String]) raises -> Int:
    """`_process_dict_properties`: a new object with nested values resolved, except
    under `$defs` (copied as they are)."""
    var n = _Node(JsonKind.OBJECT)
    for j in range(len(keys)):
        var child = kids[j]
        var ck = tree.kind(child)
        if keys[j] != "$defs" and (ck == JsonKind.OBJECT or ck == JsonKind.ARRAY):
            child = _deref(tree, child, full, processing)
        _set_member(n, keys[j], child)
    return tree.add(n^)


def _deref(mut tree: _Tree, idx: Int, full: Int, mut processing: List[String]) raises -> Int:
    """`_dereference_refs_helper` with `skip_keys=["$defs"]`, `shallow_refs=True`."""
    var kind = tree.kind(idx)
    if kind == JsonKind.OBJECT:
        var keys = tree.nodes[idx].keys.copy()
        var kids = tree.nodes[idx].kids.copy()
        var ref_pos = -1
        for j in range(len(keys)):
            if keys[j] == "$ref":
                ref_pos = j
        if ref_pos < 0:
            return _process_members(tree, keys, kids, full, processing)
        var ref_node = kids[ref_pos]
        if tree.kind(ref_node) != JsonKind.STRING:
            raise qa_error(E_MCP, "invalid inputSchema: a $ref value is not a string", "")
        var ref_path = tree.nodes[ref_node].text
        var extra_keys = List[String]()
        var extra_kids = List[Int]()
        for j in range(len(keys)):
            if j != ref_pos:
                extra_keys.append(keys[j])
                extra_kids.append(kids[j])
        for p in processing:
            if p == ref_path:
                # A cycle: keep only the keys next to the $ref.
                return _process_members(tree, extra_keys, extra_kids, full, processing)
        processing.append(ref_path)
        var target = _retrieve_ref(tree, ref_path, full)
        var resolved = _deref(tree, target, full, processing)
        _ = processing.pop()
        if len(extra_keys) == 0:
            return resolved
        var merged = _Node(JsonKind.OBJECT)
        if tree.kind(resolved) == JsonKind.OBJECT:
            merged.keys = tree.nodes[resolved].keys.copy()
            merged.kids = tree.nodes[resolved].kids.copy()
        var extra = _process_members(tree, extra_keys, extra_kids, full, processing)
        var ek = tree.nodes[extra].keys.copy()
        var ev = tree.nodes[extra].kids.copy()
        for j in range(len(ek)):
            _set_member(merged, ek[j], ev[j])
        return tree.add(merged^)
    if kind == JsonKind.ARRAY:
        var items = tree.nodes[idx].kids.copy()
        var n = _Node(JsonKind.ARRAY)
        for j in range(len(items)):
            n.kids.append(_deref(tree, items[j], full, processing))
        return tree.add(n^)
    return idx


def _rm_titles(mut tree: _Tree, idx: Int, prev_key: String) -> Int:
    """`_rm_titles` on object `idx` (SP/langchain_core/utils/function_calling.py:87-117)."""
    var keys = tree.nodes[idx].keys.copy()
    var kids = tree.nodes[idx].kids.copy()
    var n = _Node(JsonKind.OBJECT)
    for j in range(len(keys)):
        var child = kids[j]
        var is_obj = tree.kind(child) == JsonKind.OBJECT
        if keys[j] == "title":
            if is_obj and prev_key == "properties":
                n.keys.append(keys[j])
                n.kids.append(_rm_titles(tree, child, keys[j]))
            # any other `title` is removed
        elif is_obj:
            n.keys.append(keys[j])
            n.kids.append(_rm_titles(tree, child, keys[j]))
        else:
            n.keys.append(keys[j])
            n.kids.append(child)
    return tree.add(n^)


def _pop(mut tree: _Tree, obj: Int, key: String) -> Int:
    """Remove `key` from object `obj` in place; returns its value node or -1."""
    var pos = tree.find(obj, key)
    if pos < 0:
        return -1
    var v = tree.nodes[obj].kids[pos]
    _ = tree.nodes[obj].keys.pop(pos)
    _ = tree.nodes[obj].kids.pop(pos)
    return v


def _write(tree: _Tree, idx: Int, mut w: JsonWriter):
    ref n = tree.nodes[idx]
    if n.kind == JsonKind.OBJECT:
        w.begin_object()
        for j in range(len(n.keys)):
            w.key(n.keys[j])
            _write(tree, n.kids[j], w)
        w.end_object()
    elif n.kind == JsonKind.ARRAY:
        w.begin_array()
        for j in range(len(n.kids)):
            _write(tree, n.kids[j], w)
        w.end_array()
    else:
        w.raw(n.scalar)


def transform_input_schema(schema_json: String, tool_description: String) raises -> Tuple[String, String]:
    """(parameters_json, description) as Python's `convert_to_openai_tool` produces
    them for an MCP tool with this `inputSchema` and description."""
    var parsed: JsonValue
    try:
        parsed = parse_json(schema_json)
    except e:
        raise qa_error(E_MCP, "invalid inputSchema: " + String(e), "")
    if not parsed.is_object():
        raise qa_error(E_MCP, "invalid inputSchema: expected a JSON object, got " + parsed.kind_name(), "")
    var tree = _Tree()
    var root = _import(tree, parsed)
    var processing = List[String]()
    var schema = _deref(tree, root, root, processing)
    if tree.kind(schema) != JsonKind.OBJECT:
        raise qa_error(E_MCP, "invalid inputSchema: the top-level $ref does not resolve to an object", "")
    # _deref built a fresh top-level node, so popping in place does not touch `root`.
    if schema == root:
        var copy_node = _Node(JsonKind.OBJECT)
        copy_node.keys = tree.nodes[root].keys.copy()
        copy_node.kids = tree.nodes[root].kids.copy()
        schema = tree.add(copy_node^)
    _ = _pop(tree, schema, "definitions")
    _ = _pop(tree, schema, "$defs")
    _ = _pop(tree, schema, "title")
    var default_desc = _pop(tree, schema, "description")
    var cleaned = _rm_titles(tree, schema, "")
    var w = JsonWriter()
    _write(tree, cleaned, w)
    var description = tool_description
    if description.byte_length() == 0 and default_desc >= 0 and tree.kind(default_desc) == JsonKind.STRING:
        description = tree.nodes[default_desc].text
    return (w.finish(), description^)
