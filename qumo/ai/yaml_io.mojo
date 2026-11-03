"""Safe YAML 1.2 flow style for Pauli sums: the QUMO side of qilisdk interchange.

qilisdk loads YAML with ruamel's unsafe loader, so a server response can name a
Python class in a tag and run code. QUMO has no such path. It reads and writes
only the JSON-compatible subset of YAML 1.2: {...} mappings, [...] sequences,
and quoted or plain scalars. Every such document is valid YAML, so any YAML
loader can read what QUMO writes.

QUMO does not read block style, tags (!), anchors (&), aliases (*), a --- separator,
or a second document. `refuse_unsafe` raises QM-E915 for each of them before any
parsing. Its string rules match `parse_json`, so a tag or anchor character inside
a double-quoted string is data.
"""

from std.math import isinf, isnan

from qumo.ai.json import JsonValue, JsonWriter, json_number, parse_json
from qumo.ir.pauli import PauliSum


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------
def _is_blank(c: Int) -> Bool:
    return c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D


def _is_name_break(c: Int) -> Bool:
    """Bytes that end a tag, anchor or alias name."""
    return c <= 0x20 or c >= 0x7F or c == 0x2C or c == 0x5B or c == 0x5D or c == 0x7B or c == 0x7D


def _token(bs: Span[Byte, _], start: Int) -> String:
    """The name at `start` (a tag, anchor or alias): printable ASCII, at most 40 bytes."""
    var n = len(bs)
    var end = start + 1
    while end < n and not _is_name_break(Int(bs[end])):
        end += 1
    var stop = min(end, start + 41)
    var out = String(unsafe_from_utf8=bs[start:stop])
    if end > stop:
        out += "..."
    return out^


def _char_desc(bs: Span[Byte, _], pos: Int) -> String:
    var c = Int(bs[pos])
    if c >= 0x21 and c <= 0x7E:
        return "'" + String(unsafe_from_utf8=bs[pos:pos + 1]) + "'"
    return "a non-printable or non-ASCII byte"


def _is_separator(bs: Span[Byte, _], pos: Int) -> Bool:
    if Int(bs[pos]) != 0x2D or pos + 2 >= len(bs):
        return False
    return Int(bs[pos + 1]) == 0x2D and Int(bs[pos + 2]) == 0x2D


def _skip_string(bs: Span[Byte, _], start: Int) -> Int:
    """`start` is an opening quote. Returns the index after the closing quote, or len(bs) if unterminated."""
    var n = len(bs)
    var i = start + 1
    while i < n:
        var c = Int(bs[i])
        if c == 0x22:
            return i + 1
        if c == 0x5C:
            i += 2
        else:
            i += 1
    return n


def _construct_error(bs: Span[Byte, _], pos: Int) -> Error:
    """The QM-E915 error for a tag, anchor or alias at `pos`."""
    var name = _token(bs, pos)
    if Int(bs[pos]) == 0x21:
        return Error(
            "QM-E915: tag " + name + ": QUMO does not read tags, because a tag can name a class. "
            + "Hint: write the document in flow style without tags."
        )
    var what = "anchor " if Int(bs[pos]) == 0x26 else "alias "
    return Error(
        "QM-E915: " + what + name + ": QUMO does not read anchors or aliases, because an alias "
        + "can reuse a node the reader does not check. Hint: write each value out in full, without & or *."
    )


def _separator_error() -> Error:
    return Error(
        "QM-E915: document separator ---: QUMO reads one document per text. "
        + "Hint: remove the --- line and send one flow-style document."
    )


def _start_error(bs: Span[Byte, _], pos: Int) -> Error:
    """The first significant byte is not { or [. Name what it starts."""
    var c = Int(bs[pos])
    if c == 0x21 or c == 0x26 or c == 0x2A:
        return _construct_error(bs, pos)
    if _is_separator(bs, pos):
        return _separator_error()
    if c == 0x23:
        return Error("QM-E915: a comment before the document. Hint: remove the comment and start the document with { or [.")
    if c == 0x25:
        return Error("QM-E915: a directive (%...) before the document. Hint: remove the directive and start the document with { or [.")
    return Error(
        "QM-E915: block style or a bare scalar: the document starts with " + _char_desc(bs, pos)
        + ", not with { or [. Hint: write the whole document as one flow-style mapping, e.g. {\"num_qubits\": 2, \"terms\": []}."
    )


def refuse_unsafe(text: String) raises:
    """Raise QM-E915 unless `text` is one flow-style {...} or [...] document.

    Refuses tags, anchors, aliases, a --- separator, a second top-level document,
    and any document that does not start with { or [. Run it before parse_json.
    """
    var bs = text.as_bytes()
    var n = len(bs)
    var i = 0
    var started = False
    var closed = False
    var depth = 0
    while i < n:
        var c = Int(bs[i])
        if _is_blank(c):
            i += 1
            continue
        if not started:
            if c != 0x7B and c != 0x5B:
                raise _start_error(bs, i)
            started = True
        if c == 0x22 and not closed:
            i = _skip_string(bs, i)
            continue
        if c == 0x21 or c == 0x26 or c == 0x2A:
            raise _construct_error(bs, i)
        if _is_separator(bs, i):
            raise _separator_error()
        if closed:
            raise Error(
                "QM-E915: a second top-level document starts with " + _char_desc(bs, i)
                + " after the first one closed. Hint: send one flow-style document per text."
            )
        if c == 0x7B or c == 0x5B:
            depth += 1
        elif c == 0x7D or c == 0x5D:
            depth -= 1
            if depth == 0:
                closed = True
        i += 1
    if not started:
        raise Error("QM-E915: empty document. Hint: write one flow-style document, e.g. {\"num_qubits\": 1, \"terms\": []}.")


def yaml_load_flow(text: String) raises -> JsonValue:
    """Generic safe reader: refuse_unsafe, then parse_json. Other modules build on it."""
    refuse_unsafe(text)
    return parse_json(text)


# ---------------------------------------------------------------------------
# Pauli sums
# ---------------------------------------------------------------------------
def pauli_sum_to_yaml(h: PauliSum) raises -> String:
    """Write `h` as {"num_qubits": n, "terms": [{"coeff": c, "paulis": "XZI"}, ...]}.

    n is h.num_qubits(), or 1 when every term is a constant, because the reader needs n >= 1.
    Raises QM-E913 for a coefficient that is NaN or infinite.
    """
    var n = max(1, h.num_qubits())
    var w = JsonWriter()
    w.begin_object()
    w.kv_int("num_qubits", n)
    w.key("terms")
    w.begin_array()
    for t in h.terms:
        if isnan(t.coeff) or isinf(t.coeff):
            raise Error(
                "QM-E913: a Pauli term has a coefficient that is NaN or infinite, and YAML has no number for it. "
                + "Hint: pass finite coefficients."
            )
        w.begin_object()
        w.key("coeff")
        w.raw(json_number(t.coeff))
        w.kv_string("paulis", t.label(n))
        w.end_object()
    w.end_array()
    w.end_object()
    return w.finish()


def _shape(key: String, problem: String, hint: String) -> Error:
    return Error("QM-E915: key '" + key + "': " + problem + ". Hint: " + hint)


def pauli_sum_from_yaml(text: String) raises -> PauliSum:
    """Read a flow-style Pauli sum. Raises QM-E915 for unsafe YAML or a bad shape; the message names the key.

    Checks: num_qubits is an integer from 1 to 64 (the width of a PauliTerm). Each term has
    a finite numeric coeff and a paulis string of num_qubits letters over I, X, Y, Z.
    The declared width is only checked: PauliSum keeps no width, so num_qubits() of the
    result can be smaller when the top qubits are all I.
    """
    var doc = yaml_load_flow(text)
    if not doc.is_object():
        raise Error(
            "QM-E915: the document must be a mapping with num_qubits and terms, got " + doc.kind_name()
            + ". Hint: write {\"num_qubits\": 2, \"terms\": []}."
        )

    var nv = doc.get("num_qubits")
    if not nv.is_defined():
        raise _shape("num_qubits", "missing", "add \"num_qubits\": n, the qubit count from 1 to 64.")
    if not nv.is_number():
        raise _shape("num_qubits", "must be an integer, got " + nv.kind_name(), "write the qubit count as a number such as 3.")
    var nf = nv.as_float()
    var nq = 0
    if nf >= 1.0 and nf <= 64.0:
        nq = Int(nf)
    if nq == 0 or Float64(nq) != nf:
        raise _shape("num_qubits", "must be an integer from 1 to 64, got " + String(nf), "use a width such as \"num_qubits\": 3.")

    var tv = doc.get("terms")
    if not tv.is_defined():
        raise _shape("terms", "missing", "add \"terms\": [...], with [] for a sum with no terms.")
    if not tv.is_array():
        raise _shape("terms", "must be an array, got " + tv.kind_name(), "write \"terms\": [{\"coeff\": 1.0, \"paulis\": \"Z\"}].")

    var h = PauliSum()
    for k in range(len(tv)):
        var t = tv.at(k)
        var at_term = "terms[" + String(k) + "]"
        if not t.is_object():
            raise _shape(at_term, "must be an object with coeff and paulis, got " + t.kind_name(), "write each term as {\"coeff\": 1.0, \"paulis\": \"XZI\"}.")

        var cv = t.get("coeff")
        if not cv.is_defined():
            raise _shape(at_term + ".coeff", "missing", "add \"coeff\": c, a finite number.")
        if not cv.is_number():
            raise _shape(at_term + ".coeff", "must be a number, got " + cv.kind_name(), "write the coefficient as a number such as 0.5.")
        var coeff = cv.as_float()
        if isnan(coeff) or isinf(coeff):
            raise _shape(at_term + ".coeff", "must be a finite number", "use a finite number such as 0.5.")

        var pv = t.get("paulis")
        if not pv.is_defined():
            raise _shape(at_term + ".paulis", "missing", "add \"paulis\": \"...\", one letter per qubit.")
        if not pv.is_string():
            raise _shape(at_term + ".paulis", "must be a string, got " + pv.kind_name(), "write the Pauli string in quotes, e.g. \"XZI\".")
        var label = pv.as_string()
        if label.byte_length() != nq:
            raise _shape(
                at_term + ".paulis",
                "must have " + String(nq) + " letters, one per qubit, got " + String(label.byte_length()),
                "write num_qubits letters; the leftmost letter is the highest qubit.",
            )
        var lb = label.as_bytes()
        for j in range(nq):
            var ch = Int(lb[j])
            if ch != 0x49 and ch != 0x58 and ch != 0x59 and ch != 0x5A:  # I X Y Z
                raise _shape(
                    at_term + ".paulis",
                    "character " + _char_desc(lb, j) + " at position " + String(j) + " is not I, X, Y or Z",
                    "use only the letters I, X, Y and Z; the leftmost letter is the highest qubit.",
                )
        h.add_label(coeff, label)
    return h^
