"""Write pyfmt.json: Python format vectors for util/pyfmt.mojo (test_util_pyfmt).

Dev-time only, standard library only:

    agents/.venv/bin/python -I agents/mojo/tests/golden/gen_pyfmt.py

The output is ASCII JSON (ensure_ascii=True), so the Mojo test can read it with
qumo.ai.json.parse_json whatever characters the vectors hold. The "c1" section
reproduces the Python-format vectors of agents/mojo/DESIGN.md Appendix C.1.
"""

import json
import math
import os
import sys
from xml.sax.saxutils import escape, quoteattr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyfmt.json")

# Strings whose repr is exact under DESIGN.md 2.2.1's printable approximation.
REPR_INPUTS = [
    "", "a", "it's", '"q"', "both ' \"", "a\tb\nc\\", "\r\x00\x1f", "\x7f\x80\x9f",
    "é\xa0 \x7f\x01", "日本", "emoji \U0001F600", "soft\xadhyphen", "​‏",
    "⁦x⁩", "﻿bom", "private", "\U000F0000", "　wide space",
    " ogham", "tab\there", "back\\slash'quote", "only'single", 'only"double',
    "\x1b[0m", "é́", "line1\nline2\r\n", "\u0085next", "️",
]

# JSON documents: json.dumps(json.loads(doc)) in every form, and repr(json.loads(doc)).
JSON_DOCS = [
    '{}', '[]', '""', 'null', 'true', 'false', '0', '-0', '7', '-12', '1.0', '1e2', '2.5',
    '0.1', '1e-05', '1e16', '123456789012345678', '-9223372036854775808', '18446744073709551615',
    '3.141592653589793', '1e300', '-0.0',
    '{"path": "/é", "n": [1, 2.5, null, true], "s": "a\\n \\u007f\\u0001\\""}',
    '{"d": "é—\\ud83d\\ude00"}',
    '{"a": [], "b": {}, "c": [1, {"x": "é"}]}',
    '{"nested": {"deep": {"deeper": [[], [{}], [1, [2, [3]]]]}}}',
    '["\\u2028", "\\u2029", "\\u0085", "\\u00a0", "tab\\there", "quote\\"", "back\\\\slash"]',
    '{"name": "ls", "args": {"path": "/é", "n": 1.5, "ok": true, "z": null}, "id": "call_1", "type": "tool_call"}',
    '{"ctrl": "\\b\\f\\n\\r\\t\\u0000\\u001f", "del": "\\u007f", "slash": "a/b"}',
    '{"kéy": "v", "": "empty key", "it\'s": "x"}',
    '[1, -1, 0.5, -0.5, 1e-7, 123.456, 1e21, 1e22]',
]

STRINGS_FOR_DUMPS = [
    "", "plain", "é", " ", "\x7f", "\x00\x1f", "\"\\", "\U0001F600", "a\tb", "日本語",
]

FLOATS = [
    1e-05, 1e16, 1e15, 0.1 + 0.2, -0.0, 1.2345678901234568e17, 2.5, 100.0, 0.0, 1.0, -1.5,
    1e-7, 1e21, 1e22, 123456789.0, 0.1, 1 / 3, float(2 ** 53), 5e-324, 1.7976931348623157e308,
    9007199254740993.0, 1e100, 12345.678,
]

SPLITLINES = [
    "", "a", "a\n", "a\nb", "a\r\nb\rc\x0bd\x0ce\x1cf\x85g h\n", "\n\n", "x\r", "x\r\r\n",
    "a b", "a\x1db\x1ec", "no newline at all", "\r\n",
]

SPLIT_WS = [
    "", "   ", " a b　c\t", "a\x1cb\x1fc", "one", "  lead and trail  ", "a\xa0b c​d",
    "x\n\ny", " a b c",
]

STRIP = [
    "", "  x  ", "\t\nx\r\n", "　x\xa0", "x", "  ", "​x​", "\x1cx\x1f", "a b",
]

SPACE_CPS = [
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680, 0x2000,
    0x200A, 0x200B, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0x41, 0x00, 0x08, 0xFEFF, 0x180E,
]

LISTS = [[], ["a"], ["it's", '"q"', "é"], ["a\nb", ""]]

CP_SLICES = [
    ["héllo wörld", 0, 3], ["héllo wörld", -5, 11], ["héllo wörld", 2, -2], ["héllo wörld", 5, 2],
    ["héllo wörld", -100, 100], ["日本語テキスト", 1, 4], ["", 0, 5], ["a\U0001F600b", 1, 2],
]

UNIVERSAL = ["a\r\nb\rc\nd", "\r\r\n\n", "plain", "x\r"]

XML = ['hi <x> & "q"', "a'b", "a\"b'c", "tab\tnl\ncr\r", "plain", "&amp;"]


def c1_vectors():
    vec = {}
    vec["repr"] = [[s, repr(s)] for s in ["it's", '"q"', "both ' \"", "a\tb\nc\\", "é\xa0 \x7f\x01", "日本"]]
    vec["json_dumps_ensure_ascii_false"] = json.dumps(
        {"path": "/é", "n": [1, 2.5, None, True], "s": "a\n \x7f\x01\""}, ensure_ascii=False)
    vec["json_dumps_ensure_ascii_true"] = json.dumps({"d": "é—\U0001F600"})
    vec["json_dumps_indent1"] = json.dumps({"a": [], "b": {}, "c": [1, {"x": "é"}]}, indent=1, ensure_ascii=False)
    vec["float_repr"] = [[x, repr(x)] for x in [1e-05, 1e16, 1e15, 0.1 + 0.2, -0.0, 1.2345678901234568e17, 2.5, 100.0]]
    vec["splitlines"] = ["a\r\nb\rc\x0bd\x0ce\x1cf\x85g h\n".splitlines()]
    vec["split_ws"] = " a b　c\t".split()
    vec["repr_tool_calls"] = repr(
        [{"name": "ls", "args": {"path": "/é", "n": 1.5, "ok": True, "z": None}, "id": "call_1", "type": "tool_call"}])
    return vec


def nonprintable_table():
    """The `_NONPRINTABLE` literal of util/pyfmt.mojo: every range of code points at or
    above U+0080 that str.isprintable() rejects, as fixed-width records
    "SSSSSSEEEEEE" (hex, inclusive) separated by one space or newline."""
    ranges = []
    start = None
    for c in range(0x80, 0x110000):
        bad = not chr(c).isprintable()
        if bad and start is None:
            start = c
        elif not bad and start is not None:
            ranges.append((start, c - 1))
            start = None
    if start is not None:
        ranges.append((start, 0x10FFFF))
    recs = ["%06x%06x" % r for r in ranges]
    lines = [" ".join(recs[i:i + 8]) for i in range(0, len(recs), 8)]
    return len(ranges), "\n".join(lines)


def main():
    if "--table" in sys.argv:
        n, text = nonprintable_table()
        print(f"# {n} ranges, Unicode {__import__('unicodedata').unidata_version}")
        print(text)
        return
    out = {}
    out["repr"] = [[s, repr(s)] for s in REPR_INPUTS]
    out["json_docs"] = []
    for doc in JSON_DOCS:
        v = json.loads(doc)
        out["json_docs"].append({
            "doc": doc,
            "dumps_false": json.dumps(v, ensure_ascii=False),
            "dumps_true": json.dumps(v, ensure_ascii=True),
            "indent0": json.dumps(v, ensure_ascii=False, indent=0),
            "indent2": json.dumps(v, ensure_ascii=False, indent=2),
            "indent1_ascii": json.dumps(v, ensure_ascii=True, indent=1),
            "repr": repr(v),
        })
    out["dumps_string"] = [[s, json.dumps(s, ensure_ascii=False), json.dumps(s)] for s in STRINGS_FOR_DUMPS]
    out["float_repr"] = [[x, repr(x), json.dumps(x)] for x in FLOATS]
    out["float_special"] = [[name, repr(v), json.dumps(v)] for name, v in
                            [["nan", math.nan], ["inf", math.inf], ["-inf", -math.inf]]]
    out["splitlines"] = [[s, s.splitlines()] for s in SPLITLINES]
    out["split_ws"] = [[s, s.split()] for s in SPLIT_WS]
    out["strip"] = [[s, s.strip(), s.rstrip()] for s in STRIP]
    out["isspace"] = [[cp, chr(cp).isspace()] for cp in SPACE_CPS]
    out["list_repr"] = [[items, str(items)] for items in LISTS]
    out["cp_slice"] = [[s, a, b, s[a:b], len(s)] for s, a, b in CP_SLICES]
    out["cp_head_tail"] = [[s, n, s[:n], s[-n:] if n > 0 else ""] for s, n in
                           [["héllo", 2], ["héllo", 0], ["héllo", 9], ["héllo", -1], ["日本語", 1]]]
    out["universal_newlines"] = [[s, s.replace("\r\n", "\n").replace("\r", "\n")] for s in UNIVERSAL]
    out["xml"] = [[s, escape(s), quoteattr(s)] for s in XML]
    out["c1"] = c1_vectors()
    with open(OUT, "w", encoding="ascii") as f:
        json.dump(out, f, indent=1, ensure_ascii=True)
        f.write("\n")
    count = sum(len(v) for k, v in out.items() if isinstance(v, list))
    print(f"wrote {OUT} ({count} vectors, Python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()
