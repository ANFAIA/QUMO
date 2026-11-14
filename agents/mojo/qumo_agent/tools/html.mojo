"""Plain text of an HTML page, as `fetch_page` returns it (DESIGN.md 2.8.4).

Python (`qumo_agents/tools/research.py`, fetch_page):

    text = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())[:20000]

The two substitutions are reproduced by scanning, with Python's matching rules: the tag name
is matched case-insensitively with `re`'s Unicode rules (`s` also matches U+017F, `i` also
U+0130 and U+0131), the closing tag must repeat the opening name under `re`'s
case-insensitive back-reference rule (simple lower-case of each character equal), and the
shortest body wins. Entities are not decoded.
"""

from qumo_agent.util.pyfmt import cp_head, py_split_ws

comptime FETCH_PAGE_MAX_CHARS = 20000


def _cp_at(bs: Span[Byte, _], i: Int) -> Tuple[Int, Int]:
    """(code point, byte length) of the UTF-8 sequence at i (input is valid UTF-8)."""
    var b0 = Int(bs[i])
    if b0 < 0x80:
        return (b0, 1)
    if b0 < 0xE0 and i + 1 < len(bs):
        return (((b0 & 0x1F) << 6) | (Int(bs[i + 1]) & 0x3F), 2)
    if b0 < 0xF0 and i + 2 < len(bs):
        return (((b0 & 0x0F) << 12) | ((Int(bs[i + 1]) & 0x3F) << 6) | (Int(bs[i + 2]) & 0x3F), 3)
    if i + 3 < len(bs):
        return (((b0 & 0x07) << 18) | ((Int(bs[i + 1]) & 0x3F) << 12) | ((Int(bs[i + 2]) & 0x3F) << 6) | (Int(bs[i + 3]) & 0x3F), 4)
    return (b0, 1)


def _pattern_letter_matches(letter: Int, cp: Int) -> Bool:
    """re.I match of one ASCII lower-case pattern letter against a code point (Python 3.12
    sre: the letter, its upper case, and U+017F for s, U+0130/U+0131 for i)."""
    if cp == letter or cp == letter - 32:
        return True
    if letter == 0x73 and cp == 0x17F:
        return True
    if letter == 0x69 and (cp == 0x130 or cp == 0x131):
        return True
    return False


def _simple_lower(cp: Int) -> Int:
    """The one-character lower case sre uses for IGNORECASE back-references, for the code
    points that can occur in a matched `script`/`style` name or be compared with one."""
    if cp >= 0x41 and cp <= 0x5A:
        return cp + 32
    if cp == 0x130:
        return 0x69
    return cp


def _match_word(bs: Span[Byte, _], at: Int, word: String) -> Int:
    """End offset when `word` (ASCII lower case) matches at `at` under re.I, else -1."""
    var w = word.as_bytes()
    var i = at
    for k in range(len(w)):
        if i >= len(bs):
            return -1
        var c = _cp_at(bs, i)
        if not _pattern_letter_matches(Int(w[k]), c[0]):
            return -1
        i += c[1]
    return i


def _match_backref(bs: Span[Byte, _], at: Int, g0: Int, g1: Int) -> Int:
    """End offset when the group text bs[g0:g1] repeats at `at` (case-insensitively, as sre
    compares back-references), else -1."""
    var i = at
    var j = g0
    while j < g1:
        if i >= len(bs):
            return -1
        var a = _cp_at(bs, i)
        var b = _cp_at(bs, j)
        if _simple_lower(a[0]) != _simple_lower(b[0]):
            return -1
        i += a[1]
        j += b[1]
    return i


def _find_byte(bs: Span[Byte, _], c: Int, start: Int) -> Int:
    for i in range(start, len(bs)):
        if Int(bs[i]) == c:
            return i
    return -1


def _script_style_end(bs: Span[Byte, _], at: Int) -> Int:
    """End offset of a match of `<(script|style)[^>]*>.*?</\\1>` starting at `at` (a '<'),
    or -1."""
    var name_end = _match_word(bs, at + 1, "script")
    if name_end < 0:
        name_end = _match_word(bs, at + 1, "style")
    if name_end < 0:
        return -1
    var gt = _find_byte(bs, 0x3E, name_end)   # [^>]*>
    if gt < 0:
        return -1
    var q = gt + 1
    while True:
        var lt = _find_byte(bs, 0x3C, q)
        if lt < 0:
            return -1
        if lt + 1 < len(bs) and Int(bs[lt + 1]) == 0x2F:   # "</"
            var e = _match_backref(bs, lt + 2, at + 1, name_end)
            if e >= 0 and e < len(bs) and Int(bs[e]) == 0x3E:
                return e + 1
        q = lt + 1


def _remove_script_style(text: String) -> String:
    var bs = text.as_bytes()
    var out = List[UInt8](capacity=len(bs))
    var i = 0
    while i < len(bs):
        if Int(bs[i]) == 0x3C:
            var e = _script_style_end(bs, i)
            if e >= 0:
                out.append(0x20)
                i = e
                continue
        out.append(bs[i])
        i += 1
    return String(unsafe_from_utf8=Span(out))


def _remove_tags(text: String) -> String:
    """re.sub(r"<[^>]+>", " ", text)."""
    var bs = text.as_bytes()
    var out = List[UInt8](capacity=len(bs))
    var i = 0
    while i < len(bs):
        if Int(bs[i]) == 0x3C:
            var gt = _find_byte(bs, 0x3E, i + 1)
            if gt > i + 1:
                out.append(0x20)
                i = gt + 1
                continue
            if gt < 0:
                # no '>' anywhere after: no later '<' can match either
                for k in range(i, len(bs)):
                    out.append(bs[k])
                break
        out.append(bs[i])
        i += 1
    return String(unsafe_from_utf8=Span(out))


def strip_html(raw: String) -> String:
    """fetch_page's text: scripts and styles removed, tags replaced by spaces, whitespace
    collapsed, first 20,000 code points."""
    var text = _remove_tags(_remove_script_style(raw))
    var words = py_split_ws(text)
    var joined = String(" ").join(words)
    return cp_head(joined, FETCH_PAGE_MAX_CHARS)
