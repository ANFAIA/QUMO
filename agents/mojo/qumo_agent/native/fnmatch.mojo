"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

Shell-style pattern matching with Python's semantics (DESIGN.md 2.1.6):

- `fnmatch_py`: `fnmatch.fnmatchcase` (Python 3.12 `fnmatch._translate` rules).
- `wc_glob_match`: deepagents' `compile_grep_include_glob` (wcmatch `BRACE | GLOBSTAR`,
  no `DOTMATCH`), with its error texts.
- `expand_braces`: wcmatch/bracex brace expansion (`{a,b}`, nested, `{1..9..2}`,
  `{a..e}`), at most 1000 patterns.
- `pathlib_glob_match`: `pathlib.Path.rglob(pattern)` (Python 3.12) for files.

Matching is done here over Unicode code points rather than with libc `fnmatch`:
libc matches bytes in the C locale (so `?` would not match `é`), and Python's
bracket rules (`[^a]` is a set of `^` and `a`, empty ranges are dropped, an
unterminated `[` is literal) differ from libc's in several corners. The behaviour is
checked against vectors produced by Python 3.12 and wcmatch 12.0.
"""

from qumo_agent.util.pyfmt import py_str_repr

comptime _LIT = 0
comptime _ANY = 1
comptime _STAR = 2
comptime _SET = 3
comptime _NEVER = 4

comptime _PATTERN_LIMIT = 1000


def _cps(s: String) -> List[Int]:
    var out = List[Int]()
    var bs = s.as_bytes()
    var i = 0
    var n = len(bs)
    while i < n:
        var b0 = Int(bs[i])
        var cp: Int
        var w: Int
        if b0 < 0x80:
            cp = b0
            w = 1
        elif b0 < 0xE0:
            cp = ((b0 & 0x1F) << 6) | (Int(bs[i + 1]) & 0x3F)
            w = 2
        elif b0 < 0xF0:
            cp = ((b0 & 0x0F) << 12) | ((Int(bs[i + 1]) & 0x3F) << 6) | (Int(bs[i + 2]) & 0x3F)
            w = 3
        else:
            cp = ((b0 & 0x07) << 18) | ((Int(bs[i + 1]) & 0x3F) << 12) | ((Int(bs[i + 2]) & 0x3F) << 6) | (Int(bs[i + 3]) & 0x3F)
            w = 4
        out.append(cp)
        i += w
    return out^


def _from_cps(cps: List[Int], a: Int, b: Int) -> String:
    var out = String()
    for i in range(a, b):
        out += String(Codepoint(unsafe_unchecked_codepoint=UInt32(cps[i])))
    return out^


# ---------------------------------------------------------------------------
# Character sets and compiled patterns
# ---------------------------------------------------------------------------
@fieldwise_init
struct _CharSet(Copyable):
    var negate: Bool
    var lo: List[Int]        # ranges lo[i]..hi[i] (single characters have lo == hi)
    var hi: List[Int]
    var classes: List[Int]   # POSIX classes (wcmatch), see _class_id

    def contains(self, c: Int) -> Bool:
        var hit = False
        for i in range(len(self.lo)):
            if c >= self.lo[i] and c <= self.hi[i]:
                hit = True
                break
        if not hit:
            for k in self.classes:
                if _in_class(k, c):
                    hit = True
                    break
        return hit != self.negate


def _empty_set(negate: Bool) -> _CharSet:
    return _CharSet(negate, List[Int](), List[Int](), List[Int]())


comptime _CLASS_NAMES = "alnum alpha ascii blank cntrl digit graph lower print punct space upper word xdigit"


def _class_id(name: String) -> Int:
    var k = 0
    for n in String(_CLASS_NAMES).split(" "):
        if String(n) == name:
            return k
        k += 1
    return -1


def _in_class(k: Int, c: Int) -> Bool:
    var digit = c >= 0x30 and c <= 0x39
    var upper = c >= 0x41 and c <= 0x5A
    var lower = c >= 0x61 and c <= 0x7A
    var alpha = upper or lower
    var punct = (c >= 0x21 and c <= 0x2F) or (c >= 0x3A and c <= 0x40) or (c >= 0x5B and c <= 0x60) or (c >= 0x7B and c <= 0x7E)
    if k == 0:
        return alpha or digit
    if k == 1:
        return alpha
    if k == 2:
        return c < 0x80
    if k == 3:
        return c == 0x20 or c == 0x09
    if k == 4:
        return c < 0x20 or c == 0x7F
    if k == 5:
        return digit
    if k == 6:
        return c > 0x20 and c < 0x7F
    if k == 7:
        return lower
    if k == 8:
        return c >= 0x20 and c < 0x7F
    if k == 9:
        return punct
    if k == 10:
        return c == 0x20 or (c >= 0x09 and c <= 0x0D)
    if k == 11:
        return upper
    if k == 12:
        return alpha or digit or c == 0x5F
    if k == 13:
        return digit or (c >= 0x41 and c <= 0x46) or (c >= 0x61 and c <= 0x66)
    return False


struct _Pat(Movable):
    var kinds: List[Int]
    var vals: List[Int]      # _LIT: code point; _SET: index into sets
    var sets: List[_CharSet]

    def __init__(out self):
        self.kinds = List[Int]()
        self.vals = List[Int]()
        self.sets = List[_CharSet]()

    def add(mut self, kind: Int, val: Int):
        if kind == _STAR and len(self.kinds) > 0 and self.kinds[len(self.kinds) - 1] == _STAR:
            return
        self.kinds.append(kind)
        self.vals.append(val)

    def add_set(mut self, var s: _CharSet):
        self.sets.append(s^)
        self.add(_SET, len(self.sets) - 1)


def _tok_matches(p: _Pat, t: Int, c: Int) -> Bool:
    var k = p.kinds[t]
    if k == _LIT:
        return p.vals[t] == c
    if k == _ANY:
        return True
    if k == _SET:
        return p.sets[p.vals[t]].contains(c)
    return False


def _match(p: _Pat, text: List[Int], leading_dot: Bool) -> Bool:
    """Glob matching with `*` backtracking. With `leading_dot`, a "." at text index
    0 is matched only by a literal "." (no `*`, `?` or set)."""
    var n = len(text)
    var m = len(p.kinds)
    var ti = 0
    var si = 0
    var star_t = -1
    var star_s = 0
    while si < n:
        var blocked = leading_dot and si == 0 and text[0] == 0x2E
        if ti < m and p.kinds[ti] == _STAR:
            star_t = ti
            star_s = si
            ti += 1
            continue
        if ti < m and p.kinds[ti] != _NEVER and _tok_matches(p, ti, text[si]) and (not blocked or p.kinds[ti] == _LIT):
            ti += 1
            si += 1
            continue
        if star_t >= 0:
            # the star takes one more character
            if leading_dot and star_s == 0 and n > 0 and text[0] == 0x2E:
                return False
            star_s += 1
            si = star_s
            ti = star_t + 1
            continue
        return False
    while ti < m and p.kinds[ti] == _STAR:
        ti += 1
    return ti == m


# ---------------------------------------------------------------------------
# fnmatch (Python's fnmatch._translate)
# ---------------------------------------------------------------------------
@fieldwise_init
struct _SetTok(Copyable):
    var cp: Int
    var join: Bool   # an unescaped '-' produced by fnmatch's chunk join (forms ranges)


def _py_set(stuff_in: List[Int]) -> Tuple[Int, _CharSet]:
    """Python fnmatch's handling of a bracket body: returns (kind, set) with kind
    _SET, _NEVER (empty set) or _ANY (negated empty set)."""
    var toks = List[_SetTok]()
    var has_hyphen = False
    for c in stuff_in:
        if c == 0x2D:
            has_hyphen = True
    if not has_hyphen:
        for c in stuff_in:
            toks.append(_SetTok(c, False))
    else:
        var chunks = List[List[Int]]()
        var i = 0
        var j = len(stuff_in)
        var k = 2 if stuff_in[0] == 0x21 else 1
        while True:
            var f = -1
            var q = k
            while q < j:
                if stuff_in[q] == 0x2D:
                    f = q
                    break
                q += 1
            if f < 0:
                break
            var ch = List[Int]()
            for x in range(i, f):
                ch.append(stuff_in[x])
            chunks.append(ch^)
            i = f + 1
            k = f + 3
        var last = List[Int]()
        for x in range(i, j):
            last.append(stuff_in[x])
        if len(last) > 0:
            chunks.append(last^)
        else:
            chunks[len(chunks) - 1].append(0x2D)
        # Remove empty ranges.
        var kk = len(chunks) - 1
        while kk > 0:
            var prev = chunks[kk - 1].copy()
            var cur = chunks[kk].copy()
            if prev[len(prev) - 1] > cur[0]:
                var merged = List[Int]()
                for x in range(len(prev) - 1):
                    merged.append(prev[x])
                for x in range(1, len(cur)):
                    merged.append(cur[x])
                chunks[kk - 1] = merged^
                _ = chunks.pop(kk)
            kk -= 1
        for ci in range(len(chunks)):
            if ci > 0:
                toks.append(_SetTok(0x2D, True))
            for c in chunks[ci]:
                toks.append(_SetTok(c, False))
    if len(toks) == 0:
        return (_NEVER, _empty_set(False))
    if len(toks) == 1 and toks[0].cp == 0x21 and not toks[0].join:
        return (_ANY, _empty_set(False))
    var negate = toks[0].cp == 0x21 and not toks[0].join
    var s = _empty_set(negate)
    var idx = 1 if negate else 0
    while idx < len(toks):
        var this = toks[idx].cp
        idx += 1
        if idx < len(toks) and toks[idx].join:
            if idx + 1 >= len(toks):
                s.lo.append(this)
                s.hi.append(this)
                s.lo.append(0x2D)
                s.hi.append(0x2D)
                idx += 1
            else:
                var that = toks[idx + 1].cp
                if this <= that:
                    s.lo.append(this)
                    s.hi.append(that)
                idx += 2
        else:
            s.lo.append(this)
            s.hi.append(this)
    return (_SET, s^)


def _compile_py(pattern: String) -> _Pat:
    var p = _cps(pattern)
    var out = _Pat()
    var i = 0
    var n = len(p)
    while i < n:
        var c = p[i]
        i += 1
        if c == 0x2A:
            out.add(_STAR, 0)
        elif c == 0x3F:
            out.add(_ANY, 0)
        elif c == 0x5B:
            var j = i
            if j < n and p[j] == 0x21:
                j += 1
            if j < n and p[j] == 0x5D:
                j += 1
            while j < n and p[j] != 0x5D:
                j += 1
            if j >= n:
                out.add(_LIT, 0x5B)
            else:
                var stuff = List[Int]()
                for x in range(i, j):
                    stuff.append(p[x])
                i = j + 1
                var r = _py_set(stuff)
                if r[0] == _SET:
                    out.add_set(r[1].copy())
                else:
                    out.add(r[0], 0)
        else:
            out.add(_LIT, c)
    return out^


def fnmatch_py(name: String, pattern: String) -> Bool:
    """`fnmatch.fnmatchcase(name, pattern)`: case-sensitive; `*` and `?` match any
    character including "/" and a leading "."; `[!...]` negates, `[^...]` does not."""
    return _match(_compile_py(pattern), _cps(name), False)


# ---------------------------------------------------------------------------
# wcmatch segments (BRACE | GLOBSTAR, no DOTMATCH)
# ---------------------------------------------------------------------------
def _compile_wc_segment(seg: List[Int]) -> _Pat:
    """One path segment of a wcmatch glob: `*`, `?`, `[...]` (`!` or `^` negates,
    POSIX classes, ranges), backslash escapes."""
    var out = _Pat()
    var i = 0
    var n = len(seg)
    while i < n:
        var c = seg[i]
        if c == 0x5C:   # backslash
            if i + 1 >= n:
                out.add(_NEVER, 0)   # a trailing backslash never matches (wcmatch)
                i += 1
                continue
            out.add(_LIT, seg[i + 1])
            i += 2
            continue
        if c == 0x2A:
            out.add(_STAR, 0)
            i += 1
            continue
        if c == 0x3F:
            out.add(_ANY, 0)
            i += 1
            continue
        if c == 0x5B:
            var parsed = _wc_set(seg, i)
            if parsed[0] < 0:
                out.add(_LIT, 0x5B)
                i += 1
            else:
                out.add_set(parsed[1].copy())
                i = parsed[0]
            continue
        out.add(_LIT, c)
        i += 1
    return out^


def _wc_set(seg: List[Int], start: Int) -> Tuple[Int, _CharSet]:
    """Parses a wcmatch bracket at seg[start] == '['; (index past ']', set), or
    (-1, _) when it is not terminated (then '[' is literal)."""
    var n = len(seg)
    var i = start + 1
    var negate = False
    if i < n and (seg[i] == 0x21 or seg[i] == 0x5E):
        negate = True
        i += 1
    var s = _empty_set(negate)
    var first = True
    var pending = -1      # last single character, a possible range start
    while i < n:
        var c = seg[i]
        if c == 0x5D and not first:
            if pending >= 0:
                s.lo.append(pending)
                s.hi.append(pending)
            return (i + 1, s^)
        first = False
        # POSIX class [:name:]
        if c == 0x5B and i + 1 < n and seg[i + 1] == 0x3A:
            var e = i + 2
            while e + 1 < n and not (seg[e] == 0x3A and seg[e + 1] == 0x5D):
                e += 1
            if e + 1 < n:
                var name = _from_cps(seg, i + 2, e)
                var k = _class_id(name)
                if k >= 0:
                    if pending >= 0:
                        s.lo.append(pending)
                        s.hi.append(pending)
                        pending = -1
                    s.classes.append(k)
                    i = e + 2
                    continue
        var cp = c
        var adv = 1
        if c == 0x5C and i + 1 < n:
            cp = seg[i + 1]
            adv = 2
        # range: pending '-' cp
        if c == 0x2D and adv == 1 and pending >= 0 and i + 1 < n and seg[i + 1] != 0x5D:
            var hi_cp = seg[i + 1]
            var hadv = 1
            if hi_cp == 0x5C and i + 2 < n:
                hi_cp = seg[i + 2]
                hadv = 2
            if pending <= hi_cp:
                s.lo.append(pending)
                s.hi.append(hi_cp)
            pending = -1
            i += 1 + hadv
            continue
        if pending >= 0:
            s.lo.append(pending)
            s.hi.append(pending)
        pending = cp
        i += adv
    return (-1, s^)


def _split_keep(s: String) -> List[String]:
    var out = List[String]()
    for part in s.split("/"):
        out.append(String(part))
    return out^


def _wc_segments(pattern: String) -> List[String]:
    """Pattern segments: empty segments dropped (`a//b` is `a/b`), except a trailing
    one (`a/` matches only a path ending in "/")."""
    var raw = _split_keep(pattern)
    var out = List[String]()
    for i in range(len(raw)):
        if raw[i].byte_length() > 0:
            out.append(raw[i])
        elif i == len(raw) - 1 and i > 0:
            out.append(String())
    return out^


def _wc_match_from(pats: List[String], comp: List[_Pat], pi: Int, path: List[String], si: Int) -> Bool:
    if pi == len(pats):
        return si == len(path) or (si == len(path) - 1 and path[si].byte_length() == 0)
    var seg = pats[pi]
    if seg == "**":
        var last = pi == len(pats) - 1
        var n = 0
        while si + n <= len(path):
            if n > 0:
                var consumed = path[si + n - 1]
                if consumed.startswith("."):
                    break
            if not (n == 0 and last and pi > 0):
                if _wc_match_from(pats, comp, pi + 1, path, si + n):
                    return True
            n += 1
        return False
    if si >= len(path):
        return False
    var text = path[si]
    if seg.byte_length() == 0:
        return text.byte_length() == 0 and _wc_match_from(pats, comp, pi + 1, path, si + 1)
    if text.byte_length() == 0:
        return False
    if not _match(comp[pi], _cps(text), True):
        return False
    return _wc_match_from(pats, comp, pi + 1, path, si + 1)


def _wc_path_match(pattern: String, path: String) -> Bool:
    var pats = _wc_segments(pattern)
    if len(pats) == 0:
        return path.byte_length() == 0
    var comp = List[_Pat]()
    for s in pats:
        comp.append(_compile_wc_segment(_cps(s)))
    return _wc_match_from(pats, comp, 0, _split_keep(path), 0)


# ---------------------------------------------------------------------------
# Brace expansion (bracex, as wcmatch BRACE uses it)
# ---------------------------------------------------------------------------
def _find_close(p: List[Int], open_at: Int) -> Int:
    """Index of the '}' matching p[open_at] == '{' (escapes and nesting honoured), or -1."""
    var depth = 0
    var i = open_at
    while i < len(p):
        var c = p[i]
        if c == 0x5C:
            i += 2
            continue
        if c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top_commas(p: List[Int], a: Int, b: Int) -> List[List[Int]]:
    var parts = List[List[Int]]()
    var cur = List[Int]()
    var depth = 0
    var i = a
    while i < b:
        var c = p[i]
        if c == 0x5C and i + 1 < b:
            cur.append(c)
            cur.append(p[i + 1])
            i += 2
            continue
        if c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
        if c == 0x2C and depth == 0:
            parts.append(cur^)
            cur = List[Int]()
        else:
            cur.append(c)
        i += 1
    parts.append(cur^)
    return parts^


def _parse_int(cps: List[Int]) -> Optional[Int]:
    if len(cps) == 0:
        return None
    var i = 0
    var neg = False
    if cps[0] == 0x2D or cps[0] == 0x2B:
        neg = cps[0] == 0x2D
        i = 1
    if i >= len(cps):
        return None
    var v = 0
    while i < len(cps):
        if cps[i] < 0x30 or cps[i] > 0x39:
            return None
        v = v * 10 + (cps[i] - 0x30)
        if v > 1000000000:
            return None
        i += 1
    return -v if neg else v


def _sequence(body: List[Int]) -> Optional[List[String]]:
    """`{x..y[..incr]}` with integers (zero padded when an end has a leading 0) or
    single characters; None when `body` is not a sequence."""
    # split on ".."
    var parts = List[List[Int]]()
    var cur = List[Int]()
    var i = 0
    while i < len(body):
        if i + 1 < len(body) and body[i] == 0x2E and body[i + 1] == 0x2E:
            parts.append(cur^)
            cur = List[Int]()
            i += 2
            continue
        cur.append(body[i])
        i += 1
    parts.append(cur^)
    if len(parts) != 2 and len(parts) != 3:
        return None
    var step = 1
    if len(parts) == 3:
        var st = _parse_int(parts[2])
        if not st:
            return None
        step = abs(st.value())
        if step == 0:
            step = 1
    var out = List[String]()
    var a = _parse_int(parts[0])
    var b = _parse_int(parts[1])
    if a and b:
        var x = a.value()
        var y = b.value()
        var width = 0
        for side in range(2):
            var s = parts[side].copy()
            var off = 1 if (len(s) > 0 and (s[0] == 0x2D or s[0] == 0x2B)) else 0
            if len(s) - off > 1 and s[off] == 0x30:
                width = max(width, len(s))
        var v = x
        var count = 0
        while (v <= y) if x <= y else (v >= y):
            var txt = String(abs(v))
            var pad = width - (1 if v < 0 else 0)
            while txt.byte_length() < pad:
                txt = "0" + txt
            out.append(("-" + txt) if v < 0 else txt)
            v += step if x <= y else -step
            count += 1
            if count > _PATTERN_LIMIT + 1:
                break
        return out^
    if len(parts[0]) == 1 and len(parts[1]) == 1:
        var x = parts[0][0]
        var y = parts[1][0]
        var v = x
        while (v <= y) if x <= y else (v >= y):
            out.append(String(Codepoint(unsafe_unchecked_codepoint=UInt32(v))))
            v += step if x <= y else -step
        return out^
    return None


def _expand(p: List[Int]) raises -> List[String]:
    var i = 0
    while i < len(p):
        var c = p[i]
        if c == 0x5C:
            i += 2
            continue
        if c == 0x7B:
            var close = _find_close(p, i)
            if close < 0:
                i += 1
                continue
            var alts = List[String]()
            var parts = _split_top_commas(p, i + 1, close)
            var is_brace = False
            if len(parts) >= 2:
                is_brace = True
                for part in parts:
                    for e in _expand(part):
                        alts.append(e)
            else:
                var body = List[Int]()
                for x in range(i + 1, close):
                    body.append(p[x])
                var seq = _sequence(body)
                if seq:
                    is_brace = True
                    alts = seq.value().copy()
            if not is_brace:
                i += 1
                continue
            var prefix = _from_cps(p, 0, i)
            var rest = List[Int]()
            for x in range(close + 1, len(p)):
                rest.append(p[x])
            var tails = _expand(rest)
            var out = List[String]()
            for a in alts:
                for t in tails:
                    out.append(prefix + a + t)
                    if len(out) > _PATTERN_LIMIT:
                        raise Error("Pattern limit exceeded the limit of " + String(_PATTERN_LIMIT))
            return out^
        i += 1
    var whole = List[String]()
    whole.append(_from_cps(p, 0, len(p)))
    return whole^


def expand_braces(pattern: String) raises -> List[String]:
    """wcmatch BRACE expansion: `{a,b}` (nested, empty alternatives allowed),
    `{1..10..3}`, `{01..3}`, `{a..e}`; a brace with no top-level comma that is not a
    sequence, or an unbalanced one, stays literal. More than 1000 patterns raise
    `Pattern limit exceeded the limit of 1000`."""
    var out = _expand(_cps(pattern))
    if len(out) > _PATTERN_LIMIT:
        raise Error("Pattern limit exceeded the limit of " + String(_PATTERN_LIMIT))
    return out^


def _basename(path: String) -> String:
    """PurePosixPath(path).name."""
    var parts = path.split("/")
    var i = len(parts) - 1
    while i >= 0:
        var s = String(parts[i])
        if s.byte_length() > 0 and s != ".":
            return s
        if s.byte_length() > 0:
            return String()
        i -= 1
    return String()


def wc_glob_match(rel_path: String, pattern: String) raises -> Bool:
    """deepagents `compile_grep_include_glob(pattern)(rel_path)`: a pattern without
    "/" matches the basename at any depth; with "/" it matches the whole root-relative
    path (a leading "/" anchors it); `**` matches zero or more segments that do not
    start with "."; wildcards never match a leading "."; a `..` segment raises
    `Path traversal not allowed in glob pattern '<p>'`; brace expansion over the limit
    raises `Invalid glob pattern '<p>': Pattern limit exceeded the limit of 1000`.
    Unbalanced `[` or `{` are literal (they compile and match only themselves)."""
    var segs = pattern.replace("\\", "/").split("/")
    for s in segs:
        if String(s) == "..":
            raise Error("Path traversal not allowed in glob pattern " + py_str_repr(pattern))
    var anchored = pattern.find("/") >= 0
    var stripped = pattern
    while stripped.startswith("/"):
        var cut = String(stripped[byte=1:])
        stripped = cut^
    var expanded: List[String]
    try:
        expanded = expand_braces(stripped)
    except e:
        raise Error("Invalid glob pattern " + py_str_repr(pattern) + ": " + String(e))
    var target = rel_path if anchored else _basename(rel_path)
    for p in expanded:
        if _wc_path_match(p, target):
            return True
    return False


# ---------------------------------------------------------------------------
# pathlib rglob
# ---------------------------------------------------------------------------
def _rglob_from(pats: List[String], comp: List[_Pat], pi: Int, parts: List[String], si: Int) -> Bool:
    if pi == len(pats):
        return si == len(parts)
    if pats[pi] == "**":
        for n in range(0, len(parts) - si + 1):
            if _rglob_from(pats, comp, pi + 1, parts, si + n):
                return True
        return False
    if si >= len(parts):
        return False
    if not _match(comp[pi], _cps(parts[si]), False):
        return False
    return _rglob_from(pats, comp, pi + 1, parts, si + 1)


def pathlib_glob_match(rel_parts: List[String], pattern: String) -> Bool:
    """Whether the file at `rel_parts` (path parts relative to the search root) is
    yielded by `Path(root).rglob(pattern)` (Python 3.12): the pattern is
    `"**/" + pattern`; `**` matches any number of directories (hidden ones too);
    other segments use `fnmatch_py` (wildcards match a leading "."). A pattern whose
    last segment is `**` yields only directories in pathlib, so it matches no file
    here; an absolute pattern matches nothing (pathlib raises)."""
    if pattern.startswith("/") or len(rel_parts) == 0:
        return False
    var pats = List[String]()
    pats.append("**")
    for s in pattern.split("/"):
        var seg = String(s)
        if seg.byte_length() == 0 or seg == ".":
            continue
        pats.append(seg)
    if pats[len(pats) - 1] == "**":
        return False
    var comp = List[_Pat]()
    for s in pats:
        comp.append(_compile_py(s))
    return _rglob_from(pats, comp, 0, rel_parts, 0)
