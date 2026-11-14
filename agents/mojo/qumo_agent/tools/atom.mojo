"""arXiv's Atom feed as `search_arxiv` reads it (DESIGN.md 2.8.4).

Python (`qumo_agents/tools/research.py`, search_arxiv_raw) parses the feed with
`xml.etree.ElementTree.fromstring` and, for every Atom `entry` child of the root, reads
`id`, `title`, `published`, up to six `author/name` and `summary` with `findtext`.

This module is a small XML reader with the parts of expat/ElementTree those reads depend on:
an optional BOM, the XML declaration and other processing instructions, comments, a DOCTYPE
(skipped), CDATA sections, the five named entities and numeric character references, line-end
normalisation (`\\r\\n` and `\\r` become `\\n`), attribute values, and namespaces (`xmlns`,
`xmlns:p`; only elements in the Atom namespace count). `Element.text` is the character data
before the first child element; comments and processing instructions inside it are dropped and
the data around them joined, as ElementTree's TreeBuilder does.

Errors raise `ParseError: <expat message>: line L, column C` with expat's texts and positions
(line from 1, column in code points from 0) for the cases probed against Python 3.12: mismatched
tag, no element found, junk after document element, undefined entity, unbound prefix,
duplicate attribute, reference to invalid character number, and not well-formed (invalid token).
Other malformed input may be reported with a different message than expat would give.
"""

from qumo.ai.json import JsonWriter, parse_json

from qumo_agent.util.pyfmt import cp_head, py_json_dumps, py_split_ws

comptime ATOM_NS = "http://www.w3.org/2005/Atom"
comptime XML_NS = "http://www.w3.org/XML/1998/namespace"
comptime MAX_AUTHORS = 6
comptime SUMMARY_CHARS = 900


@fieldwise_init
struct ArxivEntry(Copyable):
    var id: String
    var title: String
    var published: String
    var authors: List[String]
    var authors_null: List[Bool]      # findtext returned None
    var summary: String
    var url: String
    var url_null: Bool


@fieldwise_init
struct _Node(Copyable):
    var ns: String           # namespace URI ("" when none)
    var local: String
    var parent: Int
    var children: List[Int]
    var text: String         # character data before the first child element


@fieldwise_init
struct _NsBinding(Copyable):
    var prefix: String       # "" for the default namespace
    var uri: String
    var depth: Int


def _is_ws(c: Int) -> Bool:
    return c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D


def _is_name_end(c: Int) -> Bool:
    return _is_ws(c) or c == 0x2F or c == 0x3E or c == 0x3D or c == 0x3C or c == 0x22 or c == 0x27 or c == 0x26


def _is_xml_char(cp: Int) -> Bool:
    if cp == 0x09 or cp == 0x0A or cp == 0x0D:
        return True
    if cp < 0x20:
        return False
    if cp >= 0xD800 and cp <= 0xDFFF:
        return False
    if cp == 0xFFFE or cp == 0xFFFF:
        return False
    return cp <= 0x10FFFF


def _put_cp(mut out: List[UInt8], cp: Int):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


struct _Parser(Movable):
    var src: String
    var pos: Int
    var nodes: List[_Node]
    var ns: List[_NsBinding]
    var stack: List[Int]           # open element indices
    var raw_names: List[String]    # qualified names of the open elements
    var root: Int

    def __init__(out self, src: String):
        self.src = src
        self.pos = 0
        self.nodes = List[_Node]()
        self.ns = List[_NsBinding]()
        self.stack = List[Int]()
        self.raw_names = List[String]()
        self.root = -1

    # ----------------------------------------------------------------- positions, errors
    def _error(self, msg: String, at: Int) -> Error:
        """`ParseError: <msg>: line L, column C` for the byte offset `at`."""
        var bs = self.src.as_bytes()
        var line = 1
        var col = 0
        var i = 0
        var stop = min(at, len(bs))
        while i < stop:
            var c = Int(bs[i])
            if c == 0x0D:
                line += 1
                col = 0
                if i + 1 < stop and Int(bs[i + 1]) == 0x0A:
                    i += 1
            elif c == 0x0A:
                line += 1
                col = 0
            elif c < 0x80 or c >= 0xC0:
                col += 1
            i += 1
        return Error("ParseError: " + msg + ": line " + String(line) + ", column " + String(col))

    def _byte(self, i: Int) -> Int:
        var bs = self.src.as_bytes()
        if i < 0 or i >= len(bs):
            return -1
        return Int(bs[i])

    def _starts(self, i: Int, s: String) -> Bool:
        var bs = self.src.as_bytes()
        var t = s.as_bytes()
        if i + len(t) > len(bs):
            return False
        for k in range(len(t)):
            if bs[i + k] != t[k]:
                return False
        return True

    def _find(self, s: String, start: Int) -> Int:
        var bs = self.src.as_bytes()
        var t = s.as_bytes()
        var i = start
        while i + len(t) <= len(bs):
            var ok = True
            for k in range(len(t)):
                if bs[i + k] != t[k]:
                    ok = False
                    break
            if ok:
                return i
            i += 1
        return -1

    def _len(self) -> Int:
        return self.src.byte_length()

    def _skip_ws(mut self):
        while self.pos < self._len() and _is_ws(self._byte(self.pos)):
            self.pos += 1

    def _slice(self, a: Int, b: Int) -> String:
        return String(self.src[byte=a:b])

    # ----------------------------------------------------------------- markup pieces
    def _skip_comment(mut self) raises:
        """At "<!--"."""
        var end = self._find("-->", self.pos + 4)
        if end < 0:
            raise self._error("unclosed token", self.pos)
        self.pos = end + 3

    def _skip_pi(mut self) raises:
        """At "<?"."""
        var end = self._find("?>", self.pos + 2)
        if end < 0:
            raise self._error("unclosed token", self.pos)
        self.pos = end + 2

    def _skip_doctype(mut self) raises:
        """At "<!DOCTYPE": skipped, including an internal subset in [...]."""
        var i = self.pos + 9
        var depth = 0
        var quote = -1
        while i < self._len():
            var c = self._byte(i)
            if quote >= 0:
                if c == quote:
                    quote = -1
            elif c == 0x22 or c == 0x27:
                quote = c
            elif c == 0x5B:
                depth += 1
            elif c == 0x5D:
                depth -= 1
            elif c == 0x3E and depth <= 0:
                self.pos = i + 1
                return
            i += 1
        raise self._error("unclosed token", self.pos)

    def _name(mut self) raises -> String:
        var start = self.pos
        while self.pos < self._len() and not _is_name_end(self._byte(self.pos)):
            self.pos += 1
        if self.pos == start:
            raise self._error("not well-formed (invalid token)", start)
        return self._slice(start, self.pos)

    def _reference(mut self, mut out: List[UInt8]) raises:
        """At '&': a named or numeric reference, appended to `out`."""
        var amp = self.pos
        var semi = -1
        var i = amp + 1
        while i < self._len():
            var c = self._byte(i)
            if c == 0x3B:
                semi = i
                break
            if _is_name_end(c) or c == 0x3B:
                break
            i += 1
        if semi < 0 or semi == amp + 1:
            raise self._error("not well-formed (invalid token)", amp + 1)
        var name = self._slice(amp + 1, semi)
        if name.startswith("#"):
            var digits = String(name[byte=1:])
            var base = 10
            if digits.startswith("x"):
                base = 16
                var rest = String(digits[byte=1:])
                digits = rest^
            var cp = 0
            var ok = digits.byte_length() > 0
            for b in digits.as_bytes():
                var d = Int(b)
                var v = -1
                if d >= 0x30 and d <= 0x39:
                    v = d - 0x30
                elif base == 16 and d >= 0x61 and d <= 0x66:
                    v = d - 0x61 + 10
                elif base == 16 and d >= 0x41 and d <= 0x46:
                    v = d - 0x41 + 10
                if v < 0:
                    ok = False
                    break
                cp = cp * base + v
                if cp > 0x10FFFF:
                    cp = 0x110000
            if not ok:
                raise self._error("not well-formed (invalid token)", amp)
            if not _is_xml_char(cp):
                raise self._error("reference to invalid character number", amp)
            _put_cp(out, cp)
        elif name == "amp":
            out.append(0x26)
        elif name == "lt":
            out.append(0x3C)
        elif name == "gt":
            out.append(0x3E)
        elif name == "quot":
            out.append(0x22)
        elif name == "apos":
            out.append(0x27)
        else:
            raise self._error("undefined entity", amp)
        self.pos = semi + 1

    def _attr_value(mut self) raises -> String:
        """At the opening quote. Entities expanded, \\t \\n \\r normalised to spaces."""
        var q = self._byte(self.pos)
        if q != 0x22 and q != 0x27:
            raise self._error("not well-formed (invalid token)", self.pos)
        self.pos += 1
        var out = List[UInt8]()
        while True:
            if self.pos >= self._len():
                raise self._error("unclosed token", self.pos)
            var c = self._byte(self.pos)
            if c == q:
                self.pos += 1
                break
            if c == 0x3C:
                raise self._error("not well-formed (invalid token)", self.pos)
            if c == 0x26:
                self._reference(out)
                continue
            if c == 0x0D and self._byte(self.pos + 1) == 0x0A:
                self.pos += 1
            if c == 0x09 or c == 0x0A or c == 0x0D:
                out.append(0x20)
            else:
                out.append(UInt8(c))
            self.pos += 1
        return String(unsafe_from_utf8=Span(out))

    def _resolve(self, qname: String, is_element: Bool, at: Int) raises -> Tuple[String, String]:
        """(namespace URI, local name) of a qualified name."""
        var colon = qname.find(":")
        var prefix = String()
        var local = qname
        if colon >= 0:
            prefix = String(qname[byte=0:colon])
            local = String(qname[byte=colon + 1:])
        elif not is_element:
            return (String(), qname)
        if prefix == "xml":
            return (String(XML_NS), local)
        var i = len(self.ns) - 1
        while i >= 0:
            if self.ns[i].prefix == prefix:
                return (self.ns[i].uri, local)
            i -= 1
        if prefix.byte_length() > 0:
            raise self._error("unbound prefix", at)
        return (String(), local)

    def _start_tag(mut self) raises:
        """At '<' of a start tag."""
        var lt = self.pos
        self.pos += 1
        var qname = self._name()
        var depth = len(self.stack) + 1
        var attr_names = List[String]()
        var self_closing = False
        while True:
            var had_ws = self.pos < self._len() and _is_ws(self._byte(self.pos))
            self._skip_ws()
            if self.pos >= self._len():
                raise self._error("unclosed token", lt)
            var c = self._byte(self.pos)
            if c == 0x3E:
                self.pos += 1
                break
            if c == 0x2F:
                if self._byte(self.pos + 1) != 0x3E:
                    raise self._error("not well-formed (invalid token)", self.pos)
                self.pos += 2
                self_closing = True
                break
            if not had_ws:
                raise self._error("not well-formed (invalid token)", self.pos)
            var name_at = self.pos
            var aname = self._name()
            for a in attr_names:
                if a == aname:
                    raise self._error("duplicate attribute", name_at)
            attr_names.append(aname)
            self._skip_ws()
            if self._byte(self.pos) != 0x3D:
                raise self._error("not well-formed (invalid token)", self.pos)
            self.pos += 1
            self._skip_ws()
            var value = self._attr_value()
            if aname == "xmlns":
                self.ns.append(_NsBinding(String(), value, depth))
            elif aname.startswith("xmlns:"):
                self.ns.append(_NsBinding(String(aname[byte=6:]), value, depth))
        for a in attr_names:
            if a.find(":") >= 0 and not a.startswith("xmlns:"):
                _ = self._resolve(a, False, lt)
        var r = self._resolve(qname, True, lt)
        var parent = -1
        if len(self.stack) > 0:
            parent = self.stack[len(self.stack) - 1]
        var idx = len(self.nodes)
        self.nodes.append(_Node(r[0], r[1], parent, List[Int](), String()))
        if parent >= 0:
            self.nodes[parent].children.append(idx)
        else:
            self.root = idx
        self.stack.append(idx)
        self.raw_names.append(qname)
        if self_closing:
            self._close()

    def _close(mut self):
        var depth = len(self.stack)
        _ = self.stack.pop()
        _ = self.raw_names.pop()
        while len(self.ns) > 0 and self.ns[len(self.ns) - 1].depth >= depth:
            _ = self.ns.pop()

    def _end_tag(mut self) raises:
        """At "</"."""
        var name_at = self.pos + 2
        self.pos += 2
        var qname = self._name()
        self._skip_ws()
        if self._byte(self.pos) != 0x3E:
            raise self._error("not well-formed (invalid token)", self.pos)
        if len(self.stack) == 0 or self.raw_names[len(self.raw_names) - 1] != qname:
            raise self._error("mismatched tag", name_at)
        self.pos += 1
        self._close()

    def _text(mut self, mut out: List[UInt8]) raises:
        """Character data up to the next '<', appended to `out`."""
        while self.pos < self._len():
            var c = self._byte(self.pos)
            if c == 0x3C:
                return
            if c == 0x26:
                self._reference(out)
                continue
            if c == 0x5D and self._starts(self.pos, "]]>"):
                raise self._error("not well-formed (invalid token)", self.pos + 2)
            if c == 0x0D:
                out.append(0x0A)
                self.pos += 1
                if self._byte(self.pos) == 0x0A:
                    self.pos += 1
                continue
            if c < 0x20 and c != 0x09 and c != 0x0A:
                raise self._error("not well-formed (invalid token)", self.pos)
            out.append(UInt8(c))
            self.pos += 1

    # ----------------------------------------------------------------- document
    def parse(mut self) raises:
        if self._byte(0) == 0xEF and self._byte(1) == 0xBB and self._byte(2) == 0xBF:
            self.pos = 3
        var data = List[UInt8]()       # pending character data of the open element
        var data_owner = -1            # element whose text the data belongs to, -1 = none
        while True:
            if self.pos >= self._len():
                if len(self.stack) > 0 or self.root < 0:
                    raise self._error("no element found", self._len())
                return
            var c = self._byte(self.pos)
            if c != 0x3C:
                if len(self.stack) == 0:
                    if _is_ws(c):
                        self.pos += 1
                        continue
                    if self.root >= 0:
                        raise self._error("junk after document element", self.pos)
                    raise self._error("not well-formed (invalid token)", self.pos)
                var top = self.stack[len(self.stack) - 1]
                if data_owner == top:
                    self._text(data)
                else:
                    var ignored = List[UInt8]()
                    self._text(ignored)
                continue
            if self._starts(self.pos, "<!--"):
                self._skip_comment()
                continue
            if self._starts(self.pos, "<?"):
                self._skip_pi()
                continue
            if self._starts(self.pos, "<![CDATA["):
                if len(self.stack) == 0:
                    raise self._error("not well-formed (invalid token)", self.pos)
                var end = self._find("]]>", self.pos + 9)
                if end < 0:
                    raise self._error("unclosed CDATA section", self.pos)
                var top = self.stack[len(self.stack) - 1]
                if data_owner == top:
                    var body = self._slice(self.pos + 9, end)
                    var bs = body.as_bytes()
                    var k = 0
                    while k < len(bs):
                        if Int(bs[k]) == 0x0D:
                            data.append(0x0A)
                            if k + 1 < len(bs) and Int(bs[k + 1]) == 0x0A:
                                k += 1
                        else:
                            data.append(bs[k])
                        k += 1
                self.pos = end + 3
                continue
            if self._starts(self.pos, "<!DOCTYPE"):
                if self.root >= 0 or len(self.stack) > 0:
                    raise self._error("not well-formed (invalid token)", self.pos)
                self._skip_doctype()
                continue
            if self._starts(self.pos, "</"):
                if len(self.stack) == 0:
                    raise self._error("not well-formed (invalid token)", self.pos)
                var top = self.stack[len(self.stack) - 1]
                if data_owner == top:
                    self.nodes[top].text = String(unsafe_from_utf8=Span(data))
                    data = List[UInt8]()
                    data_owner = -1
                self._end_tag()
                continue
            # start tag
            if len(self.stack) == 0 and self.root >= 0:
                raise self._error("junk after document element", self.pos)
            if len(self.stack) > 0:
                var top = self.stack[len(self.stack) - 1]
                if data_owner == top:
                    self.nodes[top].text = String(unsafe_from_utf8=Span(data))
            var depth_before = len(self.stack)
            self._start_tag()
            # A new open element collects its own text until its first child or its end;
            # after a self-closing one, the data that follows is its tail (not kept).
            data = List[UInt8]()
            if len(self.stack) > depth_before:
                data_owner = self.stack[len(self.stack) - 1]
            else:
                data_owner = -1

    # ----------------------------------------------------------------- ElementTree reads
    def _is_atom(self, idx: Int, local: String) -> Bool:
        return self.nodes[idx].ns == ATOM_NS and self.nodes[idx].local == local

    def find(self, parent: Int, local: String) -> Int:
        """Element.find("{atom}local"): first direct child, -1 when none."""
        for c in self.nodes[parent].children:
            if self._is_atom(c, local):
                return c
        return -1

    def findall(self, parent: Int, local: String) -> List[Int]:
        var out = List[Int]()
        for c in self.nodes[parent].children:
            if self._is_atom(c, local):
                out.append(c)
        return out^

    def findtext(self, parent: Int, local: String) -> Optional[String]:
        var c = self.find(parent, local)
        if c < 0:
            return None
        return self.nodes[c].text


def _collapse(s: String) -> String:
    return String(" ").join(py_split_ws(s))


def _after_last_slash(s: String) -> String:
    """s.rsplit("/", 1)[-1]."""
    var k = s.rfind("/")
    if k < 0:
        return s
    return String(s[byte=k + 1:])


def parse_arxiv_atom(xml: String) raises -> List[ArxivEntry]:
    """search_arxiv_raw's entries. Raises "ParseError: <msg>" (Python's
    `f"{type(e).__name__}: {e}"`)."""
    var p = _Parser(xml)
    p.parse()
    var out = List[ArxivEntry]()
    for e in p.findall(p.root, "entry"):
        var id_text = p.findtext(e, "id")
        var id_str = String()
        var id_null = True
        if id_text:
            id_str = id_text.value()
            id_null = False
        var title = p.findtext(e, "title")
        var published = p.findtext(e, "published")
        var summary = p.findtext(e, "summary")
        var authors = List[String]()
        var nulls = List[Bool]()
        for a in p.findall(e, "author"):
            if len(authors) >= MAX_AUTHORS:
                break
            var name = p.findtext(a, "name")
            if name:
                authors.append(name.value())
                nulls.append(False)
            else:
                authors.append(String())
                nulls.append(True)
        out.append(ArxivEntry(
            _after_last_slash(id_str),
            _collapse(title.value() if title else String()),
            cp_head(published.value() if published else String(), 10),
            authors^,
            nulls^,
            cp_head(_collapse(summary.value() if summary else String()), SUMMARY_CHARS),
            id_str,
            id_null,
        ))
    return out^


def arxiv_entries_json(entries: List[ArxivEntry]) -> String:
    """json.dumps(entries, indent=1, ensure_ascii=False), keys id, title, published, authors,
    summary, url (None as null)."""
    var w = JsonWriter()
    w.begin_array()
    for e in entries:
        w.begin_object()
        w.kv_string("id", e.id)
        w.kv_string("title", e.title)
        w.kv_string("published", e.published)
        w.key("authors")
        w.begin_array()
        for i in range(len(e.authors)):
            if e.authors_null[i]:
                w.null()
            else:
                w.string(e.authors[i])
        w.end_array()
        w.kv_string("summary", e.summary)
        if e.url_null:
            w.kv_null("url")
        else:
            w.kv_string("url", e.url)
        w.end_object()
    w.end_array()
    try:
        return py_json_dumps(parse_json(w.finish()), False, 1)
    except:
        return "[]"   # unreachable: JsonWriter output always parses
