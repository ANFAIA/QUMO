"""Research tools: new papers (arXiv), SDK releases (PyPI, GitHub) and plain pages
(DESIGN.md 2.8.4).

Ported from `qumo_agents/tools/research.py`. Every tool returns text for the model and never
raises: a network, HTTP or parse failure becomes `{"error": "..."}` (or, for `fetch_page`, the
same JSON), as in Python, and the cases Python lets crash (unknown `module`, X4) are error
results too. Results are `json.dumps(obj, indent=1, ensure_ascii=False)`.

`research_get` is the one HTTP GET: https only (plus `http://export.arxiv.org/`, which is what
the arXiv API is spoken over), urllib's redirect rules, a byte cap, TLS EOFs accepted as urllib
does. Its errors are raised with the text Python's `f"{type(e).__name__}: {e}"` would give,
so a tool can put them straight into its error result:

- `HTTPError: HTTP Error <code>: <reason>` for a non-2xx answer after redirects,
- `URLError: <urlopen error <detail>>` for DNS, connect and TLS failures,
- `TimeoutError: timed out` for a read or write that stalls,
- `IncompleteRead: ...` for a body that ends early,
- `ValueError: only https URLs are fetched` for a refused URL.

The detail after `URLError` is this port's own wording (it has no errno), not CPython's.
Error texts pass through `redact_all`, so a GitHub token never reaches the model.
"""

from std.os import getenv

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.net.http import (
    HttpClient, HttpRequest, HttpResponse, HttpOptions, REDIRECT_URLLIB,
)
from qumo_agent.net.url import quote, urlencode
from qumo_agent.tools.atom import parse_arxiv_atom, arxiv_entries_json
from qumo_agent.tools.config import LocalToolsCatalog, Modules
from qumo_agent.tools.html import strip_html
from qumo_agent.util.pyfmt import cp_head, py_json_dumps
from qumo_agent.util.redact import redact_all, secret_env_names

comptime ARXIV_API = "http://export.arxiv.org/api/query"
comptime ARXIV_PLAIN_PREFIX = "http://export.arxiv.org/"
comptime PYPI_BASE = "https://pypi.org/pypi/"
comptime GITHUB_API = "https://api.github.com/repos/"
# X6: sdk_releases reads whole PyPI JSON documents (qiskit's is several MiB).
comptime SDK_RELEASES_MAX_BYTES = 16 * 1024 * 1024
comptime RELEASES_KEPT = 8
comptime GITHUB_BODY_CHARS = 4000


# ---------------------------------------------------------------------------
# Python-style error texts
# ---------------------------------------------------------------------------
def _redact(text: String) -> String:
    return redact_all(text, secret_env_names(""))


def _drop_code(msg: String) -> String:
    """`QA-E104: x` -> `x`."""
    if msg.startswith("QA-E") and msg.byte_length() >= 9:
        var bs = msg.as_bytes()
        if Int(bs[7]) == 0x3A and Int(bs[8]) == 0x20:
            return String(msg[byte=9:msg.byte_length()])
    return msg


def py_net_error(e: Error) -> String:
    """`<Type>: <message>` as Python would report the failure of a urllib GET."""
    var msg = _drop_code(String(e))
    if msg.find("read timed out") >= 0 or msg.find("write timed out") >= 0:
        return "TimeoutError: timed out"
    if msg.startswith("body truncated"):
        return "IncompleteRead: " + msg
    if msg.startswith("invalid URL"):
        return "ValueError: " + msg
    if msg.startswith("response body larger than"):
        return "ValueError: " + msg
    if msg.find("timed out") >= 0:
        return "URLError: <urlopen error timed out>"
    return "URLError: <urlopen error " + msg + ">"


def _error_type(py_text: String) -> String:
    """`TimeoutError: timed out` -> `TimeoutError`."""
    var i = py_text.find(":")
    if i < 0:
        return py_text
    return String(py_text[byte=0:i])


def _json_text(v: String) -> String:
    """json.dumps(parse(v), indent=1, ensure_ascii=False) of JsonWriter output."""
    try:
        return py_json_dumps(parse_json(v), False, 1)
    except:
        return v   # unreachable: JsonWriter output always parses


def error_json(msg: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("error", _redact(msg))
    w.end_object()
    return _json_text(w.finish())


# ---------------------------------------------------------------------------
# HTTP GET
# ---------------------------------------------------------------------------
def research_get_prefixed(mut http: HttpClient, url: String, header_names: List[String], header_values: List[String],
                  max_bytes: Int, user_agent: String, timeout_s: Int, plain_prefix: String) raises -> List[UInt8]:
    """`research_get` with the plain-http prefix as a parameter (tests point it at a local
    fixture server; `research_get` fixes it to http://export.arxiv.org/)."""
    var plain = plain_prefix.byte_length() > 0 and url.startswith(plain_prefix)
    if not url.startswith("https://") and not plain:
        raise Error("ValueError: only https URLs are fetched")
    var names = List[String]()
    var values = List[String]()
    names.append("User-Agent")
    values.append(user_agent)
    for i in range(min(len(header_names), len(header_values))):
        # dict semantics: a caller header replaces User-Agent in place
        var replaced = False
        if header_names[i] == "User-Agent":
            values[0] = header_values[i]
            replaced = True
        if not replaced:
            names.append(header_names[i])
            values.append(header_values[i])
    var opts = HttpOptions(timeout_s * 1000, timeout_s * 1000, max_bytes, True, REDIRECT_URLLIB, 10, plain, True)
    var resp: HttpResponse
    try:
        resp = http.request(HttpRequest("GET", url, names^, values^, String()), opts)
    except e:
        raise Error(_redact(py_net_error(e)))
    if resp.status < 200 or resp.status >= 300:
        raise Error(_redact("HTTPError: HTTP Error " + String(resp.status) + ": " + resp.reason))
    return resp.body.copy()


def research_get(mut http: HttpClient, url: String, header_names: List[String], header_values: List[String],
                 max_bytes: Int, user_agent: String, timeout_s: Int) raises -> List[UInt8]:
    """GET `url` (https, or http://export.arxiv.org/) and return at most `max_bytes` of the
    body. Raises `<Type>: <message>` texts (see the module doc)."""
    return research_get_prefixed(http, url, header_names, header_values, max_bytes, user_agent, timeout_s, ARXIV_PLAIN_PREFIX)


def _lossy(body: List[UInt8]) -> String:
    return String(from_utf8_lossy=Span(body))


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------
def _clamp(v: Int, lo: Int, hi: Int) -> Int:
    return max(lo, min(v, hi))


def _arxiv_url(base: String, query: String, categories: List[String], max_results: Int) -> String:
    var terms: String
    if query.find(" ") >= 0:
        terms = "all:\"" + query + "\""
    else:
        terms = "all:" + query
    var cats = String()
    for i in range(len(categories)):
        if i > 0:
            cats += " OR "
        cats += "cat:" + categories[i]
    var q = terms
    if cats.byte_length() > 0:
        q = "(" + terms + ") AND (" + cats + ")"
    var names: List[String] = ["search_query", "sortBy", "sortOrder", "max_results"]
    var values: List[String] = [q, "submittedDate", "descending", String(_clamp(max_results, 1, 50))]
    return base + "?" + urlencode(names, values)


def arxiv_query_url(query: String, categories: List[String], max_results: Int) -> String:
    """The arXiv API URL `search_arxiv` fetches: newest first, `max_results` clamped to 1..50."""
    return _arxiv_url(ARXIV_API, query, categories, max_results)


# ---------------------------------------------------------------------------
# PyPI and GitHub JSON
# ---------------------------------------------------------------------------
def _kv_value(mut w: JsonWriter, key: String, v: JsonValue):
    """`key: <v>` with a missing value as null."""
    if v.is_defined():
        w.kv_raw(key, v.to_json())
    else:
        w.kv_null(key)


def _is_truthy_string(v: JsonValue) -> Bool:
    try:
        return v.is_string() and v.as_string().byte_length() > 0
    except:
        return False


def _ascii_lower(s: String) -> String:
    var bs = s.as_bytes()
    var out = List[UInt8](capacity=len(bs))
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c >= 0x41 and c <= 0x5A:
            c += 0x20
        out.append(UInt8(c))
    return String(from_utf8_lossy=Span(out))


def _matches_changelog_key(key: String) -> Bool:
    """re.search("change|release|news", key, re.I), ASCII case folding."""
    var k = _ascii_lower(key)
    return k.find("change") >= 0 or k.find("release") >= 0 or k.find("news") >= 0


def _json_error(e: Error) -> String:
    """`JSONDecodeError: <parser message>`; the parser's wording is this port's own."""
    var msg = String(e)
    var nl = msg.find("\n")
    if nl >= 0:
        var first = String(msg[byte=0:nl])
        msg = first^
    return "JSONDecodeError: " + _drop_code(msg)


def pypi_info_json(package: String, doc_text: String) raises -> String:
    """The `sdk_releases` object for a PyPI JSON document (`pypi_info_raw`, then
    `json.dumps(indent=1, ensure_ascii=False)`). Raises `<Type>: <message>` texts."""
    var data: JsonValue
    try:
        data = parse_json(doc_text)
    except e:
        raise Error(_json_error(e))
    if not data.is_object():
        raise Error("TypeError: PyPI answered a JSON " + data.kind_name() + ", not an object")
    var info = data.get("info")
    if not info.is_defined():
        raise Error("KeyError: 'info'")
    if not info.is_object():
        raise Error("AttributeError: 'info' is a " + info.kind_name() + ", not an object")
    # releases with at least one file: (version, date), stable sort by date descending
    var versions = List[String]()
    var dates = List[String]()
    var rels = data.get("releases")
    if rels.is_defined() and not rels.is_null():
        if not rels.is_object():
            raise Error("AttributeError: 'releases' is a " + rels.kind_name() + ", not an object")
        for i in range(len(rels)):
            var files = rels.at(i)
            if not files.is_array() or len(files) == 0:
                continue
            var first = files.at(0)
            if not first.is_object():
                raise Error("TypeError: release file entry is a " + first.kind_name() + ", not an object")
            var ut = first.get("upload_time")
            if not ut.is_defined():
                raise Error("KeyError: 'upload_time'")
            if not ut.is_string():
                raise Error("TypeError: 'upload_time' is a " + ut.kind_name() + ", not a string")
            var date = cp_head(ut.as_string(), 10)
            var v = rels.key_at(i)
            var j = len(dates)
            versions.append(v)
            dates.append(date)
            while j > 0 and dates[j - 1] < dates[j]:
                var td = dates[j - 1]
                dates[j - 1] = dates[j]
                dates[j] = td
                var tv = versions[j - 1]
                versions[j - 1] = versions[j]
                versions[j] = tv
                j -= 1
    var urls = info.get("project_urls")
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("package", package)
    _kv_value(w, "latest", info.get("version"))
    _kv_value(w, "summary", info.get("summary"))
    w.key("recent_releases")
    w.begin_array()
    for i in range(min(len(versions), RELEASES_KEPT)):
        w.begin_object()
        w.kv_string("version", versions[i])
        w.kv_string("date", dates[i])
        w.end_object()
    w.end_array()
    _kv_value(w, "requires_python", info.get("requires_python"))
    var src = String()
    var src_found = False
    var chg = String()
    var chg_found = False
    if urls.is_defined() and not urls.is_null():
        if not urls.is_object():
            raise Error("AttributeError: 'project_urls' is a " + urls.kind_name() + ", not an object")
        for i in range(len(urls)):
            var u = urls.at(i)
            if not src_found and u.is_string() and u.as_string().find("github.com") >= 0:
                src = u.to_json()
                src_found = True
            if not chg_found and _matches_changelog_key(urls.key_at(i)):
                chg = u.to_json()
                chg_found = True
    if src_found:
        w.kv_raw("source", src)
    else:
        w.kv_null("source")
    if chg_found:
        w.kv_raw("changelog", chg)
    else:
        w.kv_null("changelog")
    w.end_object()
    return _json_text(w.finish())


def _is_repo_char(c: Int) -> Bool:
    return (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F or c == 0x2E or c == 0x2D


def is_github_repo(repo: String) -> Bool:
    """X7: `[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+`, ASCII only."""
    var bs = repo.as_bytes()
    var slash = -1
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c == 0x2F:
            if slash >= 0:
                return False
            slash = i
        elif not _is_repo_char(c):
            return False
    return slash > 0 and slash < len(bs) - 1


def _string_or_empty(v: JsonValue, cap: Int) raises -> String:
    """`(v or "")[:cap]` for a string or missing value."""
    if not v.is_defined() or v.is_null():
        return String()
    if not v.is_string():
        raise Error("TypeError: expected a string, got " + v.kind_name())
    return cp_head(v.as_string(), cap)


def github_releases_json(doc_text: String) raises -> String:
    """The release list `github_release_notes` returns for a GitHub API document. Raises
    `<Type>: <message>` texts."""
    var rel: JsonValue
    try:
        rel = parse_json(doc_text)
    except e:
        raise Error(_json_error(e))
    if not rel.is_array():
        raise Error("AttributeError: GitHub answered a JSON " + rel.kind_name() + ", not a list")
    var w = JsonWriter()
    w.begin_array()
    for i in range(len(rel)):
        var r = rel.at(i)
        if not r.is_object():
            raise Error("AttributeError: release entry is a " + r.kind_name() + ", not an object")
        w.begin_object()
        _kv_value(w, "tag", r.get("tag_name"))
        w.kv_string("date", _string_or_empty(r.get("published_at"), 10))
        _kv_value(w, "name", r.get("name"))
        w.kv_string("body", _string_or_empty(r.get("body"), GITHUB_BODY_CHARS))
        _kv_value(w, "url", r.get("html_url"))
        w.end_object()
    w.end_array()
    return _json_text(w.finish())


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------
struct ResearchTools(Movable):
    var http: HttpClient
    var modules: Modules
    var catalog: LocalToolsCatalog
    # Where the requests go. The defaults are the real services; tests point them at a local
    # fixture server (and set `plain_prefix` so that its http URL is accepted).
    var arxiv_api: String
    var pypi_base: String
    var github_api: String
    var plain_prefix: String

    def __init__(out self, var http: HttpClient, modules: Modules, catalog: LocalToolsCatalog):
        self.http = http^
        self.modules = modules.copy()
        self.catalog = catalog.copy()
        self.arxiv_api = String(ARXIV_API)
        self.pypi_base = String(PYPI_BASE)
        self.github_api = String(GITHUB_API)
        self.plain_prefix = String(ARXIV_PLAIN_PREFIX)

    def _get(mut self, url: String, header_names: List[String], header_values: List[String], max_bytes: Int) raises -> List[UInt8]:
        return research_get_prefixed(self.http, url, header_names, header_values, max_bytes,
                             self.catalog.research_user_agent(), self.catalog.research_timeout_s(), self.plain_prefix)

    def search_arxiv(mut self, query: String, module: String, max_results: Int) -> String:
        var cats = List[String]()
        if module.byte_length() > 0:
            try:
                cats = self.modules.get(module).arxiv_categories.copy()
            except e:
                return error_json(String(e))
        else:
            cats.append("quant-ph")
        try:
            var body = self._get(_arxiv_url(self.arxiv_api, query, cats, max_results), List[String](), List[String](),
                                 self.catalog.research_max_bytes())
            return arxiv_entries_json(parse_arxiv_atom(_lossy(body)))
        except e:
            return error_json("arXiv query failed: " + String(e))

    def sdk_releases(mut self, package: String) -> String:
        try:
            var body = self._get(self.pypi_base + quote(package, "/") + "/json", List[String](), List[String](),
                                 SDK_RELEASES_MAX_BYTES)
            return pypi_info_json(package, _lossy(body))
        except e:
            return error_json("PyPI lookup failed for " + package + ": " + String(e))

    def github_release_notes(mut self, repo: String, count: Int) -> String:
        if not is_github_repo(repo):
            return error_json("repo must look like owner/name")
        var names: List[String] = ["Accept"]
        var values: List[String] = ["application/vnd.github+json"]
        var token = getenv("GITHUB_TOKEN", "")
        if token.byte_length() > 0:
            names.append("Authorization")
            values.append("Bearer " + token)
        var text: String
        try:
            var body = self._get(self.github_api + repo + "/releases?per_page=" + String(_clamp(count, 1, 10)),
                                 names, values, self.catalog.research_max_bytes())
            text = _lossy(body)
            # json.loads sits inside the same try in Python: a bad document is a lookup failure.
            _ = parse_json(text)
        except e:
            return error_json("GitHub lookup failed for " + repo + ": " + _error_type(String(e)))
        try:
            return github_releases_json(text)
        except e:
            return error_json("GitHub lookup failed for " + repo + ": " + _error_type(String(e)))

    def module_watchlist(self, module: String) -> String:
        try:
            var m = self.modules.get(module)
            var w = JsonWriter()
            w.begin_object()
            w.kv_string("module", m.name)
            w.kv_string("summary", m.summary)
            w.kv_string_list("sdks", m.sdks)
            w.kv_string_list("arxiv_categories", m.arxiv_categories)
            w.kv_string_list("arxiv_queries", m.arxiv_queries)
            w.kv_string_list("mcp_tools", m.mcp_tools)
            w.kv_string_list("docs", m.docs)
            w.end_object()
            return _json_text(w.finish())
        except e:
            return error_json(String(e))

    def fetch_page(mut self, url: String) -> String:
        try:
            var body = self._get(url, List[String](), List[String](), self.catalog.research_max_bytes())
            return strip_html(_lossy(body))
        except e:
            return error_json("fetch failed: " + String(e))
