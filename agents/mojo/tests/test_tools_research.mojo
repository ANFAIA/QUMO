"""Tests for tools/research.mojo (DESIGN.md 2.8.4, 8.2).

Pure parts against Python goldens (tests/fixtures/tools/research_net_golden.json and
research_golden.json, written by gen_research_net_golden.py and gen_research_golden.py with
agents/.venv): the PyPI and GitHub JSON, the arXiv query URL, module_watchlist. Network paths
go through tests/support/http_fixture_server on 127.0.0.1 (the tools' base URLs are fields
of ResearchTools): request paths and headers, redirects, caps, error texts, the 16 MiB
sdk_releases path (X6) and secret redaction. https itself is covered by the live checks.
"""

from std.os import setenv, unsetenv
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, JsonWriter, parse_json
from qumo_agent.native.libc import CStr, c_rename
from qumo_agent.native.spawn import (
    SpawnSpec, Child, spawn, run_capture, minimal_child_env, environ_list, ignore_sigpipe,
    STDIO_NULL, STDIO_PIPE, STDIO_INHERIT,
)
from qumo_agent.native.fs import make_temp_dir, make_dirs, write_text, read_text_lossy, list_dir
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.rand import uuid4_hex
from std.os import stat
from qumo_agent.net.http import HttpClient
from qumo_agent.tools.config import Modules, LocalToolsCatalog
from qumo_agent.tools.research import (
    ResearchTools, research_get, arxiv_query_url, pypi_info_json, github_releases_json,
    is_github_repo, py_net_error, research_get_prefixed,
)
from qumo_agent.util.pyfmt import py_json_dumps

comptime FX = "agents/mojo/tests/fixtures/tools"
comptime TOKEN = "ghp_qa_t2_token_value_0123456789"


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# http_fixture_server helper (each test that needs the server carries a copy)
# ---------------------------------------------------------------------------
comptime _FIXTURE_SRC = "agents/mojo/tests/support/http_fixture_server.mojo"
comptime _FIXTURE_BIN = "build/test-fixtures/http_fixture_server"


def _mtime(path: String) raises -> Int:
    var st = stat(path)
    return Int(st.st_mtimespec.tv_sec) * 1000000000 + Int(st.st_mtimespec.tv_nsec)


def _newest_source() raises -> Int:
    var newest = _mtime(_FIXTURE_SRC)
    newest = max(newest, _mtime("qumo/ai/json.mojo"))
    var dirs: List[String] = ["agents/mojo/qumo_agent/net", "agents/mojo/qumo_agent/native"]
    for d in dirs:
        for name in list_dir(d):
            if name.endswith(".mojo"):
                newest = max(newest, _mtime(d + "/" + name))
    return newest


def fixture_binary() raises -> String:
    """Builds build/test-fixtures/http_fixture_server unless it is newer than its sources."""
    if exists(_FIXTURE_BIN) and _mtime(_FIXTURE_BIN) > _newest_source():
        return _FIXTURE_BIN
    make_dirs("build/test-fixtures")
    var tmp = _FIXTURE_BIN + ".tmp-" + uuid4_hex()
    var argv: List[String] = [".venv/bin/mojo", "build", "-I", ".", "-I", "agents/mojo", "-o", tmp, _FIXTURE_SRC]
    var r = run_capture(SpawnSpec(".venv/bin/mojo", argv^, environ_list(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, False),
                        600000, 1 << 16, "")
    if r.exit_code != 0:
        raise Error("building the fixture server failed: " + r.stdout + r.stderr)
    var a = CStr(tmp)
    var b = CStr(_FIXTURE_BIN)
    _ = c_rename(a.addr(), b.addr())
    a.keep()
    b.keep()
    return _FIXTURE_BIN


struct Fixture(Movable):
    var child: Child
    var dir: String
    var port: Int

    def __init__(out self, script: String) raises:
        var bin = fixture_binary()
        self.dir = make_temp_dir("qumo-research-fixture-")
        write_text(self.dir + "/script.json", script)
        var argv: List[String] = [bin, "--script", self.dir + "/script.json", "--port-file", self.dir + "/port",
                                  "--log", self.dir + "/log.jsonl"]
        self.child = spawn(SpawnSpec(bin, argv^, minimal_child_env(), "", STDIO_NULL, STDIO_INHERIT, STDIO_INHERIT, True))
        var deadline = monotonic_ms() + 30000
        while not exists(self.dir + "/port"):
            if monotonic_ms() > deadline:
                raise Error("the fixture server did not write its port")
            sleep_ms(10)
        self.port = Int(String(read_text_lossy(self.dir + "/port").strip()))

    def url(self, path: String) -> String:
        return "http://127.0.0.1:" + String(self.port) + path

    def prefix(self) -> String:
        return "http://127.0.0.1:" + String(self.port) + "/"

    def log(self) raises -> List[JsonValue]:
        var out = List[JsonValue]()
        if not exists(self.dir + "/log.jsonl"):
            return out^
        for line in read_text_lossy(self.dir + "/log.jsonl").split("\n"):
            if String(line).byte_length() > 0:
                out.append(parse_json(String(line)))
        return out^

    def finish(mut self) raises -> Int:
        var deadline = monotonic_ms() + 5000
        while monotonic_ms() < deadline:
            var code = self.child.poll_exit()
            if code:
                return code.value()
            sleep_ms(10)
        return self.child.terminate(200)


# One scripted response as JSON text (the fixture server's format).
def resp(path: String, status: Int, reason: String, body: String, headers: List[List[String]] = List[List[String]]()) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.key("match")
    w.begin_object()
    w.kv_string("path", path)
    w.end_object()
    w.kv_int("status", status)
    w.kv_string("reason", reason)
    w.key("headers")
    w.begin_object()
    for h in headers:
        w.kv_string(h[0], h[1])
    w.end_object()
    w.kv_string("body", body)
    w.end_object()
    return w.finish()


def script(responses: List[String]) -> String:
    return "{\"responses\": [" + String(",").join(responses) + "]}"


def make_tools(arxiv: String, pypi: String, github: String, plain: String) raises -> ResearchTools:
    var t = ResearchTools(HttpClient(None), Modules("."), LocalToolsCatalog("."))
    t.arxiv_api = arxiv
    t.pypi_base = pypi
    t.github_api = github
    t.plain_prefix = plain
    return t^


def atom_case(golden: JsonValue, name: String) raises -> JsonValue:
    for i in range(len(golden)):
        if golden.at(i).get_string("name", "") == name:
            return golden.at(i)
    raise Error("no atom case " + name)


def pad_doc(doc: String, n: Int) -> String:
    """The PyPI document with an `info.description` of n bytes (never part of the result)."""
    var pad = String("x") * n
    return doc.replace("\"info\":{", "\"info\":{\"description\":\"" + pad + "\",")


def query_of(url: String) -> String:
    var i = url.find("/api/query")
    return String(url[byte=i + 10:url.byte_length()])


def test_pure() raises:
    var net = parse_json(read_text(FX + "/research_net_golden.json"))
    var pypi = net.get("pypi")
    assert_true(len(pypi) >= 8)
    var ok_cases = 0
    for i in range(len(pypi)):
        var c = pypi.at(i)
        var name = c.get_string("name", "")
        var expected = c.get_string("expected", "")
        var got: String
        try:
            got = pypi_info_json(c.get_string("package", ""), json_text(c.get("doc")))
        except e:
            # the tool wraps the raised "<Type>: <message>" into the error result
            var w = JsonWriter()
            w.begin_object()
            w.kv_string("error", "PyPI lookup failed for " + c.get_string("package", "") + ": " + String(e))
            w.end_object()
            got = pretty(w.finish())
        assert_equal(got, expected, "pypi case " + name)
        ok_cases += 1
    assert_equal(ok_cases, len(pypi))
    # github: the documents do not depend on the count
    var gh = net.get("github")
    assert_true(len(gh) >= 4)
    for i in range(len(gh)):
        var c = gh.at(i)
        assert_equal(github_releases_json(json_text(c.get("doc"))), c.get_string("expected", ""),
                     "github case " + c.get_string("name", ""))
    # the arXiv query URL, byte for byte as urllib.parse.urlencode builds it
    var urls = net.get("arxiv_urls")
    assert_true(len(urls) >= 5)
    for i in range(len(urls)):
        var c = urls.at(i)
        var cats = List[String]()
        var cv = c.get("categories")
        for k in range(len(cv)):
            cats.append(cv.at(k).as_string())
        assert_equal(arxiv_query_url(c.get_string("query", ""), cats, c.get_int("max_results", 0)),
                     c.get_string("url", ""), "arxiv url " + String(i))
    # module_watchlist for every module and the unknown ones
    var tools = make_tools("", "", "", "")
    var wl = net.get("watchlist")
    assert_true(len(wl) >= 10)
    for i in range(len(wl)):
        var c = wl.at(i)
        assert_equal(tools.module_watchlist(c.get_string("module", "")), c.get_string("expected", ""),
                     "watchlist " + c.get_string("module", ""))
    # repo names (X7: ASCII only)
    assert_true(is_github_repo("Qiskit/qiskit"))
    assert_true(is_github_repo("a_b/c.d-e"))
    assert_false(is_github_repo("a/b/c"))
    assert_false(is_github_repo("abc"))
    assert_false(is_github_repo("/x"))
    assert_false(is_github_repo("x/"))
    assert_false(is_github_repo("a b/c"))
    assert_false(is_github_repo("é/x"))
    assert_false(is_github_repo("a/b\n"))
    # Python-style error texts
    assert_equal(py_net_error(Error("QA-E104: read timed out")), "TimeoutError: timed out")
    assert_equal(py_net_error(Error("QA-E104: write timed out")), "TimeoutError: timed out")
    assert_equal(py_net_error(Error("QA-E104: 127.0.0.1:9: connect timed out")), "URLError: <urlopen error timed out>")
    assert_true(py_net_error(Error("QA-E104: body truncated (3 bytes missing)")).startswith("IncompleteRead: "))
    assert_equal(py_net_error(Error("QA-E104: refusing redirect from https to http")),
                 "URLError: <urlopen error refusing redirect from https to http>")


def json_text(v: JsonValue) -> String:
    return v.to_json()


def pretty(compact: String) raises -> String:
    return py_json_dumps(parse_json(compact), False, 1)


def test_unreachable() raises:
    """Nothing listens on port 1: a connection error is a URLError result, not a raise."""
    var tools = make_tools("http://127.0.0.1:1/api/query", "http://127.0.0.1:1/pypi/", "http://127.0.0.1:1/repos/",
                           "http://127.0.0.1:1/")
    var r = tools.search_arxiv("x", "", 5)
    assert_true(r.startswith("{\n \"error\": \"arXiv query failed: URLError: <urlopen error "), r)
    var p = tools.sdk_releases("qiskit")
    assert_true(p.startswith("{\n \"error\": \"PyPI lookup failed for qiskit: URLError: <urlopen error "), p)
    assert_equal(tools.github_release_notes("a/b", 1), "{\n \"error\": \"GitHub lookup failed for a/b: URLError\"\n}")
    var f = tools.fetch_page("http://127.0.0.1:1/x")
    assert_true(f.startswith("{\n \"error\": \"fetch failed: URLError: <urlopen error "), f)
    # fetch_page only reads https pages (the test seam allows the loopback prefix above)
    var plain_tools = make_tools("", "", "", "")
    var plain = plain_tools.fetch_page("http://example.org/")
    assert_equal(plain, "{\n \"error\": \"fetch failed: ValueError: only https URLs are fetched\"\n}")


def test_refusals() raises:
    var http = HttpClient(None)
    var none = List[String]()
    for u in ["http://example.com/x", "ftp://example.com/x", "HTTPS://pypi.org/x", "http://export.arxiv.org.evil.example/x", "/relative", ""]:
        var msg = String()
        try:
            _ = research_get(http, String(u), none, none, 1000, "ua", 5)
        except e:
            msg = String(e)
        assert_equal(msg, "ValueError: only https URLs are fetched", u)
    assert_equal(http.requests, 0)


def test_network() raises:
    var net = parse_json(read_text(FX + "/research_net_golden.json"))
    var atoms = parse_json(read_text(FX + "/research_golden.json")).get("atom")
    var arxiv_like = atom_case(atoms, "arxiv_like")
    var pypi = net.get("pypi")
    var gh = net.get("github")
    var qiskit = pypi.at(0)
    assert_equal(qiskit.get_string("name", ""), "qiskit_like")
    var missing_info = pypi.at(6)
    assert_equal(missing_info.get_string("name", ""), "missing_info")
    var gh_two = gh.at(0)
    assert_equal(gh_two.get_string("name", ""), "two_releases")
    var other_atom = atom_case(atoms, "err_mismatched")
    var html = parse_json(read_text(FX + "/research_golden.json")).get("html")
    var basic = html.at(0)
    assert_equal(basic.get_string("name", ""), "basic")

    # A second server plays "another host": the server is on 127.0.0.1, "localhost" is another
    # host for the redirect rules.
    var fxb = Fixture(script([resp("/moved", 200, "OK", "[]")]))
    var rs = List[String]()
    # 0 search_arxiv ok
    rs.append(resp("/api/query", 200, "OK", arxiv_like.get_string("xml", ""), [["Content-Type", "application/atom+xml"]]))
    # 1 search_arxiv HTTP 404
    rs.append(resp("/api/query", 404, "Not Found", "nope"))
    # 2 search_arxiv bad XML
    rs.append(resp("/api/query", 200, "OK", other_atom.get_string("xml", "")))
    # 3 sdk_releases ok
    rs.append(resp("/pypi/qiskit/json", 200, "OK", qiskit.get("doc").to_json()))
    # 4 sdk_releases, a 3 MiB document (X6: beyond the 400,000-byte cap of the other GETs)
    rs.append(resp("/pypi/qiskit/json", 200, "OK", pad_doc(qiskit.get("doc").to_json(), 3 * 1024 * 1024)))
    # 5 sdk_releases, document without info
    rs.append(resp("/pypi/noinfo/json", 200, "OK", missing_info.get("doc").to_json()))
    # 6 sdk_releases 404
    rs.append(resp("/pypi/nope/json", 404, "Not Found", "{}"))
    # 7 sdk_releases not JSON
    rs.append(resp("/pypi/bad/json", 200, "OK", "<html>not json</html>"))
    # 8 github ok with token
    rs.append(resp("/repos/Qiskit/qiskit/releases", 200, "OK", gh_two.get("doc").to_json()))
    # 9 github, redirect to another host (localhost)
    rs.append(resp("/repos/a/b/releases", 302, "Found", "", [["Location", "http://localhost:" + String(fxb.port) + "/moved"]]))
    # 10 github 403
    rs.append(resp("/repos/a/b/releases", 403, "rate limit exceeded", "{}"))
    # 11 fetch_page with a redirect
    rs.append(resp("/page", 301, "Moved Permanently", "", [["Location", "/page2"]]))
    # 12 the page
    rs.append(resp("/page2", 200, "OK", basic.get_string("html", "")))
    # 13 a reason line that contains the token
    rs.append(resp("/leak", 500, "Oops " + TOKEN, "x"))
    # 14 a 600,000-byte page: only the first 400,000 bytes are read
    rs.append(resp("/big", 200, "OK", String("y") * 600000))
    # 15 search_arxiv for a module: its categories reach the query
    rs.append(resp("/api/query", 200, "OK", "<feed xmlns=\"http://www.w3.org/2005/Atom\"/>"))

    var fx = Fixture(script(rs))
    _ = setenv("GITHUB_TOKEN", TOKEN)
    var tools = make_tools(fx.url("/api/query"), fx.prefix() + "pypi/", fx.prefix() + "repos/", fx.prefix())

    # search_arxiv
    assert_equal(tools.search_arxiv("routing", "", 10), arxiv_like.get_string("expected", ""))
    var e404 = tools.search_arxiv("routing", "", 10)
    assert_equal(e404, "{\n \"error\": \"arXiv query failed: HTTPError: HTTP Error 404: Not Found\"\n}")
    var bad = tools.search_arxiv("routing", "", 10)
    assert_equal(bad, arxiv_like_error(other_atom.get_string("expected", "")))
    # an unknown module never reaches the network (X4)
    var um = tools.search_arxiv("x", "nope", 10)
    assert_true(um.startswith("{\n \"error\": \"unknown module 'nope'; choose one of ir, sim,"))
    assert_equal(len(fx.log()), 3)

    # sdk_releases
    assert_equal(tools.sdk_releases("qiskit"), qiskit.get_string("expected", ""))
    assert_equal(tools.sdk_releases("qiskit"), qiskit.get_string("expected", ""))
    assert_equal(tools.sdk_releases("noinfo"), missing_info.get_string("expected", ""))
    assert_equal(tools.sdk_releases("nope"), "{\n \"error\": \"PyPI lookup failed for nope: HTTPError: HTTP Error 404: Not Found\"\n}")
    var notjson = tools.sdk_releases("bad")
    assert_true(notjson.startswith("{\n \"error\": \"PyPI lookup failed for bad: JSONDecodeError: "), notjson)

    # github_release_notes
    assert_equal(tools.github_release_notes("Qiskit/qiskit", 3), gh_two.get_string("expected", ""))
    assert_equal(tools.github_release_notes("a/b", 3), "[]")
    assert_equal(tools.github_release_notes("a/b", 3), "{\n \"error\": \"GitHub lookup failed for a/b: HTTPError\"\n}")
    assert_equal(tools.github_release_notes("a/b/c", 3), "{\n \"error\": \"repo must look like owner/name\"\n}")

    # fetch_page: redirect followed, tags stripped
    assert_equal(tools.fetch_page(fx.url("/page")), basic.get_string("expected", ""))
    var leak = tools.fetch_page(fx.url("/leak"))
    assert_false(leak.find(TOKEN) >= 0, leak)
    assert_true(leak.startswith("{\n \"error\": \"fetch failed: HTTPError: HTTP Error 500: Oops "), leak)
    # the 400,000-byte cap of the generic GET
    var http = HttpClient(None)
    var body = research_get_plain(http, fx.url("/big"), fx.prefix())
    assert_equal(len(body), 400000)
    assert_equal(tools.search_arxiv("gpu simulation", "gpu", 7), "[]")
    assert_equal(fx.finish(), 0)

    # What the server saw
    var log = fx.log()
    assert_equal(len(log), 16)
    assert_equal(log[15].get_string("path", ""),
                 "/api/query?search_query=%28all%3A%22gpu+simulation%22%29+AND+%28cat%3Aquant-ph+OR+cat%3Acs.DC%29&sortBy=submittedDate&sortOrder=descending&max_results=7")
    var expected_q = query_of(net.get("arxiv_urls").at(0).get_string("url", ""))
    assert_equal(log[0].get_string("path", ""), "/api/query" + expected_q)
    var h0 = log[0].get("headers")
    assert_equal(h0.get_string("user-agent", ""), "qumo-agents/0.1 (+https://github.com/ANFAIA/QUMO)")
    assert_equal(h0.get_string("accept-encoding", ""), "identity")
    assert_false(h0.has("authorization"))
    assert_equal(log[3].get_string("path", ""), "/pypi/qiskit/json")
    assert_equal(log[8].get_string("path", ""), "/repos/Qiskit/qiskit/releases?per_page=3")
    var h8 = log[8].get("headers")
    assert_equal(h8.get_string("accept", ""), "application/vnd.github+json")
    assert_equal(h8.get_string("authorization", ""), "Bearer " + TOKEN)
    assert_equal(h8.get_string("user-agent", ""), "qumo-agents/0.1 (+https://github.com/ANFAIA/QUMO)")
    # the cross-host redirect keeps the User-Agent and drops the token
    assert_equal(log[9].get_string("path", ""), "/repos/a/b/releases?per_page=3")
    assert_true(log[9].get("headers").has("authorization"))
    assert_equal(fxb.finish(), 0)
    var logb = fxb.log()
    assert_equal(len(logb), 1)
    assert_equal(logb[0].get_string("path", ""), "/moved")
    assert_equal(logb[0].get("headers").get_string("host", ""), "localhost:" + String(fxb.port))
    assert_false(logb[0].get("headers").has("authorization"))
    assert_equal(logb[0].get("headers").get_string("user-agent", ""), "qumo-agents/0.1 (+https://github.com/ANFAIA/QUMO)")
    _ = unsetenv("GITHUB_TOKEN")


def research_get_plain(mut http: HttpClient, url: String, prefix: String) raises -> List[UInt8]:
    """`research_get` with the fixture's plain prefix."""
    return research_get_prefixed(http, url, List[String](), List[String](), 400000, "ua", 5, prefix)


def arxiv_like_error(expected_parse_error: String) -> String:
    """Python's tool result for a feed whose parse fails with `expected_parse_error`."""
    return "{\n \"error\": \"arXiv query failed: " + expected_parse_error + "\"\n}"


def main() raises:
    ignore_sigpipe()
    test_pure()
    test_refusals()
    test_unreachable()
    test_network()
    print("test_tools_research OK")
