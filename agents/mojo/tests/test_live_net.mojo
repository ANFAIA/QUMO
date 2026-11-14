"""Live check 8.3.1 and 8.3.2: `doctor` and TLS with the default trust store to every host the
agents talk to. Needs QUMO_LIVE=1, network access and OPENROUTER_API_KEY; SKIPs otherwise.

Run: make agents-mojo-live
"""

from std.os import getenv
from std.testing import assert_true

from std.memory import ArcPointer

from qumo_agent.net.tls import TlsContext
from qumo_agent.net.http import HttpClient, HttpOptions, HttpRequest, REDIRECT_URLLIB

from live import skip_reason, report_skip, run_live, fail_text

comptime NAME = "test_live_net"


def _is_set(name: String) -> Bool:
    return getenv(name, "").byte_length() > 0


def test_doctor() raises:
    """With ALPHAXIV_API_KEY and GITHUB_TOKEN set, doctor prints only [ok] lines."""
    var r = run_live(["doctor"], "", 120000)
    assert_true(r.exit_code == 0, fail_text(r) + r.stdout)
    assert_true(r.stdout.find("[ok]") >= 0, r.stdout)
    if _is_set("ALPHAXIV_API_KEY") and _is_set("GITHUB_TOKEN"):
        assert_true(r.stdout.find("[--]") < 0, "doctor reported advice:\n" + r.stdout)


def test_tls_hosts() raises:
    """A TLS handshake and one GET per host. The status may be 401/403/404 (no key is sent);
    the check is that the certificate verifies and an HTTP response comes back."""
    var urls: List[String] = [
        "https://openrouter.ai/api/v1/models/count",
        "https://pypi.org/pypi/qiskit/json",
        "https://api.github.com/zen",
        "https://export.arxiv.org/api/query?search_query=all:qubit&max_results=1",
        "https://api.alphaxiv.org/",
    ]
    var tls: Optional[ArcPointer[TlsContext]] = None
    var http = HttpClient(tls)
    for url in urls:
        var names: List[String] = ["user-agent", "accept"]
        var values: List[String] = ["qumo-agent-live-check", "*/*"]
        var resp = http.request(HttpRequest("GET", url, names^, values^, String()),
                                HttpOptions(15000, 20000, 1 << 20, True, REDIRECT_URLLIB, 10, False, True))
        assert_true(resp.status >= 100 and resp.status < 600, url + " answered " + String(resp.status))
        print("  " + url + " -> HTTP " + String(resp.status))


def main() raises:
    var why = skip_reason(NAME, ["OPENROUTER_API_KEY"])
    if why.byte_length() > 0:
        report_skip(NAME, why)
        return
    test_doctor()
    test_tls_hosts()
    print(NAME + " OK")
