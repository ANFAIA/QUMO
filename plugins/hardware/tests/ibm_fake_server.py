"""Fake IBM Cloud IAM + IBM Quantum Runtime REST API for the native-client tests.

argv: <log file> <state file>. Prints the port, then serves on 127.0.0.1 until killed.
Every request is logged as one JSON line (method, path, selected headers, body). The state
file (JSON) is read on every request:

  api_key                accepted IAM API key
  backends, configuration, properties    canned backend data (one backend: "ibm_testline_5")
  job_state: {"status", "reason"}        answer of GET /api/v1/jobs/<id>
  samples: ["0x0", ...]                  answer of GET /api/v1/jobs/<id>/results
  submit_status, submit_body             override for POST /api/v1/jobs
  require_user_agent: true               refuse requests without User-Agent (Cloudflare 1010)
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG, STATE = sys.argv[1], sys.argv[2]


def load() -> dict:
    with open(STATE, encoding="utf-8") as f:
        return json.load(f)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, status: int, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, method: str):
        url = urllib.parse.urlsplit(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        hdr = {k: self.headers.get(k) for k in ("Authorization", "Service-CRN", "IBM-API-Version", "User-Agent",
                                                   "Content-Type")}
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"method": method, "path": self.path, "headers": hdr,
                                "body": raw.decode("utf-8", "replace")}) + "\n")
        st = load()
        if st.get("require_user_agent") and not self.headers.get("User-Agent"):
            return self._reply(403, {"title": "Error 1010: Access denied"})
        if method == "POST" and url.path == "/identity/token":
            form = urllib.parse.parse_qs(raw.decode("utf-8"))
            if form.get("grant_type") != ["urn:ibm:params:oauth:grant-type:apikey"] or \
                    form.get("apikey") != [st["api_key"]]:
                return self._reply(400, {"errorMessage": "Provided API key could not be found"})
            return self._reply(200, {"access_token": "iam-token-1", "token_type": "Bearer", "expires_in": 3600})
        if self.headers.get("Authorization") != "Bearer iam-token-1" or not self.headers.get("Service-CRN"):
            return self._reply(401, {"errors": [{"message": "unauthorized", "code": "1100"}]})
        p = url.path
        if method == "GET" and p == "/api/v1/backends":
            return self._reply(200, {"devices": st.get("backends", [])})
        if method == "GET" and p.startswith("/api/v1/backends/"):
            parts = p.split("/")
            if len(parts) == 6 and parts[4] == "ibm_testline_5":
                return self._reply(200, st[parts[5]])
            return self._reply(404, {"errors": [{"message": "backend not found"}]})
        if method == "POST" and p == "/api/v1/jobs":
            if st.get("submit_status"):
                return self._reply(st["submit_status"], st.get("submit_body", {}))
            return self._reply(200, {"id": "d3fakejob0001", "backend": "ibm_testline_5"})
        if method == "GET" and p == "/api/v1/jobs/d3fakejob0001":
            return self._reply(200, {"id": "d3fakejob0001", "state": st.get("job_state", {"status": "Completed"})})
        if method == "GET" and p == "/api/v1/jobs/d3fakejob0001/results":
            return self._reply(200, {"results": [{"data": {"c": {"samples": st.get("samples", []), "num_bits": 2}}}]})
        return self._reply(404, {"errors": [{"message": "not found"}]})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")


def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
