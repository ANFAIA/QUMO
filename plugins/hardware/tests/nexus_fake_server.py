"""Fake Quantinuum Nexus API for the native-client tests: plain HTTP on 127.0.0.1, own process.

argv: <log file> <state file>. Prints the port, then serves until killed. Every request is
logged (method, path, Cookie header, body). The state file (JSON) is read on every request:

  refresh_token          accepted myqos_oat cookie
  access_token           currently valid myqos_id cookie
  refreshed_access       access token handed out by POST /auth/tokens/refresh ("" = refuse)
  projects: [ids]        existing projects named as requested
  job_status, job_message, error_detail, result_id
  backend_result         attributes of GET /api/results/v1beta3/<id>
  devices                answer of GET /api/v5/available_devices
  job_http: [status, body]   override for POST /api/jobs/v1beta3
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG, STATE = sys.argv[1], sys.argv[2]
JOB = "7a1e2b3c-0000-4000-8000-00000000j0b1".replace("j0b1", "0b10")


def load() -> dict:
    with open(STATE, encoding="utf-8") as f:
        return json.load(f)


def save(st: dict) -> None:
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(st, f)


def cookies(header: str | None) -> dict:
    out = {}
    for part in (header or "").split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            out[k] = v
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, status: int, payload=None, headers=None):
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        for k, v in (headers or []):
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, method: str):
        url = urllib.parse.urlsplit(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"method": method, "path": self.path, "cookie": self.headers.get("Cookie"),
                                "user_agent": self.headers.get("User-Agent"),
                                "body": raw.decode("utf-8", "replace")}) + "\n")
        st = load()
        ck = cookies(self.headers.get("Cookie"))
        if method == "POST" and url.path == "/auth/tokens/refresh":
            if ck.get("myqos_oat") != st["refresh_token"] or not st.get("refreshed_access"):
                return self._reply(401, {"detail": "refresh token invalid"})
            st["access_token"] = st["refreshed_access"]
            save(st)
            return self._reply(200, {}, [("Set-Cookie", "other=x; Path=/"),
                                         ("Set-Cookie", f"myqos_id={st['access_token']}; Path=/; HttpOnly; Secure")])
        if ck.get("myqos_oat") != st["refresh_token"] or ck.get("myqos_id") != st["access_token"]:
            return self._reply(401, {"detail": "not authenticated"})
        p = url.path
        if method == "GET" and p == "/api/projects/v1beta2":
            ids = st.get("projects", [])
            return self._reply(200, {"data": [{"id": i, "type": "project", "attributes": {}} for i in ids], "meta": {}})
        if method == "POST" and p == "/api/projects/v1beta2":
            st["projects"] = ["11111111-2222-4333-8444-555555555555"]
            save(st)
            return self._reply(201, {"data": {"id": st["projects"][0], "type": "project", "attributes": {}}})
        if method == "POST" and p == "/api/circuits/v1beta2":
            return self._reply(201, {"data": {"id": "c1c1c1c1-0000-4000-8000-000000000001", "type": "circuit"}})
        if method == "POST" and p == "/api/jobs/v1beta3":
            if st.get("job_http"):
                return self._reply(*st["job_http"])
            return self._reply(202, {"data": {"id": JOB, "type": "job"}})
        if method == "GET" and p == f"/api/jobs/v1beta3/{JOB}":
            status = {"status": st.get("job_status", "COMPLETED"), "message": st.get("job_message", "")}
            if st.get("error_detail"):
                status["error_detail"] = st["error_detail"]
            items = [{"program_id": "c1c1c1c1-0000-4000-8000-000000000001", "status": status}]
            if st.get("result_id"):
                items[0]["result_id"] = st["result_id"]
            return self._reply(200, {"data": {"id": JOB, "attributes": {"status": status,
                                                                       "definition": {"items": items}}}})
        if method == "GET" and p.startswith("/api/jobs/v1beta3/"):
            return self._reply(404, {"errors": [{"detail": "job not found"}]})
        if method == "GET" and p == f"/api/results/v1beta3/{st.get('result_id')}":
            return self._reply(200, {"data": {"id": st["result_id"], "attributes": st["backend_result"],
                                              "relationships": {}}})
        if method == "GET" and p == "/api/v5/available_devices":
            return self._reply(200, st.get("devices", []))
        return self._reply(404, {"errors": [{"detail": "not found"}]})

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
