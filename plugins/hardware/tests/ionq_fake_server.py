"""Fake IonQ API v0.4 for the offline tests: plain HTTP on 127.0.0.1, one process.

Run as a script: prints the port on stdout, then serves until killed. Every request is
appended as one JSON line to the log file given as argv[1] (method, path, headers, body).
argv[2] is a JSON file with canned behaviour:

  {"job_status": "completed", "histogram": {"0": 0.5, "3": 0.5},
   "submit_status": 200, "submit_body": {...}, "backends": [...]}

The test suite starts it with subprocess, because the native client blocks the calling
thread inside the qumo_py extension while it waits for the response.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG = sys.argv[1]
CONF = sys.argv[2]


def conf() -> dict:
    with open(CONF, encoding="utf-8") as f:
        return json.load(f)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _record(self, body: bytes) -> None:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"method": self.command, "path": self.path,
                                "headers": {k.lower(): v for k, v in self.headers.items()},
                                "body": body.decode("utf-8", "replace")}) + "\n")

    def _send(self, status: int, obj, headers: dict | None = None) -> None:
        raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._record(b"")
        c = conf()
        if self.path == "/v0.4/backends":
            return self._send(200, c.get("backends", []))
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["v0.4", "jobs"]:
            seen = sum(1 for line in open(LOG, encoding="utf-8") if f'"path": "{self.path}"' in line)
            if seen <= c.get("status_fail_first", 0):
                return self._send(503, {"message": "try again"})
            job = {"id": parts[2], "status": c.get("job_status", "completed")}
            job.update(c.get("job_extra", {}))
            return self._send(c.get("status_code", 200), job)
        if len(parts) == 5 and parts[:2] == ["v0.4", "jobs"] and parts[3:] == ["results", "probabilities"]:
            return self._send(200, c.get("histogram", {}))
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        self._record(body)
        c = conf()
        if self.path == "/v0.4/jobs":
            body = c.get("submit_body", {"id": "0a1b2c3d-0000-4000-8000-000000000042", "status": "submitted"})
            if "submit_raw" in c:
                body = c["submit_raw"].encode("utf-8")
            return self._send(c.get("submit_status", 200), body, c.get("submit_headers"))
        return self._send(404, {"error": "not found"})


def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
