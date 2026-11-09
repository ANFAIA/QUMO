"""Fake SpeQtrum API for the native-client tests: plain HTTP on 127.0.0.1, its own process.

argv: <log file> <state file>. Prints the port, then serves until killed. Every request is
appended to the log as one JSON line (method, path, authorization, body). The state file is
a JSON object read on every request and rewritten when the server changes it:

  credentials: {username, api_key, audience}   accepted login claims
  devices, job                                 canned responses
  execute_http: [status, body] | null          override for POST /execute
  redirect_login_to: url | null                answer the login with a 302
  reject_next: n                               answer n authenticated requests with 401
  logins: n                                    token-<n> is the valid token

It runs in a separate process because the native client holds the calling thread (and the
GIL) inside the qumo_py extension while it waits for the response.
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG, STATE = sys.argv[1], sys.argv[2]
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"


def load() -> dict:
    with open(STATE, encoding="utf-8") as f:
        return json.load(f)


def save(state: dict) -> None:
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, status: int, payload=None, headers=None):
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        for k, v in (headers or {}).items():
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
            f.write(json.dumps({"method": method, "path": self.path, "authorization": self.headers.get("Authorization"),
                                "body": raw.decode("utf-8", "replace")}) + "\n")
        st = load()
        if method == "POST" and url.path == "/authorisation-tokens":
            if st.get("redirect_login_to"):
                return self._reply(302, {"message": "moved"}, {"Location": st["redirect_login_to"]})
            body = json.loads(raw or b"{}")
            assertion = str(body.get("assertion", ""))
            try:
                claims = json.loads(base64.urlsafe_b64decode(assertion + "=" * (-len(assertion) % 4)))
            except ValueError:
                return self._reply(400, {"message": "malformed assertion"})
            c = st["credentials"]
            ok = (body.get("grantType") == GRANT_TYPE and body.get("scope") == "user profile"
                  and claims.get("api_key") == c["api_key"] and claims.get("username") == c["username"]
                  and claims.get("audience") == c["audience"] and isinstance(claims.get("iat"), int))
            if not ok:
                return self._reply(401, {"message": "invalid credentials"})
            st["logins"] = st.get("logins", 0) + 1
            save(st)
            return self._reply(200, {"accessToken": f"token-{st['logins']}", "expiresIn": 3600, "tokenType": "Bearer"})
        if self.headers.get("Authorization") != f"Bearer token-{st.get('logins', 0)}":
            return self._reply(401, {"message": "missing or expired token"})
        if st.get("reject_next", 0) > 0:
            st["reject_next"] -= 1
            save(st)
            return self._reply(401, {"message": "token expired"})
        if method == "GET" and url.path == "/devices":
            return self._reply(200, {"items": st.get("devices", [])})
        if method == "POST" and url.path == "/execute":
            if st.get("execute_http"):
                return self._reply(*st["execute_http"])
            return self._reply(200, {"id": 42, "message": "queued"})
        if method == "GET" and url.path.startswith("/jobs/"):
            if url.path != "/jobs/42":
                return self._reply(404, {"message": "no such job"})
            return self._reply(200, st.get("job", {}))
        return self._reply(404, {"message": "unknown route"})

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
