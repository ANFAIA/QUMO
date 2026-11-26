"""Native HTTP/1.1 + TLS client (OpenSSL 3 loaded at run time): libc bindings, sockets, TLS,
URLs, HTTP/1.1, the URL guard and SSE.

This is the one implementation. `agents/mojo/qumo_agent/native/libc.mojo` and
`agents/mojo/qumo_agent/net/fdutil.mojo` re-export `libc` and `fdutil` from here. The modules
that raise coded errors (`url`, `ossl`, `tls`, `socket`, `http`, `guard`, `sse`) are mirrored
into `agents/mojo/qumo_agent/net/` with the agent harness's codes (QA-E104 / QA-E105 for
QM-E959 / QM-E951): edit them here, then run
`QUMO_UPDATE_NET=1 .venv/bin/mojo run -I . tests/test_net_mirror.mojo`. That test (part of
`make test`) fails when a mirror differs from what this package generates.
"""
