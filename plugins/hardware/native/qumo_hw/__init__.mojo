"""Native Mojo clients of hardware vendors and aggregators (vendor code, outside the core).

Nothing under `qumo/` imports this package (docs/ARCHITECTURE.md, "Vendor code";
`make check-neutrality`). It is on the import path only for the builds that need it
(`-I plugins/hardware/native`): the `qumo_py` extension (`make py`), which the Python plugins
of `plugins/hardware/qumo_hardware` call, and the hardware examples.

* `ionq_api`, `ibm_api`, `quantinuum_api`, `speqtrum_api`, `qbraid_api`, `pasqal_api` - one REST
  client per service, over QUMO's own HTTP/TLS client (`qumo.net`).
* `pasqal_pulser` - Pulser's abstract representation of a `qumo.ir.analog` program and the
  device specs Pasqal Cloud returns.
* `common` - what the clients share (`loopback_host`).
"""
