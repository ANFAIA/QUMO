# Native vendor clients (`qumo_hw`)

Mojo clients of hardware services. This is vendor code, so it is outside `qumo/`, and nothing under
`qumo/` or `cli/` imports it (`make check-neutrality`; [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md#4c-vendor-code-outside-qumo)).
The package imports `qumo.core`, `qumo.ir` and `qumo.net`.

| Module | Content | Environment variables it reads |
|---|---|---|
| `ibm_api` | IBM Quantum Runtime REST | `QISKIT_IBM_TOKEN`, `QISKIT_IBM_INSTANCE`, `QISKIT_IBM_URL`, else the account file (`QUMO_IBM_ACCOUNT_FILE`, default `~/.qiskit/qiskit-ibm.json`); `QUMO_IBM_API_URL`, `IAM_URL` (endpoints) |
| `ionq_api` | IonQ API v0.4 | `IONQ_API_KEY`; `IONQ_API_URL` |
| `quantinuum_api` | Quantinuum Nexus (H2) | tokens of `qnx login` in `~/.qnx/auth` (`QUMO_NEXUS_AUTH_DIR`); `QUMO_NEXUS_URL`, `NEXUS_DOMAIN` / `NEXUS_HOST`. It refuses to run if `NEXUS_PORT`, `NEXUS_PROTOCOL`, `NEXUS_TOKEN_PATH` or `NEXUS_CONFIG_FILE` is set |
| `speqtrum_api` | SpeQtrum | none: the Python provider passes the login and the URL |
| `qbraid_api` | qBraid Runtime API v2 | `QBRAID_API_KEY`; `QUMO_QBRAID_URL` |
| `pasqal_api` | Pasqal Cloud (analog programs) | `PASQAL_USERNAME` / `PASQAL_user`, `PASQAL_PASSWORD` / `PASQAL_pass`, `PASQAL_PROJECT_ID` / `PASQAL_project`; `QUMO_PASQAL_URL`, `QUMO_PASQAL_AUTH_URL` |
| `pasqal_pulser` | Pulser's abstract representation of a `qumo.ir.analog` program (`pulser_json`) and device specs (`AnalogDevice`, `check_against_device`) | none |
| `common` | `loopback_host`: 127.0.0.1, localhost, ::1 | none |

A URL given to a client must be https; plain http is accepted only for a loopback host
(`common.loopback_host`), which the tests use for local fake servers. The HTTP stack loads OpenSSL 3
at run time; `QUMO_LIBSSL` and `QUMO_LIBCRYPTO` select the library.

## Building

The package is not on the import path of the core. A build that uses it adds `-I plugins/hardware/native`
(paths from the repository root):

```bash
make py        # build/qumo_py.so: every client above except Pasqal, for the Python plugins
.venv/bin/mojo run -I . -I plugins/hardware/native examples/hardware/pasqal_mis.mojo    # local simulation only
.venv/bin/mojo build -I . -I plugins/hardware/native -o build/qbraid_all examples/hardware/qbraid_all.mojo
```

In your own file: `from qumo_hw.ionq_api import ionq_submit`. Without the second `-I` the compiler
reports `unable to locate module 'qumo_hw'`.

The Python providers and their tests are described in [docs/HARDWARE_PLUGINS.md](../../../docs/HARDWARE_PLUGINS.md),
which also says how to add a client (section 9).
