# Hardware layer: one canonical implementation

Status: analysis and plan for section 6, item 3 of `docs/PROJECT_REVIEW.md` (finding A4). No provider
behaviour changed. The first step was judged not clearly safe and was not implemented (section 6).

## 1. Why there are two layers

Every provider exists twice:

* **Mojo**: `qumo/exec/{ibm,ionq,quantinuum,speqtrum}_api.mojo` (878, 493, 705 and 756 lines). These are
  exported to Python as `qumo_py` functions (`python/qumo_py.mojo`) and are what the CLI and the MCP server
  call.
* **Python**: `plugins/hardware/qumo_hardware/` (5,099 lines). The base classes (`ibm.py`, `ionq.py`,
  `quantinuum.py`, `speqtrum.py`) and the `*_native.py` subclasses. The registry (`registry.py`) makes the
  native subclass the default client: `NATIVE` maps `ibm`, `quantinuum`, `ionq` and `speqtrum` to a class
  that calls `qumo_py`. `PYTHON` maps them to the vendor-SDK or pure-Python client, used only on request
  (`--client python` or `QUMO_HW_CLIENT`).

Two facts make this less than a clean "native plus fallback" split:

1. **The native subclasses inherit from the Python classes.** `IBMNativeProvider(IBMProvider)`,
   `QuantinuumNativeProvider(QuantinuumProvider)` and `SpeQtrumNativeProvider(SpeQtrumProvider)` override only
   some methods. Everything they do not override (login, configuration, `public_options`, `_config`,
   confirmation, the job store) still runs the Python code.
2. **IonQ has no `*_native.py` twin.** `ionq.py` is itself the native client (`client = "native"`, calls
   `load_qumo_py()`). Its fallback is `ionq_sdk.py` (`IonQSdkProvider`, the vendor SDK). So IonQ is two layers
   only when the fallback is counted.

## 2. Inventory

### 2.1 Mojo (`qumo/exec`)

| Provider | Exported to `qumo_py` | Role |
|---|---|---|
| IBM | `ibm_backends`, `ibm_target`, `ibm_isa_payload`, `ibm_submit`, `ibm_status`, `ibm_result` | Backend list, target (calibration, fractional gates), ISA OpenQASM 3 payload, Runtime calls. The account is resolved inside Mojo by name (`ibm_account` is not exported) |
| IonQ | `ionq_request`, `ionq_counts`, `ionq_submit`, `ionq_status`, `ionq_result`, `ionq_backends` | OpenQASM 2 to IonQ `ionq.circuit.v1`, counts from probabilities, API calls with retry |
| Quantinuum | `quantinuum_native_circuit`, `quantinuum_submit`, `quantinuum_status`, `quantinuum_result`, `quantinuum_devices` | Native-gate circuit, Nexus session and project, submit/status/result, device list. `qtm_hqc` is imported into `qumo_py.mojo` but never registered, so Python cannot call it (see 4.3) |
| SpeQtrum | `speqtrum_devices`, `speqtrum_execute`, `speqtrum_status`, `speqtrum_result`, `speqtrum_unsafe_construct`, `speqtrum_decode_result`, `speqtrum_payload_document` | Calls with auth, payload document, unsafe-YAML check, result decoding. `checked_base` and `login_assertion` are not exported: login is done by Python |

### 2.2 Python (`plugins/hardware/qumo_hardware`)

| Module | Class or functions | Client | Notes |
|---|---|---|---|
| `ibm.py` (617) | `IBMProvider`, `target_from_qiskit`, `compile_for_ibm`, `fake_provider` | Python (qiskit-ibm-runtime) | Also the base class of the native client; `fake_*` backends are delegated here |
| `ibm_native.py` (~200) | `IBMNativeProvider` | native | Overrides `list_backends`, `target`, `prepare`, `validate`, `estimate_cost`, `submit`, `status`, `result` |
| `ionq.py` (~250) | `IonQProvider` | native (calls `qumo_py`) | The native client itself; `prepare` and `validate` run Mojo |
| `ionq_sdk.py` (~240) | `IonQSdkProvider` | Python (vendor SDK) | Fallback; overrides `prepare`, `validate`, `submit`, `status`, `result` |
| `quantinuum.py` (591) | `QuantinuumProvider`, `hqc_cost`, `to_native`, `hugr_program`, `native_gate_counts` | Python (pytket, qnx) | Also the base class of the native client; `hqc_cost` is used by the native client |
| `quantinuum_native.py` (~190) | `QuantinuumNativeProvider` | native | Overrides `list_backends`, `prepare`, `validate`, `estimate_cost`, `submit`, `status`, `result` |
| `speqtrum.py` (546) | `SpeQtrumProvider`, `unsafe_construct`, `decode_result`, `map_status`, `login_assertion`, `urllib_transport` | Python (urllib) | Also the base class of the native client |
| `speqtrum_native.py` (~130) | `SpeQtrumNativeProvider` | native | Overrides `list_backends`, `encode` (calls Mojo `speqtrum_unsafe_construct`), `submit`, `status`, `result` |
| `local.py` (320) | `LocalEmulatorProvider`, `parse_noise`, `inject_noise`, `sample_noisy` | Python plus `qumo_py` | Not a remote provider; no native twin needed |
| `base.py`, `runner.py`, `cli.py`, `registry.py`, `compile.py`, `jobs.py`, `errors.py`, `qasm_bridge.py` | shared types and orchestration | Python | Out of scope for the duplication; they should stay Python |

## 3. Feature matrix

`M` = implemented in Mojo and used by the native client. `P` = implemented in Python only. `B` = both
implementations, and the Python one is the one the native client calls for that feature. `-` = absent.

| Feature | IBM | IonQ | Quantinuum | SpeQtrum |
|---|---|---|---|---|
| List backends (offline) | M | M | P (device table) | P (device table) |
| List backends (online) | M | M | M (`quantinuum_devices`) | M (`speqtrum_devices`) |
| Target and calibration | M (`ibm_target`) | P (`target`) | P (`target`, `to_native`) | P (`target`) |
| Payload translation | M (`ibm_isa_payload`) | M (`ionq_request`) | M (`quantinuum_native_circuit`) | B (`speqtrum_unsafe_construct` in Mojo, `payload_document` and `encode` in Python) |
| Cost estimate | P (`estimate_cost`, qiskit) | P | B (Mojo gate counts; Python `hqc_cost` for the formula; Mojo `qtm_hqc` not exposed) | P |
| Submit with confirmation | M | M | M | M |
| Status mapping | M | M | M | M on the native client (`speqtrum_status`); `map_status` on the fallback |
| Result counts | M (`ibm_result`) | B (`ionq_counts` in Mojo, `counts_from_sdk` in Python) | M (`quantinuum_result`) | M on the native client (`speqtrum_result`); `decode_result` on the fallback |
| Input safety (unsafe YAML) | - | - | - | B (`unsafe_construct` in both) |
| Local fake server tests | yes | yes | yes | yes |
| Vendor-SDK fallback | yes (qiskit) | yes (`ionq_sdk`) | yes (pytket) | no |
| Python-only features | `compile_for_ibm`, `fake_*` backends | - | `hugr_program`, TKET comparison payloads | - |

Test coverage that already exists for both sides: `plugins/hardware/tests/test_ibm_native.py`,
`test_quantinuum_native.py`, `test_speqtrum_native.py`, `test_ionq.py`, `test_ibm.py`,
`test_quantinuum.py`, `test_speqtrum.py`, `test_clients.py` (client selection). Each provider has a fake
HTTP server that logs every request (`ibm_fake_server.py`, `ionq_fake_server.py`,
`nexus_fake_server.py`, `speqtrum_fake_server.py`).

## 4. Source of truth per provider

### 4.1 IBM
Default behaviour is Mojo for the wire protocol, the ISA payload, the target and the counts. Python is the
truth for `fake_*` backends and for `compile_for_ibm` (qiskit transpile). Cost estimates and circuit stats
come from qiskit on the fallback path only. The two layers can produce different physical circuits, so
payload parity cannot be byte-level: it must be defined on the fields IBM actually reads (qubits, basis,
shots, cost cap).

### 4.2 IonQ
Mojo builds the request and computes counts on the default path. `ionq_sdk.py` is the only Python
implementation of those steps and is used when the fallback is requested. Counts use different
input shapes (`ionq_counts` takes probabilities, `counts_from_sdk` takes the SDK dictionary), so the
parity test must compare the decoded counts, not the input.

### 4.3 Quantinuum
Two implementations of the same cost formula exist: `hqc_cost` in `quantinuum.py`, which
`quantinuum_native.estimate_cost` calls, and `qtm_hqc` in `qumo/exec/quantinuum_api.mojo`. `qtm_hqc` is
imported into `python/qumo_py.mojo` but never registered, so the native client cannot reach it. The two
formulas agree by inspection only. The gate translation also exists twice: `to_native` (pytket) in Python
and `quantinuum_native_circuit` in Mojo; the native path uses Mojo for the circuit counts. Exporting
`qtm_hqc` is a one-line Mojo change and removes the Python formula from the default path. This is the
clearest duplicate: the arithmetic has no SDK dependency, so it can live in Mojo while the fallback keeps
its own copy.

### 4.4 SpeQtrum
The native client decodes and maps in Mojo: `status` calls `speqtrum_status`, `result` calls
`speqtrum_result` (which decodes), `encode` calls `speqtrum_unsafe_construct`. The Python
`unsafe_construct`, `decode_result` and `map_status` in `speqtrum.py` are therefore used only by the
fallback client and by the base-class code the native subclass does not override. The duplicates are real
code, but on the default path only Mojo runs them. Redaction and loopback checks are implemented in both
layers (`_clean`, `_failure`, `checked_base`); they are security code and must stay identical until the
fallback is removed. Login is performed in Python and passed to Mojo as auth, because `login_assertion` and
`checked_base` are not exported.

## 5. Recommendation

Make **Mojo the one canonical implementation** of every wire call, payload translation, decoder and cost
formula. It is the layer the CLI, the MCP server and the native client already use, and it is the one the
binary ships.

Python keeps:
* the registry, the CLI and the shared types (`base.py`, `runner.py`, `cli.py`, `jobs.py`, `errors.py`);
* the vendor-SDK fallbacks (`ibm.py` qiskit paths, `ionq_sdk.py`, `quantinuum.py` pytket paths), which
  need the SDKs and cannot be replaced by Mojo. These stay as **explicit** fallbacks, not as a second
  implementation of the same protocol.

What goes away: the Python protocol code that only the fallback uses. For SpeQtrum that is
`speqtrum.py` `decode_result` and `map_status` (the native client already decodes in Mojo); `quantinuum.py`
`hqc_cost` goes once `qtm_hqc` is exported and the native client calls it. The Python base classes should
be split so that the native subclass no longer inherits protocol code it does not use.

Owner decision needed (section 7): whether the vendor-SDK fallbacks stay indefinitely, stay with a
deprecation warning, or go.

## 6. Migration order

Each step is small, testable and independently revertible.

1. **Parity harness (tests only).** For each provider, run the same scenario (one submit, one status, one
   result, one `estimate_cost`) through the native and the fallback clients against the existing fake
   server, and compare: HTTP method and path sequence; the semantic fields of each request body (for
   example IBM `shots` and the cost cap, IonQ `shots` and `noise.model`, SpeQtrum `device` and `shots`,
   Quantinuum `device` and `shots`); and the decoded counts for the same fixed input. Bodies are not
   byte-compared, because the layers build different circuits (see 4.1). Status: not implemented.
2. **Move arithmetic to Mojo.** Register `qtm_hqc` in `python/qumo_py.mojo` and make
   `quantinuum_native.estimate_cost` call it; keep `hqc_cost` for the pure-Python fallback only. A parity
   test asserts equality over a grid of counts and shot values. Behaviour is unchanged if the formulas
   agree; the test shows that they do. This needs `make py` before the Python tests run.
3. **Keep the fallback decoder faithful.** The native SpeQtrum path already decodes in Mojo
   (`speqtrum_native.result`). Add a test that runs the Python `decode_result` and the Mojo
   `speqtrum_decode_result` on the same fixtures, including the malformed cases in `test_speqtrum.py`, so
   the fallback stays correct until it is removed.
4. **Redaction and safety parity.** One test per layer that feeds the same hostile responses (YAML tags,
   a 500 that echoes the key, a redirect) and asserts the same error code and no secret in the message.
5. **Split the base classes.** Move protocol code out of the Python base classes that the native subclass
   inherits but does not use. Keep the public names (`IBMProvider`, `QuantinuumProvider`, `SpeQtrumProvider`)
   because `registry.py` and user code import them.
6. **Owner decision on the fallbacks** (section 7), then either keep, warn, or remove each fallback.
7. **Shared JSON and HTTP helper** for the Python side, only after step 5, so that it has one caller.

## 7. Parity tests that prove it

| Test | Proves |
|---|---|
| Parity harness per provider (step 1) | Same requests and counts from both clients on the same fixture |
| `hqc_cost` vs `qtm_hqc` over a grid (step 2) | Cost formula is one implementation or two that agree |
| Decoder fixtures run by both decoders (step 3) | SpeQtrum decoding is identical |
| Hostile-response tests run per layer (step 4) | Redaction and YAML refusal do not drift |
| Existing `test_clients.py` | Client selection and job store keep the client used |
| Existing native tests (`test_*_native.py`) | Default path still works after each step |

Run the existing suites after each step: `make test-python PY=bench/.venv/bin/python`
(hardware tests need the fake servers and the optional SDKs; tests whose SDK is missing skip).

## 8. Risks

1. **Public behaviour.** The default client is already native. Removing a fallback changes error hints
   (`--client python`) and breaks users who pass it. Jobs store the client used, so old jobs need the
   fallback to be readable, or a clear error.
2. **The fallback is the only path for some features.** `fake_*` IBM backends, `compile_for_ibm`, and TKET
   comparison payloads have no Mojo implementation. Moving them would be a feature project, not a cleanup.
3. **Different circuits.** IBM's Mojo ISA payload and the qiskit transpile produce different physical
   circuits and therefore different estimated costs. A parity test on costs will fail for legitimate
   reasons unless it compares a defined field only.
4. **Security code in two places.** Redaction, loopback checks, and the unsafe-YAML check are duplicated.
   A fix in one layer must be applied to the other until one is removed.
5. **Build dependency.** Mojo changes need `make py` (about 60 s) before the Python tests see them. CI must
   build `qumo_py.so` before running the hardware tests.
6. **Mojo-only defects are harder to fix quickly.** A bug in a native path needs a rebuild; the fallback
   gives a workaround today.

## 9. Decisions for the owner

1. Keep, deprecate, or remove the vendor-SDK fallbacks (`ibm.py` qiskit paths, `ionq_sdk.py`,
   `quantinuum.py` pytket paths).
2. Accept that payload parity means "the same fields", not "the same bytes".
3. Approve step 1 (parity harness) as the first change; it touches only tests.

## 10. First step: why it was not implemented

The obvious first step is a shared helper or the parity harness. The harness needs a definition of
equivalence for each provider (decision 2 above), and the fallback clients need their optional SDKs
installed to run. Without owner input on equivalence, a harness written now would either pass trivially
or fail for legitimate differences. The change would therefore not be clearly safe, so no code was
changed for item 3.
