# tsim and bloqade-analog: research and integration plan

Reviewed on 2026-11-26. **This is a proposal. Nothing here is implemented.** The summary is in [ANALYSIS.md §14](ANALYSIS.md#14-additional-references-tsim-and-bloqade-analog); candidate status is tracked in [FRAMEWORK_UPDATES.md](FRAMEWORK_UPDATES.md).

Method: repositories, release notes, PyPI metadata, the tsim 1.0 blog and paper abstract, and QUMO's source. Neither package was installed or run. All QuEra performance figures are vendor-reported. Statements about QUMO come from reading the code, except the Stim-import behaviour in §1.3, which was run.

## 1. tsim

Repository: <https://github.com/QuEraComputing/tsim>. Package: `bloqade-tsim`.

### 1.1 What it is and its status

A Python sampler for noisy circuits with non-Clifford gates, aimed at QEC studies. It mirrors Stim's API and reads Stim circuit text. Releases: 1.0.0 on 2026-11-04 (breaking) and 1.1.0 on 2026-11-12. The public API follows semantic versioning from 1.0; other modules are internal.

- 1.0.0 made `active_statevector` the default, made the stabilizer-rank strategies (`cat5`, `bss`, `cutting`) provisional (they may be removed after a deprecation cycle), changed postselected sampling to return only surviving shots, replaced `pyzx-param` with `zxparam`, and requires Python 3.11 and `stim>=1.15`.
- 1.1.0 added an Apple GPU (Metal) backend, `active_width` and `stats`, and `CS`.
- Unreleased: experimental fault distance (`shortest_errors`) and determined measurements / missing detectors for non-Clifford circuits.
- It is young and moving fast (pushed on the review date).

Runtime dependencies include JAX, numba, NumPy, Stim, `zxparam`, equinox and, on Apple silicon, a Metal binding. `zxparam` 0.1.0 is on PyPI as Apache-2.0, with platform wheels and an sdist; the GitHub URL in its metadata returned 404 on the review date, so we have not verified a public source repository.

### 1.2 Method

1. The circuit becomes a ZX diagram; each Pauli noise location is a boolean parameter.
2. The diagram is reduced once. Outputs that are parities of noise bits are sampled directly from the noise model. The remaining connected components are the quantum part.
3. Default: each component is read from the reduced diagram as Pauli rotations and measurements and simulated on a dense array of 2^w amplitudes, w being the active width (following [Clifft](https://arxiv.org/abs/2604.27058) and [SymFT](https://arxiv.org/abs/2607.28600); we read neither beyond what tsim and Clifft's abstract say). CPU (numba) or Apple GPU.
4. Provisional alternative: stabilizer-rank decomposition compiled to JAX, CPU or CUDA. The [April 2026 paper](https://arxiv.org/abs/2604.01059) abstract (cost exponential only in the non-Clifford gates) describes this path, not the current default. Do not call tsim "a stabilizer-rank simulator".

Limits: w above 30 is rejected; its documentation calls 1 to about 16 comfortable and about 17 to 26 slow. The Metal backend takes w ≤ 11 and computes in float32. A seed reproduces samples only with the same versions and backend. Detector and observable values are raw outcomes (they can be random), not Stim's flips against a reference, unless the reference-sample flags are set. `detector_error_model()` is built by Stim from a Clifford stand-in and is only guaranteed for Clifford circuits; native DEMs are an open issue ([#247](https://github.com/QuEraComputing/tsim/issues/247)). Another open issue reports float32 underflow in the stabilizer-rank path above about 250 qubits ([#308](https://github.com/QuEraComputing/tsim/issues/308)).

Vendor-reported ([blog](https://bloqade.quera.com/blog/introducing-tsim-1-0-0/), not reproduced): on d=5 end-to-end cultivation with postselection, one core of a Ryzen 9 9900X, 903k attempted shots/s, against 67k for Clifft 0.12.0 and 48.8k for SymFT 0.1.1. The blog identifies the measured "Tsim 1.0.0" as a pre-release git build.

Format: beyond Stim's instructions it defines `T`, `T_DAG`, `R_X/Y/Z`, `U3`, `TPP`, `R_XX/YY/ZZ`, `R_PAULI`, `CS`, `CCZ` and `CCX`, with angles in half-turns. It stores them as tagged Stim instructions (for example `S[T]`, `I[R_Z(theta=0.25*pi)]`). Plain Stim ignores the tag and sees a Clifford stand-in.

### 1.3 What QUMO has

| Piece | QUMO | Gap |
|---|---|---|
| Near-Clifford simulation | `qumo/sim/branching.mojo`: stabilizer frame plus sparse amplitudes; ≤ 512 qubits, ≤ 2^20 branches (QM-E412) | Every shot redoes the tableau and amplitudes; `sample_branching` takes ≤ 62 classical bits and returns no detector samples |
| Detectors and noise | `qumo/sim/frame.mojo`, 512-shot batches | Clifford only (QM-E701) |
| DEM | `qumo/qec/dem.mojo` | Built on frame propagation: Clifford only |
| Determined parities | `qumo/qec/lint.mojo` | From frame samples: Clifford only |
| Post-selection | `run_physical` in `qumo/ft/emulate.mojo` filters shots after running them | No early abort |
| Stim text | `qumo/qec/stim_text.mojo` | No `T`, tags or rotations. `T`, `T_DAG`, `S[T]` and `R_Z(0.25)` fail with QM-E734. `I[R_Z(theta=0.25*pi)]` also fails, but with an uncoded "String is not convertible to float" error |
| Pauli-based form | `to_pbc` in `qumo/compiler/pbc.mojo` | Rejects detectors, observables, noise, reset and conditions (QM-E623) |
| ZX | none; `phase_fold` works on phase polynomials | No ZX graph or rewrite engine |
| GPU | Pauli-frame sampler | No GPU near-Clifford sampler |

### 1.4 Licence

Apache-2.0, "Copyright 2026 QuEra Computing Inc."; compatible with QUMO. A port that follows their code is a derivative work: keep notices and attribute. Two exceptions seen: `src/tsim/external/vec_sim/` is adapted from the Gidney, Jones and Shutty dataset under **CC BY 4.0**, and `test/data/determined/chan_d5a19_uncorrected_T.stim` comes from a repository its README gives as **MIT**. We did not audit other circuit files; check each before copying.

### 1.5 Options

| | Option | Effort | Files | New dependencies | Validation | Main risks |
|---|---|---|---|---|---|---|
| A | Import (and explicitly export) the non-Clifford dialect, starting with `T`, `T_DAG`, `S[T]`, `S_DAG[T]` | S | `qumo/qec/stim_text.mojo`, `qumo/qec/codes.mojo`, `tests/test_stim_interop.mojo`, `docs/QEC.md` | none | Gate unitaries against QUMO's state vector; round trip | It is tsim's dialect, not Stim's: Stim reads a stand-in, so export must be opt-in |
| B | Raw detector and observable sampling for non-Clifford circuits on the branching simulator, with early abort on post-selected detectors | S-M | `qumo/sim/branching.mojo` or a new module in `qumo/qec/`, example, `docs/SIMULATION.md` | none | Exact distribution from the state vector or density matrix on small circuits | Slow per shot; branch limit |
| C | Compile-once active-state-vector sampler in Mojo | L | `qumo/compiler/pbc.mojo` (noise, detectors, reset, feed-forward), new `qumo/sim/active.mojo`, dispatch, benchmarks | none | B and dense simulation; tsim or Clifft as optional comparison | Without ZX reduction QUMO's active width may be larger than tsim's |
| D | Parametric ZX engine in Mojo | L, the largest | new compiler subpackage | none | Tensor contraction on small graphs | Large surface; only justified if C shows the width gap comes from missing ZX reduction |
| E | Stabilizer-rank sampling | L | new | none | Dense simulation | Provisional upstream |
| F | Optional Python bridge that calls tsim | S-M | `plugins/` or `python/`, never `qumo/` | JAX, numba, Stim, zxparam and others | - | Heavy dependencies, young API |
| G | Optional cross-check script | S | next to `bench/bench_qec.py`, which already uses Stim; skipped when tsim is absent | tsim, development only | Marginals within binomial error; use the same raw-value convention | Shares Stim parsing, so it does not check the format independently |

Neutrality: no option makes `qumo/` import a vendor module, and C to E add no dependency. Option A adds support for a text dialect defined by one vendor's tool to a core parser. We think it is acceptable because it is a circuit text format, not a device or service format, but this is a judgement for the maintainers. The alternative is to translate the dialect outside `qumo/`.

Ranking: **adopt now** A with B (one slice); **prototype** C, starting with a planner that only reports the active width (a cheap go/no-go); **defer** D, G until B exists (then add it as an optional script), and non-Clifford fault distance and determined measurements (unreleased and experimental upstream); **skip** E, F and a GPU backend for now.

### 1.6 Recommended first slice

`from_stim` accepts `T`, `T_DAG`, `S[T]` and `S_DAG[T]`; other tagged forms fail with a coded error. A new sampling function returns raw detector and observable bits (the XOR of the listed records, with this convention documented) from the branching simulator, takes a post-selection mask and stops a shot at the first masked detector that fires. It must not go through the 62-bit `SampleResult` path. Ship a noisy Clifford+T example and a section in `SIMULATION.md`.

**Success criterion.** On at least three circuits of at most 12 qubits with `T` gates, mid-circuit measurement and Pauli noise, the joint detector and observable histogram from 10^6 shots passes a chi-square test, with the threshold fixed in advance, against an exact distribution computed without `branching.mojo` or its tableau (dense simulation with noise enumerated or as a density matrix, measurements deferred). On Clifford inputs, rates agree within sampling error with `FrameSimulator` after XOR with the reference. For one circuit with hundreds of qubits and tens of `T` gates, report shots/s and the peak branch count, with no target against tsim.

**Why it could fail.** Per-shot tableau work and dictionary allocation may be far too slow for logical-error-rate statistics, and peak branch counts on cultivation-like circuits have not been measured; they could reach 2^20. B would then remain only as a check for C.

## 2. bloqade-analog

Repository: <https://github.com/QuEraComputing/bloqade-analog>.

### 2.1 What it is and its status

A Python SDK for analog Hamiltonian simulation on neutral atoms: a builder (geometry, then Rydberg or hyperfine coupling, then detuning or Rabi amplitude and phase, with uniform, per-site or scaled fields and a waveform), variables and batches, an IR with validation passes, a local emulator and submission to hardware. The emulator uses SciPy ODE solvers (`dop853` by default, `lsoda`, `dopri5`), has two- and three-level atom types and can restrict the basis with a blockade radius.

- Hamiltonian: global or local Ω, φ, Δ and a C6/r^6 interaction. Its documentation says Aquila restricts local control to detuning.
- Hardware: through Amazon Braket to Aquila, which the [AWS device page](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html) lists as an AHS device (live availability not checked). A direct QuEra path imports `qcs.api_client`, which is not a declared dependency; we have not verified that it is publicly usable.
- Local detuning: release 0.16.9 builds it with `ShiftingField`. The Braket SDK (checked at 1.128.2) keeps that name only as a deprecated alias of `LocalDetuning`. An issue reporting a local-detuning rejection ([#995](https://github.com/QuEraComputing/bloqade-analog/issues/995)) has been open since 2024.
- Julia: `juliacall` is a dependency, but the Julia type module we opened is entirely commented out. We do not treat Julia as an available backend.
- Release 0.16.9 (2026-03-02) is the latest; `main` says 0.17.0-dev. Apart from bot updates, the later commits we saw are a docs fix and a CI fix. Calling this maintenance mode is our inference, not a QuEra statement. The `bloqade` 0.38.0 meta-package (2026-11-20) still requires `bloqade-analog~=0.16.9`.
- Python 3.10 to 3.13. Dependencies include the Braket SDK, `juliacall`, numba, NumPy, SciPy, pandas, bokeh and pydantic.
- Licence: Apache-2.0, "Copyright (c) 2023 QuEra Computing Inc.."; GitHub's detector reports no standard licence. We did not audit the licences of data or documentation assets.
- Its bundled `capabilities.json` is a snapshot, not current device truth. Among its values: at most 256 sites, 4 um minimum spacing, 1,000 shots, 4 us duration, Ω ≤ 15.8 rad/us, |Δ| ≤ 125 rad/us, and a local-detuning section.

### 2.2 The C6 difference

bloqade-analog uses `RB_C6 = 2π × 862690 = 5420441.13` rad um^6/us for the 70S level of 87Rb. QUMO's `c6_for_level(70)` is `5420158.53` (as Pulser tabulates it). QUMO's value is lower by 282.6, a relative 5.2e-5. Any cross-check must give both sides the same value.

### 2.3 What QUMO has

| bloqade-analog | QUMO | Gap |
|---|---|---|
| Register with Ω, Δ, φ | `qumo/ir/analog.mojo` (um, ns, rad/us; constant, ramp and evenly spaced interpolated waveforms; one phase per pulse) | Global drive only. Other breakpoints need a run of ramp pulses |
| Local detuning, per-site scales | none | A gap in the model |
| Variables and batches | none in `AnalogProgram` (a function and a loop can do it) | A builder language is not worth copying |
| Emulator | `qumo/sim/rydberg.mojo`: second-order Trotter, ≤ 20 atoms, noiseless | No blockade subspace, local detuning or third level |
| Device validation | `check_against_device` for Pasqal specs, in the plugin | No AHS limits |
| Vendor formats | `plugins/hardware/native/qumo_hw/`: Pulser export and Pasqal client | No Braket AHS format (`qbraid.py` refuses `braket_ahs`) |
| Example | `examples/hardware/pasqal_mis.mojo` | Can serve as the AHS fixture |

### 2.4 Options

| | Option | Effort | Files | New dependencies | Validation | Main risks |
|---|---|---|---|---|---|---|
| A | AHS JSON export (`braket.ir.ahs.program`, version 1) with a limits check | S | new `plugins/hardware/native/qumo_hw/ahs.mojo`, tests, example, `docs/HARDWARE_PLUGINS.md` | none | Golden file; optional, user-run load with Amazon's schema classes and local AHS simulator | AHS gives each field a time series, so a jump between pulses may not be expressible; phase and slew rules must come from the current device schema (not checked) |
| B | AHS JSON import | S after A | same | none | Round trip | Local detuning, vacant sites and unsupported breakpoints must be refused or modelled |
| C | Local detuning in the neutral model: per-site weights times one waveform | M | `qumo/ir/analog.mojo`, `qumo/sim/rydberg.mojo`, `pasqal_pulser.mojo` (refuse or map; Pulser's mapping not studied), docs, example | none | `evolve_schedule` (a separate integrator) and analytic non-interacting atoms | The core model must not copy one device's sign or range limits |
| D | Blockade-subspace emulator | M | `qumo/sim/` | none | Full space at large C6, reporting the truncation error | An approximation that must be labelled; the current drive kernel does not carry over |
| E | Parameter-sweep or builder language | M | `qumo/ir/analog.mojo` | none | - | Little value in Mojo |
| F | Python bridge to bloqade-analog | M | plugin | Braket SDK, `juliacall` and others | - | Heavy dependencies, little upstream activity |
| G | Optional emulator cross-check | S after A | `bench/` | Braket SDK or bloqade-analog, development only | Their solver against QUMO's | Align C6 first |
| H | Native Braket AHS client | L | `qumo_hw`, request signing | AWS account | Hardware | Needs credentials and paid shots |

Ranking: **adopt now** A; **prototype** C; **defer** B, D, G (until A exists) and H; **skip** E, F and three-level or hyperfine emulation.

### 2.5 Recommended first slice

`ahs_json(AnalogProgram)` in `qumo_hw/ahs.mojo`: convert um to m, ns to s and rad/us to rad/s; join pulses into piecewise-linear Ω and Δ series; raise on a discontinuity or a phase it cannot represent exactly. The limits check takes a device-capabilities JSON as input, so no Aquila number is built in. Export the 5-atom MIS program next to the existing Pulser path. This tests that one neutral model serves two vendor formats. Nothing under `qumo/` changes, so `make check-neutrality` keeps passing.

**Success criterion.** The output equals a golden file derived by hand from the published schema; discontinuous input raises; and, in a check the user runs, Amazon's schema classes load the file and the Braket local AHS simulator, given the same C6, agrees with `simulate_analog` on 10^5 shots within a total-variation distance fixed in advance from the sampling noise over 32 outcomes.

**Why it could fail.** The Braket simulator was not run for this review; it may fix its own C6 or interpolate differently. The schema does not encode all device rules, so a schema-valid file can still be rejected by the device, which cannot be tested without a paid submission.

## 3. Not verified

Installation, performance and correctness of both packages. `zxparam`'s public source. tsim's input paths other than Stim text. Clifft and SymFT beyond what is stated above. Current Aquila availability, live capabilities and AHS phase and slew rules. Any QuEra statement about bloqade-analog's future.
