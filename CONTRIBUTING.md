# Contributing to QUMO

QUMO is a research prototype (v0.1.0, Apache-2.0). Bug reports, examples, benchmark results and code are welcome. This page says how to set up, how to check your change, and what a reproducible report looks like.

## Ways to contribute

| You have | Do this |
|---|---|
| A bug | Open an issue with the "Bug report" template. The smallest OpenQASM file or Mojo script that shows it is worth more than a description. |
| An idea | Open an issue with the "Feature request" template before writing much code. |
| A benchmark result, good or bad | Open an issue with the "Benchmark result" template. See [Reporting benchmark results](#reporting-benchmark-results). |
| A new example | See [Adding an example](#adding-an-example). |
| A code change | Fork, branch, run the checks below, open a pull request. |

## Setup

You need macOS on Apple silicon or Linux (x86-64 or aarch64), and [uv](https://docs.astral.sh/uv/). Details and platform limits: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

```bash
git clone https://github.com/ANFAIA/QUMO.git && cd QUMO
make setup        # .venv with the pinned Mojo toolchain (about 850 MB)
make cli          # build/qumo
make py           # build/qumo_py.so (needed by the Python package qumo, Python examples and plugin tests)
make setup-py     # optional: bench/.venv with Qiskit, Aer, TKET, Cirq, Stim (about 1.5 GB)
make setup-py-locked  # the same environment at the exact versions in bench/requirements.lock (use this to reproduce a reported result)
make help         # every documented target
```

## Mojo dialect

QUMO pins one Mojo version (see `MOJO_VERSION` in the `Makefile`) because the language changes between releases. Read [docs/MOJO_GUIDE.md](docs/MOJO_GUIDE.md) before writing Mojo for this repository: it lists the constructs that compile on the pinned toolchain and the common traps. One trap you will meet on day one: `Circuit` is not implicitly copyable, so write `var out = r.circuit.copy()`.

## Running the tests

```bash
make test                                       # every tests/test_*.mojo, plus the fast RL tests; prints PASS/FAIL per file
.venv/bin/mojo run -I . tests/test_sim.mojo     # one suite
make examples                                   # fast-tier examples, verified by their CHECK lines
make test-python                                # tests of the Python package qumo, python/tests (builds build/qumo_py.so; needs NumPy and pytest)
make test-py                                    # every Python test: the package, examples runner, hardware plugins (needs make py and make setup-py)
```

- A test file is a Mojo program whose last printed line contains `OK` (for example `test_sim OK`). `make test` looks at that last line. A new module gets a new `tests/test_<module>.mojo`.
- Tests of numerical code compare against an independent reference (state vectors, an analytic value, another simulator), not against QUMO's own earlier output.
- Mojo prints `Failed to initialize Crashpad...` first; ignore it.
- If you add or change an error code, regenerate [docs/ERRORS.md](docs/ERRORS.md): `QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo`. Do not edit that file by hand.
- GPU code is compiled only with `-D QUMO_GPU`. `make gpu-check` cross-compiles the kernels without a GPU; `make test-gpu` needs one.

## Adding an example

Examples live in `examples/<category>/` and are checked by machine: each prints `CHECK <label>: PASS|FAIL ...` lines and ends with `ALL CHECKS PASSED (n)`. `examples/run_examples.py` runs them and fails on any `FAIL`.

1. Pick the category folder by what the example shows, not by the SDK it comes from.
2. Start the file with a docstring header. The first line is the title. If it reproduces something, name it and link the exact original. `example-meta` tells the runner how to run it:

   ```mojo
   """Short title of what the example shows.

   Reproduces: Qiskit - CHSH inequality
   Source: https://exact.url/of/the/original
   Framework: Qiskit (IBM) - https://github.com/Qiskit/qiskit

   Run: .venv/bin/mojo run -I . examples/<category>/<file>.mojo
   example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 60, "requires": [], "args": []}
   """
   ```

   An example of your own says `Framework: QUMO - https://github.com/ANFAIA/QUMO` and `"framework": "qumo"`, and has no `Reproduces:` or `Source:` lines. The runner checks the `Framework:` link against its table; copy the line from an existing example of the same origin.
3. Check results with `CheckLog` from `qumo.testing.checks`:

   ```mojo
   from qumo.testing.checks import CheckLog

   def main() raises:
       var log = CheckLog("category/my_example")
       # ... compute value ...
       _ = log.close("energy matches exact diagonalisation", value, expected, 1e-9)   # |value - expected| <= tol
       _ = log.within("sampled probability", p, 0.48, 0.52)                           # lo <= p <= hi
       _ = log.check("only two outcomes", n == 2, "found " + String(n))               # any condition
       log.finish()                                                                   # prints ALL CHECKS PASSED, or raises
   ```

   Labels must not contain `:` (the runner splits each CHECK line on the first `: `).

   Check against something independent: an analytic value, exact diagonalisation, or a number published by the original, with its source stated in the header. `fast` tier examples should finish within their timeout on a laptop; put long ones in `"tier": "slow"`.
4. Run it, then regenerate and check the generated indexes (`examples/CATALOG.md` and the index block in `examples/README.md`; do not edit those by hand):

   ```bash
   bench/.venv/bin/python examples/run_examples.py --only examples/<category>/<file>.mojo
   bench/.venv/bin/python examples/run_examples.py --write-index
   bench/.venv/bin/python examples/run_examples.py --check-index
   ```

   The runner uses only the Python standard library, so a plain `python3` works in place of `bench/.venv/bin/python` for Mojo and OpenQASM examples.

Do not copy text or code from another project's tutorial beyond what its licence allows; reproduce the experiment and link the original.

## Changing the library

- Keep QUMO vendor-neutral. No vendor is a default; presets are named by modality or topology. Vendor-specific code belongs in `plugins/hardware`.
- Compiler passes must keep the unitary exact (global phase tracked) and are tested by state vector. Where the extension points are, and which need core edits: [docs/EXTENDING.md](docs/EXTENDING.md).
- Errors carry a stable `QM-Exxx` code and a hint that says how to fix the input. Add new codes to the catalogue in `qumo/ai/diagnostics.mojo`.
- Performance claims need a measurement (next section). If a change makes something slower, say so in the pull request.
- Update the docs your change makes wrong. Code snippets in docs must be runnable; say so if you could not run one.

## Reporting benchmark results

Numbers are only useful if someone else can get them again. This applies to results in pull requests, in issues and in the docs. A report needs:

1. **What ran.** The exact command (for example `make bench-sv`, or `bench/.venv/bin/python bench/neutral/run_neutral.py ...`), the QUMO commit (`git rev-parse HEAD`) and whether the tree had local changes.
2. **Versions.** `.venv/bin/mojo --version`, and the version of every SDK compared (`bench/.venv/bin/python -m pip freeze | grep -i -E "qiskit|tket|cirq|stim|pymatching"`). `make setup-py` installs the latest versions, so they drift: always record them. `make setup-py-locked` installs the versions pinned in `bench/requirements.lock` (a freeze of the environment used for the current reports); if you use it, say so.
3. **Machine.** CPU model and core count, RAM, OS version; for GPU runs the GPU model, driver and precision (fp32 or fp64).
4. **Conditions.** Thread count if you set one, what else was running (load average), number of repetitions, and whether you report the median, the mean or the best. Prefer medians of at least 3 runs. For A/B comparisons alternate the two binaries.
5. **Equal work.** Say how you know both sides did the same thing: same circuits, same basis and coupling map, same precision, same shots; and how correctness was checked (state-vector equivalence, counts within statistical error).
6. **Raw output.** Attach the result file (`bench/results/*.jsonl`) or the full log, not only a summary table.
7. **Everything, not the best.** Include the cases where QUMO loses or fails. If the code was tuned on the benchmark being reported, say so; the existing reports do ([docs/BENCHPRESS_BIAS_REPORT.md](docs/BENCHPRESS_BIAS_REPORT.md), [docs/NEUTRAL_BENCHMARK_REPORT.md](docs/NEUTRAL_BENCHMARK_REPORT.md)).

How to run the existing harnesses: "Reproduce the benchmarks" in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) and [docs/BENCHMARKING_ON_CLOUD_GPUS.md](docs/BENCHMARKING_ON_CLOUD_GPUS.md).

## Pull requests

- One topic per pull request, with a description of what changed and how you checked it (which tests, which examples, which measurements).
- `make test` and `make examples` pass. The CI workflow in `.github/workflows/ci.yml` runs the same commands; it is new and has not completed a run yet, so run them locally.
- Add a line to [CHANGELOG.md](CHANGELOG.md) under "Unreleased".
- By contributing you agree that your contribution is licensed under the [Apache License 2.0](LICENSE).

## Citing

If QUMO helps your research, cite it with [CITATION.cff](CITATION.cff) and state the commit you used.
