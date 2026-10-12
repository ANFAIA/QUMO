# QUMO — Mojo 1.1 Contributor Guide

Mojo changes fast. This file records the **exact dialect QUMO compiles with**
(Mojo 1.1.0 + MAX 26.6 Mojo libs) so humans and coding agents stop guessing.
Toolchain lives in `.venv/` (`uv pip install mojo max-mojo-libs max-core`).

```bash
.venv/bin/mojo run -I . tests/test_ir.mojo        # run a test
.venv/bin/mojo build -I . -o build/qumo cli/main.mojo   # build CLI (a main file named qumo.mojo would shadow the package)
.venv/bin/mojo build -I . --emit shared-lib -o build/qumo_py.so python/qumo_py.mojo   # Python extension
make test                                         # all tests
```

Ignore the `Failed to initialize Crashpad` line — harmless.

## Language rules (1.1)

| Topic | Rule |
|---|---|
| Functions | `def` only. `fn` is **removed**. `def` does not imply `raises`; add `raises` explicitly. |
| Constants | `comptime NAME = value` (`alias` removed). Struct-scoped: `struct Op: comptime H = 4`. |
| Compile-time control flow | `comptime for k in range(4):`, `comptime if cond:` (`@parameter for/if` removed). |
| Variables | Must be declared with `var` (implicit declaration is an error). |
| Arg conventions | `imm` (default, read-only), `mut`, `var` (owned), `out` (constructors), `deinit`. `read` is an error. |
| Transfer | `x^` moves. Collections are `Copyable` but **not implicitly**: call `.copy()`. |
| Structs | `@fieldwise_init struct P(ImplicitlyCopyable, Writable): var a: Int`. Structs are `Movable` by default. Constructors: `def __init__(out self, ...)`. Copy ctor: `def __init__(out self, *, copy: Self)`. |
| Printing | Implement `Writable`: `def write_to(self, mut w: Some[Writer]): w.write(...)`. |
| Traits | `trait Pass: def name(self) -> String: ...` ; generic: `def f[B: Backend](b: B)`. |
| Implicit conversion | `@implicit def __init__(out self, v: Float64)`. Only ONE implicit hop (FloatLiteral→Float64→Param fails, so add a FloatLiteral overload). |
| Unused results | Returning `ref[self] Self` for chaining triggers "value is unused" warnings → QUMO builders return `None`. |
| Closures | Capture lists: `def body(i: Int) {imm a, mut total}:`. `{}` = no captures, bare name = `imm`. Function-type params: `def f[F: def(Int) -> None](g: F)`. |
| Parallel | `from max.algorithm import parallelize` then `parallelize(closure, n)` (needs `max-core` runtime). Prefer `qumo.sim.statevector.run_chunks(body, total)` with `body(lo, hi)`. `from std.runtime import parallelism_level`. |
| Pointers | `UnsafePointer` → `Pointer`. Raw buffer from a list: `var p = lst.unsafe_ptr().as_unsafe_any_origin()` (type `MutPointer[Float64, MutAnyOrigin]`). Index: `p[unsafe_offset=i]`. SIMD: `p.unsafe_load[width=8](i)`, `p.unsafe_store(i, vec)`. |
| SIMD | `SIMD[DType.float64, 4](1, 2, 3, 4)`; `.length` not `.size`. `Int` is `Scalar[DType.int]`. |
| Arrays | `[1, 2, 3]` literal is `Array[Int, 3]`; write `var xs: List[Int] = [1, 2, 3]` for a List. `List[Int](length=n, fill=0)`, `List[T](capacity=n)`. |
| Slicing | Out-of-range / negative slices **abort**. No negative indexing. |
| Mutating iteration | `for ref x in lst: x += 1`. |
| Dict | `d[k] = v`, `d.get(k, default)`, `for kv in d.items(): kv.key, kv.value`, `for k in d.keys()`. |
| Optional | `Optional[Int]`, `if o: o.value()`, `return None`. |
| Tuples | `-> Tuple[Float64, Float64]`, `return (a, b)`, `t[0]`. |
| Sorting | `sort(lst)` for Comparable; `sort(span, cmp_fn)` custom. |
| Errors | `raise Error("QM-E123: message. Hint: ...")`; `try: ... except e: String(e)`. |
| **Lifetimes (ASAP destruction)** | A value is destroyed right after its *last syntactic use*. A raw pointer taken with `lst.unsafe_ptr()` does **not** keep `lst` alive: if a closure (e.g. inside `parallelize`) reads through the pointer after the last use of `lst`, it reads freed memory. Add `_ = len(lst)` after the parallel call. |
| Reserved words | `fn` is still reserved (cannot be a variable name), as are `class`, `del`, `match`, `yield` for free functions. |
| Read-only kernels | Take `self` (not `mut self`) and use `self.re.unsafe_ptr()` (immutable pointer) so `evolve(c).expectation(h)` works on temporaries. |

## Strings (UTF-8 strict)

* `len(s)` is a **compile error**. Use `s.byte_length()`.
* Byte access: `var bs = s.as_bytes()` → `Span[Byte]`; compare `Int(bs[i]) == ord("x")`.
* Substring from bytes: `String(unsafe_from_utf8=bs[a:b])`, or `s[byte=a:b]`.
* Methods: `split`, `strip`, `startswith`, `endswith`, `find`, `replace`, `lower`, `upper`, `join`.
* Parse numbers: `Int("42")`, `Float64("3.5")` (raise on bad input). Template strings: `String(t"{x} and {y}")`.

## Imports

* Stdlib is `std.*`: `from std.collections import List, Dict, Optional`,
  `from std.math import sqrt, cos, sin, pi, log, exp, atan2`, `from std.bit import pop_count`,
  `from std.time import perf_counter_ns`, `from std.sys import argv`, `from std.testing import assert_equal, assert_true, assert_almost_equal`.
* QUMO absolute imports with `-I .`: `from qumo.ir.circuit import Circuit`.

## QUMO conventions

* Error messages start with a stable code `QM-Exxx` and end with a `Hint:` — agents key on these (`docs/ERRORS.md`).
* Every random process takes an explicit `seed` → bit-identical reruns (`qumo.sim.rng.Rng`, use `rng.split(k)` per worker).
* Qubit order: little-endian (bit k of basis index = qubit k); bitstring keys are Qiskit-style (leftmost = highest clbit).
* Hot loops: split re/im `List[Float64]`, raw pointers, `W = 8` SIMD lanes, `run_chunks` for threads.
* Tests: one `tests/test_<module>.mojo` per module with `def main() raises:` and prints `test_<module> OK`.
