"""f32 matrices, GEMM family and vector kernels for the native RL stack.

STUB (W-1): the signatures of MOJO_PORT.md section 5 only. Component C4 replaces
this file; every behavioural body aborts with a "STUB" message instead of
pretending to work.
"""

from std.os import abort

# The one pointer alias of the contract; every other qumo.rl module imports it from here.
comptime FPtr = MutPointer[Float32, MutAnyOrigin]
comptime LANES = 8
comptime CHUNK_ROWS = 512
comptime VW = 16


struct Mat(Movable, Copyable):
    """Row-major f32 matrix."""

    var rows: Int
    var cols: Int
    var data: List[Float32]

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = List[Float32](length=rows * cols, fill=0.0)

    def ptr(mut self) -> FPtr:
        return self.data.unsafe_ptr().as_unsafe_any_origin()


def gemm_nt(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    """C[m,n] (+)= A[m,k] . B[n,k]^T."""
    abort("qumo.rl.tensor.gemm_nt: STUB (component C4)")


def gemm_nn(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    """C[m,n] (+)= A[m,k] . B[k,n]."""
    abort("qumo.rl.tensor.gemm_nn: STUB (component C4)")


def gemm_tn(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    """C[m,n] (+)= A[k,m]^T . B[k,n]."""
    abort("qumo.rl.tensor.gemm_tn: STUB (component C4)")


def relu_backward(dx: FPtr, pre: FPtr, n: Int):
    """dx *= (pre > 0); gradient 0 at exactly 0 (torch rule)."""
    abort("qumo.rl.tensor.relu_backward: STUB (component C4)")


def col_sum_acc(g: FPtr, m: Int, n: Int, dst: FPtr):
    """dst[j] += sum_i g[i, j]."""
    abort("qumo.rl.tensor.col_sum_acc: STUB (component C4)")


def sumsq_f64(p: FPtr, n: Int) -> Float64:
    """Sum of squares accumulated in f64."""
    abort("qumo.rl.tensor.sumsq_f64: STUB (component C4)")


# --- helpers named in MOJO_PORT.md 4.2 / 11.3 (not part of the section 5 contract) ---
def add_bias_rows(x: FPtr, rows: Int, cols: Int, bias: FPtr):
    """x[i, :] += bias for every row."""
    abort("qumo.rl.tensor.add_bias_rows: STUB (component C4)")


def relu_inplace(x: FPtr, n: Int):
    abort("qumo.rl.tensor.relu_inplace: STUB (component C4)")


def saxpy(y: FPtr, x: FPtr, alpha: Float32, n: Int):
    """y += alpha * x."""
    abort("qumo.rl.tensor.saxpy: STUB (component C4)")
