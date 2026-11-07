"""Adam, gradient-norm clipping and the cosine learning-rate schedule.

STUB (W-1): the signatures of MOJO_PORT.md section 5 only. Component C4 replaces
this file; every behavioural body aborts with a "STUB" message.
"""

from std.os import abort


struct Adam(Movable):
    var n_params: Int
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var t: Int
    var m: List[Float32]
    var v: List[Float32]

    def __init__(
        out self, n_params: Int, lr: Float64, beta1: Float64 = 0.9, beta2: Float64 = 0.999, eps: Float64 = 1e-8
    ):
        self.n_params = n_params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = List[Float32](length=n_params, fill=0.0)
        self.v = List[Float32](length=n_params, fill=0.0)

    def set_frozen(mut self, off: Int, length: Int):
        """Grad is None for the slice: no step, no state."""
        abort("qumo.rl.optim.Adam.set_frozen: STUB (component C4)")

    def mark_sparse_rows(mut self, off: Int, row_len: Int, rows: Int):
        """Skip never-touched rows (exact)."""
        abort("qumo.rl.optim.Adam.mark_sparse_rows: STUB (component C4)")

    def step(mut self, mut params: List[Float32], grads: List[Float32], lr: Float64):
        abort("qumo.rl.optim.Adam.step: STUB (component C4)")

    def steps_taken(self) -> Int:
        return self.t


def clip_grad_norm(mut grads: List[Float32], ranges: List[Int], max_norm: Float64) -> Float64:
    """Returns the pre-clip norm."""
    abort("qumo.rl.optim.clip_grad_norm: STUB (component C4)")


def cosine_epoch_lr(base: Float64, epoch: Int, total_epochs: Int) -> Float64:
    abort("qumo.rl.optim.cosine_epoch_lr: STUB (component C4)")
