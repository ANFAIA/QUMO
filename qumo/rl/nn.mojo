"""Parameter store and the synthesis network (BasicPolicy-shaped MLP) with hand-derived backward.

STUB (W-1): the signatures of MOJO_PORT.md section 5 only. Component C5 replaces
this file; every behavioural body aborts or raises with a "STUB" message.
"""

from std.os import abort
from qumo.compiler.policy import ModelFile
from qumo.rl.tensor import FPtr


struct ParamStore(Movable, Copyable):
    """One flat parameter buffer plus named (rows x cols) tensors."""

    var names: List[String]
    var shapes: List[Int]  # rows, cols per tensor
    var offsets: List[Int]
    var data: List[Float32]

    def __init__(out self):
        self.names = List[String]()
        self.shapes = List[Int]()
        self.offsets = List[Int]()
        self.data = List[Float32]()

    def add(mut self, name: String, rows: Int, cols: Int) -> Int:
        abort("qumo.rl.nn.ParamStore.add: STUB (component C5)")

    def total(self) -> Int:
        return len(self.data)

    def offset(self, name: String) raises -> Int:
        raise Error("QM-E640: qumo.rl.nn.ParamStore.offset is a STUB (component C5)")


struct SparseBatch(Movable):
    """CSR rows of active observation indices."""

    var off: List[Int32]  # length N+1
    var idx: List[Int32]
    var perm: List[Int32]  # -1 = no twist

    def __init__(out self):
        self.off = List[Int32]()
        self.idx = List[Int32]()
        self.perm = List[Int32]()

    def rows(self) -> Int:
        return len(self.perm)


@fieldwise_init
struct NetSpec(Copyable):
    var obs_size: Int
    var num_actions: Int
    var emb: Int
    var common: List[Int]
    var policy_hidden: List[Int]
    var value_hidden: List[Int]
    var obs_perms: List[List[Int]]
    var act_perms: List[List[Int]]

    def __init__(out self):
        self.obs_size = 0
        self.num_actions = 0
        self.emb = 0
        self.common = List[Int]()
        self.policy_hidden = List[Int]()
        self.value_hidden = List[Int]()
        self.obs_perms = List[List[Int]]()
        self.act_perms = List[List[Int]]()

    def params(self) -> Int:
        """Parameter count (MOJO_PORT.md 4.4)."""
        abort("qumo.rl.nn.NetSpec.params: STUB (component C5)")


struct ChunkCache(Movable):
    """Per virtual worker scratch for up to CHUNK_ROWS rows."""

    var rows: Int
    var logits: List[Float32]  # post-twist logits of the chunk, rows x num_actions
    var values: List[Float32]

    def __init__(out self, spec: NetSpec):
        self.rows = 0
        self.logits = List[Float32]()
        self.values = List[Float32]()


struct ActorScratch(Movable):
    """Per task scratch for forward_one."""

    var buf: List[Float32]

    def __init__(out self, spec: NetSpec):
        self.buf = List[Float32]()


struct BasicNet(Movable):
    var spec: NetSpec
    var ps: ParamStore

    def __init__(out self, spec: NetSpec, seed: UInt64) raises:
        self.spec = spec.copy()
        self.ps = ParamStore()
        raise Error("QM-E640: qumo.rl.nn.BasicNet is a STUB (component C5)")

    @staticmethod
    def from_model(m: ModelFile) raises -> BasicNet:
        """Warm start from any container (f16 decoded)."""
        raise Error("QM-E640: qumo.rl.nn.BasicNet.from_model is a STUB (component C5)")

    def forward_one(self, idx: Span[Int32, _], twist: Int, mut sc: ActorScratch, logits: FPtr) -> Float32:
        abort("qumo.rl.nn.BasicNet.forward_one: STUB (component C5)")

    def forward_chunk(self, sb: SparseBatch, lo: Int, hi: Int, mut cc: ChunkCache):
        """Logits and values land in `cc`."""
        abort("qumo.rl.nn.BasicNet.forward_chunk: STUB (component C5)")

    def backward_chunk(
        self, sb: SparseBatch, lo: Int, hi: Int, mut cc: ChunkCache, dz: FPtr, dv: FPtr, grad: FPtr
    ):
        abort("qumo.rl.nn.BasicNet.backward_chunk: STUB (component C5)")

    def param_count(self) -> Int:
        return self.ps.total()
