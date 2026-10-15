"""QEC architectures: code + primitives + logical API + passes + runtime policies.

A Guppy FT "QEC architecture" bundles a code definition, its logical gadgets
and their physical implementations, the passes that encode programs, and the
policies that manage resources at runtime (state factories, QEC-cycle
insertion). `Architecture` is QUMO's equivalent configuration record; the
passes are free functions in `qumo.ft.encode` (`compile`, `implement_ops`,
`encode`) and the gadget library is `qumo.ft.gadgets`.

Builders mirror Guppy FT's `SteaneBuilder().build(n_blocks=...)`:

```mojo
var steane = SteaneBuilder().with_qec_policy(QECPolicy("knill", 2.0).with_cost("h", 1.0).with_cost("cx", 1.0))
                            .with_zero_factory_conf(FactoryConf(2, 5)).build(3)
var ice = IcebergBuilder(4).build(2)          # [[6,4,2]] blocks
```

Policies (all resolved at compile time, see docs/FT.md "static vs runtime"):

* `QECPolicy(style, threshold)`: every logical op adds `cost(op)` to the
  counter of each block it touches; at `threshold` a QEC cycle of `style`
  ("steane" = two one-bit-teleportation cycles, "knill" = Bell-pair
  teleportation, "syndrome" = Iceberg stabilizer measurement, "none") is
  inserted on that block. `before_non_clifford` also inserts one before every
  T / rotation. Guppy FT default: costs 0, threshold 1 (no automatic cycles).
* `FactoryConf(size, max_attempts)`: `size` states are prepared ahead of use
  (in parallel with the computation, refilled after each consumption);
  `max_attempts` repeat-until-success attempts per state, each verified by its
  flag measurements; a shot whose attempts all fail is discarded (post-selected),
  like Guppy FT's `exit("StateFactory ran out of attempts!")`.
* `feed_forward`: "frame" (default) keeps Pauli corrections in a compile-time
  Pauli frame of classical expressions (zero quantum cost, decoded values read
  only when a non-Pauli operation needs them); "eager" applies every
  correction immediately as classically-conditioned physical gates.
* `magic_prep`: "ft" (flag-verified T|+> preparation, arXiv:1804.06995 /
  10.1038/srep19578 circuits as in Guppy FT) or "nonft" (encoder only).
* `rz_mode`: "static" (exact for multiples of pi/4, otherwise the shortest
  Clifford+T word within `rz_epsilon` up to `rz_max_t` T gates),
  "comparator" (arXiv:2404.05618 repeat-until-success circuit, also for
  angles only known at run time), or "native" (Iceberg: physical RZZ/RXX).

Idea from Guppy FT (Apache-2.0), files src/guppyft/code/steane/encode.py
(`SteaneBuilder`, `QECPolicy`, `QECStyle`, `RUSStateFactoryConf`) and
docs/architecture_dev.md; re-derived.

Error codes: QM-E727 (invalid architecture or policy option).
"""

from qumo.ft.code import StabilizerCode, steane_code, iceberg_code
from qumo.ft.logical import LOp, lop_name
from qumo.ai.json import JsonWriter


struct QECPolicy(Copyable, Writable):
    var style: String
    var threshold: Float64
    var costs: List[Float64]  # per LOp kind
    var before_non_clifford: Bool

    def __init__(out self, style: String = "steane", threshold: Float64 = 1.0, before_non_clifford: Bool = False):
        self.style = style
        self.threshold = threshold
        self.costs = List[Float64](length=LOp.NUM, fill=0.0)
        self.before_non_clifford = before_non_clifford

    def with_cost(self, op: String, cost: Float64) raises -> QECPolicy:
        """Cost of a logical op ("h", "cx", "t", "prep_zero" = "alloc", "prep_t", ...)."""
        if cost < 0.0:
            raise Error("QM-E727: QEC op cost cannot be negative (got " + String(cost) + "). Hint: costs are >= 0.")
        var name = op.lower()
        if name == "prep_zero":
            name = "alloc"
        var out = self.copy()
        for k in range(LOp.NUM):
            if lop_name(k) == name:
                out.costs[k] = cost
                return out^
        raise Error("QM-E727: unknown QEC cost key '" + op + "'. Hint: use a logical op name such as h, cx, t, s, x, alloc (prep_zero), prep_t, inject_t, rz, rzz.")

    def active(self) -> Bool:
        if self.style == "none":
            return False
        if self.before_non_clifford:
            return True
        for c in self.costs:
            if c > 0.0:
                return True
        return False

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_string("style", self.style)
        w.kv_float("threshold", self.threshold)
        w.kv_bool("before_non_clifford", self.before_non_clifford)
        w.key("costs")
        w.begin_object()
        for k in range(LOp.NUM):
            if self.costs[k] > 0.0:
                w.kv_float(lop_name(k), self.costs[k])
        w.end_object()
        w.end_object()

    def write_to(self, mut w: Some[Writer]):
        w.write("QECPolicy(", self.style, ", threshold=", self.threshold, ")")


@fieldwise_init
struct FactoryConf(ImplicitlyCopyable, Writable):
    var size: Int  # states prepared ahead of use (0 = on demand)
    var max_attempts: Int  # repeat-until-success attempts per state

    def write_to(self, mut w: Some[Writer]):
        w.write("FactoryConf(size=", self.size, ", max_attempts=", self.max_attempts, ")")


struct Architecture(Copyable, Writable):
    var kind: String  # "steane" | "iceberg"
    var code: StabilizerCode
    var n_blocks: Int
    var qec: QECPolicy
    var zero_factory: FactoryConf
    var magic_factory: FactoryConf
    var feed_forward: String
    var magic_prep: String
    var zero_prep: String
    var rz_mode: String
    var rz_epsilon: Float64
    var rz_max_t: Int
    var comparator_attempts: Int

    def __init__(out self, kind: String, var code: StabilizerCode, n_blocks: Int):
        self.kind = kind
        self.code = code^
        self.n_blocks = n_blocks
        self.qec = QECPolicy(String("steane") if kind == "steane" else String("syndrome"), 1.0)
        self.zero_factory = FactoryConf(0, 1)
        self.magic_factory = FactoryConf(0, 1)
        self.feed_forward = "frame"
        self.magic_prep = "nonft"
        self.zero_prep = String("ft") if kind == "steane" else String("nonft")
        self.rz_mode = String("static") if kind == "steane" else String("native")
        self.rz_epsilon = 1e-2
        self.rz_max_t = 16
        self.comparator_attempts = 4

    def name(self) -> String:
        return self.kind + "[" + self.code.name + "] x" + String(self.n_blocks)

    def logical_gate_set(self) -> List[String]:
        if self.kind == "steane":
            var s: List[String] = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap (relabel)", "t / tdg (magic-state injection)", "measure (decoded)", "qec_cycle", "prep_t", "inject_t", "inject_tdg"]
            return s^
        var s: List[String] = ["x", "y", "z", "rz(theta)", "rx(theta)", "rzz(theta)", "rxx(theta)", "swap (relabel)", "measure (decoded, error-detecting)", "qec_cycle (syndrome)"]
        return s^

    def validate(self) raises:
        if self.n_blocks < 1:
            raise Error("QM-E727: n_blocks must be >= 1 (got " + String(self.n_blocks) + "). Hint: one logical block per simultaneously live logical qubit (k per block).")
        var st = self.qec.style
        if st != "none" and st != "steane" and st != "knill" and st != "syndrome":
            raise Error("QM-E727: unknown QEC style '" + st + "'. Hint: steane, knill (Steane architecture), syndrome (Iceberg) or none.")
        if self.kind == "iceberg" and (st == "steane" or st == "knill") and self.qec.active():
            raise Error("QM-E727: QEC style '" + st + "' needs a distance-3 CSS code with a transversal Clifford group. Hint: use style 'syndrome' for Iceberg.")
        if self.kind == "steane" and st == "syndrome" and self.qec.active():
            raise Error("QM-E727: QEC style 'syndrome' is the Iceberg detection cycle. Hint: use 'steane' or 'knill' for the Steane architecture.")
        if self.qec.threshold <= 0.0:
            raise Error("QM-E727: QEC threshold must be > 0. Hint: e.g. 1.0.")
        if self.feed_forward != "frame" and self.feed_forward != "eager":
            raise Error("QM-E727: feed_forward must be 'frame' or 'eager' (got '" + self.feed_forward + "').")
        if self.magic_prep != "ft" and self.magic_prep != "nonft":
            raise Error("QM-E727: magic_prep must be 'ft' or 'nonft'.")
        if self.zero_prep != "ft" and self.zero_prep != "nonft":
            raise Error("QM-E727: zero_prep must be 'ft' or 'nonft'.")
        if self.zero_factory.size < 0 or self.zero_factory.max_attempts < 1 or self.magic_factory.size < 0 or self.magic_factory.max_attempts < 1:
            raise Error("QM-E727: factory size must be >= 0 and max_attempts >= 1. Hint: FactoryConf(size=1, max_attempts=5).")
        if self.kind == "steane" and self.rz_mode != "static" and self.rz_mode != "comparator" and self.rz_mode != "exact":
            raise Error("QM-E727: rz_mode for Steane must be 'static', 'comparator' or 'exact' (got '" + self.rz_mode + "').")
        if self.kind == "iceberg" and self.rz_mode != "native":
            raise Error("QM-E727: Iceberg implements rotations natively; rz_mode must be 'native'.")
        if self.rz_epsilon <= 0.0 or self.rz_epsilon >= 1.0:
            raise Error("QM-E727: rz_epsilon must lie in (0, 1). Hint: 1e-2.")
        if self.rz_max_t < 0 or self.rz_max_t > 24:
            raise Error("QM-E727: rz_max_t must lie in [0, 24] (exhaustive Clifford+T search). Hint: 16.")

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_string("architecture", self.kind)
        w.kv_string("name", self.name())
        w.key("code")
        self.code.write_json(w)
        w.kv_int("n_blocks", self.n_blocks)
        w.kv_int("logical_capacity", self.n_blocks * self.code.k)
        w.kv_string_list("logical_gate_set", self.logical_gate_set())
        w.key("qec_policy")
        self.qec.write_json(w)
        w.key("zero_factory")
        w.begin_object()
        w.kv_int("size", self.zero_factory.size)
        w.kv_int("max_attempts", self.zero_factory.max_attempts)
        w.end_object()
        w.key("magic_factory")
        w.begin_object()
        w.kv_int("size", self.magic_factory.size)
        w.kv_int("max_attempts", self.magic_factory.max_attempts)
        w.end_object()
        w.kv_string("feed_forward", self.feed_forward)
        w.kv_string("magic_prep", self.magic_prep)
        w.kv_string("zero_prep", self.zero_prep)
        w.kv_string("rz_mode", self.rz_mode)
        w.kv_float("rz_epsilon", self.rz_epsilon)
        w.kv_int("rz_max_t", self.rz_max_t)
        w.end_object()

    def to_json(self) -> String:
        var w = JsonWriter()
        self.write_json(w)
        return w.finish()

    def write_to(self, mut w: Some[Writer]):
        w.write("Architecture(", self.name(), ", qec=", self.qec, ", feed_forward=", self.feed_forward, ")")


struct SteaneBuilder(Copyable):
    """Guppy FT `SteaneBuilder`: chain `with_*` calls, then `build(n_blocks)`."""

    var arch: Architecture

    def __init__(out self) raises:
        self.arch = Architecture("steane", steane_code(), 1)

    def with_qec_policy(self, p: QECPolicy) -> SteaneBuilder:
        var b = self.copy()
        b.arch.qec = p.copy()
        return b^

    def with_zero_factory_conf(self, c: FactoryConf) -> SteaneBuilder:
        var b = self.copy()
        b.arch.zero_factory = c
        return b^

    def with_magic_factory_conf(self, c: FactoryConf) -> SteaneBuilder:
        var b = self.copy()
        b.arch.magic_factory = c
        return b^

    def with_feed_forward(self, mode: String) -> SteaneBuilder:
        var b = self.copy()
        b.arch.feed_forward = mode
        return b^

    def with_magic_prep(self, mode: String) -> SteaneBuilder:
        var b = self.copy()
        b.arch.magic_prep = mode
        return b^

    def with_zero_prep(self, mode: String) -> SteaneBuilder:
        var b = self.copy()
        b.arch.zero_prep = mode
        return b^

    def with_rz(self, mode: String, epsilon: Float64 = 1e-2, max_t: Int = 16, comparator_attempts: Int = 4) -> SteaneBuilder:
        var b = self.copy()
        b.arch.rz_mode = mode
        b.arch.rz_epsilon = epsilon
        b.arch.rz_max_t = max_t
        b.arch.comparator_attempts = comparator_attempts
        return b^

    def build(self, n_blocks: Int) raises -> Architecture:
        var a = self.arch.copy()
        a.n_blocks = n_blocks
        a.validate()
        return a^


struct IcebergBuilder(Copyable):
    """Iceberg [[k+2,k,2]] architecture (k even): `IcebergBuilder(k).build(n_blocks)`."""

    var arch: Architecture

    def __init__(out self, k: Int) raises:
        self.arch = Architecture("iceberg", iceberg_code(k), 1)

    def with_qec_policy(self, p: QECPolicy) -> IcebergBuilder:
        var b = self.copy()
        b.arch.qec = p.copy()
        return b^

    def with_feed_forward(self, mode: String) -> IcebergBuilder:
        var b = self.copy()
        b.arch.feed_forward = mode
        return b^

    def build(self, n_blocks: Int) raises -> Architecture:
        var a = self.arch.copy()
        a.n_blocks = n_blocks
        a.validate()
        return a^


def architecture_by_name(name: String, n_blocks: Int) raises -> Architecture:
    """"steane" or "iceberg_<k>" (k even)."""
    var k = name.lower()
    if k == "steane":
        return SteaneBuilder().build(n_blocks)
    if k.startswith("iceberg_"):
        var kk: Int
        try:
            kk = Int(String(k[byte=8:]))
        except:
            raise Error("QM-E727: bad Iceberg size in '" + name + "'. Hint: iceberg_2, iceberg_4, ...")
        return IcebergBuilder(kk).build(n_blocks)
    raise Error("QM-E727: unknown architecture '" + name + "'. Hint: 'steane' or 'iceberg_<even k>'.")
