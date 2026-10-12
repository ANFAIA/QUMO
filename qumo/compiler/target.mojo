"""Hardware-agnostic compilation targets.

A `Target` is plain data, in the spirit of Qiskit's `Target`, Braket device
properties and PennyLane's TOML device capabilities: qubit count, undirected
coupling graph, native basis (list of opcodes, empty = "any gate"), per-qubit
1q / readout errors, per-edge 2q errors, T1/T2 and per-opcode durations.

Vendor neutral by construction: presets are named by *topology* (`line`,
`ring`, `grid`, `torus`, `tree`, `heavy_hex`, `all_to_all`, `from_edges`) and
by *modality* (`superconducting_heavy_hex`, `superconducting_grid`,
`superconducting_grid_iswap`, `trapped_ion`, `neutral_atom`). The default
target is `Target.all_to_all(n)` with basis "any". Basis presets are named by
gate family ("generic" = {u, cx}, "sc_cz" = {rz, sx, x, cz}, "ion_ms" =
{rz, rx, ry, rxx}, ...) and arbitrary gate lists are accepted ("rz,ry,rzz").

`heavy_hex(d)` reproduces rustworkx `heavy_hex_graph(d)` edge-for-edge (node
order: d^2 data qubits, (d^2-1)/2 syndrome, d(d-1) flag qubits; verified
against rustworkx for d = 1..11) so coupling maps line up with
Qiskit/Benchpress; `from_edges` takes any graph.

All-pairs hop distances are precomputed by BFS (O(n (n + m))) for routing.
`describe()` returns a compact JSON object for agents; `parse_target` accepts a
one-line spec such as "grid:5x5/sc_cz" or "heavy_hex:7/rz,sx,x,ecr".
"""

from std.collections import Dict
from std.math import sqrt
from qumo.ir.gates import Op, op_name, op_from_name, op_arity, op_is_unitary
from qumo.ir.circuit import Circuit
from qumo.compiler.common import basis_str, fmt_num


def _lower_trim(s: String) -> String:
    return String(s.strip()).lower()


def basis_presets() -> List[String]:
    var out: List[String] = [
        "any", "generic", "sc_cx", "sc_cz", "sc_ecr", "sc_iswap",
        "ion_ms", "ion_zz", "atom_cz", "clifford_t",
    ]
    return out^


def parse_basis(spec: String) raises -> List[Int]:
    """Gate-list or preset name -> opcode list (empty list = any gate).

    Presets: any, generic {u,cx}, sc_cx {rz,sx,x,cx}, sc_cz {rz,sx,x,cz},
    sc_ecr {rz,sx,x,ecr}, sc_iswap {rz,rx,ry,iswap}, ion_ms {rz,rx,ry,rxx},
    ion_zz {rz,ry,rzz}, atom_cz {rz,rx,ry,cz}, clifford_t {h,s,sdg,t,tdg,x,y,z,cx}.
    """
    var s = _lower_trim(spec)
    var out = List[Int]()
    if s == "" or s == "any" or s == "all" or s == "*":
        return out^
    if s == "generic" or s == "u_cx":
        out = [Op.U, Op.CX]
        return out^
    if s == "sc_cx":
        out = [Op.RZ, Op.SX, Op.X, Op.CX]
        return out^
    if s == "sc_cz" or s == "superconducting":
        out = [Op.RZ, Op.SX, Op.X, Op.CZ]
        return out^
    if s == "sc_ecr":
        out = [Op.RZ, Op.SX, Op.X, Op.ECR]
        return out^
    if s == "sc_iswap":
        out = [Op.RZ, Op.RX, Op.RY, Op.ISWAP]
        return out^
    if s == "ion_ms" or s == "ion" or s == "trapped_ion":
        out = [Op.RZ, Op.RX, Op.RY, Op.RXX]
        return out^
    if s == "ion_zz":
        out = [Op.RZ, Op.RY, Op.RZZ]
        return out^
    if s == "atom_cz" or s == "atom" or s == "neutral_atom":
        out = [Op.RZ, Op.RX, Op.RY, Op.CZ]
        return out^
    if s == "clifford_t":
        out = [Op.H, Op.S, Op.SDG, Op.T, Op.TDG, Op.X, Op.Y, Op.Z, Op.CX]
        return out^
    # explicit gate list: separators , ; + whitespace
    var cleaned = s.replace(";", ",").replace("+", ",").replace(" ", ",").replace("{", "").replace("}", "")
    for part in cleaned.split(","):
        var name = String(part.strip())
        if name == "":
            continue
        var k = op_from_name(name)
        if k < 0:
            raise Error("QM-E610: unknown gate '" + name + "' in basis '" + spec + "'. Hint: use gate names like rz,sx,x,cx,cz,ecr,iswap,rxx,rzz,u or a preset: " + String(", ").join(basis_presets()) + ".")
        if not op_is_unitary(k):
            continue
        var dup = False
        for e in out:
            if e == k:
                dup = True
        if not dup:
            out.append(k)
    return out^


struct Target(Copyable, Writable):
    var name: String
    var modality: String
    var num_qubits: Int
    var full: Bool  # all-to-all connectivity (no coupling constraint)
    var edges: List[Int]  # flat undirected pairs (a0, b0, a1, b1, ...), a < b
    var adj_start: List[Int]  # CSR adjacency (empty when full)
    var adj: List[Int]
    var dist: List[Int]  # n*n hop distances (empty when full)
    var basis: List[Int]  # native opcodes; empty = any gate
    var error_1q: List[Float64]
    var error_2q: List[Float64]  # parallel to edges
    var default_error_2q: Float64
    var readout_error: List[Float64]
    var t1_us: List[Float64]
    var t2_us: List[Float64]
    var duration_ns: List[Float64]  # per opcode
    var edge_index: Dict[Int, Int]

    def __init__(out self, num_qubits: Int, edges: List[Int], basis: List[Int],
                 name: String = "custom", full: Bool = False) raises:
        if num_qubits < 1:
            raise Error("QM-E612: target needs at least one qubit. Hint: pass num_qubits >= 1.")
        self.name = name
        self.modality = "generic"
        self.num_qubits = num_qubits
        self.full = full or num_qubits == 1
        self.edges = List[Int]()
        self.adj_start = List[Int]()
        self.adj = List[Int]()
        self.dist = List[Int]()
        self.basis = basis.copy()
        self.error_1q = List[Float64](length=num_qubits, fill=0.0)
        self.error_2q = List[Float64]()
        self.default_error_2q = 0.0
        self.readout_error = List[Float64](length=num_qubits, fill=0.0)
        self.t1_us = List[Float64](length=num_qubits, fill=0.0)
        self.t2_us = List[Float64](length=num_qubits, fill=0.0)
        self.duration_ns = List[Float64](length=Op.NUM_OPS, fill=0.0)
        self.edge_index = Dict[Int, Int]()
        if self.full:
            return
        if len(edges) % 2 != 0:
            raise Error("QM-E612: edge list must contain pairs (flat [a0, b0, a1, b1, ...]). Hint: got an odd length " + String(len(edges)) + ".")
        for i in range(len(edges) // 2):
            var a = edges[2 * i]
            var b = edges[2 * i + 1]
            if a < 0 or b < 0 or a >= num_qubits or b >= num_qubits or a == b:
                raise Error("QM-E612: invalid coupling edge (" + String(a) + ", " + String(b) + ") for " + String(num_qubits) + " qubits. Hint: endpoints must be distinct and in 0.." + String(num_qubits - 1) + ".")
            var lo = min(a, b)
            var hi = max(a, b)
            var key = lo * num_qubits + hi
            if key in self.edge_index:
                continue
            self.edge_index[key] = len(self.edges) // 2
            self.edges.append(lo)
            self.edges.append(hi)
        self.error_2q = List[Float64](length=len(self.edges) // 2, fill=0.0)
        self._build_adjacency()
        self._build_distances()

    def _build_adjacency(mut self):
        var n = self.num_qubits
        var deg = List[Int](length=n, fill=0)
        var m = len(self.edges) // 2
        for i in range(m):
            deg[self.edges[2 * i]] += 1
            deg[self.edges[2 * i + 1]] += 1
        self.adj_start = List[Int](length=n + 1, fill=0)
        for q in range(n):
            self.adj_start[q + 1] = self.adj_start[q] + deg[q]
        self.adj = List[Int](length=2 * m, fill=0)
        var fillp = List[Int](length=n, fill=0)
        for i in range(m):
            var a = self.edges[2 * i]
            var b = self.edges[2 * i + 1]
            self.adj[self.adj_start[a] + fillp[a]] = b
            fillp[a] += 1
            self.adj[self.adj_start[b] + fillp[b]] = a
            fillp[b] += 1
        # sort each neighbour list (deterministic iteration order)
        for q in range(n):
            var lo = self.adj_start[q]
            var hi = self.adj_start[q + 1]
            for i in range(lo + 1, hi):
                var v = self.adj[i]
                var j = i - 1
                while j >= lo and self.adj[j] > v:
                    self.adj[j + 1] = self.adj[j]
                    j -= 1
                self.adj[j + 1] = v

    def _build_distances(mut self):
        var n = self.num_qubits
        var inf = n + 1
        self.dist = List[Int](length=n * n, fill=inf)
        var queue = List[Int](length=n, fill=0)
        for s in range(n):
            var base = s * n
            self.dist[base + s] = 0
            var head = 0
            var tail = 0
            queue[tail] = s
            tail += 1
            while head < tail:
                var u = queue[head]
                head += 1
                var du = self.dist[base + u]
                for k in range(self.adj_start[u], self.adj_start[u + 1]):
                    var v = self.adj[k]
                    if self.dist[base + v] > du + 1:
                        self.dist[base + v] = du + 1
                        queue[tail] = v
                        tail += 1

    # ------------------------------------------------------------ topology
    @staticmethod
    def all_to_all(n: Int, basis: String = "any") raises -> Target:
        var t = Target(n, List[Int](), parse_basis(basis), "all_to_all_" + String(n), full=True)
        return t^

    @staticmethod
    def from_edges(n: Int, edges: List[Int], basis: String = "any", name: String = "custom") raises -> Target:
        var t = Target(n, edges, parse_basis(basis), name)
        return t^

    @staticmethod
    def line(n: Int, basis: String = "any") raises -> Target:
        var e = List[Int]()
        for i in range(n - 1):
            e.append(i)
            e.append(i + 1)
        var t = Target(n, e, parse_basis(basis), "line_" + String(n))
        return t^

    @staticmethod
    def ring(n: Int, basis: String = "any") raises -> Target:
        var e = List[Int]()
        for i in range(n - 1):
            e.append(i)
            e.append(i + 1)
        if n > 2:
            e.append(n - 1)
            e.append(0)
        var t = Target(n, e, parse_basis(basis), "ring_" + String(n))
        return t^

    @staticmethod
    def _grid_edges(rows: Int, cols: Int, periodic: Bool) -> List[Int]:
        var e = List[Int]()
        for r in range(rows):
            for c in range(cols):
                var q = r * cols + c
                if c + 1 < cols:
                    e.append(q)
                    e.append(q + 1)
                elif periodic and cols > 2:
                    e.append(q)
                    e.append(r * cols)
                if r + 1 < rows:
                    e.append(q)
                    e.append(q + cols)
                elif periodic and rows > 2:
                    e.append(q)
                    e.append(c)
        return e^

    @staticmethod
    def grid(rows: Int, cols: Int, basis: String = "any") raises -> Target:
        var t = Target(rows * cols, Target._grid_edges(rows, cols, False), parse_basis(basis),
                       "grid_" + String(rows) + "x" + String(cols))
        return t^

    @staticmethod
    def torus(rows: Int, cols: Int, basis: String = "any") raises -> Target:
        var t = Target(rows * cols, Target._grid_edges(rows, cols, True), parse_basis(basis),
                       "torus_" + String(rows) + "x" + String(cols))
        return t^

    @staticmethod
    def tree(n: Int, branching: Int = 2, basis: String = "any") raises -> Target:
        var e = List[Int]()
        var b = max(branching, 1)
        for i in range(1, n):
            e.append((i - 1) // b)
            e.append(i)
        var t = Target(n, e, parse_basis(basis), "tree_" + String(n))
        return t^

    @staticmethod
    def heavy_hex_edges(d: Int) raises -> List[Int]:
        """Edges of rustworkx `heavy_hex_graph(d)` (d odd): (5d^2 - 2d - 1)/2 nodes."""
        if d < 1 or d % 2 == 0:
            raise Error("QM-E612: heavy_hex distance must be odd and >= 1, got " + String(d) + ". Hint: use d = 3, 5, 7, ... ((5d^2-2d-1)/2 qubits).")
        var e = List[Int]()
        if d == 1:
            return e^
        var num_data = d * d
        var num_syn = (d - 1) * (d + 1) // 2
        var syn0 = num_data
        var flag0 = num_data + num_syn
        var chunk = (d + 1) // 2
        # data <-> flag along each row
        for i in range(d):
            for j in range(d - 1):
                var f = flag0 + i * (d - 1) + j
                e.append(i * d + j)
                e.append(f)
                e.append(f)
                e.append(i * d + j + 1)
        # data <-> syndrome at the row boundaries (left for even row pairs,
        # right for odd ones), exactly as rustworkx heavy_hex_graph
        for i in range(d - 1):
            var base = syn0 + i * chunk
            if i % 2 == 0:
                e.append(i * d)
                e.append(base)
                e.append((i + 1) * d)
                e.append(base)
            else:
                var s = base + chunk - 1
                e.append(i * d + (d - 1))
                e.append(s)
                e.append((i + 1) * d + (d - 1))
                e.append(s)
        # flag <-> syndrome
        for i in range(d - 1):
            var base = syn0 + i * chunk
            if i % 2 == 0:
                for j in range(1, chunk):
                    e.append(base + j)
                    e.append(flag0 + i * (d - 1) + 2 * (j - 1) + 1)
                    e.append(base + j)
                    e.append(flag0 + (i + 1) * (d - 1) + 2 * (j - 1) + 1)
            else:
                for j in range(chunk - 1):
                    e.append(base + j)
                    e.append(flag0 + i * (d - 1) + 2 * j)
                    e.append(base + j)
                    e.append(flag0 + (i + 1) * (d - 1) + 2 * j)
        return e^

    @staticmethod
    def heavy_hex(d: Int, basis: String = "any") raises -> Target:
        var n = (5 * d * d - 2 * d - 1) // 2
        var t = Target(n, Target.heavy_hex_edges(d), parse_basis(basis), "heavy_hex_d" + String(d))
        return t^

    # ------------------------------------------------------------ modalities
    def _set_noise(mut self, modality: String, e1: Float64, e2: Float64, ro: Float64,
                   t1: Float64, t2: Float64, d1: Float64, d2: Float64, dmeas: Float64,
                   virtual_z: Bool):
        self.modality = modality
        for q in range(self.num_qubits):
            self.error_1q[q] = e1
            self.readout_error[q] = ro
            self.t1_us[q] = t1
            self.t2_us[q] = t2
        for i in range(len(self.error_2q)):
            self.error_2q[i] = e2
        self.default_error_2q = e2
        for k in range(Op.NUM_OPS):
            var a = op_arity(k)
            if op_is_unitary(k):
                self.duration_ns[k] = d1 if a == 1 else (d2 if a == 2 else 3 * d2)
        self.duration_ns[Op.I] = 0.0
        if virtual_z:
            for k in [Op.Z, Op.S, Op.SDG, Op.T, Op.TDG, Op.RZ, Op.P]:
                self.duration_ns[k] = 0.0
        self.duration_ns[Op.MEASURE] = dmeas
        self.duration_ns[Op.RESET] = dmeas

    @staticmethod
    def superconducting_heavy_hex(d: Int, basis: String = "sc_cz") raises -> Target:
        """Heavy-hex lattice (degree <= 3), fixed-frequency transmon style."""
        var t = Target.heavy_hex(d, basis)
        t.name = "superconducting_heavy_hex_d" + String(d)
        t._set_noise("superconducting", 2.5e-4, 7e-3, 1.5e-2, 200.0, 150.0, 35.0, 300.0, 1500.0, True)
        return t^

    @staticmethod
    def superconducting_grid(rows: Int, cols: Int, basis: String = "sc_cz") raises -> Target:
        """Square lattice (degree <= 4), tunable-coupler transmon style."""
        var t = Target.grid(rows, cols, basis)
        t.name = "superconducting_grid_" + String(rows) + "x" + String(cols)
        t._set_noise("superconducting", 1e-3, 5e-3, 1e-2, 80.0, 60.0, 25.0, 40.0, 600.0, True)
        return t^

    @staticmethod
    def superconducting_grid_iswap(rows: Int, cols: Int, basis: String = "sc_iswap") raises -> Target:
        """Square lattice with an iSWAP-family entangler."""
        var t = Target.grid(rows, cols, basis)
        t.name = "superconducting_grid_iswap_" + String(rows) + "x" + String(cols)
        t._set_noise("superconducting", 1e-3, 6e-3, 1e-2, 80.0, 60.0, 25.0, 32.0, 600.0, True)
        return t^

    @staticmethod
    def trapped_ion(n: Int, basis: String = "ion_ms") raises -> Target:
        """All-to-all ion chain with Molmer-Sorensen (RXX) entangler."""
        var t = Target.all_to_all(n, basis)
        t.name = "trapped_ion_" + String(n)
        t._set_noise("trapped_ion", 5e-5, 3e-3, 3e-3, 1e9, 1e6, 10000.0, 200000.0, 100000.0, False)
        return t^

    @staticmethod
    def neutral_atom(rows: Int, cols: Int, radius: Float64 = 1.5, basis: String = "atom_cz") raises -> Target:
        """Rydberg array: sites on a grid, CZ between atoms within `radius` lattice units."""
        var e = List[Int]()
        var r = Int(radius)
        var r2 = radius * radius + 1e-9
        for a_r in range(rows):
            for a_c in range(cols):
                for dr in range(0, r + 1):
                    for dc in range(-r, r + 1):
                        if dr == 0 and dc <= 0:
                            continue
                        var b_r = a_r + dr
                        var b_c = a_c + dc
                        if b_r >= rows or b_c < 0 or b_c >= cols:
                            continue
                        if Float64(dr * dr + dc * dc) <= r2:
                            e.append(a_r * cols + a_c)
                            e.append(b_r * cols + b_c)
        var t = Target(rows * cols, e, parse_basis(basis), "neutral_atom_" + String(rows) + "x" + String(cols))
        t._set_noise("neutral_atom", 5e-4, 5e-3, 1e-2, 1e7, 1e6, 500.0, 400.0, 5e6, False)
        return t^

    # ------------------------------------------------------------ queries
    def with_basis(self, basis: String) raises -> Target:
        var t = self.copy()
        t.basis = parse_basis(basis)
        return t^

    def supports(self, kind: Int) -> Bool:
        """True when `kind` may appear in compiled output (non-unitary ops always may)."""
        if not op_is_unitary(kind):
            return True
        if len(self.basis) == 0:
            return True
        for k in self.basis:
            if k == kind:
                return True
        return False

    @always_inline
    def distance(self, a: Int, b: Int) -> Int:
        if a == b:
            return 0
        if self.full:
            return 1
        return self.dist[a * self.num_qubits + b]

    @always_inline
    def are_connected(self, a: Int, b: Int) -> Bool:
        if a == b or a < 0 or b < 0 or a >= self.num_qubits or b >= self.num_qubits:
            return False
        if self.full:
            return True
        return self.dist[a * self.num_qubits + b] == 1

    def num_edges(self) -> Int:
        if self.full:
            return self.num_qubits * (self.num_qubits - 1) // 2
        return len(self.edges) // 2

    def degree(self, q: Int) -> Int:
        if self.full:
            return self.num_qubits - 1
        return self.adj_start[q + 1] - self.adj_start[q]

    def is_connected_graph(self) -> Bool:
        if self.full:
            return True
        for q in range(self.num_qubits):
            if self.dist[q] > self.num_qubits:
                return False
        return True

    def diameter(self) -> Int:
        if self.full:
            return 1 if self.num_qubits > 1 else 0
        var d = 0
        for v in self.dist:
            if v <= self.num_qubits:
                d = max(d, v)
        return d

    def edge_error(self, a: Int, b: Int) -> Float64:
        if self.full:
            return self.default_error_2q
        var key = min(a, b) * self.num_qubits + max(a, b)
        var i = self.edge_index.get(key, -1)
        if i < 0:
            return self.default_error_2q
        return self.error_2q[i]

    def estimate_fidelity(self, c: Circuit) -> Float64:
        """Product of (1 - error) over gates and measurements (virtual gates free)."""
        var f = 1.0
        for g in c.ops:
            var k = g.kind
            if k == Op.MEASURE:
                if g.q0 < self.num_qubits:
                    f *= 1.0 - self.readout_error[g.q0]
                continue
            if not op_is_unitary(k) or k == Op.I:
                continue
            var a = op_arity(k)
            if a == 1:
                if self.duration_ns[k] > 0.0 and g.q0 < self.num_qubits:
                    f *= 1.0 - self.error_1q[g.q0]
            elif a == 2:
                f *= 1.0 - self.edge_error(g.q0, g.q1)
            else:
                var e = self.default_error_2q
                f *= (1.0 - e) * (1.0 - e) * (1.0 - e)
        return f

    def estimate_duration_ns(self, c: Circuit) -> Float64:
        """Critical-path duration using per-opcode durations."""
        var t = List[Float64](length=max(c.num_qubits, self.num_qubits), fill=0.0)
        var total = 0.0
        for g in c.ops:
            if g.kind == Op.BARRIER:
                var m = 0.0
                for v in t:
                    m = max(m, v)
                for i in range(len(t)):
                    t[i] = m
                continue
            var a = op_arity(g.kind)
            if a == 0 or (not op_is_unitary(g.kind) and g.kind != Op.MEASURE and g.kind != Op.RESET):
                continue
            var start = 0.0
            for j in range(a):
                start = max(start, t[g.qubit(j)])
            var end = start + self.duration_ns[g.kind]
            for j in range(a):
                t[g.qubit(j)] = end
            total = max(total, end)
        return total

    def describe(self) -> String:
        """Compact JSON description (stable keys, agent-readable)."""
        var n = self.num_qubits
        var s = String("{\"name\":\"") + self.name + "\",\"modality\":\"" + self.modality
        s += "\",\"num_qubits\":" + String(n)
        s += ",\"connectivity\":\"" + ("all_to_all" if self.full else "coupling_map") + "\""
        s += ",\"num_edges\":" + String(self.num_edges())
        var maxdeg = 0
        for q in range(n):
            maxdeg = max(maxdeg, self.degree(q))
        s += ",\"max_degree\":" + String(maxdeg)
        s += ",\"diameter\":" + String(self.diameter())
        s += ",\"connected\":" + ("true" if self.is_connected_graph() else "false")
        s += ",\"basis\":["
        if len(self.basis) == 0:
            s += "\"any\""
        for i in range(len(self.basis)):
            if i > 0:
                s += ","
            s += "\"" + op_name(self.basis[i]) + "\""
        s += "]"
        var e1 = 0.0
        var ro = 0.0
        var t1 = 0.0
        var t2 = 0.0
        for q in range(n):
            e1 += self.error_1q[q]
            ro += self.readout_error[q]
            t1 += self.t1_us[q]
            t2 += self.t2_us[q]
        var e2 = self.default_error_2q
        if not self.full and len(self.error_2q) > 0:
            e2 = 0.0
            for v in self.error_2q:
                e2 += v
            e2 /= Float64(len(self.error_2q))
        var nq_f = Float64(n)
        s += ",\"avg_error_1q\":" + fmt_num(e1 / nq_f) + ",\"avg_error_2q\":" + fmt_num(e2)
        s += ",\"avg_readout_error\":" + fmt_num(ro / nq_f)
        s += ",\"avg_t1_us\":" + fmt_num(t1 / nq_f) + ",\"avg_t2_us\":" + fmt_num(t2 / nq_f)
        if not self.full:
            var m = len(self.edges) // 2
            var shown = min(m, 512)
            s += ",\"edges\":["
            for i in range(shown):
                if i > 0:
                    s += ","
                s += "[" + String(self.edges[2 * i]) + "," + String(self.edges[2 * i + 1]) + "]"
            s += "]"
            if shown < m:
                s += ",\"edges_truncated\":true"
        s += "}"
        return s

    def write_to(self, mut w: Some[Writer]):
        w.write(self.describe())


def _parse_dims(s: String, mut a: Int, mut b: Int) raises:
    var parts = s.split("x")
    if len(parts) == 2:
        a = Int(String(parts[0].strip()))
        b = Int(String(parts[1].strip()))
    elif len(parts) == 1:
        a = Int(String(parts[0].strip()))
        b = -1
    else:
        raise Error("bad dims")


def parse_target(spec: String) raises -> Target:
    """One-line target spec: "<topology>:<size>[/<basis>]".

    Topologies: all_to_all:N, line:N, ring:N, grid:RxC, torus:RxC, tree:N,
    heavy_hex:D, superconducting_heavy_hex:D, superconducting_grid:RxC,
    superconducting_grid_iswap:RxC, trapped_ion:N, neutral_atom:RxC.
    Example: "grid:4x4/sc_cz", "heavy_hex:5/rz,sx,x,ecr", "trapped_ion:11".
    """
    var s = _lower_trim(spec)
    var basis = String("")
    var has_basis = False
    var slash = s.find("/")
    var body = s
    if slash >= 0:
        basis = String(s[byte=slash + 1:])
        body = String(s[byte=0:slash])
        has_basis = True
    var colon = body.find(":")
    if colon < 0:
        raise Error("QM-E611: target spec '" + spec + "' needs '<topology>:<size>'. Hint: e.g. \"grid:4x4/sc_cz\", \"line:20\", \"heavy_hex:5\", \"trapped_ion:11\".")
    var topo = String(body[byte=0:colon].strip())
    var size = String(body[byte=colon + 1:].strip())
    var a = 0
    var b = -1
    try:
        _parse_dims(size, a, b)
    except:
        raise Error("QM-E611: bad size '" + size + "' in target spec '" + spec + "'. Hint: use N or RxC, e.g. 'line:16', 'grid:4x5'.")
    var bs = basis if has_basis else String("any")
    if topo == "all_to_all" or topo == "full" or topo == "all":
        return Target.all_to_all(a, bs)
    if topo == "line" or topo == "linear":
        return Target.line(a, bs)
    if topo == "ring":
        return Target.ring(a, bs)
    if topo == "grid" or topo == "square":
        return Target.grid(a, b if b > 0 else a, bs)
    if topo == "torus":
        return Target.torus(a, b if b > 0 else a, bs)
    if topo == "tree":
        return Target.tree(a, 2, bs)
    if topo == "heavy_hex" or topo == "heavyhex":
        return Target.heavy_hex(a, bs)
    if topo == "superconducting_heavy_hex":
        return Target.superconducting_heavy_hex(a, basis if has_basis else String("sc_cz"))
    if topo == "superconducting_grid":
        return Target.superconducting_grid(a, b if b > 0 else a, basis if has_basis else String("sc_cz"))
    if topo == "superconducting_grid_iswap":
        return Target.superconducting_grid_iswap(a, b if b > 0 else a, basis if has_basis else String("sc_iswap"))
    if topo == "trapped_ion" or topo == "ion":
        return Target.trapped_ion(a, basis if has_basis else String("ion_ms"))
    if topo == "neutral_atom" or topo == "atom":
        return Target.neutral_atom(a, b if b > 0 else a, 1.5, basis if has_basis else String("atom_cz"))
    raise Error("QM-E611: unknown topology '" + topo + "' in target spec '" + spec + "'. Hint: valid topologies are all_to_all, line, ring, grid, torus, tree, heavy_hex, superconducting_heavy_hex, superconducting_grid, superconducting_grid_iswap, trapped_ion, neutral_atom.")
