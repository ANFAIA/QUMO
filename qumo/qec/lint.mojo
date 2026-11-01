"""QEC lint: `missing_detectors` (Stim 1.16) and `count_determined_measurements`.

QM-E703 catches a declared detector that is not deterministic. This is the other
direction: deterministic parities of measurements that nothing declares. Each one
is a check the decoder never sees, usually a forgotten boundary detector.

The noiseless outcomes of a stabilizer circuit are a coset of a linear code, so a
parity of records is deterministic exactly when it is 0 on every sample (samples
XOR the reference). Random samples span that code once there are a few more
shots than records, with failure probability about 2^-64 here. The relations are
the null space of the sample matrix; what remains after reducing them by the
declared detectors and observables is reported, as record lists. Like Stim's, the
basis is not unique and it is no detector solver: treat it as a diagnostic.
"""

from std.bit import pop_count
from qumo.ir.gates import Op, op_is_noise, op_is_unitary
from qumo.ir.circuit import Circuit
from qumo.sim.frame import FrameSimulator

comptime _EXTRA_SHOTS = 96


def _noiseless(c: Circuit, unknown_input: Bool) -> Circuit:
    var out = c.empty_like()
    out.records = c.records.copy()
    out.num_measurements = c.num_measurements
    out.num_detectors = c.num_detectors
    out.num_observables = c.num_observables
    if unknown_input:
        # a maximally mixed input: two random Paulis on every qubit
        for q in range(c.num_qubits):
            out.x_error(0.5, q)
            out.z_error(0.5, q)
    for g in c.ops:
        if not op_is_noise(g.kind):
            out.append(g)
    return out^


def _set_bit(mut v: List[UInt64], i: Int):
    v[i >> 6] ^= UInt64(1) << UInt64(i & 63)


def _bit(v: List[UInt64], i: Int) -> Bool:
    return ((v[i >> 6] >> UInt64(i & 63)) & 1) != 0


def _reduce(mut v: List[UInt64], basis: List[List[UInt64]], lead: List[Int]):
    for i in range(len(basis)):
        if _bit(v, lead[i]):
            for w in range(len(v)):
                v[w] ^= basis[i][w]


def _null_space(c: Circuit, unknown_input: Bool) raises -> List[List[UInt64]]:
    """Record parities that are 0 on every sample."""
    var M = c.num_measurements
    if M > 20000 or c.num_params() > 0 or not c.is_clifford():
        raise Error("QM-E737: missing_detectors needs a bound Clifford circuit with at most 20000 measurements. Hint: got " + String(M) + " measurements.")
    var shots = M + _EXTRA_SHOTS
    var fs = FrameSimulator(_noiseless(c, unknown_input))
    var ms = fs.sample_measurements(shots, 0x51ED2701)
    var words = (shots + 63) >> 6
    var cw = (M + 63) >> 6
    var rows = List[List[UInt64]]()
    var combo = List[List[UInt64]]()
    for r in range(M):
        var row = List[UInt64](length=words, fill=0)
        var ref_bit = fs.prog.reference[r] != 0
        for s in range(shots):
            if ms.get(s, r) != ref_bit:
                _set_bit(row, s)
        rows.append(row^)
        var cb = List[UInt64](length=cw, fill=0)
        _set_bit(cb, r)
        combo.append(cb^)
    # Gaussian elimination over the sample columns
    var rank = 0
    for col in range(shots):
        var piv = -1
        for r in range(rank, M):
            if _bit(rows[r], col):
                piv = r
                break
        if piv < 0:
            continue
        if piv != rank:
            rows.swap_elements(piv, rank)
            combo.swap_elements(piv, rank)
        for r in range(M):
            if r != rank and _bit(rows[r], col):
                for w in range(words):
                    rows[r][w] ^= rows[rank][w]
                for w in range(cw):
                    combo[r][w] ^= combo[rank][w]
        rank += 1
        if rank == M:
            break
    var out = List[List[UInt64]]()
    for r in range(rank, M):
        out.append(combo[r].copy())
    return out^


def count_determined_measurements(c: Circuit, unknown_input: Bool = False) raises -> Int:
    """Measurements whose result follows from earlier ones (and from the |0> input,
    unless `unknown_input`)."""
    return len(_null_space(c, unknown_input))


def missing_detectors(c: Circuit, unknown_input: Bool = False) raises -> List[List[Int]]:
    """Deterministic record parities not spanned by the declared detectors and
    observables. Empty when the circuit declares everything it could."""
    var rel = _null_space(c, unknown_input)
    var M = c.num_measurements
    var cw = (M + 63) >> 6
    # reduced basis of the declared parities
    var basis = List[List[UInt64]]()
    var lead = List[Int]()

    for g in c.ops:
        if g.kind != Op.DETECTOR and g.kind != Op.OBSERVABLE:
            continue
        var v = List[UInt64](length=cw, fill=0)
        for j in range(g.q1):
            _set_bit(v, c.records[g.q0 + j])
        _reduce(v, basis, lead)
        for r in range(M):
            if _bit(v, r):
                basis.append(v^)
                lead.append(r)
                break
    var out = List[List[Int]]()
    for ref v in rel:
        _reduce(v, basis, lead)
        var at = -1
        for r in range(M):
            if _bit(v, r):
                at = r
                break
        if at < 0:
            continue
        var recs = List[Int]()
        for r in range(M):
            if _bit(v, r):
                recs.append(r)
        out.append(recs^)
        basis.append(v.copy())
        lead.append(at)
    return out^
