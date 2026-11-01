"""General (non-Pauli) noise channels as Kraus operators.

`Op.KRAUS1` / `Op.KRAUS2` hold an index into `Circuit.kraus` (a side table: a
channel has any number of matrices, and `Instruction` is a POD). Only the state
vector trajectories and the density-matrix simulator take them; stabilizer and
Pauli-frame simulators need Pauli noise and report that with QM-E414.
"""

from std.math import sqrt, exp


struct KrausChannel(Copyable, Writable):
    """rho -> sum_k K_k rho K_k^dag; matrices are row-major, dim = 2^num_qubits."""

    var name: String
    var num_qubits: Int
    var re: List[List[Float64]]
    var im: List[List[Float64]]

    def __init__(out self, name: String, num_qubits: Int):
        self.name = name
        self.num_qubits = num_qubits
        self.re = List[List[Float64]]()
        self.im = List[List[Float64]]()

    def dim(self) -> Int:
        return 1 << self.num_qubits

    def add(mut self, var re: List[Float64], var im: List[Float64]):
        self.re.append(re^)
        self.im.append(im^)

    def add_real(mut self, var re: List[Float64]):
        var im = List[Float64](length=len(re), fill=0.0)
        self.re.append(re^)
        self.im.append(im^)

    def count(self) -> Int:
        return len(self.re)

    def validate(self) raises:
        """Trace preservation: sum_k K_k^dag K_k = identity."""
        var d = self.dim()
        if len(self.re) == 0:
            raise Error("QM-E414: Kraus channel '" + self.name + "' has no operators. Hint: pass at least one matrix.")
        for k in range(self.count()):
            if len(self.re[k]) != d * d or len(self.im[k]) != d * d:
                raise Error("QM-E414: Kraus operator " + String(k) + " of '" + self.name + "' needs " + String(d * d) + " entries. Hint: matrices are row-major with dimension 2^num_qubits.")
        for r in range(d):
            for c in range(d):
                var sr = 0.0
                var si = 0.0
                for k in range(self.count()):
                    for i in range(d):
                        var ar = self.re[k][i * d + r]
                        var ai = -self.im[k][i * d + r]
                        var br = self.re[k][i * d + c]
                        var bi = self.im[k][i * d + c]
                        sr += ar * br - ai * bi
                        si += ar * bi + ai * br
                var want = 1.0 if r == c else 0.0
                if abs(sr - want) > 1e-9 or abs(si) > 1e-9:
                    raise Error("QM-E414: Kraus channel '" + self.name + "' is not trace preserving. Hint: sum_k K_k^dag K_k must be the identity.")

    def write_to(self, mut w: Some[Writer]):
        w.write("KrausChannel(", self.name, ", qubits=", self.num_qubits, ", operators=", self.count(), ")")


def amplitude_damping(gamma: Float64) raises -> KrausChannel:
    """Relaxation to |0> with probability gamma."""
    if gamma < 0.0 or gamma > 1.0:
        raise Error("QM-E414: amplitude damping gamma " + String(gamma) + " is outside [0, 1]. Hint: gamma = 1 - exp(-t/T1).")
    var ch = KrausChannel("amplitude_damping", 1)
    ch.add_real([1.0, 0.0, 0.0, sqrt(1.0 - gamma)])
    ch.add_real([0.0, sqrt(gamma), 0.0, 0.0])
    return ch^


def phase_damping(lam: Float64) raises -> KrausChannel:
    """Pure dephasing: the off-diagonal of rho shrinks by sqrt(1 - lam)."""
    if lam < 0.0 or lam > 1.0:
        raise Error("QM-E414: phase damping lambda " + String(lam) + " is outside [0, 1]. Hint: it is a probability.")
    var ch = KrausChannel("phase_damping", 1)
    ch.add_real([1.0, 0.0, 0.0, sqrt(1.0 - lam)])
    ch.add_real([0.0, 0.0, 0.0, sqrt(lam)])
    return ch^


def thermal_relaxation(t: Float64, t1: Float64, t2: Float64) raises -> KrausChannel:
    """Exact T1 + T2 decay for time t, zero temperature, T2 <= 2 T1.

    rho_11 -> e^{-t/T1} rho_11, rho_01 -> e^{-t/T2} rho_01. The composition of
    amplitude damping with pure dephasing; the Pauli twirl of this channel is
    `qumo.qec.characterization.thermal_pauli_channel`."""
    if t < 0.0 or t1 <= 0.0 or t2 <= 0.0 or t2 > 2.0 * t1 + 1e-12:
        raise Error("QM-E414: thermal relaxation needs t >= 0, T1 > 0 and 0 < T2 <= 2 T1. Hint: got t=" + String(t) + ", T1=" + String(t1) + ", T2=" + String(t2) + ".")
    var gamma = 1.0 - exp(-t / t1)
    # off-diagonal: sqrt(1 - gamma) sqrt(1 - lam) = exp(-t/T2)
    var keep = exp(-t / t2) / sqrt(1.0 - gamma) if gamma < 1.0 else 0.0
    var lam = max(0.0, min(1.0, 1.0 - keep * keep))
    var ad = amplitude_damping(gamma)
    var pd = phase_damping(lam)
    var ch = KrausChannel("thermal_relaxation", 1)
    for a in range(ad.count()):
        for b in range(pd.count()):
            # K = PD_b . AD_a
            var m = List[Float64](length=4, fill=0.0)
            for r in range(2):
                for c in range(2):
                    m[2 * r + c] = pd.re[b][2 * r] * ad.re[a][c] + pd.re[b][2 * r + 1] * ad.re[a][2 + c]
            ch.add_real(m^)
    return ch^
