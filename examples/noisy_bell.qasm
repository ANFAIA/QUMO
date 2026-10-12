// QUMO extensions travel in pragmas that other OpenQASM tools ignore:
// Pauli noise channels (simulated as per-shot trajectories), detectors, ticks.
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
#pragma qumo depolarize1(0.02) q[0]
cx q[0], q[1];
#pragma qumo depolarize2(0.05) q[0], q[1]
#pragma qumo tick
c = measure q;
#pragma qumo detector 0, 1
