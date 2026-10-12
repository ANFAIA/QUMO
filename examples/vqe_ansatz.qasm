// Hardware-efficient VQE ansatz with symbolic parameters.
// Bind with params, e.g.:
//   qumo expect examples/vqe_ansatz.qasm "Z0 Z1 + Z1 Z2 + 0.5*X0" --params '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]'
//   or --params '{"t0": 0.1, "t1": 0.2, "t2": 0.3, "t3": 0.4, "t4": 0.5, "t5": 0.6}'
OPENQASM 3.0;
include "stdgates.inc";

input float t0;
input float t1;
input float t2;
input float t3;
input float t4;
input float t5;

qubit[3] q;

// layer 1
ry(t0) q[0];
ry(t1) q[1];
ry(t2) q[2];
cx q[0], q[1];
cx q[1], q[2];
// layer 2 (affine expressions of ONE input are allowed)
ry(t3) q[0];
ry(2*t4 + pi/4) q[1];
rz(-t5 / 2) q[2];
