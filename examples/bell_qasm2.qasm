// OpenQASM 2.0 input is accepted too (qelib1 gates, qreg/creg, measure ->).
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
u2(0, pi) q[0];
cx q[0], q[1];
measure q -> c;
