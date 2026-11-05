// Bell pair: (|00> + |11>)/sqrt(2). Expect ~50/50 counts of "00" and "11".
// Framework: QUMO - https://github.com/ANFAIA/QUMO
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
c = measure q;
