// n-qubit GHZ state with a constant for loop (unrolled at parse time).
// Expect only the all-zeros and all-ones bitstrings.
OPENQASM 3.0;
include "stdgates.inc";

const int n = 8;
qubit[n] q;
bit[n] c;

h q[0];
for uint i in [0:n-2] {      // OpenQASM 3 ranges are inclusive
    cx q[i], q[i + 1];
}
c = measure q;
