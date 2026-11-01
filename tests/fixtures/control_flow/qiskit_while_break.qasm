OPENQASM 3.0;
include "stdgates.inc";
bit[3] c;
qubit[3] q;
h q[0];
while (c == 0) {
  h q[1];
  c[1] = measure q[1];
  if (c[1]) {
    x q[2];
    break;
  } else {
    continue;
  }
  c[0] = measure q[0];
}
