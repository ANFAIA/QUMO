OPENQASM 3.0;
include "stdgates.inc";
bit[1] c;
bit[4] m;
qubit[1] q2;
uint[4] k;
k = 0;
while (!c[0] && k < 7) {
  reset q2[0];
  h q2[0];
  c[0] = measure q2[0];
  k = k + 1;
}
m = k;
