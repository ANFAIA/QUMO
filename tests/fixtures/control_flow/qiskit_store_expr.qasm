OPENQASM 3.0;
include "stdgates.inc";
bit[3] a;
bit[3] b;
qubit[3] q;
bool flag;
h q[0];
h q[1];
h q[2];
a[0] = measure q[0];
a[1] = measure q[1];
a[2] = measure q[2];
b = a ^ 5;
if (b == 3 || a >= 6) {
  x q[0];
}
flag = !(a == b);
b[0] = flag;
