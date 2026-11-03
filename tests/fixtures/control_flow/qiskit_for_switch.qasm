OPENQASM 3.0;
include "stdgates.inc";
bit[3] c;
int switch_dummy;
qubit[3] q;
for int _loop_i_0 in [1:3] {
  rx(0.1*_loop_i_0) q[0];
  c[0] = measure q[0];
  if (c[0]) {
    break;
  }
}
for int _ in {0, 2, 5} {
  h q[1];
}
switch_dummy = c;
switch (switch_dummy) {
  case 0 {
    x q[0];
  }
  case 1, 2 {
    z q[0];
  }
  default {
    h q[2];
  }
}
c[2] = measure q[2];
