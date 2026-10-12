// Quantum teleportation with mid-circuit measurement and feed-forward.
// The state ry(1.1)|0> on q[0] is teleported to q[2]:
// P(c[2] = 1) = sin^2(0.55) ~ 0.2733 regardless of c[0], c[1].
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
bit[3] c;

ry(1.1) q[0];            // state to teleport
h q[1];                  // Bell pair between q[1] and q[2]
cx q[1], q[2];
cx q[0], q[1];           // Bell-basis measurement of q[0], q[1]
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1] == 1) x q[2];   // corrections (single-bit conditions)
if (c[0] == 1) z q[2];
c[2] = measure q[2];
