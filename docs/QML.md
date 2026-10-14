# QUMO quantum machine learning (`qumo.algorithms` QML layer)

A PennyLane-class QML layer in Mojo: templates and embeddings, observables,
batched and initial-state gradients, step optimizers with PennyLane's exact
formulas, quantum kernels with an SVM, the Fubini-Study metric with quantum
natural gradient, and deterministic toy datasets. Everything is plain QUMO IR
(`Circuit`, affine `Param`, `PauliSum`) evaluated on the state-vector
simulator with adjoint gradients, and it reproduces PennyLane 0.45.1 numbers
to about 1e-15 (see [Verification](#verification)).

```
qumo/algorithms/
  qml.mojo         templates, embeddings, amplitude encoding, QCNN blocks, observables
  qml_train.mojo   StepOptimizer, lr schedules, Batcher, losses, accuracy, vjp
  qkernels.mojo    fidelity kernels (3 routes), kernel-target alignment, SVC, kernel ridge
  qgeometry.mojo   parameter layers, metric tensor, QNG direction and optimizer
  datasets.mojo    parity, DoubleCake, circle, moons, bars-and-stripes, CSV/JSON loaders, scaling
  gradients.mojo   + ExpectationEngine.set_initial_state, batch_value_and_gradient
qumo/sim/rng.mojo  + Rng.normal, Rng.normal_list
```

Tests: `tests/test_qml.mojo`, `tests/test_qkernels.mojo`, `tests/test_qgeometry.mojo`.

## Conventions

### Parameter order: weights, then features

Declare the trainable weights first and the input features second:

```mojo
var c = Circuit(4, 0, "classifier")
var w = c.parameters("w", 24)   # weights  -> parameter indices 0..23
var x = c.parameters("x", 4)    # features -> parameter indices 24..27
```

A parameter vector is then `weights ++ features`. Features are ordinary
affine `Param`s, so `RX(pi * x)` (`x * pi`) is still differentiable and one
compiled circuit serves every data point: only the parameter vector changes.
Kernels (`embedding_states`, `kernel_circuit`) rely on this order.

Non-affine feature maps (products such as `x_i * x_j` in IQP, or
`(pi - x_i)(pi - x_j)` in Qiskit's ZZ map) become extra feature parameters
whose values a helper computes (`iqp_feature_values`, `zz_feature_values`).

### Wire order

Templates take `wires: List[Int]`: QUMO qubits listed in PennyLane wire order
(`wires[0]` plays PennyLane wire 0). Expectation values of products of local
observables do not depend on the labelling, so `wires = [0, 1, ..., n-1]`
with `Z0` for `qml.PauliZ(0)` reproduces PennyLane.

When amplitudes or multi-wire probabilities matter, remember that QUMO is
little-endian (bit k of a basis index is qubit k) while PennyLane's wire 0 is
the most significant bit. Map PennyLane wire `w` to QUMO qubit `n-1-w`, i.e.
pass `wires = [n-1, ..., 0]`; basis indices are then identical
(`amplitude_state`, `bars_and_stripes` and the QCNN test use this mapping).

### PennyLane equivalence table

| PennyLane | QUMO |
|---|---|
| `Rot(phi, theta, omega)` | `append_rot(c, phi, theta, omega, q)` = RZ(phi), RY(theta), RZ(omega) in time order |
| `BasisState(x)` / `BasisEmbedding` | `append_basis_embedding` = RX(pi x) per wire (same expectation values; the state differs by a phase -i per 1-bit) |
| `AngleEmbedding(x, rotation)` | `append_angle_embedding(c, x, wires, "x"/"y"/"z")` |
| `IQPEmbedding(x, n_repeats)` | `iqp_embedding(n, n_repeats)` + `iqp_feature_values(x)`, or `append_iqp_embedding` |
| Qiskit `ZZFeatureMap(n, reps)` | `zz_feature_map(n, reps)` + `zz_feature_values(x)` |
| `AmplitudeEmbedding(f, normalize, pad_with)` (real) | `append_amplitude_embedding_real` (Mottonen tree), or `amplitude_state` as the engine's initial state |
| tutorial `get_angles(x)` | `mottonen_angles(x)` (2 qubits: identical layout) |
| `StronglyEntanglingLayers(W, ranges, imprimitive)` | `append_strongly_entangling_layers(c, W, wires, ranges, "cx"/"cz")` |
| `BasicEntanglerLayers(W, rotation)` | `append_basic_entangler_layers(c, W, wires, "x"/"y"/"z")` |
| `ArbitraryUnitary(W)` | `append_arbitrary_unitary(c, W, wires)` (Gray-code word order, `pauli_words_gray`) |
| `PauliRot(theta, word)` | `append_pauli_rotation(c, x, z, theta)` (library.mojo) |
| `IsingXX/YY/ZZ`, 2-wire `MultiRZ` | `c.rxx`, `c.ryy`, `c.rzz` |
| `U3(theta, phi, delta)` | `c.u(theta, phi, delta, q)` |
| `ctrl(U3)` / `measure` + `cond(U3)` | `append_controlled_u3(c, ctrl, target, theta, phi, lam)` |
| QCNN conv / pool (tutorial_learning_few_data) | `append_qcnn_conv`, `append_qcnn_pool` |
| `Hermitian(|0..0><0..0|)` | `projector_zero(n)`; `projector_basis(n, k)`; `local_projector_zero(n)` |
| `Hermitian(diag(v))` | `diagonal_observable(v)` |
| `qml.kernels.kernel_matrix / square_kernel_matrix` | `kernel_matrix`, `kernel_matrix_symmetric` |
| `qml.kernels.target_alignment` | `target_alignment`, `target_alignment_grad` |
| `qml.metric_tensor(approx=...)` | `metric_tensor(c, params, "block-diag"/"diag"/"full")` |
| `GradientDescent/Momentum/NesterovMomentum/Adam/Adagrad/RMSProp/QNG Optimizer` | `StepOptimizer.gd/momentum/nesterov/adam/adagrad/rmsprop`, `QNGOptimizer` |

### Optimizer formulas (`StepOptimizer`)

`g` is the gradient at the evaluation point, `t` the step count from 1.

| Kind | Update |
|---|---|
| `gd(lr)` | x <- x - lr g |
| `momentum(lr, m=0.9)` | a <- m a + lr g; x <- x - a |
| `nesterov(lr, m=0.9)` | as momentum, with g evaluated at `lookahead(x)` = x - m a |
| `adam(lr, b1=0.9, b2=0.99, eps=1e-8, form="pennylane")` | m1 <- b1 m1 + (1-b1) g; m2 <- b2 m2 + (1-b2) g^2; lr_t = lr sqrt(1-b2^t)/(1-b1^t); x <- x - lr_t m1 / (sqrt(m2) + eps) |
| `adam(..., form="optax")` | x <- x - lr (m1/(1-b1^t)) / (sqrt(m2/(1-b2^t)) + eps) (pass b2=0.999 for optax's default) |
| `adagrad(lr, eps=1e-8)` | s <- s + g^2; x <- x - lr / sqrt(s + eps) g |
| `rmsprop(lr, decay=0.9, eps=1e-8)` | s <- d s + (1-d) g^2; x <- x - lr / sqrt(s + eps) g |

The two Adam forms differ only in where `eps` enters (outside the
bias-corrected step size in PennyLane, after bias-correcting both moments in
optax); trajectories agree to about 1e-8 and are reproduced exactly by the
matching form. `qumo.algorithms.optimizers.Adam` (full-batch `minimize`,
best iterate) keeps its own epsilon placement.

## API reference

Tuples are returned for multi-output functions. Copy elements out before using
several of them in one expression: `var r = parity_dataset(4); var X = r[0].copy(); var y = r[1].copy()`.

### `qml.mojo` (QM-E830)

| API | Notes |
|---|---|
| `append_rot(mut c, phi, theta, omega, q)` | |
| `append_angle_embedding(mut c, features: List[Param], wires, rotation="x")` | fewer features than wires leaves wires untouched |
| `append_basis_embedding(mut c, features, wires)` | RX(pi x) |
| `append_iqp_embedding(mut c, x, xx, wires, n_repeats=1)`; `iqp_embedding(n, n_repeats=1) -> Circuit`; `iqp_feature_values(x)` | params `x[i]`, then `xx[i,j]` (i < j, combinations order) |
| `zz_feature_map(n, reps=2) -> Circuit`; `zz_feature_values(x)` | params `x[i]`, then `phi[i,j]`; H, P(2x), CX P(2 phi) CX, full entanglement |
| `mottonen_tree_angles(a) -> List[Float64]` | 2^n - 1 RY-tree angles, level k pattern j at index 2^k - 1 + j; signed amplitudes exact |
| `mottonen_angles(a)` | get_angles layout: [alpha_0] ++ per level k >= 1, pattern j = 2^k-1..0: (-alpha_kj/2, alpha_kj/2) |
| `append_amplitude_embedding_real(mut c, a, wires, normalize=True, pad_with=0.0)` | uniformly controlled RY tree: 2^m - 1 RY, 2^m - 2 CX; wires[0] = most significant bit; normalises when `normalize` or `pad_with != 0`, else requires unit norm |
| `amplitude_state(a, n, normalize=True, pad_with=0.0) -> StateVector` | sv.re[k] = a[k] |
| `append_strongly_entangling_layers(mut c, W, wires, ranges=[], imprimitive="cx")` | W[(l n + i) 3 + k]; default range r_l = l mod (n-1) + 1 |
| `append_basic_entangler_layers(mut c, W, wires, rotation="x")` | W[l n + i]; CNOT ring (n > 2), one CNOT (n = 2) |
| `pauli_words_gray(n)`; `append_arbitrary_unitary(mut c, W, wires)` | 4^n - 1 words; n = 2: XI YI ZI ZX IX XX YX YY ZY IY XY XZ YZ ZZ IZ (character k on wires[k]) |
| `append_controlled_u3(mut c, ctrl, target, theta, phi, lam)` | P(phi/2) P(lam/2) on ctrl, CRZ(lam), CRY(theta), CRZ(phi); symbolic and adjoint-differentiable |
| `append_qcnn_conv(mut c, W15, wires, skip_first_layer=False)` | U3 x2 (unless skipped, even pairs), IsingXX/YY/ZZ, U3 x2 on pairs (even, then odd) |
| `append_qcnn_pool(mut c, W3, sinks, sources)` | deferred measurement: controlled-U3(source -> sink) |
| `diagonal_observable(values) -> PauliSum` | fast Walsh-Hadamard into Z strings; n <= 12 |
| `projector_zero(n)`, `projector_basis(n, k)` | n <= 12 |
| `local_projector_zero(n)` | (1/n) sum_j \|0><0\|_j = 1/2 + sum_j Z_j / 2n |

### `gradients.mojo` additions

| API | Notes |
|---|---|
| `ExpectationEngine.set_initial_state(sv)` (QM-E804 on width mismatch), `clear_initial_state()` | every evaluation and all three gradient methods start from `sv` instead of \|0...0> |
| `batch_value_and_gradient(circuit, h, param_sets, method=Grad.ADJOINT, initial_states=[]) -> (values, grads)` | parallel over sets with one engine per worker for <= 14 qubits; identical to the sequential per-set result; `initial_states` empty or one per set |
| `batch_gradient(...)` | now the gradient part of `batch_value_and_gradient` (parallel; same results) |

### `qml_train.mojo` (QM-E833, QM-E831)

| API | Notes |
|---|---|
| `StepOptimizer.gd / momentum / nesterov / adam / adagrad / rmsprop` | `lookahead(x)`, `step(mut x, grad)`, `set_lr(lr)`, `reset()`, fields `t`, `lr` |
| `cosine_lr(step, total, lr0, lr_min=0)` | optax `cosine_decay_schedule(lr0, total, alpha=lr_min/lr0)` |
| `step_lr(step, every, gamma, lr0)` | lr0 gamma^(step // every) |
| `Batcher(n, batch, seed, shuffle=True, drop_last=False, replacement=False)`; `next()`; `batches_per_epoch()`; `Batcher.replay(batches)` | epochs of a seeded permutation (DataLoader); `drop_last` = PennyLane `iterate_minibatches`; `replacement` = `np.random.randint(0, n, (batch,))`; replay cycles recorded fixture batches |
| `square_loss(pred, y)`, `hinge_loss`, `l1_loss` | (mean value, d/dpred) |
| `cross_entropy(probs, labels)` | gradient w.r.t. probabilities, flattened row-major |
| `fidelity_loss(F)` | mean (1 - F)^2 |
| `accuracy(pred, y)` | sign agreement |
| `expectations(circuit, observables, params)` | all <O_j> from one evolution (a layer's forward) |
| `vjp(circuit, observables, params, cotangent)` | sum_j u_j d<O_j>/dtheta by one adjoint pass on sum_j u_j O_j (a layer's backward) |

Dataset-loop pattern (what the examples do):

```mojo
var opt = StepOptimizer.nesterov(0.5)
for step in range(steps):
    var w_eval = opt.lookahead(w)
    var sets = List[List[Float64]]()
    for i in batch: sets.append(w_eval ++ X[i])          # weights, then features
    var r = batch_value_and_gradient(model, H, sets)       # parallel adjoint passes
    var L = square_loss(predictions(r[0]), labels)         # value, d/dprediction
    var g = sum_i L[1][i] * r[1][i][:num_weights]          # chain rule
    opt.step(w, g)
```

### `qkernels.mojo` (QM-E831)

| API | Notes |
|---|---|
| `embedding_states(embed, X, weights=[])` | U(w, x_i)\|0> per row, parallel |
| `kernel_matrix(embed, X1, X2, weights=[])` | row-major len(X1) x len(X2), state route |
| `kernel_matrix_symmetric(embed, X, weights=[])` | N(N-1)/2 overlaps, unit diagonal |
| `kernel_circuit(embed, num_features)`; `kernel_circuit_params(w, xa, xb)` | U(w, a) U(w, b)^dagger; features renamed `xa[i]`, `xb[i]`, weights shared by name; params w ++ xa ++ xb |
| `kernel_value_and_grad(kc, w, xa, xb) -> (k, dk/dw)` | one adjoint pass on projector_zero (n <= 12); gradient over the weights only |
| `kernel_matrix_symmetric_and_grad(kc, w, X) -> (K, dK)` | dK[p] is N x N; pairs run in parallel |
| `sampled_kernel_entry(kc, params, shots, seed)` | frequency of 0...0: the QPU route |
| `target_alignment(K, y, assume_normalized_kernel=False, rescale_class_labels=True)`; `target_alignment_grad(K, dK, y, ...)` | PennyLane semantics (labels rescaled by class sizes) |
| `svc_fit(K, y, C=1.0, tol=1e-3, max_iter=100000) -> SVCModel`; `svc_decision(model, K_rows)`; `svc_predict` | libsvm SMO with second-order working-set selection; `dual_coef`, `intercept` as sklearn; K_rows = kernel_matrix(embed, X_test, X_train) |
| `kernel_ridge_fit(K, y, lam)`; `kernel_ridge_predict(alpha, K_rows)` | (K + lam I)^-1 y via jacobi_eigh |

### `qgeometry.mojo` (QM-E832)

| API | Notes |
|---|---|
| `parameter_layers(circuit, wrt=[])` | layers of parameter indices; a gate starts a new layer when one of its ancestors is in the current layer (PennyLane `CircuitGraph.parametrized_layers`) |
| `metric_tensor(circuit, params, approx="block-diag", wrt=[])` | P x P row-major, covariance of the generators (PennyLane convention); "block-diag", "diag" or "full" |
| `qng_direction(G, grad, lam=0.0)` | pinv(G + lam I) grad, rcond 1e-12 |
| `QNGOptimizer(lr=0.01, approx="block-diag", lam=0.0)`; `step(circuit, h, mut params) -> cost before the step` | |

Supported trainable gates: rx, ry, rz, p, cp, rxx, ryy, rzz, crx, cry, crz,
each parameter in exactly one gate (affine coefficients are allowed). Shared
parameters (QAOA's gamma) and trainable `u` angles raise QM-E832; restrict
the trainable set with `wrt`.

**Block-diagonal metrics are PennyLane-identical only for layered circuits.**
For a layered circuit (each parametrized layer's generators commute back to
the layer start, as in the QNG demo), the blocks of `"full"` equal
`"block-diag"` exactly (test: 1e-14). For other circuits `"block-diag"` is an
approximation: QUMO reproduces PennyLane's approximation (same layering rule,
same pre-layer state), and `"full"` gives the exact metric (validated against
finite-difference derivative states in the tests).

### `datasets.mojo` (QM-E831, QM-E834)

| API | Notes |
|---|---|
| `parity_dataset(n)` | all 2^n bitstrings, first character = wire 0; labels +1 odd / -1 even |
| `double_cake(num_sectors=3)` | PennyLane `make_double_cake_data`, labels +/-1, bit-identical |
| `circle_dataset(n, seed, radius=sqrt(2/pi))` | uniform in [-1,1]^2, labels 1 inside / 0 outside (QUMO RNG) |
| `moons_dataset(n, noise, seed, shuffle=True)` | sklearn `make_moons` recipe, labels 0/1 (QUMO RNG, not bitwise sklearn) |
| `bars_and_stripes(rows, cols)` | valid patterns as basis integers in tutorial_qcbm order |
| `load_csv(path, label_column=-1, skip_header=True) -> (X, y)` | numeric CSV, `#` comments |
| `load_json_matrix(path, key)` | `key` may be "a/b" into nested objects; flat arrays become one row |
| `standard_scale(X) -> (Xs, mean, std)`; `apply_standard_scale` | population std (sklearn StandardScaler) |
| `l2_normalize_rows`, `pad_rows(X, width, value)`, `train_test_split(n, test_fraction, seed)`, `take_rows`, `take_values` | |

### `Rng` additions

`Rng.normal()` (Box-Muller, two uniforms per call, no cached spare, so
`split()` streams stay reproducible) and `Rng.normal_list(n, scale=1.0)`.

## Error codes

| Code | Meaning |
|---|---|
| QM-E830 | invalid template / embedding / observable arguments |
| QM-E831 | shape mismatch in kernels, datasets, losses, batches |
| QM-E832 | metric tensor / QNG: unsupported or shared trainable parameter, unknown approx |
| QM-E833 | optimizer state or settings (length change between steps, lr < 0, beta outside [0, 1)) |
| QM-E834 | dataset file missing or malformed |

See [ERRORS.md](ERRORS.md).

## Verification

All reference numbers below are asserted in the tests (tolerance 1e-12 unless
noted). They were generated once with PennyLane 0.45.1 (`default.qubit`),
sklearn 1.9.1, numpy and Qiskit 2.5.2 in a throw-away environment; QUMO has no
runtime dependency on them.

| Check | Reference | QUMO |
|---|---|---|
| Parity classifier (BasisState + Rot + CNOT ring), w = 0.1(l+1) + 0.01q + 0.001k, x = 0101 | <Z0> 0.9065742515591111; dE/dw1 -0.10562131348797613 | 7e-16 |
| 30-step full-batch Nesterov(0.5) on 16 parity strings (seed-0 init) | 1.9992624749350514 ... 0.002525821086497758 | 6e-15 (asserted 1e-10) |
| `mottonen_angles([0.539, 0.795, 0.278, 0])` | get_angles [0.5640, -0, 0, -0.9750, 0.9750] | 8e-13 (get_angles adds 1e-12 inside its square roots) |
| StronglyEntangling / BasicEntangler / ArbitraryUnitary expectation values | PennyLane | 1e-12 |
| Qiskit ZZFeatureMap(3, reps=2) probabilities and kernel | Qiskit Statevector | 1e-12 |
| Angle kernel (iris pair) / IQP(n_repeats=2) kernel | 0.24695866074501246 / 0.280132222844837 | 1e-12 |
| QCNN forward from amplitude-embedded digit | p = [0.4247292672981652, 0.5752707327018325], dp/dW, dp/dlast | 1e-11 / 1e-10 |
| DoubleCake kernel (5 qubits x 6 layers, seed 1359): K, dk01/dtheta, KTA, dKTA/dtheta | PennyLane | K 1.6e-14, KTA 5e-15, dKTA 1.4e-16 |
| C-SVC decision values vs sklearn `SVC(kernel="precomputed")` | sklearn | 2e-8 (asserted 1e-6) |
| QNG demo: metric diagonal, g23; 200 QNG steps; 200 GD steps | PennyLane | 1e-16 / 2e-16 (asserted 1e-14 / 1e-10) |
| Adam (both forms), Momentum, Nesterov, Adagrad, RMSProp, 3 steps on a quadratic | PennyLane optimizers / optax rule | 2e-16 |

Performance is printed by the tests (not asserted): 64 parallel
value-and-gradient evaluations of a 4-qubit, 28-parameter model take about
0.5-2 ms; 3000 DoubleCake kernel-entry gradients about 0.2-0.8 s on a laptop.
