# Quantum machine learning examples (PennyLane demos, reproduced with QUMO)

Eleven PennyLane QML demos, rebuilt on QUMO's QML layer
([docs/QML.md](../../docs/QML.md): templates, embeddings, adjoint and batched
gradients, step optimizers, quantum kernels, the metric tensor). PennyLane is
not used at run time. Every example compares its numbers with a committed
PennyLane 0.45.1 fixture and prints machine-readable `CHECK` lines. Each file's
docstring cites the original demo, lists its checks and states any deviation.

```bash
.venv/bin/mojo run -I . examples/qml/variational_classifier.mojo
bench/.venv/bin/python examples/run_examples.py --only examples/qml      # all, with timeouts
.venv/bin/mojo run -I . examples/qml/barren_plateaus.mojo --max-qubits 12  # larger sweep (about 40 s)
```

Every example is in the fast tier: it runs offline in under 60 s on a laptop
CPU, and most of that time is Mojo compilation. `hybrid_torch_layer.py` needs
torch and the phase-2 `qumo_py` bindings (`value_and_gradient`, `vjp`), so the
runner reports SKIP until they exist.

| Demo | URL | QUMO file | What is checked | Reference numbers (provenance) |
|---|---|---|---|---|
| Basic tutorial: qubit rotation | https://pennylane.ai/qml/demos/tutorial_qubit_rotation | `qubit_rotation.mojo` | initial cost and gradient (1e-12); 100-step GD(0.4) trajectory of costs and parameters (1e-12); cost < -0.999999 first at step 37; final [0, pi] | fixture (trajectory, step 37 from a rerun) |
| Variational classifier (+ Qibo 3-class Iris) | https://pennylane.ai/qml/demos/tutorial_variational_classifier, https://qibo.science/qibo/stable/code-examples/tutorials/variational_classifier/README.html | `variational_classifier.mojo` | parity forward 0.9065742515591111 and 24 gradients (1e-12); 30-step full-batch Nesterov trajectory (1e-10); stochastic run with replayed batches (1e-8), train/test accuracy 1.0; Iris: get_angles == mottonen_angles (1e-11), <Z0> = 0.45177693213489395 and gradients (1e-12), 60 replayed Nesterov steps (1e-8), validation accuracy 1.0; Qibo-style 3-class Iris (4 qubits, 5 layers) test accuracy >= 0.9 | fixture; page final cost 1.74e-5 (quoted); Qibo page about 0.73 test accuracy with Powell and sampling (quoted; QUMO measures 0.921 with Adam and exact gradients) |
| Data-reuploading classifier (+ Qibo) | https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier, https://qibo.science/qibo/stable/code-examples/tutorials/reuploading_classifier/README.html | `data_reuploading.mojo` | initial fidelity 0.4214046402715622 (1e-12); 60 minibatch losses (first 20 at 1e-10, all at 1e-8); final parameters (1e-8); per-epoch accuracies and all 2000 test predictions equal; QPU circuit-run cost model printed | fixture rerun 0.89 / 0.847; page 0.870 / 0.779 (quoted: another NumPy draw order) |
| Kernel-based training with scikit-learn | https://pennylane.ai/qml/demos/tutorial_kernel_based_training | `quantum_kernel_svm.mojo` | K_train sum 1849.5885793209322 (1e-9); compute-uncompute circuit == state route (1e-12); IQP k = 0.280132222844837 (1e-12); sampled entries within 4 sigma; QUMO C-SVC vs sklearn SVC: dual objective (1e-4 relative), intercept (2e-3), decisions (5e-3); test accuracy 1.0; cost model 7500 vs 96025 circuit runs | fixture (sklearn 1.9.1); the page text says 74025, its formula 24 x 100 x 20 x 2 + 25 gives 96025 |
| Training and evaluating quantum kernels | https://pennylane.ai/qml/demos/tutorial_kernels_module | `kernel_alignment.mojo` | K_init (1e-12), k01 = 0.029983453404801606, KTA_init = 0.07433831454242083, dk01/dtheta and dKTA/dtheta (1e-12); 500 replayed GD(0.2) steps: alignment every 50 steps 0.0904 -> 0.2661 (1e-6), final parameters (1e-8); SVM accuracy 0.833 -> 1.000 | fixture; page 0.074, 0.0904, 0.2661 |
| Quantum natural gradient | https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient | `quantum_natural_gradient.mojo` | block-diag metric diagonal and g23 (1e-14); diag and full (aux-wire) metrics; 200 QNG steps at 1/10/50/100/200 and full trajectory (1e-10); 200 GD steps -0.3737189660016403 (1e-10) | fixture |
| Barren plateaus + local cost functions | https://pennylane.ai/qml/demos/tutorial_barren_plateaus, https://pennylane.ai/qml/demos/tutorial_local_cost_functions | `barren_plateaus.mojo` | values and gradients for 10 exported gate sequences (1e-12); global variance falls with n; ln-variance slope in [-1.3, -0.5] (2000 samples per width, QUMO RNG); local cost decays more slowly | fixture slope -0.955 (200 samples); PennyLane reruns -0.77 to -0.96; QUMO -0.873 (n = 2..6), -0.944 (n = 2..12) |
| Quantum models as Fourier series | https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series | `fourier_expressivity.mojo` | serial f(0) = 0.7068878706839132 (1e-12); 50-step Adam trajectory with the demo's 0.5 * MSE (1e-8); Fourier coefficients equal the fixture (1e-8) and vanish above degree r (1e-12); 3-qubit parallel model trajectory (1e-8) | fixture; research-rerun plain-MSE values 0.22856 ... 0.00026741 are twice the fixture's (1e-3 relative) |
| Quantum circuit Born machines | https://pennylane.ai/qml/demos/tutorial_qcbm | `qcbm.mojo` | initial MMD and probabilities (1e-12), gradient (1e-10) and finite differences (1e-6); 100-step optax-Adam MMD and KL trajectories (1e-8); final KL < 0.2, valid-pattern probability > 0.85 | fixture (JAX float64): KL 0.0750, valid 0.929 |
| Generalization in QML from few training data | https://pennylane.ai/qml/demos/tutorial_learning_few_data | `qcnn.mojo` | QCNN forward [0.4247292672981652, 0.5752707327018325] (1e-11), dp/dW and dp/dlast (1e-10), Mottonen-prepared circuit agrees (1e-12); all 40 training runs (sizes 5/10/20/40 x 10): cost trajectories (1e-8) and final accuracies equal; mean test accuracy at 40 >= 0.9 | fixture (JAX float64): mean test accuracy 0.74 / 0.871 / 0.962 / 0.946 |
| Turning quantum nodes into Torch layers | https://pennylane.ai/qml/demos/tutorial_qnn_module_torch | `hybrid_torch_layer.py` (phase 2) | layer forward and Jacobian (1e-10); per-epoch average loss equals PennyLane's (1e-6); accuracy equals the fixture | fixture; page 0.4943 -> 0.1666, accuracy 0.855 |

## Timings (indicative)

QUMO compute time inside each example (Apple M-series laptop CPU, compilation
excluded) against the PennyLane 0.45.1 `default.qubit` times recorded in the
fixture by the generator. They are different engines and different code
(PennyLane runs these demos per sample through autograd, or through JAX
`jit` where the demo does), so read them as orders of magnitude, not a benchmark.

| Example | QUMO | PennyLane (fixture) |
|---|---|---|
| qubit_rotation, 100 GD steps | < 0.001 s | 1.2 s |
| variational_classifier: parity full-batch / stochastic / Iris | 0.017 / 0.11 / 0.11 s | 115 / 137 / 295 s |
| data_reuploading, 10 epochs | 0.05 s | 702 s |
| quantum_kernel_svm, 75 x 75 Gram matrix angle / IQP | 0.001 / 0.001 s | 61 / 151 s |
| kernel_alignment, 500 alignment steps | 0.47 s | 1382 s (page: about 317 s) |
| quantum_natural_gradient, 200 QNG / GD steps | 0.002 / 0.001 s | 17.7 / 2.7 s |
| barren_plateaus, gradient statistics | 0.12 s (2000 samples x 2 costs, n = 2..6) | 42 s (200 samples) |
| fourier_expressivity, serial / parallel training | 0.04 / 0.03 s | 160 / 661 s |
| qcbm, 100 iterations (JAX jit) | 0.06 s | 15 s |
| qcnn, 40 training runs (JAX jit) | 1.5 s | 286 s |

GPU: these circuits have 1 to 9 qubits, far below the size at which QUMO's
GPU state vector pays off (see docs/GPU.md). The examples run on the CPU state
vector; their speed comes from adjoint gradients and from batching parameter
sets in parallel (`batch_value_and_gradient`), not from the GPU.

## Reference fixtures

`reference/pennylane_reference.json` holds, per example, the data, initial
parameters, batch indices, forward values, gradients, per-step trajectories,
final metrics and PennyLane wall times. It is generated by
`reference/pennylane_reference.py` in a separate environment. PennyLane,
scikit-learn, JAX, optax and torch are not QUMO dependencies:

```bash
uv venv /tmp/qumo-qml-ref --python 3.12
uv pip install --python /tmp/qumo-qml-ref/bin/python -r examples/qml/reference/requirements.txt
/tmp/qumo-qml-ref/bin/python examples/qml/reference/pennylane_reference.py --write-data   # rewrite data/ (no download)
/tmp/qumo-qml-ref/bin/python examples/qml/reference/pennylane_reference.py               # every section (about 1 h)
/tmp/qumo-qml-ref/bin/python examples/qml/reference/pennylane_reference.py --only qcbm   # one section
```

Where PennyLane's own reruns do not reproduce the page numbers (data
re-uploading, barren-plateau slopes, the research rerun of the Fourier demo),
the examples assert the committed fixture by exact replay of its random draws
(initial weights, batch indices, data subsets) and quote the page values
without asserting them. Where QUMO's random stream necessarily differs
(barren-plateau sampling), the checks are statistical bands.

## Data

`data/` is described in [data/README.md](data/README.md). Everything there was
generated from scikit-learn's bundled copies of the UCI data sets or from
deterministic synthetic recipes; nothing is downloaded.

## Conventions

* Weights first, then features, in every parameter vector (docs/QML.md).
* PennyLane `Rot(phi, theta, omega)` is RZ(phi), RY(theta), RZ(omega); `BasisState(x)` is RX(pi x).
* When amplitudes or multi-wire probabilities matter (QCBM, QCNN), PennyLane wire `w` is QUMO qubit `n-1-w`.
* `_common.mojo` is a helper module (fixture reader, table loader), not an example.
