# PennyLane signature examples, reproduced with QUMO

The PennyLane demos live in [examples/qml](../../qml/README.md), next to the
QML library they exercise ([docs/QML.md](../../../docs/QML.md)). Each one is
checked against a committed PennyLane 0.45.1 fixture
(`examples/qml/reference/pennylane_reference.json`).

```bash
bench/.venv/bin/python examples/run_examples.py --only examples/qml
```

| PennyLane demo | QUMO file |
|---|---|
| [Basic tutorial: qubit rotation](https://pennylane.ai/qml/demos/tutorial_qubit_rotation) | [`../../qml/qubit_rotation.mojo`](../../qml/qubit_rotation.mojo) |
| [Variational classifier](https://pennylane.ai/qml/demos/tutorial_variational_classifier) | [`../../qml/variational_classifier.mojo`](../../qml/variational_classifier.mojo) |
| [Data-reuploading classifier](https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier) | [`../../qml/data_reuploading.mojo`](../../qml/data_reuploading.mojo) |
| [Kernel-based training of quantum models with scikit-learn](https://pennylane.ai/qml/demos/tutorial_kernel_based_training) | [`../../qml/quantum_kernel_svm.mojo`](../../qml/quantum_kernel_svm.mojo) |
| [Training and evaluating quantum kernels](https://pennylane.ai/qml/demos/tutorial_kernels_module) | [`../../qml/kernel_alignment.mojo`](../../qml/kernel_alignment.mojo) |
| [Quantum natural gradient](https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient) | [`../../qml/quantum_natural_gradient.mojo`](../../qml/quantum_natural_gradient.mojo) |
| [Barren plateaus](https://pennylane.ai/qml/demos/tutorial_barren_plateaus) and [local cost functions](https://pennylane.ai/qml/demos/tutorial_local_cost_functions) | [`../../qml/barren_plateaus.mojo`](../../qml/barren_plateaus.mojo) |
| [Quantum models as Fourier series](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series) | [`../../qml/fourier_expressivity.mojo`](../../qml/fourier_expressivity.mojo) |
| [Quantum circuit Born machines](https://pennylane.ai/qml/demos/tutorial_qcbm) | [`../../qml/qcbm.mojo`](../../qml/qcbm.mojo) |
| [Generalization in QML from few training data](https://pennylane.ai/qml/demos/tutorial_learning_few_data) | [`../../qml/qcnn.mojo`](../../qml/qcnn.mojo) |
| [Turning quantum nodes into Torch layers](https://pennylane.ai/qml/demos/tutorial_qnn_module_torch) | [`../../qml/hybrid_torch_layer.py`](../../qml/hybrid_torch_layer.py) (needs torch and the phase-2 `qumo_py` bindings) |

PennyLane-to-QUMO translation (templates, optimizers, wire order) is in the
equivalence table of [docs/QML.md](../../../docs/QML.md#pennylane-equivalence-table).
