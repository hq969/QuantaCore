# QuantaCore: Quantum Computing Architectures for AI Acceleration

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch_%7C_PennyLane-orange.svg)
![Status](https://img.shields.io/badge/Status-Active_Research-success.svg)

**QuantaCore** is a hybrid orchestration framework designed to bridge classical deep learning infrastructure with Quantum Processing Units (QPUs). By offloading intractable optimization bottlenecks and highly entangled tensor operations to quantum circuits, QuantaCore accelerates AI workloads that are fundamentally bounded by classical silicon constraints.

This repository demonstrates the practical application of Noisy Intermediate-Scale Quantum (NISQ) algorithms—specifically Variational Quantum Circuits (VQCs) and the Quantum Approximate Optimization Algorithm (QAOA)—seamlessly integrated into classical PyTorch pipelines.

---

## 🚀 Core Objectives & Capabilities

* **Seamless Hybridization:** Integrates Parameterized Quantum Circuits (PQCs) directly into classical neural network layers. Classical CPUs/GPUs handle data loading and feature extraction, while the QPU acts as a specialized layer for complex pattern recognition.
* **Quantum Gradient Descent:** Utilizes the parameter-shift rule, allowing classical optimizers (e.g., Adam, SGD) to calculate exact gradients of quantum circuits and backpropagate across the quantum-classical boundary.
* **Combinatorial Optimization (QAOA):** Includes dedicated modules for solving NP-Hard graph problems (routing, scheduling, max-cut) using quantum state mixers and cost Hamiltonians.
* **Hardware-Agnostic Execution:** Built on PennyLane, allowing seamless switching between local classical simulators (`default.qubit`) and physical quantum hardware (e.g., AWS Braket, IBM Qiskit, Google Cirq).

---

## 🏗️ System Architecture



QuantaCore relies on a modular, hardware-aware architecture:
1. **Classical Front-End:** Dimensionality reduction and feature encoding using PyTorch.
2. **Quantum Embedding:** Mapping classical continuous variables into high-dimensional Hilbert space via Angle Embedding.
3. **Quantum Processing:** Applying Strongly Entangling Layers to exploit quantum superposition and entanglement.
4. **Classical Back-End:** Measurement (Pauli-Z expectation values) decoded by classical linear layers for final classification or regression.

---

## 🗂️ Repository Structure

```text
quantacore/
├── quantum/
│   ├── __init__.py
│   ├── circuits.py        # Variational Quantum Circuits (VQCs) & PQCs
│   └── qaoa.py            # Quantum Approximate Optimization Algorithm modules
├── models/
│   ├── __init__.py
│   └── hybrid_nn.py       # PyTorch nn.Module combining CPU/GPU and QPU layers
├── training/
│   ├── __init__.py
│   └── trainer.py         # Optimization loop handling hybrid backpropagation
├── config.py              # Centralized hyperparameters and QPU device target settings
├── main.py                # Entry point for Hybrid Neural Network classification
├── run_qaoa.py            # Entry point for NP-Hard routing/optimization tasks
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.10+
* A virtual environment manager (e.g., `venv` or `conda`)

### 1. Clone the Repository

```bash
git clone [https://github.com/hq969/quantacore.git](https://github.com/hq969/quantacore.git)
cd quantacore

```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

*(Key dependencies include `torch`, `pennylane`, `numpy`, `scikit-learn`, and `networkx`.)*

---

## 🏃‍♂️ Usage Examples

QuantaCore provides two distinct execution paths depending on the workload: Machine Learning (Classification) and Combinatorial Optimization.

### 1. Hybrid Quantum-Classical Deep Learning

To train the hybrid neural network on a highly non-linear dataset (synthetic moons), run the main entry point. This demonstrates continuous backpropagation across the PyTorch-PennyLane bridge.

```bash
python main.py

```

*Expected Output:* The system will initialize the quantum simulator and output epoch-by-epoch loss and accuracy metrics as the classical Adam optimizer updates the quantum gate rotation angles.

### 2. Quantum Optimization (QAOA)

To run the Quantum Approximate Optimization Algorithm on a simulated logistics/graph problem, execute the QAOA script:

```bash
python run_qaoa.py

```

*Expected Output:* The optimizer will iteratively minimize the Cost Hamiltonian, eventually outputting the optimized parameters (Gammas and Betas) required to solve the target graph separation.

---

## 🛠️ Technology Stack

* **Deep Learning Framework:** PyTorch
* **Quantum Compiler & Simulator:** PennyLane (Xanadu)
* **Optimization & Graphing:** NetworkX, SciPy
* **Data Processing:** Scikit-Learn, NumPy

---

## 🔮 Future Roadmap

* **Hardware Integration:** Add direct configuration flags to execute circuits on physical QPUs (e.g., IonQ, Rigetti) via cloud providers.
* **VQE Implementation:** Expand the `quantum/` module to include Variational Quantum Eigensolvers for materials science simulations.
* **Error Mitigation:** Implement Zero-Noise Extrapolation (ZNE) wrappers around the quantum nodes to improve fidelity on actual NISQ hardware.

---

## 🤝 Contributing

Contributions are welcome, particularly in expanding hardware backend support and designing novel ansatzes for the quantum layers. Please submit a pull request or open an issue for major architectural changes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---
