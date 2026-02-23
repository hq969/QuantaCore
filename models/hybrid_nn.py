import torch
import torch.nn as nn
import pennylane as qml
from quantum.circuits import quantum_neural_network, get_quantum_weight_shapes
from config import config

class QuantaCoreHybridModel(nn.Module):
    """PyTorch Module mapping CPU/GPU tensors to QPU circuits."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        # Classical feature extraction mapping to the number of available qubits
        self.classical_in = nn.Linear(input_dim, config.N_QUBITS)
        self.relu = nn.ReLU()
        
        # Quantum Torch Layer (handles exact gradient calculations natively)
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_neural_network, 
            weight_shapes=get_quantum_weight_shapes()
        )
        
        # Classical decoder
        self.classical_out = nn.Linear(config.N_QUBITS, output_dim)

    def forward(self, x):
        x = self.classical_in(x)
        x = self.relu(x)
        x = x * torch.pi  # Scale inputs to quantum phase angles [-pi, pi]
        
        x = self.quantum_layer(x)
        x = self.classical_out(x)
        return x
