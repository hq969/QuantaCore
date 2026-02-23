import pennylane as qml
from config import config

# Initialize the Quantum Device
dev = qml.device(config.QUANTUM_DEVICE, wires=config.N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_neural_network(inputs, weights):
    """Parameterized Quantum Circuit (PQC) for feature processing."""
    # Encode classical data into quantum phase angles
    qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS))
    
    # Highly entangled trainable quantum layers
    qml.StronglyEntanglingLayers(weights, wires=range(config.N_QUBITS))
    
    # Extract classical values via Pauli-Z expectation
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(config.N_QUBITS)]

def get_quantum_weight_shapes():
    """Returns the tensor shape required for the quantum weights."""
    return {"weights": (config.Q_LAYERS, config.N_QUBITS, 3)}
