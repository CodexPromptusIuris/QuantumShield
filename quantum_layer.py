import pennylane as qml
from pennylane import numpy as np

NUM_QUBITS = 4
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    for i in range(NUM_QUBITS):
        qml.RY(inputs[i], wires=i)

    for i in range(NUM_QUBITS):
        qml.RX(weights[i], wires=i)

    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]