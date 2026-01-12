import pennylane as qml
import tensorflow as tf

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

weight_shapes = {"weights": (NUM_QUBITS,)}

quantum_layer = qml.qnn.KerasLayer(
    quantum_circuit,
    weight_shapes,
    output_dim=NUM_QUBITS,
)