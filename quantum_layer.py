import pennylane as qml
import tensorflow as tf

# Número de qubits
NUM_QUBITS = 4
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    # Codificación de datos
    for i in range(NUM_QUBITS):
        qml.RY(inputs[i], wires=i)

    # Ansatz (parametrizado)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i], wires=i)

    # Entrelazamiento (opcional)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    # Medidas
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

# Especificar formas de los parámetros del circuito
weight_shapes = {"weights": (NUM_QUBITS,)}

# Capa de Keras con salida NUM_QUBITS
quantum_layer = qml.qnn.KerasLayer(
    quantum_circuit, weight_shapes, output_dim=NUM_QUBITS
)