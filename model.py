import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers, models
from quantum_layer import quantum_circuit, NUM_QUBITS

weight_shapes = {"weights": (NUM_QUBITS,)}

quantum_layer = qml.qnn.KerasLayer(
    quantum_circuit,
    weight_shapes,
    output_dim=NUM_QUBITS
)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(NUM_QUBITS,)),
        quantum_layer,
        layers.Dense(8, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])
    return model