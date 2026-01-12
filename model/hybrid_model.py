import tensorflow as tf
from quantum.quantum_layer import quantum_layer, NUM_QUBITS

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_QUBITS,)),
        quantum_layer,
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model