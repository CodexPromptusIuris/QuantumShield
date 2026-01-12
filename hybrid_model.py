import pennylane.numpy as np
from quantum_layer import quantum_circuit, NUM_QUBITS

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def build_params(seed=0):
    rng = np.random.default_rng(seed)

    # Capa clásica: proyecta 4->4 para alimentar el quantum
    W_in = np.array(rng.normal(0, 0.5, size=(NUM_QUBITS, NUM_QUBITS)), requires_grad=True)
    b_in = np.array(rng.normal(0, 0.1, size=(NUM_QUBITS,)), requires_grad=True)

    # Pesos del circuito cuántico (4,)
    w_q = np.array(rng.normal(0, 0.2, size=(NUM_QUBITS,)), requires_grad=True)

    # Capa salida clásica: 4->1
    W_out = np.array(rng.normal(0, 0.5, size=(NUM_QUBITS, 1)), requires_grad=True)
    b_out = np.array(rng.normal(0, 0.1, size=(1,)), requires_grad=True)

    return W_in, b_in, w_q, W_out, b_out

def forward(x, params):
    W_in, b_in, w_q, W_out, b_out = params

    # (4,) -> (4,)
    x_proj = np.dot(x, W_in) + b_in

    # limitar rango (opcional) para evitar inputs enormes al circuito
    x_proj = np.tanh(x_proj) * np.pi

    # quantum -> (4,)
    q_out = quantum_circuit(x_proj, w_q)

    # salida -> (1,)
    logit = np.dot(q_out, W_out) + b_out
    yhat = sigmoid(logit)[0]
    return yhat