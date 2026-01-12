import pennylane.numpy as np
from pennylane import grad
from hybrid_model import build_params, forward

def bce(y, yhat, eps=1e-7):
    yhat = np.clip(yhat, eps, 1 - eps)
    return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

def make_dataset(n=120, seed=1):
    rng = np.random.default_rng(seed)

    # X en rango [0, 2pi] con 4 features
    X = rng.uniform(0, 2*np.pi, size=(n, 4))

    # Labels dummy: clase 1 si suma de features supera umbral (toy problem)
    y = (np.sum(X, axis=1) > (4*np.pi)).astype(float)

    return X, y

def loss_fn(params, X, y):
    # promedio BCE
    total = 0.0
    for i in range(len(X)):
        yhat = forward(X[i], params)
        total = total + bce(y[i], yhat)
    return total / len(X)

def accuracy(params, X, y):
    correct = 0
    for i in range(len(X)):
        yhat = forward(X[i], params)
        pred = 1.0 if yhat >= 0.5 else 0.0
        correct += (pred == y[i])
    return correct / len(X)

def train(epochs=15, lr=0.2, seed=0):
    X, y = make_dataset()
    params = build_params(seed=seed)

    # gradientes w.r.t. todos los params
    dloss = grad(loss_fn, argnum=0)

    for epoch in range(1, epochs + 1):
        L = loss_fn(params, X, y)
        acc = accuracy(params, X, y)

        grads = dloss(params, X, y)

        # SGD update: params y grads son tuplas
        params = tuple(p - lr * g for p, g in zip(params, grads))

        print(f"Epoch {epoch:02d} | loss={L:.4f} | acc={acc:.3f}")

    print("OK ✅ entrenamiento completado")
    return params

if __name__ == "__main__":
    train()