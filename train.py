import numpy as np
from model.hybrid_model import build_model

# Datos dummy: 100 muestras, 4 features (uno por qubit)
X = np.random.uniform(0, 2 * np.pi, size=(100, 4)).astype("float32")
y = np.random.randint(0, 2, size=(100, 1)).astype("float32")

model = build_model()
model.summary()

history = model.fit(
    X, y,
    epochs=3,
    batch_size=8,
    verbose=1,
)

print("OK ✅ entrenamiento completado")