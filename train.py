import numpy as np
from model.hybrid_model import build_model

# Dummy data (ejemplo de "shielding")
X = np.random.uniform(0, 2 * np.pi, size=(100, 4))
y = np.random.randint(0, 2, size=(100, 1))

model = build_model()
model.summary()

model.fit(
    X,
    y,
    epochs=10,
    batch_size=8
)