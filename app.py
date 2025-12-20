QuantumShield/app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from src.model import build_model
from src.quantum_layer import NUM_QUBITS

# -------------------------
# Configuración inicial
# -------------------------
st.set_page_config(page_title="QuantumShield", layout="centered")
st.title("🛡️ QuantumShield – Detección Cuántica de Ataques")

# -------------------------
# Datos de ejemplo
# -------------------------
texts = [
    "SQL injection attempt detected",
    "Normal user login",
    "DDoS attack from multiple IPs",
    "User accessed dashboard",
    "Cross-site scripting payload",
    "File upload successful"
]

labels = [1, 0, 1, 0, 1, 0]

# -------------------------
# Preprocesamiento
# -------------------------
vectorizer = TfidfVectorizer(max_features=50)
X = vectorizer.fit_transform(texts).toarray()

pca = PCA(n_components=NUM_QUBITS)
X_reduced = pca.fit_transform(X)

# -------------------------
# Modelo
# -------------------------
model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_reduced, np.array(labels), epochs=10, verbose=0)

# -------------------------
# Interfaz
# -------------------------
user_input = st.text_area("Introduce un log o evento de seguridad:")

if st.button("Analizar"):
    if user_input.strip() == "":
        st.warning("Ingresa un texto para analizar.")
    else:
        vec = vectorizer.transform([user_input]).toarray()
        vec_pca = pca.transform(vec)

        prediction = model.predict(vec_pca)
        attack_prob = prediction[0][1]

        if attack_prob > 0.5:
            st.error(f"🚨 Ataque detectado ({attack_prob:.2f})")
        else:
            st.success(f"✅ Tráfico normal ({1-attack_prob:.2f})")