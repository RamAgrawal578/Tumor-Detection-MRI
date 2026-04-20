import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow import keras

from src.config import IMG_SIZE
from src.utils import load_class_names, preprocess_pil_image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.prediction {
    font-size: 28px;
    font-weight: bold;
    color: #38bdf8;
}

.confidence {
    font-size: 18px;
    color: #22c55e;
}

</style>
""", unsafe_allow_html=True)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_classifier.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)
    return model, class_names

model, class_names = load_model()

# ---------------- LABEL FIX ----------------
def prettify_label(label: str):
    cleaned = label.lower().replace("-", "").replace("_", "")
    if cleaned == "notumor":
        return "No Tumor"
    return label.title()

# ---------------- HEADER ----------------
st.markdown('<div class="title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload MRI image to classify tumor type</div>', unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded MRI", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    batch = preprocess_pil_image(image, IMG_SIZE)

    with st.spinner("Analyzing MRI..."):
        preds = model.predict(batch)[0]

    best_idx = int(np.argmax(preds))
    prediction = prettify_label(class_names[best_idx])
    confidence = round(float(preds[best_idx]) * 100, 2)

    # ---------------- RESULT CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">Prediction: {prediction}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">Confidence: {confidence}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PROBABILITIES ----------------
    st.markdown("### 📊 Class Probabilities")

    for label, score in zip(class_names, preds):
        nice_label = prettify_label(label)
        percent = round(score * 100, 2)

        st.write(f"**{nice_label}** — {percent}%")
        st.progress(float(score))
