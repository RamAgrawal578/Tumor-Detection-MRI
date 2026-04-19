import os
import uuid
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow import keras
from werkzeug.utils import secure_filename

from src.config import IMG_SIZE
from src.utils import load_class_names, preprocess_pil_image

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_classifier.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 🔥 STRONG LABEL CLEANER (FIXED)
def prettify_label(label: str) -> str:
    cleaned = str(label).lower().replace("-", "").replace("_", "").replace(" ", "")

    if cleaned == "no-tumor":
        return "No Tumor"
    elif cleaned == "glioma":
        return "Glioma"
    elif cleaned == "meningioma":
        return "Meningioma"
    elif cleaned == "pituitary":
        return "Pituitary"
    
    return str(label).title()


def load_assets():
    model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)
    return model, class_names


model, class_names = load_assets()


@app.route("/")
def home():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return (
            "Model files are missing. Ensure models/ contains "
            "brain_tumor_classifier.keras and class_names.json"
        )
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return (
            "Model files are missing. Ensure models/ contains "
            "brain_tumor_classifier.keras and class_names.json"
        )

    if "file" not in request.files:
        return "No file part in request"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    if not allowed_file(file.filename):
        return "Invalid file type. Please upload JPG, JPEG, or PNG."

    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"

    relative_path = os.path.join("static", "uploads", unique_name)
    absolute_path = os.path.join(BASE_DIR, relative_path)

    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    file.save(absolute_path)

    image = Image.open(absolute_path).convert("RGB")
    batch = preprocess_pil_image(image, IMG_SIZE)

    raw_probabilities = model.predict(batch, verbose=0)[0]
    best_idx = int(np.argmax(raw_probabilities))

    # 🔥 FIXED PREDICTION
    raw_label = class_names[best_idx]
    prediction = prettify_label(raw_label)

    confidence = round(float(raw_probabilities[best_idx]) * 100, 2)

    # 🔥 FIXED PROBABILITY LIST
    probability_items = sorted(
        [
            {
                "label": prettify_label(label),
                "score": float(score),
                "percent": round(float(score) * 100, 2),
            }
            for label, score in zip(class_names, raw_probabilities.tolist())
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        image_file=relative_path,
        probabilities=probability_items,
        uploaded_name=original_name,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True) 