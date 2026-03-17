import os
import uuid
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow import keras
from werkzeug.utils import secure_filename

from src.config import CLASS_NAMES_PATH, IMG_SIZE, MODEL_PATH
from src.utils import load_class_names, preprocess_pil_image

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prettify_label(label: str) -> str:
    return str(label).replace("_", " ").replace("-", " ").title()


def load_assets():
    model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()
    return model, class_names


model, class_names = load_assets()


@app.route("/")
def home():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        return (
            "Model files are missing. Train the model first so models/ contains "
            "the saved model and class_names.json."
        )
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        return (
            "Model files are missing. Train the model first so models/ contains "
            "the saved model and class_names.json."
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
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    batch = preprocess_pil_image(image, IMG_SIZE)

    raw_probabilities = model.predict(batch, verbose=0)[0]
    best_idx = int(np.argmax(raw_probabilities))

    prediction = prettify_label(class_names[best_idx])
    confidence = round(float(raw_probabilities[best_idx]) * 100, 2)

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
        image_file=filepath,
        probabilities=probability_items,
        uploaded_name=original_name,
    )


if __name__ == "__main__":
    app.run(debug=True)