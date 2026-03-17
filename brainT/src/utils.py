import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import (
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    HISTORY_PLOT_PATH,
    MODELS_DIR,
    OUTPUT_DIR,
)


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_class_names(class_names: Iterable[str]) -> None:
    ensure_directories()
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, indent=2)


# 🔥 FIXED FUNCTION (IMPORTANT)
def load_class_names(path=None) -> list[str]:
    """
    Loads class names from JSON file.
    Works both for training (config path) and deployment (custom path).
    """
    if path is None:
        path = CLASS_NAMES_PATH

    # Convert Path → string if needed
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"class_names.json not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_pil_image(image: Image.Image, img_size: tuple[int, int]) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(img_size)
    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return array


def merge_histories(history_a: dict, history_b: dict) -> dict:
    keys = set(history_a.keys()) | set(history_b.keys())
    merged = {}
    for key in keys:
        merged[key] = history_a.get(key, []) + history_b.get(key, [])
    return merged


def plot_training_history(history: dict) -> Path:
    ensure_directories()
    epochs = range(1, len(history.get("loss", [])) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get("accuracy", []), label="Train Accuracy")
    plt.plot(epochs, history.get("val_accuracy", []), label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get("loss", []), label="Train Loss")
    plt.plot(epochs, history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    return HISTORY_PLOT_PATH


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> Path:
    ensure_directories()
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.0 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    return CONFUSION_MATRIX_PATH
