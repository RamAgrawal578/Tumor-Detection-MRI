import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from .config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    METRICS_PATH,
    MODEL_PATH,
    REPORT_PATH,
    TEST_DIR,
)
from .utils import ensure_directories, load_class_names, plot_confusion_matrix


def main() -> None:
    ensure_directories()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    class_names = load_class_names()
    model = keras.models.load_model(MODEL_PATH)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical",
    )

    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds], axis=0)
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    plot_confusion_matrix(cm, class_names)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": class_names,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation complete.")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    main()
