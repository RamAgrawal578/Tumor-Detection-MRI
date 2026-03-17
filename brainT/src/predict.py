import argparse

import numpy as np
from PIL import Image
from tensorflow import keras

from .config import IMG_SIZE, MODEL_PATH
from .utils import load_class_names, preprocess_pil_image


def predict_image(image_path: str) -> tuple[str, float, list[tuple[str, float]]]:
    model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    image = Image.open(image_path)
    batch = preprocess_pil_image(image, IMG_SIZE)
    probabilities = model.predict(batch, verbose=0)[0]

    top_index = int(np.argmax(probabilities))
    top_label = class_names[top_index]
    top_confidence = float(probabilities[top_index])

    ranked = sorted(
        zip(class_names, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return top_label, top_confidence, ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict brain tumor class from one MRI image.")
    parser.add_argument("image_path", help="Path to an image file")
    args = parser.parse_args()

    label, confidence, ranked = predict_image(args.image_path)
    print(f"Predicted class: {label}")
    print(f"Confidence: {confidence:.4f}\n")
    print("All class probabilities:")
    for class_name, score in ranked:
        print(f"- {class_name}: {score:.4f}")


if __name__ == "__main__":
    main()
