import json

import tensorflow as tf
from tensorflow import keras

from .config import (
    BATCH_SIZE,
    EPOCHS_FINE_TUNE,
    EPOCHS_HEAD,
    FINE_TUNE_AT,
    IMG_SIZE,
    LEARNING_RATE_FINE_TUNE,
    LEARNING_RATE_HEAD,
    METRICS_PATH,
    MODEL_PATH,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_SPLIT,
)
from .model_builder import build_model, compile_model, unfreeze_last_layers
from .utils import ensure_directories, merge_histories, plot_training_history, save_class_names

AUTOTUNE = tf.data.AUTOTUNE


def get_datasets():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DIR}\n"
            "Put your dataset here: data/brain_tumor_mri_dataset/Training and Testing"
        )
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Testing folder not found: {TEST_DIR}\n"
            "Put your dataset here: data/brain_tumor_mri_dataset/Training and Testing"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def main() -> None:
    ensure_directories()
    train_ds, val_ds, test_ds, class_names = get_datasets()
    save_class_names(class_names)

    model, base_model = build_model(num_classes=len(class_names))
    model = compile_model(model, LEARNING_RATE_HEAD)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    unfreeze_last_layers(base_model, fine_tune_at=FINE_TUNE_AT)
    model = compile_model(model, LEARNING_RATE_FINE_TUNE)

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FINE_TUNE,
        initial_epoch=history_head.epoch[-1] + 1,
        callbacks=callbacks,
    )

    combined_history = merge_histories(history_head.history, history_fine.history)
    plot_training_history(combined_history)

    best_model = keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": class_names,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining finished successfully.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
