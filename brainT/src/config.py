from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "brain_tumor_mri_dataset"
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODELS_DIR / "brain_tumor_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
HISTORY_PLOT_PATH = OUTPUT_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42

EPOCHS_HEAD = 8
EPOCHS_FINE_TUNE = 5
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
FINE_TUNE_AT = 20
