# Brain Tumor MRI Classifier Starter

This is a **classification** project, not segmentation. It predicts one label for each MRI image.

## Classes
- glioma
- meningioma
- pituitary
- no tumor

## Folder structure expected by the code

```text
brain_tumor_classifier_starter/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── brain_tumor_mri_dataset/
│       ├── Training/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── pituitary/
│       │   └── notumor/   # or no_tumor if your dataset uses that name consistently
│       └── Testing/
│           ├── glioma/
│           ├── meningioma/
│           ├── pituitary/
│           └── notumor/
├── models/
├── outputs/
└── src/
    ├── config.py
    ├── model_builder.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    └── utils.py
```

## 1) Create a virtual environment

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Download the dataset

### Kaggle option
```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip -p data
```

After extraction, make sure you end up with:

```text
data/brain_tumor_mri_dataset/Training
data/brain_tumor_mri_dataset/Testing
```

If the extracted folder has a different name, rename it to `brain_tumor_mri_dataset`.

## 4) Train the model

```bash
python -m src.train
```

This will save:
- `models/brain_tumor_classifier.keras`
- `models/class_names.json`
- `outputs/training_history.png`
- `outputs/test_metrics.json`

## 5) Evaluate the model again

```bash
python -m src.evaluate
```

This will save:
- `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`

## 6) Predict one image from the terminal

```bash
python -m src.predict path/to/your_image.jpg
```

## 7) Run the web app locally

```bash
streamlit run app.py
```

## 8) Deploy to Streamlit Community Cloud

1. Push this project to GitHub.
2. Make sure `models/brain_tumor_classifier.keras` and `models/class_names.json` are in the repo.
3. In Streamlit Community Cloud, click **Create app**.
4. Select your repository, branch, and `app.py` as the entrypoint file.
5. Click **Deploy**.

## Notes
- This project uses transfer learning with MobileNetV2.
- It is meant for learning and prototyping.
- It is **not** a medical device and should not be used for clinical decisions.
