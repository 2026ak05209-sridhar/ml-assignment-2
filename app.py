from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

app_folder = Path(__file__).resolve().parent

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.title("Breast Cancer Classification")
st.write("Upload a labeled test CSV and choose a trained model to evaluate.")
st.caption("Diagnosis labels: 0 = malignant, 1 = benign.")

uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])
selected_model = st.selectbox("Choose a classifier", list(model_files))

if uploaded_file is None:
    st.stop()

try:
    test_data = pd.read_csv(uploaded_file)
except (ValueError, OSError):
    st.error("Could not read the uploaded file. Please upload a valid CSV.")
    st.stop()

if test_data.empty:
    st.error("The uploaded CSV has no data rows.")
    st.stop()

# Read the feature names in the same order used during training.
feature_columns = pd.read_csv(
    app_folder / "test_data.csv", nrows=0
).columns.drop("diagnosis").tolist()

required_columns = feature_columns + ["diagnosis"]
missing_columns = [
    column for column in required_columns if column not in test_data.columns
]

if missing_columns:
    st.error("Missing columns: " + ", ".join(missing_columns))
    st.stop()

actual_labels = test_data["diagnosis"]
if not actual_labels.isin([0, 1]).all():
    st.error("The diagnosis column must contain only 0 and 1.")
    st.stop()

try:
    test_features = test_data[feature_columns].to_numpy(dtype=float)
except (ValueError, TypeError):
    st.error("All feature values must be numeric.")
    st.stop()

if not np.isfinite(test_features).all():
    st.error("The feature columns contain missing or infinite values.")
    st.stop()

# Load the selected model and predict using the uploaded measurements.
model_path = app_folder / "model" / model_files[selected_model]
model = joblib.load(model_path)
predictions = model.predict(test_features)

st.write(f"Evaluated {len(predictions)} records using {selected_model}.")

# Use malignant (label 0) as the positive class.
positive_label = 0
malignant_column = list(model.classes_).index(positive_label)
malignant_scores = model.predict_proba(test_features)[:, malignant_column]

auc = "N/A (requires both classes)"
if actual_labels.nunique() == 2:
    auc_value = roc_auc_score(actual_labels == positive_label, malignant_scores)
    auc = f"{auc_value:.4f}"

precision = precision_score(
    actual_labels, predictions, pos_label=positive_label, zero_division=0
)
recall = recall_score(
    actual_labels, predictions, pos_label=positive_label, zero_division=0
)
f1 = f1_score(
    actual_labels, predictions, pos_label=positive_label, zero_division=0
)

metrics = {
    "Accuracy": f"{accuracy_score(actual_labels, predictions):.4f}",
    "AUC": auc,
    "Precision": f"{precision:.4f}",
    "Recall": f"{recall:.4f}",
    "F1": f"{f1:.4f}",
    "MCC": f"{matthews_corrcoef(actual_labels, predictions):.4f}",
}

st.subheader("Evaluation metrics")
st.table(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]))

matrix = confusion_matrix(actual_labels, predictions, labels=[0, 1])

st.subheader("Confusion matrix")
st.table(pd.DataFrame(
    matrix,
    index=["Actual malignant", "Actual benign"],
    columns=["Predicted malignant", "Predicted benign"],
))
