# ML Assignment 2 — Breast Cancer Classification
## Problem Statement
The goal of this project is to classify tumors as malignant or benign using measurements from the Wisconsin Diagnostic Breast Cancer dataset, available through scikit-learn. We evaluated five classifiers (Decision Tree, K-Nearest Neighbors, Logistic Regression, Gaussian Naive Bayes, and Support Vector Machine) on the same held-out test set of 114 records. The Streamlit app evaluates the selected model on an uploaded, labeled test CSV and displays six evaluation metrics and a confusion matrix.

## Project files

- `ML_Assignment_2.ipynb` — Model training and evaluation notebook.
- `app.py` — Streamlit app.
- `requirements.txt` — Python package versions.
- `test_data.csv` — Held-out test set with 114 records, 30 features, and a `diagnosis` column.
- `model/` — Saved files for the five trained classifiers.

## Run the app

From the project folder, run these commands in a Python 3.12 environment.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal. Upload `test_data.csv`, select a classifier, and scroll down to view the evaluation metrics and confusion matrix. In the CSV, `0` represents malignant and `1` represents benign. Precision, recall, and F1 use malignant as the positive class.

## Model comparison

All five classifiers were evaluated on the same 114 held-out records.

| Classifier | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9825 | 0.9954 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.8478 | 0.9286 | 0.8864 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9268 | 0.9048 | 0.9157 | 0.8676 |
| SVM | 0.9825 | 0.9950 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |

Logistic Regression and SVM tied for the highest accuracy at 98.25%. Logistic Regression had the highest AUC at 0.9954.
