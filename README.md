# Fraudulent Bank Account Prediction

Predicting fraudulent bank-account applications from application data, comparing
logistic regression with tree-based ensembles on a large, highly imbalanced dataset
(about 1% fraud).

---

## Problem

Fraudsters open bank accounts with stolen or synthetic identities and use them for money
laundering and scams. Banks screen new applications, and every missed fraud is costly,
but so is every legitimate customer held up for review. The goal was a model that flags
fraudulent applications at the point of opening and explains which signals drive the risk.

## Data

[Bank Account Fraud (BAF) dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022),
NeurIPS 2022: one million synthetic but realistic account applications with 30+
features (income, address history, device and session data, contact details) and a
fraud label. The `Base.csv` variant was used. The data is not included in this repository
and remains under its own licence on Kaggle.

## Approach

| Step | Notebook | What it does |
|---|---|---|
| 1 | [`Step1_Data_Cleaning.ipynb`](Step1_Data_Cleaning.ipynb) | Missing-value markers (negative values) handled per feature with median fills and missing-flags, duplicate check, uninformative columns dropped; outputs `final_dataset.csv` |
| 2 | [`Step2_EDA.ipynb`](Step2_EDA.ipynb) | Class balance, feature distributions by fraud label, correlations |
| 3a | [`Step3a_LogisticRegression_Model.ipynb`](Step3a_LogisticRegression_Model.ipynb) | Baseline, L1 and L2 logistic regression; regularisation and decision-threshold analysis |
| 3b | [`Step3b_Tree_Based_Models.ipynb`](Step3b_Tree_Based_Models.ipynb) | Decision tree, random forest, gradient boosting, XGBoost; hyperparameter search; feature importance |
| 4 | [`Step4_Result_Comparison.xlsx`](Step4_Result_Comparison.xlsx) | Side-by-side metrics and best parameters |

Common procedure: one-hot encoding of categorical features, a stratified 70/30
train/test split, **random undersampling** of the majority class on the training set
only (the test set keeps the real 1% fraud rate), and cross-validated hyperparameter
tuning (grid search for logistic regression, randomised search for the tree models). Models were compared on
recall for the fraud class and ROC-AUC, since a missed fraud costs far more than a false alarm.

## Results (test set, 298,447 applications, 3,302 fraudulent)

| Model | Recall (fraud) | Precision (fraud) | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Logistic regression (L1) | 0.79 | 0.04 | 0.80 | 0.87 |
| Decision tree (tuned) | 0.97 | 0.01 | 0.23 | 0.71 |
| Random forest | 0.95 | 0.02 | 0.44 | 0.86 |
| Gradient boosting | 0.80 | 0.04 | 0.81 | **0.88** |
| XGBoost | **0.99** | 0.01 | 0.25 | 0.86 |

**Most important features** across the tree-based models: `device_os_windows`,
`prev_address_months_count` and `current_address_months_count`. Applications from
Windows devices and applicants with short or missing address histories carried markedly
higher fraud risk.

## Reading the results

The project selected **XGBoost**, which catches 99% of fraud. That recall comes at a price
the table makes visible: the model flags about 76% of all applications for review
(224,491 false alarms against 3,255 frauds caught). Logistic regression and gradient
boosting catch about 80% of fraud while flagging about 20%, which is a more workable
operating point for a review team, and gradient boosting has the best ROC-AUC.

The choice between them is a business decision about review capacity, not a property of
the models. Natural next steps:

- **Report recall at a fixed false-positive rate.** The BAF paper's standard is recall at
  5% FPR, which compares models at an operating point a bank could actually run.
- **Tune the decision threshold per model** against the cost of a missed fraud versus a
  manual review, rather than tuning for recall alone.
- **Investigate the gap between cross-validated and test ROC-AUC** (XGBoost: 0.99 vs
  0.86). ROC-AUC barely depends on class balance, so the undersampled folds alone don't
  explain it; overfitting during the search or a distribution difference between the
  folds and the test set are the first things to check.

## Running the notebooks

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download `Base.csv` from the Kaggle page above into a `data/` folder, then run the
notebooks in order. Step 1 writes `data/final_dataset.csv`, which the later steps read.

## Report

The full write-up, covering background, method, results and discussion, is in
[`Project_Report.pdf`](Project_Report.pdf).
