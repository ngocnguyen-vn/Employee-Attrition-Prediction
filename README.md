# HR Analytics — Predicting Job Change Intention of Data Scientists

## Overview

This project builds a **binary classification model** to predict whether a data scientist candidate is actively looking for a new job. The goal is to help HR teams proactively identify and retain high-flight-risk employees.

**Dataset:** [Kaggle — HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)

---

## Project Structure

```
├── aug_train.csv          # Training data (19,158 rows, labeled)
├── aug_test.csv           # Test data (2,129 rows, unlabeled)
├── HR_Analysis.ipynb      # Main analysis notebook
└── submission_final.csv   # Final predictions on test set
```

---

## Problem Statement

Companies invest heavily in recruiting and training data scientists, only to lose them shortly after. This model helps predict **which candidates are looking for a job change** so organizations can act early.

**Target variable:**
- `0` — Candidate is **NOT** looking for a job change (~75%)
- `1` — Candidate **IS** looking for a job change (~25%)

> The dataset is imbalanced (75/25), so **Recall** is the primary metric to minimize missed detections.

---

## Key Features

| Feature | Type | Description |
|---|---|---|
| `city_development_index` | Numeric | Development level of candidate's city (0–1) |
| `last_new_job` | Categorical | Years since last job change |
| `experience` | Categorical | Total years of work experience |
| `education_level` | Categorical | Highest education level |
| `enrolled_university` | Categorical | Current university enrollment status |
| `relevent_experience` | Categorical | Has relevant data science experience? |
| `company_size` | Categorical | Size of current employer |
| `training_hours` | Numeric | Total training hours completed |

---

## Methodology

| Step | Description |
|---|---|
| Data Preprocessing | Handle missing values, clean columns |
| EDA | Explore distributions and feature correlations |
| Feature Engineering | Ordinal encoding, MinMax scaling |
| Modeling | Logistic Regression, Random Forest, Gradient Boosting |
| Evaluation | Stratified K-Fold cross-validation, prioritize Recall |
| Tuning | RandomizedSearchCV on best model |
| Prediction | Generate probabilities and labels for test set |

---

## Models & Evaluation

All models are evaluated using **Stratified 5-Fold Cross-Validation** with `class_weight='balanced'`.

**Metrics tracked:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Best model** is selected based on highest **Recall** on class 1 (job-seeking candidates).

---

## Research Hypotheses

- **H1:** `last_new_job` (job mobility history) is a strong predictor of turnover intent.
- **H2:** Work experience features (`experience`, `relevent_experience`) significantly predict attrition.
- **H3:** `city_development_index` (regional development) contributes significantly to prediction.

---

## Output

`submission_final.csv` contains predictions for each candidate in the test set:

| Column | Description |
|---|---|
| `enrollee_id` | Candidate ID |
| `leave_proba` | Predicted probability of job-seeking (class 1) |
| `predicted_target` | Final predicted label (0 or 1) |

---

## Libraries Used

- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling, preprocessing, evaluation
