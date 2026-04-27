# Employee Attrition Prediction & Salary Impact Estimation

A multi-step machine learning pipeline that predicts employee attrition, simulates future salaries, and estimates the financial cost of employee exits — built on the IBM HR Analytics dataset.

---

## Overview

Most HR analytics projects stop at classifying whether an employee will leave. This project goes further by combining classification and regression into a 5-part pipeline that answers: **who will leave, what are they worth, and how much will it cost?**

---

## Pipeline

| Part | Task | Approach |
|------|------|----------|
| 1 | Attrition Prediction | Classification (LR, Decision Tree, SVM) |
| 2 | Future Salary Simulation | Rule-based increment on `PerformanceRating` |
| 3 | Salary Prediction | Regression on likely-to-stay employees |
| 4 | Identify Likely Stayers | Probability threshold (`P_stay > 0.6`) |
| 5 | Financial Loss Estimation | `Expected Loss = P_leave × FutureSalary` |

---

## Results

**Classification (Part 1)**

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 0.799 |
| Decision Tree | 0.649 |
| **SVM** | **0.792** |

SVM selected as the final classifier based on F1-score and AUC-ROC. Custom threshold of 0.3 applied to improve recall for at-risk employees (dataset has ~16% attrition rate).

**Regression (Part 3)** — trained only on employees with `P_stay > 0.6`

| Model | RMSE | R² | MAPE |
|-------|------|----|------|
| Random Forest | 137.17 | 0.9992 | 1.26% |
| Ridge | 85.06 | 0.9997 | 1.09% |
| **Lasso** | **81.29** | **0.9997** | **1.00%** |
| SVR | 5047.62 | -0.139 | 50.43% |

**Financial Loss Estimation (Part 5)**

> **Total Expected Salary Loss: ₹13,97,383.14**

---

## Dataset

[IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employee records with features including age, job role, monthly income, performance rating, years at company, and attrition status.

---

## Tech Stack

- **Language:** Python (Google Colab)
- **Libraries:** scikit-learn, Pandas, NumPy, Matplotlib
- **Models:** Logistic Regression, Decision Tree, SVM, Random Forest, Ridge, Lasso, SVR

---

## How to Run

1. Clone the repo and open `attrition_pipeline.py` in Google Colab (or Jupyter)
2. Upload `hr_project.csv` when prompted (or load from Kaggle)
3. Run cells sequentially — Parts 1 through 5 are clearly labelled

---

## Key Design Decisions

- **Custom threshold (0.3):** Default 0.5 threshold misses too many at-risk employees given class imbalance. Lowering to 0.3 improves attrition recall significantly.
- **Salary simulation:** The IBM dataset has no future salary column. We simulate it using `FutureSalary = MonthlyIncome × Increment`, where increment is 1.10 for top performers (rating 4) and 1.05 otherwise.
- **Regression on stayers only:** Training the salary model on employees likely to leave would introduce noise — only employees with `P_stay > 0.6` are used for regression training.
