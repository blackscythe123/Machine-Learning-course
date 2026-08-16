# Experiment 4
### Binary Classification using Linear and Kernel-Based Models

## Aim
Classify spam/ham emails with Logistic Regression and SVM, compare L1/L2 regularization and the four SVM kernels, and tune both models via GridSearchCV/RandomizedSearchCV.

## Hyperparameter Tuning Summary

| Model | Search Method | Best Parameters | Best CV Accuracy |
|---|---|---|---|
| Logistic Regression | Grid | C=100, L1, liblinear | 0.9261 |
| Logistic Regression | Random | C=100, L1, liblinear | 0.9261 |
| SVM | Grid | C=10, γ=scale, RBF | **0.9356** |
| SVM | Random | C=1, γ=auto, RBF | 0.9321 |

## SVM Kernel-wise Performance (Test Set, Default Hyperparameters)

| Kernel | Accuracy | F1-Score | Training Time (s) |
|---|---|---|---|
| Linear | 0.9294 | **0.9093** | 1.70 |
| Polynomial | 0.7796 | 0.6220 | 1.45 |
| RBF | **0.9273** | 0.9055 | 1.10 |
| Sigmoid | 0.8849 | 0.8528 | 1.20 |

## Tuned Model Performance (Test Set)

| Metric | Logistic Regression | SVM |
|---|---|---|
| Accuracy | 0.9262 | 0.9207 |
| Precision | 0.9202 | 0.9143 |
| Recall | 0.8898 | 0.8815 |
| F1-Score | 0.9048 | 0.8976 |
| ROC-AUC | 0.9679 | **0.9702** |

## 5-Fold Cross-Validation (Training Set)

| Fold | Logistic Regression | SVM |
|---|---|---|
| 1 | 0.9416 | 0.9443 |
| 2 | 0.9253 | 0.9416 |
| 3 | 0.9348 | 0.9348 |
| 4 | 0.9117 | 0.9307 |
| 5 | 0.9171 | 0.9266 |
| **Average** | **0.9261 ± 0.0110** | **0.9356 ± 0.0066** |

## Learning Outcomes
- SVM (RBF) beats Logistic Regression on 5-fold CV accuracy and is more stable across folds, even though LR edges it slightly on the one held-out test split.
- L1 regularization matched or beat L2 at every C tested, and accuracy kept climbing up to C=100, so the informative features here are sparse enough for L1's selection to help rather than hurt.
- The untuned Polynomial kernel was the clear loser (recall 0.46): not every kernel is a safe default without tuning.
- Linear and RBF perform almost identically, which suggests the true decision boundary is close to linear with only mild non-linear structure.
