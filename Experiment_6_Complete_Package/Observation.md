# Experiment 6
### Bagging, Boosting, and Stacked Ensemble Models

## Aim
Understand and compare three ensemble strategies on the breast cancer dataset: Bagging, Boosting (AdaBoost / Gradient Boosting), and a Stacked Ensemble, each tuned via 5-fold cross-validation.

## Bagging Hyperparameter Evaluation

| n_estimators | max_samples | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|
| 10 | 0.5 | 95.16 | 0.9610 |
| **50** | **0.5** | **96.04** | **0.9681** |
| 100 | 0.5 | 95.38 | 0.9630 |
| 10 | 0.7 | 94.95 | 0.9594 |
| 50 | 0.7 | 95.82 | 0.9665 |
| 100 | 0.7 | 96.04 | 0.9683 |
| 10 | 1.0 | 95.60 | 0.9648 |
| 50 | 1.0 | 96.04 | 0.9683 |
| 100 | 1.0 | 96.04 | 0.9684 |

## Boosting Hyperparameter Evaluation (AdaBoost)

| n_estimators | learning_rate | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|
| 50 | 0.01 | 92.53 | 0.9407 |
| 100 | 0.01 | 93.41 | 0.9480 |
| 200 | 0.01 | 93.63 | 0.9496 |
| 50 | 0.10 | 95.38 | 0.9633 |
| 100 | 0.10 | 96.04 | 0.9686 |
| 200 | 0.10 | 97.14 | 0.9773 |
| 50 | 1.00 | 96.70 | 0.9740 |
| **100** | **1.00** | **98.02** | **0.9844** |
| 200 | 1.00 | 97.80 | 0.9826 |

*Gradient Boosting's best config (n_estimators=200, learning_rate=0.2, max_depth=2) reached 97.36% CV accuracy, close but not enough to beat AdaBoost, so AdaBoost was carried forward as "Boosting."*

## Stacked Ensemble Evaluation

| Final Estimator C | Passthrough | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|
| 0.1 | True | 94.29 | 0.9547 |
| 0.1 | False | 94.95 | 0.9601 |
| **1.0** | **True** | **95.16** | **0.9615** |
| 1.0 | False | 94.95 | 0.9597 |
| 10.0 | True | 95.16 | 0.9615 |
| 10.0 | False | 94.95 | 0.9597 |

*Base models: SVM, Naive Bayes, Decision Tree (fixed). Meta-learner: Logistic Regression.*

## Performance Comparison (Test Set)

| Metric | Bagging | AdaBoost | Stacking |
|---|---|---|---|
| Accuracy | 0.9561 | 0.9561 | **0.9649** |
| Precision | 0.9589 | 0.9467 | **0.9595** |
| Recall | 0.9722 | **0.9861** | **0.9861** |
| F1 Score | 0.9655 | 0.9660 | **0.9726** |
| ROC-AUC | 0.9931 | 0.9818 | **0.9957** |

## 5-Fold Cross-Validation (Training Set)

| Fold | Bagging | AdaBoost | Stacking |
|---|---|---|---|
| 1 | 0.9560 | 0.9890 | 0.9890 |
| 2 | 0.9560 | **1.0000** | 0.9341 |
| 3 | 0.9341 | 0.9451 | 0.9451 |
| 4 | 0.9670 | 0.9780 | 0.9451 |
| 5 | 0.9890 | 0.9890 | 0.9451 |
| **Average** | 0.9604 ± 0.0179 | **0.9802 ± 0.0189** | 0.9516 ± 0.0192 |

## Learning Outcomes
- Bagging only cuts variance, not bias, which is why it stayed the weakest of the three across every metric here.
- Boosting (AdaBoost) got the best cross-validated accuracy by directly targeting bias through sequential error-correction, beating Bagging by nearly 2 points.
- Stacking won on the test set and ROC-AUC despite having the lowest CV mean, which says more about small-sample fold variance than about one model being definitively better.
- Neither AdaBoost nor Stacking is a clean winner here; picking between them honestly needs a bigger dataset, not just one CV run.
