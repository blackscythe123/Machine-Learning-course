# Experiment 5
### Decision Tree and Random Forest: A Comparative Classification Study

## Aim
Implement a Decision Tree classifier, extend it into a Random Forest ensemble, study how hyperparameters affect overfitting and generalization, and compare the two models using 5-fold cross-validation.

## Hyperparameter Tuning Summary

| Model | Best Parameters | Best CV Accuracy |
|---|---|---|
| Decision Tree | gini, max_depth=5, min_samples_split=2, min_samples_leaf=4 | 0.9385 |
| Random Forest | n_estimators=100, max_depth=10, max_features=log2, bootstrap=True | **0.9648** |

## Test Set Performance

| Metric | Decision Tree (tuned) | Random Forest (tuned) |
|---|---|---|
| Accuracy | 0.9035 | **0.9561** |
| Precision | 0.9420 | **0.9589** |
| Recall | 0.9028 | **0.9722** |
| F1-Score | 0.9220 | **0.9655** |
| ROC-AUC | 0.9358 | **0.9924** |

## 5-Fold Cross-Validation (Training Set)

| Fold | Decision Tree (tuned) | Random Forest (tuned) |
|---|---|---|
| 1 | 0.9451 | 0.9670 |
| 2 | 0.9451 | 0.9780 |
| 3 | 0.9121 | 0.9341 |
| 4 | 0.9231 | 0.9560 |
| 5 | 0.9670 | 0.9890 |
| **Average** | **0.9385 ± 0.0192** | **0.9648 ± 0.0189** |

## Learning Outcomes
- Decision Tree training accuracy hits 1.0000 by depth 7 while test accuracy peaks around depth 3–4 and then drops off: textbook overfitting once the tree starts fitting noise instead of signal.
- Random Forest wins on every metric and every CV fold, mainly by cutting variance through bagging and random feature subsets rather than fixing any single tree's bias.
- Depth has the biggest effect on the Decision Tree's CV score; for Random Forest, n_estimators and max_depth both matter but with clear diminishing returns past 100 trees or depth 10.
- Ensembling isn't automatic. It works here because bagging decorrelates trees whose errors weren't already correlated, giving a real reduction in variance.
