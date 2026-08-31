# Experiment 7
### Dimensionality Reduction and Model Evaluation (With and Without PCA)

## Aim
Train and 5-fold cross-validate ten classifiers on the Spambase dataset, once on the original 57 features and once on their PCA projection (48 components, 95% variance), to see which model families benefit from PCA and which don't.

## Best Hyperparameters and Cross-Validated Accuracy

| Model | Setting | Best Parameters | Best CV Accuracy (%) |
|---|---|---|---|
| SVM | No-PCA | C=10 | 93.56 |
| SVM | With-PCA | C=10 | 93.32 |
| Naive Bayes | No-PCA | var_smoothing=1e-7 | 81.49 |
| Naive Bayes | With-PCA | var_smoothing=1e-9 | 80.46 |
| KNN | No-PCA | k=11, distance | 92.15 |
| KNN | With-PCA | k=11, distance | 92.07 |
| Logistic Regression | No-PCA | C=10 | 92.45 |
| Logistic Regression | With-PCA | C=10 | 92.31 |
| Decision Tree | No-PCA | max_depth=10 | 92.23 |
| Decision Tree | With-PCA | max_depth=5 | 88.75 |
| Random Forest | No-PCA | 50 trees, no depth limit | **95.30** |
| Random Forest | With-PCA | 100 trees, no depth limit | 93.18 |
| AdaBoost | No-PCA | 100 estimators | 93.56 |
| AdaBoost | With-PCA | 100 estimators | 90.46 |
| Gradient Boosting | No-PCA | 100 estimators | 94.59 |
| Gradient Boosting | With-PCA | 100 estimators | 92.64 |
| XGBoost | No-PCA | 100 estimators, lr=0.3 | **95.54** |
| XGBoost | With-PCA | 100 estimators, lr=0.1 | 93.89 |
| Stacking | No-PCA | final C=0.1 | 94.08 |
| Stacking | With-PCA | final C=1 | 93.23 |

## Test Set Performance: No-PCA vs With-PCA

| Model | No-PCA Acc | No-PCA F1 | With-PCA Acc | With-PCA F1 |
|---|---|---|---|---|
| SVM | 0.9207 | 0.8976 | 0.9273 | 0.9058 |
| Naive Bayes | 0.8328 | 0.8188 | 0.8339 | 0.8123 |
| KNN | 0.9240 | 0.9033 | 0.9218 | 0.9000 |
| Logistic Regression | 0.9262 | 0.9048 | 0.9294 | 0.9096 |
| Decision Tree | 0.9088 | 0.8824 | 0.8762 | 0.8496 |
| Random Forest | **0.9425** | **0.9252** | 0.9197 | 0.8966 |
| AdaBoost | 0.9273 | 0.9058 | 0.8958 | 0.8644 |
| Gradient Boosting | 0.9392 | 0.9216 | 0.9153 | 0.8917 |
| XGBoost | 0.9403 | 0.9243 | 0.9273 | 0.9076 |
| Stacking | 0.9294 | 0.9101 | 0.9262 | 0.9042 |

## Learning Outcomes
- PCA hurt cross-validated accuracy for all ten models here; the sample-to-feature ratio (3680:57) is too generous for PCA's usual "fight overfitting" benefit to kick in.
- Tree-based models (Decision Tree, AdaBoost, Random Forest) lost the most, since axis-aligned splits work best on individually meaningful features, not on PCA's rotated linear combinations of all of them.
- Linear and distance-based models (SVM, KNN, Logistic Regression) barely moved either way, since a weighted sum over rotated features represents the same thing a weighted sum over original features does.
- XGBoost and Random Forest without PCA were the two strongest models overall (95.5% and 95.3% CV accuracy), both clearly ahead of the rest of the field.
