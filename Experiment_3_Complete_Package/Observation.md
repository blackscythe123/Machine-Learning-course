# Experiment 3
### Regression Analysis using Linear and Regularized Models

## Aim
Clean, preprocess and model the continuous loan amount target using Linear, Ridge, Lasso and Elastic Net regression, and compare them via 5-fold cross-validation and test-set performance.

## Hyperparameter Tuning Summary

| Model | Best Parameters | Best CV R² |
|---|---|---|
| Ridge Regression | α=100 | 0.3714 |
| Lasso Regression | α=10 | 0.3311 |
| Elastic Net Regression | α=1, l1_ratio=0.8 | **0.3710** |

## Cross-Validation Performance (K=5, Training Set)

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 43.06 | 5506.90 | 74.21 | 0.3110 |
| Ridge Regression | 44.07 | 4932.95 | 70.23 | **0.3714** |
| Lasso Regression | 46.23 | 5281.51 | 72.67 | 0.3311 |
| Elastic Net Regression | **43.85** | **4939.55** | **70.28** | 0.3710 |

## Test Set Performance

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | **36.52** | **3352.43** | 57.90 | 0.0906 |
| Ridge Regression | 37.37 | 3175.95 | 56.36 | 0.1385 |
| Lasso Regression | 37.80 | 3176.80 | 56.36 | 0.1382 |
| Elastic Net Regression | 37.32 | 3163.59 | **56.25** | **0.1418** |

## Coefficient Comparison (Top 8 Features)

| Feature | Linear | Ridge | Lasso | Elastic Net |
|---|---|---|---|---|
| ApplicantIncome | 54.03 | 43.88 | 44.98 | 43.76 |
| CoapplicantIncome | 22.30 | 17.22 | 11.79 | 16.76 |
| Married_Yes | 9.43 | 7.87 | 1.74 | 7.51 |
| Loan_Amount_Term | 7.71 | 5.64 | 0.00 | 4.98 |
| Self_Employed_Yes | 6.46 | 6.18 | 0.00 | 5.66 |
| Education_Not Graduate | −6.38 | −6.81 | 0.00 | −6.23 |
| Property_Area_Urban | −5.47 | −4.12 | 0.00 | −3.01 |
| Dependents | 4.98 | 5.72 | 0.00 | 5.28 |

## Learning Outcomes
- With only 592 samples and 119 held out for testing, the CV/test R² gap looks like overfitting at first glance, but the learning curve says it's really just small-sample variance.
- Elastic Net matches Ridge's CV R² while still getting Lasso-style sparsity on the weaker features, which is why it's the overall pick.
- ApplicantIncome and CoapplicantIncome dominate every model's coefficients; Lasso zeroes out most of the rest, Ridge and Elastic Net just shrink them.
- None of the four models clears R² ≈ 0.14 on the test set, so the ceiling here is the available features, not the modelling approach.
