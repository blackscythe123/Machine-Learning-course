# Experiment 6
### Bagging, Boosting, and Stacked Ensemble Models

## Aim
Compare three ensemble strategies (Bagging, AdaBoost, Stacking) on the breast cancer dataset, each tuned via 5-fold cross-validation.

## Best Configuration per Model

| Model | Best Setting | Avg CV Accuracy (%) | Avg CV F1 |
|---|---|---|---|
| Bagging | n_estimators=100, max_samples=1.0 | 96.04 | 0.9684 |
| AdaBoost | n_estimators=100, lr=1.0 | **98.02** | **0.9844** |
| Stacking | final C=1.0, passthrough=True | 95.16 | 0.9615 |

*Gradient Boosting (200 estimators, lr=0.2, depth=2) reached 97.36% CV accuracy, close but short of AdaBoost.*
 
## Test Set Performance

| Metric | Bagging | AdaBoost | Stacking |
|---|---|---|---|
| Accuracy | 0.9561 | 0.9561 | **0.9649** |
| F1 Score | 0.9655 | 0.9660 | **0.9726** |
| ROC-AUC | 0.9931 | 0.9818 | **0.9957** |

## Learning Outcomes
- Bagging only cuts variance, not bias, so it stayed the weakest of the three across every metric.
- AdaBoost won on cross-validated accuracy (98.02%) by targeting bias directly; Stacking won on the test set and ROC-AUC despite a lower CV mean, likely fold variance rather than a real edge.
- Neither AdaBoost nor Stacking is a clean winner here; telling them apart honestly would need a bigger dataset.
