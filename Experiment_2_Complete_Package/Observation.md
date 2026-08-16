# Experiment 2
### Email Spam/Ham Classification using Naive Bayes and KNN

## Aim
Build and evaluate email spam classifiers using Naive Bayes and K-Nearest Neighbours, compare the Naive Bayes variants, tune KNN hyperparameters via cross-validation, and compare KDTree against BallTree.

## Naive Bayes Comparison (Test Set)

| Metric | Gaussian | Multinomial | Bernoulli |
|---|---|---|---|
| Accuracy | 0.8328 | 0.7763 | **0.8762** |
| Precision | 0.7146 | 0.7199 | **0.8716** |
| Recall | **0.9587** | 0.7080 | 0.8044 |
| F1-Score | 0.8188 | 0.7139 | **0.8367** |
| ROC-AUC | 0.9376 | 0.8248 | **0.9496** |

## KNN: Accuracy vs k (Test Set, uniform weights, Euclidean distance)

| k | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 1 | 0.8990 | 0.8792 | 0.8623 | 0.8707 |
| 3 | 0.8979 | 0.8726 | 0.8678 | 0.8702 |
| 5 | 0.9077 | 0.8861 | 0.8788 | 0.8824 |
| 7 | 0.9088 | 0.8930 | 0.8733 | 0.8830 |
| 9 | 0.9088 | 0.8930 | 0.8733 | 0.8830 |
| 11 | **0.9099** | **0.9000** | 0.8678 | **0.8836** |

## Grid Search vs Randomised Search

| Parameter | Grid Search CV | Randomised Search CV |
|---|---|---|
| Best k | 9 | 8 |
| Metric | manhattan | manhattan |
| Weights | distance | distance |
| Algorithm | kd_tree | ball_tree |
| Best CV Accuracy | 0.9253 | 0.9255 |
| Execution Time (s) | 58.52 | 23.77 |

## KDTree vs BallTree (k=11, uniform weights)

| Metric | KDTree | BallTree |
|---|---|---|
| Accuracy | 0.9099 | 0.9099 |
| Training Time (s) | 0.0167 | **0.0115** |
| Prediction Time (s) | 0.2659 | **0.2506** |

## Tuned KNN Performance (Test Set, k=9, distance, manhattan, kd_tree)

| Metric | Value |
|---|---|
| Accuracy | 0.9207 |
| Precision | 0.9421 |
| Recall | 0.8512 |
| F1-score | 0.8944 |
| ROC-AUC | 0.9712 |

## Cross Validation (K=5)

| Fold | Bernoulli Naive Bayes | Tuned KNN |
|---|---|---|
| 1 | 0.9063 | 0.9389 |
| 2 | 0.8913 | 0.9280 |
| 3 | 0.8723 | 0.9253 |
| 4 | 0.8682 | 0.9266 |
| 5 | 0.8981 | 0.9076 |
| **Average** | **0.8872 ± 0.0147** | **0.9253 ± 0.0101** |

## Learning Outcomes
- Bernoulli NB comes out on top here because its binarized, presence/absence view fits the zero-inflated word-frequency features better than Gaussian NB's normality assumption.
- Distance-weighted KNN with the Manhattan metric consistently beat uniform-weight Euclidean, and tuning landed on k=9.
- Grid and Randomized Search converged on almost the same CV accuracy, but Randomized Search got there in under half the time.
- KDTree and BallTree give identical accuracy since both are exact searches; BallTree was marginally faster here, though at d=57 the gap is close to noise.
