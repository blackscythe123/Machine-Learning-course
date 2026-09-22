# Experiment 7
### Dimensionality Reduction and Model Evaluation (With and Without PCA)

## Aim
Train and 5-fold cross-validate ten classifiers on the Spambase dataset, comparing the raw 57-feature space against a properly justified PCA representation, to see which model families benefit and which don't.

## PCA Component Choice
Kept **48 of the 57 components**, the number needed to reach a **95% cumulative explained-variance target**, rather than guessing a count up front. This choice was checked, not assumed: on the standardized training features, mean pairwise |r| is a modest 0.061, yet Bartlett's test of sphericity is highly significant (chi-sq = 60210, p < 0.001) and the overall KMO measure is 0.84 ("meritorious"), confirming PCA is statistically appropriate here. But only 20 of 57 eigenvalues exceed 1 (Kaiser's criterion), and those 20 components capture just 57.7% of the variance, too aggressive a cut to keep the signal that matters. Re-running all ten classifiers on that 20-component representation confirmed it: every model did worse than at 48 components. So 48 is the right amount of compression for this dataset, not an arbitrary or lazy choice, and it's the only setting used below.

## Test Set Performance: No-PCA vs With-PCA

| Model | No-PCA Acc | With-PCA Acc | Change |
|---|---|---|---|
| SVM | 0.9207 | 0.9273 | +0.0066 |
| Naive Bayes | 0.8328 | 0.8339 | +0.0011 |
| KNN | 0.9240 | 0.9218 | -0.0022 |
| Logistic Regression | 0.9262 | 0.9294 | +0.0032 |
| Decision Tree | 0.9088 | 0.8762 | -0.0326 |
| Random Forest | **0.9425** | 0.9197 | -0.0228 |
| AdaBoost | 0.9273 | 0.8958 | -0.0315 |
| Gradient Boosting | 0.9392 | 0.9153 | -0.0239 |
| XGBoost | **0.9403** | 0.9273 | -0.0130 |
| Stacking | 0.9294 | 0.9262 | -0.0032 |

*Best cross-validated accuracy overall: XGBoost No-PCA (95.54%) and Random Forest No-PCA (95.30%).*

## Learning Outcomes
- A properly justified PCA representation, not an arbitrary one, still gave no net benefit over the raw feature space for most models: it hurt cross-validated accuracy for all ten models and test accuracy for eight of the ten.
- Tree-based models (Decision Tree, AdaBoost, Random Forest) lost the most, since axis-aligned splits work best on individually meaningful features, not PCA's rotated combinations of them.
- Linear and distance-based models (SVM, KNN, Logistic Regression) were largely indifferent to PCA, so a correctly-sized PCA representation costs them almost nothing while modestly shrinking the input; it's a defensible option for these models, just not a clear win for the rest.
