# Experiment 1
### Working with Python Packages – NumPy, SciPy, Scikit-Learn, Matplotlib

## Aim
Explore the core functions of NumPy, Pandas, SciPy, Scikit-learn and Matplotlib, and survey five reference datasets to identify the ML task type, a suitable feature-selection technique, and a suitable algorithm for each.

## ML Task Table

| Dataset | Type of ML Task | Feature Selection Technique | Suitable ML Algorithm |
|---|---|---|---|
| Iris | Supervised – Multi-class Classification | ANOVA F-test / SelectKBest (all 4 features informative) | Logistic Regression / KNN / SVM |
| Loan Amount Prediction | Supervised – Regression (continuous target) | Correlation analysis + domain-driven selection (income, credit history) | Linear / Ridge / Lasso / Elastic Net Regression |
| Predicting Diabetes | Supervised – Binary Classification | ANOVA F-test (top: Glucose, BMI, Age, Pregnancies) | Logistic Regression / Random Forest / SVM |
| Classification of Email Spam | Supervised – Binary Classification | Correlation with label + variance threshold (57 features) | Naive Bayes / KNN / Logistic Regression / SVM |
| Handwritten Character Recognition / MNIST | Supervised – Multi-class Classification (10 classes) | PCA (dimensionality reduction on pixel features) | CNN / SVM (RBF) / Random Forest / k-NN |

## Learning Outcomes
- Got hands-on with NumPy, Pandas, SciPy, Scikit-learn and Matplotlib as the core Python data-science stack.
- Learned to tell supervised classification, regression and image classification tasks apart just from a dataset's feature types and target variable.
- Saw how the right feature-selection technique depends on data type: ANOVA F-test for continuous features against a classification target, PCA for high-dimensional pixel data.
- Recognized load, EDA, preprocess, select features, split, evaluate as the same workflow reused in every later experiment.
