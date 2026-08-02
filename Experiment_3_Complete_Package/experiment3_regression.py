"""
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 3: Regression Analysis using Linear and Regularized Models
------------------------------------------------------------------------
Dataset : Loan Prediction dataset (614 loan applications, numerical + categorical
          features). Target: LoanAmount (continuous, loan amount sanctioned, in
          thousands).

This script performs:
  1. Data loading and cleaning (missing value imputation)
  2. Encoding of categorical variables + standardization of numeric features
  3. Exploratory Data Analysis (EDA)
  4. Linear, Ridge, Lasso and Elastic Net regression
  5. Hyperparameter tuning (GridSearchCV, 5-fold CV)
  6. Evaluation using MAE, MSE, RMSE, R2
  7. Overfitting/underfitting and bias-variance visualisations
  8. Generation of all required plots
"""

import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RNG = 42
sns.set_style("whitegrid")

# ---------------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------------
df = pd.read_csv("loan_train.csv")
df = df.drop(columns=["Loan_ID", "Loan_Status"])   # Loan_Status belongs to a different (classification) task
print("Raw shape:", df.shape)
print("Missing values before cleaning:\n", df.isnull().sum())

# Drop rows with missing target (cannot impute what we are trying to predict)
df = df.dropna(subset=["LoanAmount"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
num_cols = ["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term", "Credit_History"]

for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

print("\nMissing values after cleaning:\n", df.isnull().sum().sum())

df_enc = pd.get_dummies(df, columns=["Gender","Married","Education","Self_Employed","Property_Area"], drop_first=True)

y = df_enc["LoanAmount"].values
X = df_enc.drop(columns=["LoanAmount"])
feature_names = X.columns.tolist()
X = X.values.astype(float)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RNG)

scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# ---------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(y, bins=30, kde=True, color="teal")
plt.xlabel("Loan Amount (thousands)"); plt.title("Target Variable Distribution")
plt.tight_layout(); plt.savefig("figures/01_target_distribution.png", dpi=150); plt.close()

fig, axes = plt.subplots(1,2, figsize=(11,4.5))
axes[0].scatter(df["ApplicantIncome"], y, alpha=0.5, s=15)
axes[0].set_xlabel("Applicant Income"); axes[0].set_ylabel("Loan Amount")
axes[0].set_title("Income vs Loan Amount")
axes[1].scatter(df["CoapplicantIncome"], y, alpha=0.5, s=15, color="orange")
axes[1].set_xlabel("Co-applicant Income"); axes[1].set_ylabel("Loan Amount")
axes[1].set_title("Co-applicant Income vs Loan Amount")
plt.tight_layout(); plt.savefig("figures/02_feature_vs_target.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return dict(MAE=mae, MSE=mse, RMSE=np.sqrt(mse), R2=r2_score(y_true, y_pred))

kf5 = KFold(n_splits=5, shuffle=True, random_state=RNG)

results_test = {}
results_cv = {}
train_times = {}
models_fitted = {}

# --- Baseline Linear Regression ---
t0 = time.perf_counter()
lin = LinearRegression().fit(X_train, y_train)
train_times["Linear Regression"] = time.perf_counter() - t0
pred = lin.predict(X_test)
results_test["Linear Regression"] = evaluate(y_test, pred)
models_fitted["Linear Regression"] = lin

# --- Ridge (GridSearchCV) ---
ridge_grid = GridSearchCV(Ridge(random_state=RNG), {"alpha":[0.01,0.1,1,10,100]},
                           cv=kf5, scoring="r2", n_jobs=1)
t0 = time.perf_counter()
ridge_grid.fit(X_train, y_train)
train_times["Ridge Regression"] = time.perf_counter() - t0
ridge_best = ridge_grid.best_estimator_
results_test["Ridge Regression"] = evaluate(y_test, ridge_best.predict(X_test))
models_fitted["Ridge Regression"] = ridge_best
print("Ridge best alpha:", ridge_grid.best_params_, "CV R2:", ridge_grid.best_score_)

# --- Lasso (GridSearchCV) ---
lasso_grid = GridSearchCV(Lasso(random_state=RNG, max_iter=10000), {"alpha":[0.001,0.01,0.1,1,10]},
                           cv=kf5, scoring="r2", n_jobs=1)
t0 = time.perf_counter()
lasso_grid.fit(X_train, y_train)
train_times["Lasso Regression"] = time.perf_counter() - t0
lasso_best = lasso_grid.best_estimator_
results_test["Lasso Regression"] = evaluate(y_test, lasso_best.predict(X_test))
models_fitted["Lasso Regression"] = lasso_best
print("Lasso best alpha:", lasso_grid.best_params_, "CV R2:", lasso_grid.best_score_)

# --- Elastic Net (GridSearchCV) ---
enet_grid = GridSearchCV(ElasticNet(random_state=RNG, max_iter=10000),
                          {"alpha":[0.01,0.1,1,10], "l1_ratio":[0.2,0.5,0.8]},
                          cv=kf5, scoring="r2", n_jobs=1)
t0 = time.perf_counter()
enet_grid.fit(X_train, y_train)
train_times["Elastic Net Regression"] = time.perf_counter() - t0
enet_best = enet_grid.best_estimator_
results_test["Elastic Net Regression"] = evaluate(y_test, enet_best.predict(X_test))
models_fitted["Elastic Net Regression"] = enet_best
print("ElasticNet best params:", enet_grid.best_params_, "CV R2:", enet_grid.best_score_)

# ---------------------------------------------------------------------
# 5. CROSS-VALIDATION PERFORMANCE (K=5) for each model, using its tuned/base config
# ---------------------------------------------------------------------
from sklearn.model_selection import cross_validate
scoring = {"MAE":"neg_mean_absolute_error", "MSE":"neg_mean_squared_error", "R2":"r2"}
for name, model in models_fitted.items():
    cvres = cross_validate(model, X_train, y_train, cv=kf5, scoring=scoring)
    results_cv[name] = dict(
        MAE = -cvres["test_MAE"].mean(),
        MSE = -cvres["test_MSE"].mean(),
        RMSE = np.sqrt(-cvres["test_MSE"].mean()),
        R2 = cvres["test_R2"].mean()
    )

# ---------------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------------
# Predicted vs actual (best model = lowest test RMSE)
best_name = min(results_test, key=lambda k: results_test[k]["RMSE"])
best_model = models_fitted[best_name]
best_pred = best_model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, best_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
plt.plot(lims, lims, "r--", lw=1.5)
plt.xlabel("Actual Loan Amount"); plt.ylabel("Predicted Loan Amount")
plt.title(f"Predicted vs Actual ({best_name})")
plt.tight_layout(); plt.savefig("figures/03_predicted_vs_actual.png", dpi=150); plt.close()

# Residual plot
residuals = y_test - best_pred
plt.figure(figsize=(6,5))
plt.scatter(best_pred, residuals, alpha=0.6, edgecolor="k", linewidth=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Loan Amount"); plt.ylabel("Residual")
plt.title(f"Residual Plot ({best_name})")
plt.tight_layout(); plt.savefig("figures/04_residual_plot.png", dpi=150); plt.close()

# Training vs validation error (learning curve) for Linear Regression (illustrates over/underfitting)
train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X_train, y_train, cv=kf5,
    train_sizes=np.linspace(0.2,1.0,6), scoring="neg_mean_squared_error")
train_rmse = np.sqrt(-train_scores).mean(axis=1)
val_rmse = np.sqrt(-val_scores).mean(axis=1)
plt.figure(figsize=(6,5))
plt.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
plt.plot(train_sizes, val_rmse, marker="s", label="Validation RMSE")
plt.xlabel("Training Set Size"); plt.ylabel("RMSE")
plt.title("Training vs Validation Error (Linear Regression)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/05_train_val_error.png", dpi=150); plt.close()

# Coefficient comparison bar plot
coef_df = pd.DataFrame({
    "Linear": lin.coef_,
    "Ridge": ridge_best.coef_,
    "Lasso": lasso_best.coef_,
    "ElasticNet": enet_best.coef_,
}, index=feature_names)
top_idx = coef_df["Linear"].abs().sort_values(ascending=False).head(8).index
plt.figure(figsize=(10,6))
coef_df.loc[top_idx].plot(kind="bar", ax=plt.gca())
plt.ylabel("Coefficient Value"); plt.title("Coefficient Comparison (Top 8 Features by |Linear Coef|)")
plt.xticks(rotation=40, ha="right")
plt.tight_layout(); plt.savefig("figures/06_coefficient_comparison.png", dpi=150); plt.close()

# Test performance comparison bar chart
plt.figure(figsize=(8,5))
names = list(results_test.keys())
r2s = [results_test[n]["R2"] for n in names]
rmses = [results_test[n]["RMSE"] for n in names]
x = np.arange(len(names)); w=0.35
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.bar(x-w/2, r2s, width=w, label="R2", color="seagreen")
ax1.set_ylabel("R2 Score")
ax2 = ax1.twinx()
ax2.bar(x+w/2, rmses, width=w, label="RMSE", color="indianred")
ax2.set_ylabel("RMSE")
ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right")
plt.title("Test Set Performance Comparison")
fig.legend(loc="upper right", bbox_to_anchor=(0.9,0.88))
plt.tight_layout(); plt.savefig("figures/07_test_performance_comparison.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 7. SAVE RESULTS SUMMARY
# ---------------------------------------------------------------------
with open("results_summary.txt","w") as f:
    f.write(f"Cleaned dataset shape: {df_enc.shape}\n")
    f.write(f"Features used ({len(feature_names)}): {feature_names}\n\n")

    f.write(f"Ridge best alpha: {ridge_grid.best_params_}  CV R2: {ridge_grid.best_score_:.4f}\n")
    f.write(f"Lasso best alpha: {lasso_grid.best_params_}  CV R2: {lasso_grid.best_score_:.4f}\n")
    f.write(f"ElasticNet best params: {enet_grid.best_params_}  CV R2: {enet_grid.best_score_:.4f}\n\n")

    f.write("=== Test Set Performance ===\n")
    for n in names:
        f.write(f"{n}: {results_test[n]}  train_time={train_times[n]:.4f}s\n")

    f.write("\n=== 5-Fold CV Performance (mean) ===\n")
    for n in names:
        f.write(f"{n}: {results_cv[n]}\n")

    f.write(f"\nBest model by test RMSE: {best_name}\n")
    f.write(f"\nCoefficient table (top 8 features):\n{coef_df.loc[top_idx].to_string()}\n")

print("\nDone. Results written to results_summary.txt and figures/ directory.")
print("\n=== TEST RESULTS ===")
for n in names:
    print(n, results_test[n])
print("\n=== CV RESULTS ===")
for n in names:
    print(n, results_cv[n])
