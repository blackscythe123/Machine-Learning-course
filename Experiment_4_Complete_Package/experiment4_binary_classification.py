"""
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 4: Binary Classification using Linear and Kernel-Based Models
--------------------------------------------------------------------------
Dataset : UCI / Kaggle Spambase (4601 emails, 57 numeric features, binary label)

This script performs:
  1. Data loading and cleaning
  2. Preprocessing (standardisation, train-test split)
  3. Exploratory Data Analysis (EDA)
  4. Baseline Logistic Regression
  5. Logistic Regression hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
  6. SVM with Linear, Polynomial, RBF and Sigmoid kernels
  7. SVM hyperparameter tuning
  8. 5-Fold Cross-Validation comparison
  9. Generation of all required plots
"""

import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)

RNG = 42
sns.set_style("whitegrid")

# ---------------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------------
WORDS = ["make","address","all","3d","our","over","remove","internet","order","mail",
"receive","will","people","report","addresses","free","business","email","you","credit",
"your","font","000","money","hp","hpl","george","650","lab","labs","telnet","857","data",
"415","85","technology","1999","parts","pm","direct","cs","meeting","original","project",
"re","edu","table","conference"]
CHARS = [";","(","[","!","$","#"]
COL_NAMES = ([f"word_freq_{w}" for w in WORDS] + [f"char_freq_{c}" for c in CHARS] +
             ["capital_run_length_average","capital_run_length_longest",
              "capital_run_length_total","is_spam"])

df = pd.read_csv("spambase.data", header=None, names=COL_NAMES)
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Class distribution:\n", df["is_spam"].value_counts())

X = df.drop(columns=["is_spam"]).values
y = df["is_spam"].values

# ---------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------
plt.figure(figsize=(5,4))
sns.countplot(x=y)
plt.title("Class Distribution (0 = Ham, 1 = Spam)")
plt.xlabel("Class"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("figures/01_class_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr.iloc[-6:,-6:], annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (last 6 features + label)")
plt.tight_layout(); plt.savefig("figures/02_correlation_heatmap.png", dpi=150); plt.close()

top_feats = corr["is_spam"].drop("is_spam").abs().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2,3, figsize=(14,8))
for ax, feat in zip(axes.ravel(), top_feats):
    sns.histplot(df[feat], bins=40, ax=ax, kde=False)
    ax.set_title(feat, fontsize=9)
plt.suptitle("Distributions of Top Correlated Features")
plt.tight_layout(); plt.savefig("figures/03_histograms.png", dpi=150); plt.close()

fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax, feat in zip(axes.ravel(), top_feats[:3]):
    sns.boxplot(x=y, y=df[feat], ax=ax)
    ax.set_title(feat, fontsize=9); ax.set_xlabel("is_spam")
plt.tight_layout(); plt.savefig("figures/04_boxplots.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG)

scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

def evaluate(y_true, y_pred, y_score=None):
    d = dict(accuracy=accuracy_score(y_true, y_pred),
             precision=precision_score(y_true, y_pred),
             recall=recall_score(y_true, y_pred),
             f1=f1_score(y_true, y_pred))
    if y_score is not None:
        d["roc_auc"] = roc_auc_score(y_true, y_score)
    return d

# ---------------------------------------------------------------------
# 4. BASELINE LOGISTIC REGRESSION
# ---------------------------------------------------------------------
t0 = time.perf_counter()
lr_base = LogisticRegression(max_iter=2000, random_state=RNG).fit(X_train, y_train)
lr_base_time = time.perf_counter() - t0
lr_base_pred = lr_base.predict(X_test)
lr_base_score = lr_base.predict_proba(X_test)[:,1]
lr_base_metrics = evaluate(y_test, lr_base_pred, lr_base_score)
print("Baseline Logistic Regression:", lr_base_metrics)

# ---------------------------------------------------------------------
# 5. LOGISTIC REGRESSION HYPERPARAMETER TUNING
# ---------------------------------------------------------------------
lr_param_grid = [
    {"penalty":["l1"], "C":[0.01,0.1,1,10,100], "solver":["liblinear","saga"]},
    {"penalty":["l2"], "C":[0.01,0.1,1,10,100], "solver":["liblinear","saga"]},
]
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

t0 = time.perf_counter()
lr_grid = GridSearchCV(LogisticRegression(max_iter=2000, random_state=RNG),
                        lr_param_grid, cv=cv5, scoring="accuracy", n_jobs=1)
lr_grid.fit(X_train, y_train)
lr_grid_time = time.perf_counter() - t0
print("LR GridSearch best:", lr_grid.best_params_, lr_grid.best_score_, f"{lr_grid_time:.2f}s")

lr_param_dist = {"penalty":["l1","l2"], "C":[0.01,0.1,1,10,100], "solver":["liblinear","saga"]}
t0 = time.perf_counter()
lr_rand = RandomizedSearchCV(LogisticRegression(max_iter=2000, random_state=RNG),
                              lr_param_dist, n_iter=10, cv=cv5, scoring="accuracy",
                              random_state=RNG, n_jobs=1)
lr_rand.fit(X_train, y_train)
lr_rand_time = time.perf_counter() - t0
print("LR RandomSearch best:", lr_rand.best_params_, lr_rand.best_score_, f"{lr_rand_time:.2f}s")

lr_best = lr_grid.best_estimator_ if lr_grid.best_score_ >= lr_rand.best_score_ else lr_rand.best_estimator_
lr_best_pred = lr_best.predict(X_test)
lr_best_score = lr_best.predict_proba(X_test)[:,1]
lr_best_metrics = evaluate(y_test, lr_best_pred, lr_best_score)
print("Tuned Logistic Regression (test):", lr_best_metrics)

# ---------------------------------------------------------------------
# 6. SVM WITH DIFFERENT KERNELS (default hyperparameters, for comparison)
# ---------------------------------------------------------------------
svm_kernel_results = {}
for kernel in ["linear","poly","rbf","sigmoid"]:
    t0 = time.perf_counter()
    svc = SVC(kernel=kernel, probability=True, random_state=RNG).fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = svc.predict(X_test)
    score = svc.predict_proba(X_test)[:,1]
    m = evaluate(y_test, pred, score)
    m["train_time"] = train_time
    svm_kernel_results[kernel] = m
    print(f"SVM ({kernel}):", m)

# ---------------------------------------------------------------------
# 7. SVM HYPERPARAMETER TUNING (RBF kernel, most commonly best)
# ---------------------------------------------------------------------
svm_param_grid = {"C":[0.1,1,10,100], "gamma":["scale","auto"], "kernel":["rbf"]}
t0 = time.perf_counter()
svm_grid = GridSearchCV(SVC(probability=True, random_state=RNG), svm_param_grid,
                         cv=cv5, scoring="accuracy", n_jobs=1)
svm_grid.fit(X_train, y_train)
svm_grid_time = time.perf_counter() - t0
print("SVM GridSearch best:", svm_grid.best_params_, svm_grid.best_score_, f"{svm_grid_time:.2f}s")

svm_param_dist = {"C":[0.1,1,10,100], "gamma":["scale","auto"], "kernel":["rbf","poly","sigmoid"]}
t0 = time.perf_counter()
svm_rand = RandomizedSearchCV(SVC(probability=True, random_state=RNG), svm_param_dist,
                               n_iter=10, cv=cv5, scoring="accuracy", random_state=RNG, n_jobs=1)
svm_rand.fit(X_train, y_train)
svm_rand_time = time.perf_counter() - t0
print("SVM RandomSearch best:", svm_rand.best_params_, svm_rand.best_score_, f"{svm_rand_time:.2f}s")

svm_best = svm_grid.best_estimator_ if svm_grid.best_score_ >= svm_rand.best_score_ else svm_rand.best_estimator_
svm_best_pred = svm_best.predict(X_test)
svm_best_score = svm_best.predict_proba(X_test)[:,1]
svm_best_metrics = evaluate(y_test, svm_best_pred, svm_best_score)
print("Tuned SVM (test):", svm_best_metrics)

# ---------------------------------------------------------------------
# 8. 5-FOLD CROSS VALIDATION (best LR vs best SVM)
# ---------------------------------------------------------------------
lr_cv_scores = cross_val_score(lr_best, X_train, y_train, cv=cv5, scoring="accuracy")
svm_cv_scores = cross_val_score(svm_best, X_train, y_train, cv=cv5, scoring="accuracy")
print("LR CV folds:", lr_cv_scores, "mean:", lr_cv_scores.mean())
print("SVM CV folds:", svm_cv_scores, "mean:", svm_cv_scores.mean())

# ---------------------------------------------------------------------
# 9. PLOTS
# ---------------------------------------------------------------------
# Confusion matrices
fig, axes = plt.subplots(1,2, figsize=(10,4))
ConfusionMatrixDisplay(confusion_matrix(y_test, lr_best_pred)).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Logistic Regression (tuned)")
ConfusionMatrixDisplay(confusion_matrix(y_test, svm_best_pred)).plot(ax=axes[1], colorbar=False)
axes[1].set_title("SVM (tuned)")
plt.tight_layout(); plt.savefig("figures/05_confusion_matrices.png", dpi=150); plt.close()

# ROC curves
plt.figure(figsize=(6,5))
for name, score in [("Logistic Regression", lr_best_score), ("SVM", svm_best_score)]:
    fpr, tpr, _ = roc_curve(y_test, score)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test,score):.3f})")
plt.plot([0,1],[0,1],"k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend()
plt.tight_layout(); plt.savefig("figures/06_roc_curves.png", dpi=150); plt.close()

# SVM kernel comparison bar chart
plt.figure(figsize=(7,5))
kernels = list(svm_kernel_results.keys())
accs = [svm_kernel_results[k]["accuracy"] for k in kernels]
f1s  = [svm_kernel_results[k]["f1"] for k in kernels]
x = np.arange(len(kernels)); w = 0.35
plt.bar(x-w/2, accs, width=w, label="Accuracy")
plt.bar(x+w/2, f1s, width=w, label="F1 Score")
plt.xticks(x, kernels); plt.ylim(0.7,1.0)
plt.title("SVM Kernel-wise Performance"); plt.legend()
plt.tight_layout(); plt.savefig("figures/07_svm_kernel_comparison.png", dpi=150); plt.close()

# Training time comparison
plt.figure(figsize=(7,5))
times = [svm_kernel_results[k]["train_time"] for k in kernels]
plt.bar(kernels, times, color="salmon")
plt.ylabel("Training Time (s)"); plt.title("SVM Kernel Training Time")
plt.tight_layout(); plt.savefig("figures/08_svm_training_time.png", dpi=150); plt.close()

# GridSearch heatmap for LR (C vs penalty, l2 rows only for readability)
res = pd.DataFrame(lr_grid.cv_results_)
pivot = res.pivot_table(values="mean_test_score", index="param_C", columns="param_penalty")
plt.figure(figsize=(6,5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title("Logistic Regression GridSearchCV Accuracy")
plt.tight_layout(); plt.savefig("figures/09_lr_gridsearch_heatmap.png", dpi=150); plt.close()

# RandomSearch score distribution for SVM
plt.figure(figsize=(6,5))
plt.hist(svm_rand.cv_results_["mean_test_score"], bins=10, color="steelblue", edgecolor="k")
plt.xlabel("Mean CV Accuracy"); plt.ylabel("Count")
plt.title("SVM RandomizedSearchCV Score Distribution")
plt.tight_layout(); plt.savefig("figures/10_svm_randomsearch_distribution.png", dpi=150); plt.close()

# Cross-validation fold comparison
plt.figure(figsize=(7,5))
folds = np.arange(1,6)
plt.plot(folds, lr_cv_scores, marker="o", label="Logistic Regression")
plt.plot(folds, svm_cv_scores, marker="s", label="SVM")
plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.title("5-Fold Cross-Validation Accuracy")
plt.legend(); plt.tight_layout(); plt.savefig("figures/11_cv_fold_comparison.png", dpi=150); plt.close()

# ---------------------------------------------------------------------
# 10. SAVE SUMMARY RESULTS TO A TEXT FILE (used to fill report tables)
# ---------------------------------------------------------------------
with open("results_summary.txt","w") as f:
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Class distribution: {df['is_spam'].value_counts().to_dict()}\n\n")

    f.write(f"LR GridSearch best params: {lr_grid.best_params_}\n")
    f.write(f"LR GridSearch best CV acc: {lr_grid.best_score_:.4f}  time={lr_grid_time:.2f}s\n")
    f.write(f"LR RandomSearch best params: {lr_rand.best_params_}\n")
    f.write(f"LR RandomSearch best CV acc: {lr_rand.best_score_:.4f}  time={lr_rand_time:.2f}s\n\n")

    f.write(f"SVM GridSearch best params: {svm_grid.best_params_}\n")
    f.write(f"SVM GridSearch best CV acc: {svm_grid.best_score_:.4f}  time={svm_grid_time:.2f}s\n")
    f.write(f"SVM RandomSearch best params: {svm_rand.best_params_}\n")
    f.write(f"SVM RandomSearch best CV acc: {svm_rand.best_score_:.4f}  time={svm_rand_time:.2f}s\n\n")

    f.write(f"Baseline LR test metrics: {lr_base_metrics}  train_time={lr_base_time:.4f}s\n")
    f.write(f"Tuned LR test metrics: {lr_best_metrics}\n\n")
    f.write(f"Tuned SVM test metrics: {svm_best_metrics}\n\n")

    for k in kernels:
        f.write(f"SVM[{k}] test metrics: {svm_kernel_results[k]}\n")
    f.write("\n")

    f.write(f"LR 5-fold CV scores: {lr_cv_scores.tolist()}  mean={lr_cv_scores.mean():.4f} std={lr_cv_scores.std():.4f}\n")
    f.write(f"SVM 5-fold CV scores: {svm_cv_scores.tolist()}  mean={svm_cv_scores.mean():.4f} std={svm_cv_scores.std():.4f}\n")

print("\nDone. Results written to results_summary.txt and figures/ directory.")
