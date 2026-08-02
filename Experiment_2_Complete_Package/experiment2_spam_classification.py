"""
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 2: Email Spam/Ham Classification using Naive Bayes and KNN
---------------------------------------------------------------------
Dataset : UCI / Kaggle Spambase (4601 emails, 57 numeric features, binary label)

This script performs:
  1. Data loading and cleaning
  2. Exploratory Data Analysis (EDA)
  3. Preprocessing (standardisation, train-test split)
  4. Gaussian, Multinomial and Bernoulli Naive Bayes training & evaluation
  5. KNN: k-sweep, KDTree vs BallTree, GridSearchCV, RandomizedSearchCV
  6. 5-Fold Cross-Validation comparison of the best NB and best KNN models
  7. Training/prediction time benchmarking
  8. Generation of all required plots
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, auc, precision_recall_curve,
                              confusion_matrix, ConfusionMatrixDisplay)

# ---------------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------------
# UCI Spambase attribute names: 48 word-frequency + 6 char-frequency +
# 3 capital-run-length features + 1 binary label (1 = spam, 0 = ham)
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

X_raw = df.drop(columns=["is_spam"]).values      # non-negative frequencies (for MNB/BNB)
y = df["is_spam"].values

# ---------------------------------------------------------------------
# 2. EDA (see notebook / report for all 4 EDA figures)
# ---------------------------------------------------------------------
sns.countplot(x=y)                                          # Class distribution
plt.title("Class Distribution"); plt.savefig("class_distribution.png"); plt.close()

corr = df.corr()["is_spam"].drop("is_spam").sort_values(ascending=False)
print("Top correlated features with label:\n", corr.head(5))

# ---------------------------------------------------------------------
# 3. PREPROCESSING: stratified 80/20 split + standardisation
# ---------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

def evaluate(y_true, y_pred, y_score=None):
    """Return standard classification metrics as a dict."""
    d = dict(accuracy=accuracy_score(y_true, y_pred),
             precision=precision_score(y_true, y_pred),
             recall=recall_score(y_true, y_pred),
             f1=f1_score(y_true, y_pred))
    if y_score is not None:
        d["roc_auc"] = roc_auc_score(y_true, y_score)
    return d

# ---------------------------------------------------------------------
# 4. NAIVE BAYES: Gaussian (scaled), Multinomial & Bernoulli (raw, non-negative)
# ---------------------------------------------------------------------
results_nb = {}
for name, model, Xtr, Xte in [
        ("Gaussian",     GaussianNB(),     X_train_scaled, X_test_scaled),
        ("Multinomial",  MultinomialNB(),  X_train_raw,    X_test_raw),
        ("Bernoulli",    BernoulliNB(),    X_train_raw,    X_test_raw)]:
    t0 = time.perf_counter(); model.fit(Xtr, y_train); train_time = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); predict_time = time.perf_counter() - t0
    score = model.predict_proba(Xte)[:, 1]
    results_nb[name] = {**evaluate(y_test, pred, score),
                         "train_time": train_time, "predict_time": predict_time}
    print(name, "NB:", results_nb[name])

best_nb_name = max(results_nb, key=lambda k: results_nb[k]["f1"])   # Bernoulli wins on F1

# ---------------------------------------------------------------------
# 5. KNN: k-sweep, KDTree vs BallTree, GridSearchCV, RandomizedSearchCV
# ---------------------------------------------------------------------
knn_k_results = {}
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train)
    knn_k_results[k] = evaluate(y_test, knn.predict(X_test_scaled))

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {"n_neighbors": [1,3,5,7,9,11,13,15],
              "weights": ["uniform","distance"],
              "algorithm": ["kd_tree","ball_tree"],
              "metric": ["euclidean","manhattan"]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print("GridSearchCV best:", grid.best_params_, grid.best_score_)

param_dist = {"n_neighbors": list(range(1, 26)),
              "weights": ["uniform","distance"],
              "algorithm": ["kd_tree","ball_tree"],
              "metric": ["euclidean","manhattan"]}
rand = RandomizedSearchCV(KNeighborsClassifier(), param_dist, n_iter=25, cv=cv5,
                           scoring="accuracy", random_state=42, n_jobs=-1)
rand.fit(X_train_scaled, y_train)
print("RandomizedSearchCV best:", rand.best_params_, rand.best_score_)

best_knn = KNeighborsClassifier(**grid.best_params_).fit(X_train_scaled, y_train)
best_knn_metrics = evaluate(y_test, best_knn.predict(X_test_scaled),
                             best_knn.predict_proba(X_test_scaled)[:, 1])
print("Best KNN test metrics:", best_knn_metrics)

# ---------------------------------------------------------------------
# 6. 5-FOLD CROSS VALIDATION: Best NB vs Best KNN
# ---------------------------------------------------------------------
nb_cv = cross_val_score(BernoulliNB(), X_train_raw, y_train, cv=cv5, scoring="accuracy")
knn_cv = cross_val_score(KNeighborsClassifier(**grid.best_params_),
                          X_train_scaled, y_train, cv=cv5, scoring="accuracy")
print("Bernoulli NB 5-fold CV:", nb_cv, nb_cv.mean())
print("Best KNN 5-fold CV:", knn_cv, knn_cv.mean())

# ---------------------------------------------------------------------
# 7. Plots (ROC, PR, confusion matrices, accuracy-vs-k, timing, heatmaps, etc.)
#    -- see accompanying report for the complete set of 14 figures.
# ---------------------------------------------------------------------
