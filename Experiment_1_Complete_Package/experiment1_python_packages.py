"""
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 1: Working with Python packages - NumPy, SciPy, Scikit-Learn, Matplotlib
--------------------------------------------------------------------------------------
Part A: Explore core functions of NumPy, Pandas, SciPy, Scikit-learn and Matplotlib.
Part B: Download/load five reference datasets, identify the ML task for each, and
        the machine-learning workflow steps applied to them.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_digits
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")

# =====================================================================
# PART A: LIBRARY FUNCTION EXPLORATION
# =====================================================================
print("="*70)
print("PART A: NumPy / Pandas / SciPy / Scikit-learn / Matplotlib basics")
print("="*70)

# --- NumPy ---
a = np.arange(12).reshape(3, 4)
print("\nNumPy array:\n", a)
print("Sum:", a.sum(), " Mean:", a.mean(), " Std:", round(a.std(), 3))
print("Transpose:\n", a.T)
print("Matrix product a @ a.T:\n", a @ a.T)

# --- Pandas ---
df_demo = pd.DataFrame({
    "feature_1": np.random.RandomState(0).normal(50, 10, 10),
    "feature_2": np.random.RandomState(1).randint(1, 100, 10),
    "category": ["A","B","A","B","A","A","B","B","A","B"]
})
print("\nPandas DataFrame head:\n", df_demo.head())
print("\nGroupby mean by category:\n", df_demo.groupby("category").mean(numeric_only=True))
print("\nDescribe:\n", df_demo.describe())

# --- SciPy ---
t_stat, p_val = stats.ttest_ind(df_demo[df_demo.category=="A"]["feature_1"],
                                 df_demo[df_demo.category=="B"]["feature_1"])
print(f"\nSciPy independent t-test (feature_1, A vs B): t={t_stat:.3f}, p={p_val:.3f}")
mat = np.array([[4,2],[1,3]])
eigvals, eigvecs = linalg.eig(mat)
print("SciPy linalg eigenvalues of [[4,2],[1,3]]:", eigvals.real)

# --- Scikit-learn ---
iris_tmp = load_iris()
Xk = SelectKBest(score_func=f_classif, k=2).fit(iris_tmp.data, iris_tmp.target)
print("\nScikit-learn SelectKBest (ANOVA F-test) scores on Iris:", np.round(Xk.scores_, 2))

# --- Matplotlib ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(np.sort(df_demo["feature_1"].values), marker="o")
axes[0].set_title("Matplotlib: Line Plot")
axes[1].bar(df_demo["category"].value_counts().index, df_demo["category"].value_counts().values)
axes[1].set_title("Matplotlib: Bar Chart")
plt.tight_layout()
plt.savefig("figures/00_library_demo.png", dpi=150)
plt.close()
print("\nSaved demonstration plot to figures/00_library_demo.png")

# =====================================================================
# PART B: DATASET EXPLORATION AND ML TASK IDENTIFICATION
# =====================================================================
print("\n" + "="*70)
print("PART B: Dataset exploration")
print("="*70)

summary_rows = []

# ---------- 1. Iris Dataset ----------
iris = load_iris(as_frame=True)
iris_df = iris.frame
print("\n[Iris] shape:", iris_df.shape, " classes:", iris.target_names.tolist())
plt.figure(figsize=(6,5))
sns.scatterplot(data=iris_df, x="sepal length (cm)", y="petal length (cm)", hue="target", palette="deep")
plt.title("Iris: Sepal Length vs Petal Length by Species")
plt.tight_layout(); plt.savefig("figures/01_iris_scatter.png", dpi=150); plt.close()
summary_rows.append(dict(Dataset="Iris Dataset", Task="Supervised - Multi-class Classification",
                          FeatureSelection="ANOVA F-test / SelectKBest (all 4 features informative)",
                          Algorithm="Logistic Regression / KNN / SVM (small, linearly-separable-ish, 3 classes)"))

# ---------- 2. Loan Amount Prediction ----------
loan_df = pd.read_csv("loan_train.csv")
print("\n[Loan] shape:", loan_df.shape, " missing values:", loan_df.isnull().sum().sum())
plt.figure(figsize=(6,4))
sns.histplot(loan_df["LoanAmount"].dropna(), bins=25, kde=True)
plt.title("Loan Amount Distribution"); plt.xlabel("Loan Amount (thousands)")
plt.tight_layout(); plt.savefig("figures/02_loan_hist.png", dpi=150); plt.close()
summary_rows.append(dict(Dataset="Loan Amount Prediction", Task="Supervised - Regression (continuous LoanAmount target)",
                          FeatureSelection="Correlation analysis + domain-driven selection (income, credit history)",
                          Algorithm="Linear / Ridge / Lasso / Elastic Net Regression"))

# ---------- 3. Diabetes Prediction (Pima Indians Diabetes) ----------
pima_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
             "BMI","DiabetesPedigreeFunction","Age","Outcome"]
pima_df = pd.read_csv("pima_diabetes.csv", header=None, names=pima_cols)
print("\n[Diabetes] shape:", pima_df.shape, " class balance:\n", pima_df["Outcome"].value_counts())
plt.figure(figsize=(6,4))
sns.boxplot(x="Outcome", y="Glucose", data=pima_df)
plt.title("Glucose Level by Diabetes Outcome")
plt.tight_layout(); plt.savefig("figures/03_diabetes_box.png", dpi=150); plt.close()
chi_scores = SelectKBest(score_func=f_classif, k=4).fit(
    pima_df.drop(columns=["Outcome"]), pima_df["Outcome"])
top4 = pima_df.drop(columns=["Outcome"]).columns[np.argsort(chi_scores.scores_)[::-1][:4]].tolist()
summary_rows.append(dict(Dataset="Predicting Diabetes", Task="Supervised - Binary Classification",
                          FeatureSelection=f"ANOVA F-test (top: {', '.join(top4)})",
                          Algorithm="Logistic Regression / Random Forest / SVM"))

# ---------- 4. Email Spam Classification ----------
WORDS = ["make","address","all","3d","our","over","remove","internet","order","mail",
"receive","will","people","report","addresses","free","business","email","you","credit",
"your","font","000","money","hp","hpl","george","650","lab","labs","telnet","857","data",
"415","85","technology","1999","parts","pm","direct","cs","meeting","original","project",
"re","edu","table","conference"]
CHARS = [";","(","[","!","$","#"]
spam_cols = ([f"word_freq_{w}" for w in WORDS] + [f"char_freq_{c}" for c in CHARS] +
             ["capital_run_length_average","capital_run_length_longest",
              "capital_run_length_total","is_spam"])
spam_df = pd.read_csv("spambase.data", header=None, names=spam_cols)
print("\n[Spam] shape:", spam_df.shape, " class balance:\n", spam_df["is_spam"].value_counts())
plt.figure(figsize=(6,4))
sns.countplot(x="is_spam", data=spam_df)
plt.title("Spambase: Class Distribution")
plt.tight_layout(); plt.savefig("figures/04_spam_countplot.png", dpi=150); plt.close()
summary_rows.append(dict(Dataset="Classification of Email Spam", Task="Supervised - Binary Classification",
                          FeatureSelection="Correlation with label + variance threshold (57 numeric features)",
                          Algorithm="Naive Bayes / KNN / Logistic Regression / SVM"))

# ---------- 5. Handwritten Character Recognition (Digits, 8x8 - MNIST-style task) ----------
digits = load_digits(as_frame=False)
print("\n[Digits] shape:", digits.data.shape, " classes:", np.unique(digits.target))
fig, axes = plt.subplots(2, 5, figsize=(10,4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(str(digits.target[i])); ax.axis("off")
plt.suptitle("Handwritten Digit Samples (8x8 pixel images)")
plt.tight_layout(); plt.savefig("figures/05_digits_samples.png", dpi=150); plt.close()
summary_rows.append(dict(Dataset="Handwritten Character Recognition / MNIST",
                          Task="Supervised - Multi-class Classification (10 classes)",
                          FeatureSelection="PCA (dimensionality reduction on 64 pixel features)",
                          Algorithm="CNN / SVM (RBF kernel) / Random Forest / k-NN"))

# ---------- Summary Table ----------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("dataset_task_summary.csv", index=False)
print("\n" + "="*70)
print("SUMMARY TABLE: Dataset -> ML Task -> Feature Selection -> Suitable Algorithm")
print("="*70)
print(summary_df.to_string(index=False))

print("\nDone. Figures saved to figures/, summary table saved to dataset_task_summary.csv")
