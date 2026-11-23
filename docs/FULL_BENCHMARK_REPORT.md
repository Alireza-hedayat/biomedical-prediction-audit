<!-- Generated from the executed notebook by export_report.py. -->

# Reproducible Multi-Dataset Medical ML Benchmark

**Scope.** This executed project reference evaluates 11 classical classifiers, structured feature subsets, selection stability, clinically motivated thresholds, probability calibration, decision curves, and an exactly simulated three-qubit fidelity kernel on four public medical datasets. `SEED = 42`; all measurements shown were computed in this execution. CSVs contain full numerical precision.

**Two complementary resampling designs.** Overall performance tables use `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)` (15 correlated test folds; sample SD is descriptive, not a confidence interval). Paired tests, nested thresholding, calibration, and clinical curves use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` so each **row** has exactly one out-of-fold (OOF) prediction. For Parkinson's, a row is a recording rather than a person; a separate group-aware audit is essential.

**Interpretation boundary.** These are internal cross-validation studies, not prospective external validation, a diagnostic device, or evidence of clinical benefit. A source-identifier audit finds 139 shared identifiers between WDBC and WPBC, including 126 identical 30-feature measurements; these are not independent cohorts and must not be described as cross-dataset external validation. The Parkinson source describes 31 people but 32 distinct name stems are observed; stem grouping is a proxy requiring verification. WPBC is a censored time-to-event cohort analysed here with a crude observed recurrence label and no fixed prediction horizon. Case-control sampling affects calibration, prevalence and decision curves.

**Execution.** Run all cells in order from this repository directory with the pinned environment and the supplied `data/`. Every table, model, figure and quality check is recomputed; no network access or quantum hardware is required. Parallel numerical library threads are restricted in the supplied command-line runner to limit resource contention. All plots are both inline and exported to `figures/` as vector PDF and 300-dpi PNG.

## 0. Computational environment and output contract

All pseudo-random components use $s=42$. Library versions, computational platform and output directories are printed before any model is fitted. The wall-clock measurements that appear later are machine-specific and deliberately excluded from exact numerical regression tests. Run this cell first from the repository root; no global preprocessing has yet been learned.


<!-- Notebook code cell 2; execution count 1 -->

````python
import sys, time, json, platform, warnings, hashlib
import numpy as np
import pandas as pd
import sklearn
from pathlib import Path

warnings.filterwarnings("once")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")

SEED = 42
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

print("python      :", sys.version.split()[0])
print("platform    :", platform.platform())
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("scikit-learn:", sklearn.__version__)

import os
from IPython.display import display, Image, Markdown
np.random.seed(SEED)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)
````


**Output (stdout):**

````text
python      : 3.13.18
platform    : Linux-6.1.189+-x86_64-with-glibc2.41
numpy       : 2.3.5
pandas      : 2.2.3
scikit-learn: 1.6.1
````

**Interpretation.** The printed versions define the observed software environment. Matching the pinned versions and raw-data checksums supports reproducibility, but floating-point arithmetic, processor architecture, library implementations and timing can still differ across machines.

## 1. WDBC: clinical positive class

`load_breast_cancer` supplies $n=569$ observations and $p=30$ predictors, but its labels are `0 = malignant` and `1 = benign`. We define $Y_i=\mathbb{1}(Y_i^{\rm sklearn}=0)$, so sensitivity always refers to the clinically positive condition. Assertions establish the expected 212 malignant and 357 benign records before any evaluation. The displayed hash fingerprints the exact numerical design matrix.


<!-- Notebook code cell 5; execution count 2 -->

````python
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer(as_frame=True)
X_raw = ds.data.copy()
y_sklearn = ds.target.copy()                 # 0 = malignant, 1 = benign  (sklearn convention)
y = (y_sklearn == 0).astype(int)             # 1 = malignant  <-- our positive class

assert y.sum() == 212 and (1 - y).sum() == 357, "class counts changed"
assert X_raw.shape == (569, 30)

# fingerprint of the design matrix, so the data version is unambiguous
fp = hashlib.sha256(np.ascontiguousarray(X_raw.values)).hexdigest()[:16]
print(f"design matrix : {X_raw.shape}   sha256[:16] = {fp}")
print(f"malignant (1) : {int(y.sum())}   benign (0) : {int((1-y).sum())}")
print("\nclass balance :", dict(y.value_counts().sort_index()))

summary = X_raw.describe().T[["mean", "std", "min", "max"]]
print("\nfeature scale ranges (first 5):")
display(summary.head())
````


**Output (stdout):**

````text
design matrix : (569, 30)   sha256[:16] = 6b202a2072f9a038
malignant (1) : 212   benign (0) : 357

class balance : {0: np.int64(357), 1: np.int64(212)}

feature scale ranges (first 5):
````


**Displayed output:**

````text
                    mean      std      min       max
mean radius      14.1273   3.5240   6.9810   28.1100
mean texture     19.2896   4.3010   9.7100   39.2800
mean perimeter   91.9690  24.2990  43.7900  188.5000
mean area       654.8891 351.9141 143.5000 2501.0000
mean smoothness   0.0964   0.0141   0.0526    0.1634
````

**Interpretation.** The class count assertions prevent an inverted clinical endpoint; the descriptive means and ranges are dataset summaries, not fitted transformations. Feature magnitudes differ substantially, motivating fold-specific standardization. `The asserted WDBC sample contains 212 malignant and 357 benign observations.`

## 2. WDBC feature-block membership

Let $G_{\mathrm{Mean}},G_{\mathrm{Error}},G_{\mathrm{Worst}}$ be disjoint sets of ten measurements each, and $G_{\mathrm{All}}$ their union. The name token `mean ` or `worst ` occurs at the **start** of the scikit-learn column name, whereas ` error` occurs at the end. Membership, sizes, disjointness and exhaustiveness are asserted rather than inferred from labels. No outcome information is used to define these fixed blocks.


<!-- Notebook code cell 8; execution count 3 -->

````python
STEMS = ["radius","texture","perimeter","area","smoothness",
         "compactness","concavity","concave points","symmetry","fractal dimension"]

mean_cols  = [c for c in X_raw.columns if c.startswith("mean ")]
error_cols = [c for c in X_raw.columns if c.endswith(" error")]
worst_cols = [c for c in X_raw.columns if c.startswith("worst ")]

GROUPS = {"Mean": mean_cols, "Error": error_cols, "Worst": worst_cols,
          "All": list(X_raw.columns)}

# --- hard checks -------------------------------------------------------------
for g in ("Mean", "Error", "Worst"):
    assert len(GROUPS[g]) == 10, (g, len(GROUPS[g]))
assert len(set(mean_cols) | set(error_cols) | set(worst_cols)) == 30
assert set(mean_cols) | set(error_cols) | set(worst_cols) == set(X_raw.columns)

print("group sizes :", {k: len(v) for k, v in GROUPS.items()})
print("\nMean  block:", ", ".join(mean_cols))
print("\nError block:", ", ".join(error_cols))
print("\nWorst block:", ", ".join(worst_cols))
print("\nall groups are disjoint and jointly exhaustive: OK")
````


**Output (stdout):**

````text
group sizes : {'Mean': 10, 'Error': 10, 'Worst': 10, 'All': 30}

Mean  block: mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension

Error block: radius error, texture error, perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error

Worst block: worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension

all groups are disjoint and jointly exhaustive: OK
````

**Interpretation.** The block assertions establish a 10/10/10 decomposition of all 30 predictors; a fixed block is deterministic across folds, not a data-driven selection. These names should not be generalized to datasets with different domains.

## 3. Leakage-safe cross-validation and metric definitions

For $TP,TN,FP,FN$, sensitivity is $TP/(TP+FN)$, specificity $TN/(TN+FP)$ and precision $TP/(TP+FP)$. F1, balanced accuracy, MCC, ROC-AUC and average precision use the same positive-class convention. A `Pipeline` fits `StandardScaler` on each training fold only; scores are computed on the corresponding held-out fold. This shared engine is used by the WDBC experiments below.


<!-- Notebook code cell 11; execution count 4 -->

````python
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (confusion_matrix, recall_score, precision_score, f1_score,
                             roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef, brier_score_loss)

def build_pipe(estimator, scaler=True):
    steps = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

def metrics_from_preds(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
               sensitivity=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
               specificity=tn / (tn + fp) if (tn + fp) else np.nan,
               precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
               f1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
               accuracy=accuracy_score(y_true, y_pred),
               balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
               mcc=matthews_corrcoef(y_true, y_pred))
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return out

def cv_evaluate(estimator, X, y, cv, scaler=True, decision="auto"):
    # fit inside every fold; return per-fold metrics and OOF probabilities
    rows, oof = [], np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(cv.split(X, y), start=1):
        pipe = build_pipe(clone(estimator), scaler=scaler)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        if hasattr(pipe, "predict_proba"):
            score = pipe.predict_proba(X.iloc[te])[:, 1]
        elif decision == "decision":
            s = pipe.decision_function(X.iloc[te])
            score = 1 / (1 + np.exp(-s))
        else:
            score = pred.astype(float)
        oof[te] = score
        m = metrics_from_preds(y.iloc[te].values, pred, score)
        m["fold"] = k
        rows.append(m)
    return pd.DataFrame(rows), oof
````

**Interpretation.** The helper defines metrics, not a numerical result. Every training/test split is stratified; an OOF vector is meaningful only for a single five-fold partition, because repeated CV overwrites row scores across repeats. Repeated-CV summaries below use fold metrics instead.

## 4. WDBC experiment 1: structural blocks

Four prespecified feature blocks are crossed with logistic regression, RBF SVM and gradient boosting under the 15-fold repeated protocol. For a metric $m$, $\bar m=15^{-1}\sum_k m_k$ and $s_m$ is its sample SD. Standardization is learned within each fit. Compare models within the same block and blocks within the same model; this is exploratory rather than a superiority or equivalence test.


<!-- Notebook code cell 14; execution count 5 -->

````python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

cv_main = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

probe_models = {
    "Logistic Regression (L2)": LogisticRegression(max_iter=2000, random_state=SEED),
    "SVM (RBF)":                 SVC(kernel="rbf", probability=True, random_state=SEED),
    "Gradient Boosting":         GradientBoostingClassifier(random_state=SEED),
}

rows = []
for gname, cols in GROUPS.items():
    for mname, est in probe_models.items():
        t0 = time.perf_counter()
        df, _ = cv_evaluate(est, X_raw[cols], y, cv_main)
        dt = time.perf_counter() - t0
        rows.append({"feature_block": gname, "n_features": len(cols), "model": mname,
                     "sensitivity_mean": df.sensitivity.mean(), "sensitivity_sd": df.sensitivity.std(ddof=1),
                     "precision_mean": df.precision.mean(), "specificity_mean": df.specificity.mean(),
                     "f1_mean": df.f1.mean(), "mcc_mean": df.mcc.mean(),
                     "roc_auc_mean": df.roc_auc.mean(), "fn_mean": df.fn.mean(), "fp_mean": df.fp.mean(),
                     "fit_seconds": dt})
        print(f"{gname:>6} | {mname:<26} sens={df.sensitivity.mean():.4f} ± {df.sensitivity.std(ddof=1):.4f}  "
              f"prec={df.precision.mean():.4f}  auc={df.roc_auc.mean():.4f}  FN={df.fn.mean():.1f}")

feature_group_table = pd.DataFrame(rows).sort_values(["feature_block", "model"]).reset_index(drop=True)
feature_group_table.to_csv(RESULTS / "exp1_feature_blocks.csv", index=False)

print("\n--- feature-block x model ---")
display(feature_group_table[["feature_block","n_features","model","sensitivity_mean",
                             "sensitivity_sd","precision_mean","roc_auc_mean","fn_mean"]])
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
  Mean | Logistic Regression (L2)   sens=0.8931 ± 0.0538  prec=0.9417  auc=0.9858  FN=4.5
````


**Output (stdout):**

````text
  Mean | SVM (RBF)                  sens=0.8807 ± 0.0559  prec=0.9655  auc=0.9867  FN=5.1
````


**Output (stdout):**

````text
  Mean | Gradient Boosting          sens=0.9152 ± 0.0536  prec=0.9312  auc=0.9856  FN=3.6
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
 Error | Logistic Regression (L2)   sens=0.7547 ± 0.0593  prec=0.8868  auc=0.9426  FN=10.4
````


**Output (stdout):**

````text
 Error | SVM (RBF)                  sens=0.7264 ± 0.0462  prec=0.8809  auc=0.9388  FN=11.6
````


**Output (stdout):**

````text
 Error | Gradient Boosting          sens=0.8162 ± 0.0403  prec=0.8717  auc=0.9495  FN=7.8
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
 Worst | Logistic Regression (L2)   sens=0.9575 ± 0.0379  prec=0.9813  auc=0.9925  FN=1.8
````


**Output (stdout):**

````text
 Worst | SVM (RBF)                  sens=0.9498 ± 0.0364  prec=0.9684  auc=0.9948  FN=2.1
````


**Output (stdout):**

````text
 Worst | Gradient Boosting          sens=0.9277 ± 0.0400  prec=0.9526  auc=0.9899  FN=3.1
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
   All | Logistic Regression (L2)   sens=0.9512 ± 0.0371  prec=0.9841  auc=0.9946  FN=2.1
````


**Output (stdout):**

````text
   All | SVM (RBF)                  sens=0.9606 ± 0.0317  prec=0.9748  auc=0.9953  FN=1.7
````


**Output (stdout):**

````text
   All | Gradient Boosting          sens=0.9262 ± 0.0506  prec=0.9604  auc=0.9925  FN=3.1

--- feature-block x model ---
````


**Displayed output:**

````text
   feature_block  n_features                     model  sensitivity_mean  sensitivity_sd  precision_mean  roc_auc_mean  fn_mean
0            All          30         Gradient Boosting            0.9262          0.0506          0.9604        0.9925   3.1333
1            All          30  Logistic Regression (L2)            0.9512          0.0371          0.9841        0.9946   2.0667
2            All          30                 SVM (RBF)            0.9606          0.0317          0.9748        0.9953   1.6667
3          Error          10         Gradient Boosting            0.8162          0.0403          0.8717        0.9495   7.8000
4          Error          10  Logistic Regression (L2)            0.7547          0.0593          0.8868        0.9426  10.4000
5          Error          10                 SVM (RBF)            0.7264          0.0462          0.8809        0.9388  11.6000
6           Mean          10         Gradient Boosting            0.9152          0.0536          0.9312        0.9856   3.6000
7           Mean          10  Logistic Regression (L2)            0.8931          0.0538          0.9417        0.9858   4.5333
8           Mean          10                 SVM (RBF)            0.8807          0.0559          0.9655        0.9867   5.0667
9          Worst          10         Gradient Boosting            0.9277          0.0400          0.9526        0.9899   3.0667
10         Worst          10  Logistic Regression (L2)            0.9575          0.0379          0.9813        0.9925   1.8000
11         Worst          10                 SVM (RBF)            0.9498          0.0364          0.9684        0.9948   2.1333
````

**Interpretation.** `For L2 logistic regression, Worst-block sensitivity is 0.9575 ± 0.0379 and All-feature sensitivity is 0.9512 ± 0.0371.` The fold SD describes variability across overlapping resamples, not independent replications or a confidence interval. Predictive performance alone does not prove the fixed subset is statistically equivalent to all features.

## 5. WDBC experiment 2: classical classifier zoo

Eleven explicitly parameterized estimators are compared on all 30 features using identical repeated stratified splits and the same fold-fitted scaling. ROC-AUC measures ranking across thresholds, whereas sensitivity, specificity, precision, MCC, FN and FP describe a chosen prediction rule. The architecture table records hyperparameters; runtimes are descriptive measurements on this machine.


<!-- Notebook code cell 17; execution count 6 -->

````python
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

zoo = {
    "Logistic Regression (L2)": LogisticRegression(penalty="l2", max_iter=2000, random_state=SEED),
    "Logistic Regression (L1)": LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=SEED),
    "Linear Discriminant (LDA)": LinearDiscriminantAnalysis(),
    "SVM (linear kernel)":      SVC(kernel="linear", probability=True, random_state=SEED),
    "SVM (RBF kernel)":         SVC(kernel="rbf", probability=True, random_state=SEED),
    "k-Nearest Neighbours":     KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes":     GaussianNB(),
    "Random Forest":            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Extra Trees":              ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingClassifier(random_state=SEED),
    "HistGradient Boosting":    HistGradientBoostingClassifier(random_state=SEED),
}

ARCH_NOTES = {
    "Logistic Regression (L2)": "linear, L2 penalty; p(y=1|x)=sigma(w·x+b)",
    "Logistic Regression (L1)": "linear, L1 penalty -> embedded sparsity",
    "Linear Discriminant (LDA)": "Gaussian class densities, shared covariance",
    "SVM (linear kernel)":      "max-margin hyperplane, soft margin C=1",
    "SVM (RBF kernel)":         "max-margin in infinite-dim RKHS, C=1, gamma='scale'",
    "k-Nearest Neighbours":     "non-parametric, 5 nearest neighbours, Euclidean, majority vote",
    "Gaussian Naive Bayes":     "generative, per-feature Gaussians, conditional independence",
    "Random Forest":            "bagged CART, 300 trees, sqrt(p) split candidates",
    "Extra Trees":              "bagged CART, random split thresholds, 300 trees",
    "Gradient Boosting":        "stagewise additive CART, 100 trees, lr=0.1, log-loss",
    "HistGradient Boosting":    "histogram-binned gradient boosting (LightGBM-style)",
}

arch_rows, res_rows = [], []
for name, est in zoo.items():
    t0 = time.perf_counter()
    df, _ = cv_evaluate(est, X_raw, y, cv_main)
    dt = time.perf_counter() - t0
    res_rows.append({"model": name, "sensitivity": df.sensitivity.mean(), "sensitivity_sd": df.sensitivity.std(ddof=1),
                     "precision": df.precision.mean(), "specificity": df.specificity.mean(),
                     "f1": df.f1.mean(), "balanced_acc": df.balanced_accuracy.mean(),
                     "mcc": df.mcc.mean(), "roc_auc": df.roc_auc.mean(), "pr_auc": df.pr_auc.mean(),
                     "fn": df.fn.mean(), "fp": df.fp.mean(), "n_folds": len(df), "train_seconds": dt})
    arch_rows.append({"model": name, "architecture": ARCH_NOTES[name],
                      "pipeline": "StandardScaler -> " + type(est).__name__,
                      "key_hyperparameters": str(est.get_params())})
    print(f"{name:<28} sens={df.sensitivity.mean():.4f} prec={df.precision.mean():.4f} "
          f"auc={df.roc_auc.mean():.4f} MCC={df.mcc.mean():.4f}  ({dt:.1f}s)")

model_zoo_table = pd.DataFrame(res_rows).sort_values("sensitivity", ascending=False).reset_index(drop=True)
arch_table = pd.DataFrame(arch_rows)
model_zoo_table.to_csv(RESULTS / "exp2_model_zoo.csv", index=False)
arch_table.to_csv(RESULTS / "exp2_architectures.csv", index=False)

print("\n--- ranked by sensitivity on the malignant class ---")
display(model_zoo_table[["model","sensitivity","sensitivity_sd","precision","specificity",
                         "mcc","roc_auc","fn","fp"]])
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Logistic Regression (L2)     sens=0.9512 prec=0.9841 auc=0.9946 MCC=0.9490  (0.3s)
````


**Output (stdout):**

````text
Logistic Regression (L1)     sens=0.9450 prec=0.9743 auc=0.9951 MCC=0.9362  (0.3s)
````


**Output (stdout):**

````text
Linear Discriminant (LDA)    sens=0.8899 prec=0.9928 auc=0.9920 MCC=0.9086  (0.3s)
````


**Output (stdout):**

````text
SVM (linear kernel)          sens=0.9465 prec=0.9787 auc=0.9942 MCC=0.9411  (0.4s)
````


**Output (stdout):**

````text
SVM (RBF kernel)             sens=0.9606 prec=0.9748 auc=0.9953 MCC=0.9488  (0.5s)
````


**Output (stdout):**

````text
k-Nearest Neighbours         sens=0.9262 prec=0.9837 auc=0.9844 MCC=0.9292  (0.3s)
````


**Output (stdout):**

````text
Gaussian Naive Bayes         sens=0.8979 prec=0.9190 auc=0.9872 MCC=0.8545  (0.2s)
````


**Output (stdout):**

````text
Random Forest                sens=0.9341 prec=0.9581 auc=0.9898 MCC=0.9146  (10.1s)
````


**Output (stdout):**

````text
Extra Trees                  sens=0.9403 prec=0.9688 auc=0.9939 MCC=0.9280  (7.6s)
````


**Output (stdout):**

````text
Gradient Boosting            sens=0.9262 prec=0.9604 auc=0.9925 MCC=0.9104  (9.0s)
````


**Output (stdout):**

````text
HistGradient Boosting        sens=0.9340 prec=0.9681 auc=0.9913 MCC=0.9226  (2.9s)

--- ranked by sensitivity on the malignant class ---
````


**Displayed output:**

````text
                        model  sensitivity  sensitivity_sd  precision  specificity    mcc  roc_auc     fn     fp
0            SVM (RBF kernel)       0.9606          0.0317     0.9748       0.9850 0.9488   0.9953 1.6667 1.0667
1    Logistic Regression (L2)       0.9512          0.0371     0.9841       0.9907 0.9490   0.9946 2.0667 0.6667
2         SVM (linear kernel)       0.9465          0.0412     0.9787       0.9879 0.9411   0.9942 2.2667 0.8667
3    Logistic Regression (L1)       0.9450          0.0376     0.9743       0.9850 0.9362   0.9951 2.3333 1.0667
4                 Extra Trees       0.9403          0.0415     0.9688       0.9813 0.9280   0.9939 2.5333 1.3333
5               Random Forest       0.9341          0.0553     0.9581       0.9748 0.9146   0.9898 2.8000 1.8000
6       HistGradient Boosting       0.9340          0.0367     0.9681       0.9813 0.9226   0.9913 2.8000 1.3333
7        k-Nearest Neighbours       0.9262          0.0376     0.9837       0.9907 0.9292   0.9844 3.1333 0.6667
8           Gradient Boosting       0.9262          0.0506     0.9604       0.9767 0.9104   0.9925 3.1333 1.6667
9        Gaussian Naive Bayes       0.8979          0.0474     0.9190       0.9514 0.8545   0.9872 4.3333 3.4667
10  Linear Discriminant (LDA)       0.8899          0.0408     0.9928       0.9963 0.9086   0.9920 4.6667 0.2667
````

**Interpretation.** `The largest descriptive sensitivity is SVM (RBF kernel) at 0.9606 ± 0.0317; its mean MCC is 0.9488 and mean false negatives per fold are 1.6667.` Ranking 11 models on the same sample introduces selection uncertainty; a high ranked score is not proof of clinically meaningful superiority. Each FN/FP value is a **mean per test fold**, not the count in the full dataset.

**Reference wording correction.** The legacy architecture-note string calls Extra Trees “bagged”; scikit-learn `ExtraTreesClassifier` actually uses `bootstrap=False` by default here. The estimator was not changed.

## 6. WDBC experiment 3: fixed subsets versus data-driven selection

`SelectKBest` selects ten training-fold features by ANOVA $F$ or mutual information before scaling and model fitting. The WDBC stability statistic is the **global** intersection divided by the global union of the 15 selected sets. Stability is not applicable to the two fixed configurations in this reference table; the multi-dataset experiment later additionally computes mean pairwise Jaccard similarities.


<!-- Notebook code cell 20; execution count 7 -->

````python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def _mi_score(Xa, ya):
    return mutual_info_classif(Xa, ya, random_state=SEED)

def make_selector(kind, k=10):
    if kind == "f_classif":
        return SelectKBest(score_func=f_classif, k=k)
    if kind == "mutual_info":
        return SelectKBest(score_func=_mi_score, k=k)
    raise ValueError(kind)

cv_sel = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

sel_rows = []
configs = [("fixed: Worst block (10)", X_raw[worst_cols], build_pipe(LogisticRegression(max_iter=2000, random_state=SEED))),
           ("fixed: All (30)",         X_raw,             build_pipe(LogisticRegression(max_iter=2000, random_state=SEED))),
           ("SelectKBest f_classif (10)",  X_raw, Pipeline([("select", make_selector("f_classif")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=SEED))])),
           ("SelectKBest mutual_info (10)",X_raw, Pipeline([("select", make_selector("mutual_info")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=SEED))]))]

for label, Xd, pipe in configs:
    sens, prec, aucs, fns, picks = [], [], [], [], []
    for tr, te in cv_sel.split(Xd, y):
        p = clone(pipe)
        p.fit(Xd.iloc[tr], y.iloc[tr])
        pred = p.predict(Xd.iloc[te]); sc = p.predict_proba(Xd.iloc[te])[:, 1]
        tn, fp, fn, tp = confusion_matrix(y.iloc[te], pred, labels=[0,1]).ravel()
        sens.append(tp/(tp+fn)); prec.append(tp/(tp+fp) if (tp+fp) else 0.0)
        aucs.append(roc_auc_score(y.iloc[te], sc)); fns.append(fn)
        if "select" in p.named_steps:
            picks.append(tuple(sorted(np.array(Xd.columns)[p.named_steps["select"].get_support()])))
    jaccard = np.nan
    if picks:
        inter = set(picks[0]).intersection(*map(set, picks[1:])); union = set().union(*map(set, picks))
        jaccard = len(inter)/len(union) if union else np.nan
    sel_rows.append({"configuration": label, "sensitivity": np.mean(sens), "sensitivity_sd": np.std(sens, ddof=1),
                     "precision": np.mean(prec), "roc_auc": np.mean(aucs), "fn": np.mean(fns),
                     "fold_selection_stability_jaccard": jaccard})
    print(f"{label:<32} sens={np.mean(sens):.4f} prec={np.mean(prec):.4f} "
          f"jaccard={jaccard if not np.isnan(jaccard) else float('nan'):.3f}")

selection_table = pd.DataFrame(sel_rows)
selection_table.to_csv(RESULTS / "exp3_feature_selection.csv", index=False)
display(selection_table)
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
fixed: Worst block (10)          sens=0.9575 prec=0.9813 jaccard=nan
fixed: All (30)                  sens=0.9512 prec=0.9841 jaccard=nan
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
SelectKBest f_classif (10)       sens=0.9261 prec=0.9432 jaccard=1.000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
SelectKBest mutual_info (10)     sens=0.9244 prec=0.9541 jaccard=1.000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Displayed output:**

````text
                  configuration  sensitivity  sensitivity_sd  precision  roc_auc     fn  fold_selection_stability_jaccard
0       fixed: Worst block (10)       0.9575          0.0379     0.9813   0.9925 1.8000                               NaN
1               fixed: All (30)       0.9512          0.0371     0.9841   0.9946 2.0667                               NaN
2    SelectKBest f_classif (10)       0.9261          0.0550     0.9432   0.9886 3.1333                            1.0000
3  SelectKBest mutual_info (10)       0.9244          0.0438     0.9541   0.9909 3.2000                            1.0000
````

**Interpretation.** `ANOVA and mutual-information selection sensitivities are 0.9261 and 0.9244; their global Jaccard values are 1.0000 and 1.0000, respectively.` Global and pairwise Jaccard are different statistics and should not be interchanged. A high selection stability says little about causal relevance, and statistical selection itself must remain inside the training fold.

## 7. WDBC experiment 4: apparent versus nested thresholding

For score $s_i$ and threshold $\tau$, predict positive when $s_i\ge\tau$. The apparent threshold is chosen and assessed on the same five-fold OOF scores, making its operating characteristics optimistically selected. The nested estimate chooses $\tau$ from inner OOF scores inside each outer training set and assesses it once on the outer test fold. The selection rule maximizes precision among grid thresholds reaching 95% sensitivity; if none is feasible this reference implementation minimizes classification error over the grid.


<!-- Notebook code cell 23; execution count 8 -->

````python
from sklearn.model_selection import StratifiedKFold

def choose_tau(y_true, scores, target=0.95, grid=np.arange(0.05, 0.96, 0.01)):
    best = None
    for t in grid:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn) else 0.0
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        if sens >= target:
            if best is None or prec > best[1]:
                best = (t, prec, sens)
    if best is None:
        t = grid[np.argmin([np.mean((scores >= g).astype(int) != y_true) for g in grid])]
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        best = (t, tp/(tp+fp) if (tp+fp) else 0.0, tp/(tp+fn))
    return float(best[0])

# --- 4a. out-of-fold scores for the 10-feature Worst block, logistic regression
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
_, oof_lr = cv_evaluate(LogisticRegression(max_iter=2000, random_state=SEED), X_raw[worst_cols], y, cv5)

threshold_rows = []
for t in np.arange(0.05, 0.96, 0.01):
    pred = (oof_lr >= t).astype(int)
    m = metrics_from_preds(y.values, pred, oof_lr)
    m["threshold"] = round(float(t), 2)
    threshold_rows.append(m)
threshold_table = pd.DataFrame(threshold_rows)[
    ["threshold","sensitivity","specificity","precision","f1","accuracy","fn","fp","tp","tn"]]
threshold_table.to_csv(RESULTS / "exp4a_threshold_curve.csv", index=False)

tau_apparent = choose_tau(y.values, oof_lr)
row = threshold_table.iloc[(threshold_table.threshold - tau_apparent).abs().argmin()]
print(f"apparent-optimal tau = {tau_apparent:.2f}")
print(row.to_string())

print("\n--- threshold curve (selected rows) ---")
display(threshold_table[threshold_table.threshold.isin([0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70])])

# --- 4b. nested (honest) estimate -------------------------------------------
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
nested_rows, collected = [], []
for k, (tr, te) in enumerate(outer.split(X_raw[worst_cols], y), start=1):
    Xtr, ytr = X_raw[worst_cols].iloc[tr], y.iloc[tr]
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, inner_oof = cv_evaluate(LogisticRegression(max_iter=2000, random_state=SEED), Xtr, ytr, inner)
    tau_k = choose_tau(ytr.values, inner_oof)

    pipe = build_pipe(LogisticRegression(max_iter=2000, random_state=SEED))
    pipe.fit(Xtr, ytr)
    score_te = pipe.predict_proba(X_raw[worst_cols].iloc[te])[:, 1]
    pred_te = (score_te >= tau_k).astype(int)
    m = metrics_from_preds(y.iloc[te].values, pred_te, score_te); m["fold"] = k; m["tau"] = tau_k
    nested_rows.append(m); collected.append(pd.DataFrame({"y": y.iloc[te].values, "score": score_te, "tau": tau_k}))

nested_df = pd.DataFrame(nested_rows)
print("\n--- nested threshold evaluation (tau re-selected inside every outer fold) ---")
display(nested_df[["fold","tau","sensitivity","precision","specificity","fn","fp","roc_auc"]])
print(f"nested sensitivity = {nested_df.sensitivity.mean():.4f} ± {nested_df.sensitivity.std(ddof=1):.4f}   "
      f"precision = {nested_df.precision.mean():.4f}   FN = {nested_df.fn.mean():.1f}")
nested_df.to_csv(RESULTS / "exp4b_nested_threshold.csv", index=False)
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
apparent-optimal tau = 0.55
threshold       0.5500
sensitivity     0.9528
specificity     0.9916
precision       0.9854
f1              0.9688
accuracy        0.9772
fn             10.0000
fp              3.0000
tp            202.0000
tn            354.0000

--- threshold curve (selected rows) ---
````


**Displayed output:**

````text
    threshold  sensitivity  specificity  precision     f1  accuracy  fn  fp   tp   tn
5      0.1000       0.9858       0.8824     0.8327 0.9028    0.9209   3  42  209  315
15     0.2000       0.9764       0.9412     0.9079 0.9409    0.9543   5  21  207  336
20     0.2500       0.9764       0.9524     0.9241 0.9495    0.9613   5  17  207  340
25     0.3000       0.9764       0.9580     0.9324 0.9539    0.9649   5  15  207  342
30     0.3500       0.9717       0.9720     0.9537 0.9626    0.9719   6  10  206  347
35     0.4000       0.9670       0.9832     0.9716 0.9693    0.9772   7   6  205  351
40     0.4500       0.9623       0.9888     0.9808 0.9714    0.9789   8   4  204  353
45     0.5000       0.9623       0.9888     0.9808 0.9714    0.9789   8   4  204  353
55     0.6000       0.9434       0.9944     0.9901 0.9662    0.9754  12   2  200  355
65     0.7000       0.9151       0.9972     0.9949 0.9533    0.9666  18   1  194  356
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text

--- nested threshold evaluation (tau re-selected inside every outer fold) ---
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Displayed output:**

````text
   fold    tau  sensitivity  precision  specificity  fn  fp  roc_auc
0     1 0.4300       1.0000     0.9348       0.9577   0   3   1.0000
1     2 0.5400       0.8605     1.0000       1.0000   6   0   0.9869
2     3 0.5900       0.9286     1.0000       1.0000   3   0   0.9808
3     4 0.4300       1.0000     0.9545       0.9722   0   2   1.0000
4     5 0.4100       0.9762     1.0000       1.0000   1   0   0.9997
````


**Output (stdout):**

````text
nested sensitivity = 0.9530 ± 0.0594   precision = 0.9779   FN = 2.0
````

**Interpretation.** `The apparent grid-selected threshold is 0.55 with sensitivity 0.9528; nested mean sensitivity is 0.9530 ± 0.0594 with outer thresholds 0.41–0.59.` Neither the training-fold target nor the apparent curve guarantees future sensitivity. The five nested outer thresholds are heterogeneous; a subsequent deployment threshold must be selected using an independently specified development procedure.

## 8. WDBC experiment 5: calibration and conditional bootstrap

The Brier score is $n^{-1}\sum_i(\hat p_i-y_i)^2$; log loss penalizes confident errors. Reliability bins compare average predictions with observed event frequencies. Platt scaling fits a sigmoid. Percentile bootstrap intervals resample the already-generated nested OOF prediction/outcome pairs ($B=2000$), and are therefore **conditional on the fitted OOF predictions** rather than full retraining uncertainty. In this WDBC reference protocol the scaler is fitted on the outer training set before the calibration object's internal splits; the stricter multi-dataset audit places the entire preprocessing pipeline inside the calibration estimator.


<!-- Notebook code cell 26; execution count 9 -->

````python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

def oof_scores(estimator, X, y, cv, calibrate=False):
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y):
        est = clone(estimator)
        if calibrate:
            est = CalibratedClassifierCV(estimator=est, method="sigmoid", cv=5)
            pipe = Pipeline([("scaler", StandardScaler()), ("cal", est)])
        else:
            pipe = build_pipe(est)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return oof

score_raw = oof_scores(LogisticRegression(max_iter=2000, random_state=SEED), X_raw[worst_cols], y, cv5, calibrate=False)
score_cal = oof_scores(LogisticRegression(max_iter=2000, random_state=SEED), X_raw[worst_cols], y, cv5, calibrate=True)

cal_rows = []
for label, s in [("uncalibrated", score_raw), ("Platt-calibrated", score_cal)]:
    cal_rows.append({"variant": label,
                     "brier": brier_score_loss(y, s),
                     "log_loss": log_loss(y, s),
                     "roc_auc": roc_auc_score(y, s)})
calibration_table = pd.DataFrame(cal_rows)
print("--- probability quality on the Worst block (5-fold OOF) ---")
display(calibration_table)
calibration_table.to_csv(RESULTS / "exp5a_calibration.csv", index=False)

# reliability bins
bins = np.linspace(0, 1, 11)
rel = []
for label, s in [("uncalibrated", score_raw), ("Platt-calibrated", score_cal)]:
    idx = np.clip(np.digitize(s, bins) - 1, 0, len(bins) - 2)
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() > 0:
            rel.append({"variant": label, "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                        "n": int(m.sum()), "mean_predicted": s[m].mean(), "observed_rate": y.values[m].mean()})
reliability_table = pd.DataFrame(rel)
reliability_table.to_csv(RESULTS / "exp5b_reliability.csv", index=False)
print("\n--- reliability diagram data (Platt-calibrated) ---")
display(reliability_table[reliability_table.variant == "Platt-calibrated"].head(10))

# --- bootstrap on the nested predictions ------------------------------------
pooled = pd.concat(collected, ignore_index=True)
pooled["pred"] = (pooled.score >= pooled.tau).astype(int)
rng = np.random.default_rng(SEED)
boot = []
for _ in range(2000):
    s = pooled.iloc[rng.integers(0, len(pooled), len(pooled))]
    tp = int(((s.pred == 1) & (s.y == 1)).sum()); fn = int(((s.pred == 0) & (s.y == 1)).sum())
    tn = int(((s.pred == 0) & (s.y == 0)).sum()); fp = int(((s.pred == 1) & (s.y == 0)).sum())
    boot.append({"sensitivity": tp/(tp+fn) if (tp+fn) else np.nan,
                 "precision": tp/(tp+fp) if (tp+fp) else np.nan,
                 "specificity": tn/(tn+fp) if (tn+fp) else np.nan})
boot = pd.DataFrame(boot)

point = metrics_from_preds(pooled.y.values, pooled.pred.values, pooled.score.values)
ci = boot.quantile([0.025, 0.5, 0.975]).T
ci.columns = ["ci_2.5", "median", "ci_97.5"]
ci["point_estimate"] = [point["sensitivity"], point["precision"], point["specificity"]]
print("\n--- bootstrap 95% percentile intervals (B = 2000) ---")
display(ci)
ci.to_csv(RESULTS / "exp5c_bootstrap_ci.csv")
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
--- probability quality on the Worst block (5-fold OOF) ---
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Displayed output:**

````text
            variant  brier  log_loss  roc_auc
0      uncalibrated 0.0201    0.0891   0.9917
1  Platt-calibrated 0.0264    0.1149   0.9920
````


**Output (stdout):**

````text

--- reliability diagram data (Platt-calibrated) ---
````


**Displayed output:**

````text
             variant      bin    n  mean_predicted  observed_rate
10  Platt-calibrated  0.0-0.1  278          0.0249         0.0072
11  Platt-calibrated  0.1-0.2   46          0.1379         0.0217
12  Platt-calibrated  0.2-0.3   22          0.2369         0.0909
13  Platt-calibrated  0.3-0.4   14          0.3581         0.2143
14  Platt-calibrated  0.4-0.5    7          0.4690         0.5714
15  Platt-calibrated  0.5-0.6    8          0.5424         0.7500
16  Platt-calibrated  0.6-0.7   11          0.6472         1.0000
17  Platt-calibrated  0.7-0.8   10          0.7443         1.0000
18  Platt-calibrated  0.8-0.9   27          0.8576         1.0000
19  Platt-calibrated  0.9-1.0  146          0.9801         1.0000
````


**Output (stdout):**

````text

--- bootstrap 95% percentile intervals (B = 2000) ---
````


**Displayed output:**

````text
             ci_2.5  median  ci_97.5  point_estimate
sensitivity  0.9220  0.9531   0.9804          0.9528
precision    0.9543  0.9765   0.9951          0.9758
specificity  0.9733  0.9863   0.9972          0.9860
````

**Interpretation.** `Brier score is 0.0201 uncalibrated and 0.0264 after Platt scaling. The conditional nested sensitivity interval is [0.9220, 0.9804].` Lower Brier and log loss are better; Platt scaling does not guarantee improvement. The displayed intervals do not include model-selection or full cross-validation fitting variability and do not establish external calibration.

## 9. WDBC experiment 6A: exact three-qubit statevector

For angles $x_0,x_1,x_2$, the state is $|\psi(x)\rangle=\mathrm{CNOT}_{1\to2}\mathrm{CNOT}_{0\to1}\bigotimes_jR_y(x_j)|000\rangle$, and $k(x,x')=|\langle\psi(x)|\psi(x')\rangle|^2$. NumPy builds the actual eight-amplitude state. Norms and finite kernel bounds are checked; a later cell additionally asserts symmetry, diagonal unity, positive semidefiniteness and a closed-form identity. Simulator wall time is not a hardware claim: a train/test kernel matrix requires $\mathcal O(n_{\rm train}n_{\rm test})$ pair evaluations before any symmetry or caching savings.


<!-- Notebook code cell 29; execution count 10 -->

````python
# --- 3-qubit state preparation, pure numpy ----------------------------------
RY = lambda t: np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])

def cnot(state, control, target, n=3):
    # apply CNOT on the computational-basis statevector (q0 = most significant)
    out = np.zeros_like(state)
    for idx, amp in enumerate(state):
        bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
        if bits[control] == 1:
            bits[target] ^= 1
        j = 0
        for b in bits:
            j = (j << 1) | b
        out[j] += amp
    return out

def quantum_state(x, n=3):
    psi = np.array([1.0])
    for xi in x:
        psi = np.kron(psi, RY(xi) @ np.array([1.0, 0.0]))
    for c, t in [(0, 1), (1, 2)]:
        psi = cnot(psi, c, t, n=n)
    return psi

def fidelity_kernel_matrix(Xa, Xb):
    # k(x,x') = |<psi(x)|psi(x')>|^2 for every pair
    A = np.array([quantum_state(x) for x in Xa])
    B = np.array([quantum_state(x) for x in Xb])
    inner = A.conj() @ B.T
    return np.abs(inner) ** 2

# --- sanity checks ------------------------------------------------------------
_psi = quantum_state(np.array([0.7, 1.1, 2.0]))
print("state norm          :", np.linalg.norm(_psi))                      # must be 1
K = fidelity_kernel_matrix(np.array([[0.3,0.5,0.9]]), np.array([[0.3,0.5,0.9],[1.0,2.0,0.1]]))
print("k(x,x)              :", K[0,0])                                     # must be 1
print("finite / in [0,1] :", bool(np.isfinite(K).all() and (K >= 0).all() and (K <= 1+1e-9).all()))

# --- timing of the kernel evaluation ----------------------------------------
_t0 = time.perf_counter()
_ = fidelity_kernel_matrix(np.random.rand(200, 3) * np.pi, np.random.rand(200, 3) * np.pi)
_dt = time.perf_counter() - _t0
print(f"\nkernel evaluations simulated: {200*200} pairs in {_dt:.4f} s  ({_dt/(200*200)*1e6:.1f} us/pair)")
print(f"extrapolated cost for one 5-fold CV over 569 samples: "
      f"~{sum(len(te)*len(tr) for tr, te in cv5.split(X_raw, y)) * (_dt/(200*200)) :.3f} s of simulation")
````


**Output (stdout):**

````text
state norm          : 1.0
k(x,x)              : 0.9999999999999991
finite / in [0,1] : True

kernel evaluations simulated: 40000 pairs in 0.0454 s  (1.1 us/pair)
extrapolated cost for one 5-fold CV over 569 samples: ~0.294 s of simulation
````

**Interpretation.** The printed timing measures classical simulation on this machine, not quantum-device latency, energy, or advantage. A fixed, input-independent unitary after the embedding cancels in an inner product; the analytical consequence is verified below.

## 10. WDBC experiment 6B: fidelity SVM and classical controls

The simulated model uses fold-fitted standardization, three principal components, training-fitted angular rescaling and an SVM with a precomputed fidelity kernel. The reference classical PCA-3 control uses standardization and PCA but does **not** include the angular rescaling; consequently it is not a strictly identical full front end. A raw-feature RBF-SVM provides a broader classical baseline. The comparison is on the same single five-fold partition.


<!-- Notebook code cell 32; execution count 11 -->

````python
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class QuantumKernelSVM(BaseEstimator, ClassifierMixin):
    # StandardScaler -> PCA(3) -> MinMax to [0,pi] -> fidelity kernel -> SVC(kernel='precomputed')
    def __init__(self, n_components=3, C=1.0):
        self.n_components = n_components; self.C = C
    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=self.n_components, random_state=SEED).fit(Z)
        Zp = self.pca_.transform(Z)
        self.angle_ = MinMaxScaler(feature_range=(0, np.pi)).fit(Zp)
        A = self.angle_.transform(Zp)
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=True, random_state=SEED)
        self.svc_.fit(fidelity_kernel_matrix(A, A), y)
        self.train_angles_ = A
        return self
    def decision_function(self, X):
        A = self.angle_.transform(self.pca_.transform(self.scaler_.transform(X)))
        return self.svc_.decision_function(fidelity_kernel_matrix(A, self.train_angles_))
    def predict_proba(self, X):
        A = self.angle_.transform(self.pca_.transform(self.scaler_.transform(X)))
        return self.svc_.predict_proba(fidelity_kernel_matrix(A, self.train_angles_))
    def predict(self, X):
        return self.svc_.predict(fidelity_kernel_matrix(
            self.angle_.transform(self.pca_.transform(self.scaler_.transform(X))), self.train_angles_))

class ClassicalPCA3SVM(BaseEstimator, ClassifierMixin):
    # Original PCA-3 RBF reference: no angular scaling, unlike the quantum path.
    def __init__(self, C=1.0):
        self.C = C
    def fit(self, X, y):
        self.pipe_ = Pipeline([("scaler", StandardScaler()),
                               ("pca", PCA(n_components=3, random_state=SEED)),
                               ("svc", SVC(kernel="rbf", C=self.C, probability=True, random_state=SEED))])
        self.pipe_.fit(X, y); return self
    def predict(self, X): return self.pipe_.predict(X)
    def predict_proba(self, X): return self.pipe_.predict_proba(X)
    def decision_function(self, X): return self.pipe_.decision_function(X)

quantum_rows, quantum_curves = [], {}
for label, est in [("Quantum kernel SVM (3 qubits)", QuantumKernelSVM()),
                   ("Classical RBF-SVM (same PCA-3 front end)", ClassicalPCA3SVM()),
                   ("Classical RBF-SVM (30 raw features)", SVC(kernel="rbf", probability=True, random_state=SEED))]:
    t0 = time.perf_counter()
    df, oofq = cv_evaluate(est, X_raw, y, cv5, scaler=isinstance(est, SVC))
    dt = time.perf_counter() - t0
    quantum_curves[label] = oofq
    quantum_rows.append({"model": label, "sensitivity": df.sensitivity.mean(), "sensitivity_sd": df.sensitivity.std(ddof=1),
                         "precision": df.precision.mean(), "specificity": df.specificity.mean(),
                         "f1": df.f1.mean(), "balanced_acc": df.balanced_accuracy.mean(), "mcc": df.mcc.mean(),
                         "roc_auc": df.roc_auc.mean(), "pr_auc": df.pr_auc.mean(),
                         "fn": df.fn.mean(), "fp": df.fp.mean(), "wall_seconds_5fold": dt})
    print(f"{label:<44} sens={df.sensitivity.mean():.4f} prec={df.precision.mean():.4f} "
          f"auc={df.roc_auc.mean():.4f}  ({dt:.1f}s)")

quantum_table = pd.DataFrame(quantum_rows)
print("\n--- quantum vs classical, identical splits ---")
display(quantum_table)
quantum_table.to_csv(RESULTS / "exp6_quantum_vs_classical.csv", index=False)

# paired comparison across folds: same test folds, so a paired test is legitimate
q = quantum_curves["Quantum kernel SVM (3 qubits)"]
c = quantum_curves["Classical RBF-SVM (same PCA-3 front end)"]
from scipy.stats import wilcoxon
d = np.abs(q - c)
print(f"\nmean |score difference| quantum vs classical-PCA3 : {d.mean():.4f}")
print("Reference comparison: both models use StandardScaler and PCA-3; the quantum path additionally rescales PCA coordinates to angles. The matched comparison below aligns that step as well.")
````


**Output (stdout):**

````text
Quantum kernel SVM (3 qubits)                sens=0.8963 prec=0.9471 auc=0.9904  (1.1s)
Classical RBF-SVM (same PCA-3 front end)     sens=0.8915 prec=0.9387 auc=0.9870  (0.1s)
````


**Output (stdout):**

````text
Classical RBF-SVM (30 raw features)          sens=0.9576 prec=0.9810 auc=0.9945  (0.2s)

--- quantum vs classical, identical splits ---
````


**Displayed output:**

````text
                                      model  sensitivity  sensitivity_sd  precision  specificity     f1  balanced_acc    mcc  roc_auc  pr_auc     fn     fp  wall_seconds_5fold
0             Quantum kernel SVM (3 qubits)       0.8963          0.0585     0.9471       0.9692 0.9196        0.9328 0.8768   0.9904  0.9866 4.4000 2.2000              1.1492
1  Classical RBF-SVM (same PCA-3 front end)       0.8915          0.0541     0.9387       0.9637 0.9130        0.9276 0.8658   0.9870  0.9835 4.6000 2.6000              0.1228
2       Classical RBF-SVM (30 raw features)       0.9576          0.0419     0.9810       0.9888 0.9688        0.9732 0.9514   0.9945  0.9928 1.8000 0.8000              0.1639
````


**Output (stdout):**

````text

mean |score difference| quantum vs classical-PCA3 : 0.0263
Reference comparison: both models use StandardScaler and PCA-3; the quantum path additionally rescales PCA coordinates to angles. The matched comparison below aligns that step as well.
````

**Interpretation.** `Quantum sensitivity is 0.8963 ± 0.0585 versus 0.8915 ± 0.0541 for the PCA-3 classical reference.` The reference PCA controls differ in one preprocessing step; the multi-dataset quantum experiment below explicitly matches that step and separately evaluates matched predictions. Simulation accuracy is not evidence of computational or clinical quantum advantage.

## 11. WDBC supplementary statistics and plotting setup

We reconstruct the same WDBC matrix and clinical labels, initialize figure directories, and define a dual-format save function. An image is exported as PDF and as a 300-dpi PNG. This setup does not estimate model parameters and does not alter the resampling design.


<!-- Notebook code cell 35; execution count 12 -->

````python
import numpy as np, pandas as pd, time, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("once")
SEED = 42
RES = Path("results"); RES.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     learning_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.stats import chi2

ds = load_breast_cancer(as_frame=True)
X = ds.data.copy()
y = (ds.target == 0).astype(int)              # 1 = malignant
worst = [c for c in X.columns if c.startswith("worst ")]

def savefig2(fig, name):
    # write the figure as PDF (for LaTeX) and PNG (for inline display)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    return FIG / f"{name}.png"

print("data:", X.shape, "| malignant:", int(y.sum()))
print("output dirs:", RES.resolve().name, "/", FIG.resolve().name)
````


**Output (stdout):**

````text
data: (569, 30) | malignant: 212
output dirs: results / figures
````

**Interpretation.** Dataset dimensions and positive-class counts are checked again. The resulting PNG is for inspection and the PDF is a scalable figure for manuscript typesetting; neither format changes the underlying data.

## 12. WDBC experiment 7A: paired McNemar comparisons

With the same OOF patients, $b$ counts cases correct only for model A and $c$ those correct only for model B. The continuity-corrected McNemar statistic is $(|b-c|-1)^2/(b+c)$ when discordances are present. Pairing requires one prediction per case, hence the shuffled five-fold OOF partition rather than the 15-fold repeated-CV output. These tests are exploratory and conditional on the fitted OOF models.


<!-- Notebook code cell 38; execution count 13 -->

````python
def oof_pred(est, Xd, cv=None):
    cv = cv or StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(Xd, y):
        p = Pipeline([("s", StandardScaler()), ("m", est)])
        p.fit(Xd.iloc[tr], y.iloc[tr])
        oof[te] = p.predict_proba(Xd.iloc[te])[:, 1]
        pred[te] = p.predict(Xd.iloc[te])
    return oof, pred

def mcnemar(y_true, p1, p2):
    b = int(((p1 == y_true) & (p2 != y_true)).sum())
    c = int(((p1 != y_true) & (p2 == y_true)).sum())
    n = b + c
    if n == 0:
        return dict(b=b, c=c, stat=0.0, p=1.0)
    stat = (abs(b - c) - 1) ** 2 / n
    return dict(b=b, c=c, stat=float(stat), p=float(1 - chi2.cdf(stat, 1)))

_, p_all   = oof_pred(LogisticRegression(max_iter=2000, random_state=SEED), X)
_, p_worst = oof_pred(LogisticRegression(max_iter=2000, random_state=SEED), X[worst])
_, p_svm   = oof_pred(SVC(kernel="rbf", probability=True, random_state=SEED), X)

tests = {
    "Worst(10) vs All(30), logistic regression": mcnemar(y.values, p_worst, p_all),
    "RBF-SVM vs logistic regression, all 30 features": mcnemar(y.values, p_svm, p_all),
}

print("=== McNEMAR (continuity-corrected, df = 1) ===")
for k, v in tests.items():
    print(f"{k}\n   b={v['b']}  c={v['c']}  chi2={v['stat']:.3f}  p={v['p']:.4f}")

pd.DataFrame([{"comparison": k, **v} for k, v in tests.items()]).to_csv(
    RES / "exp7_mcnemar.csv", index=False)
print("\nwritten:", RES / "exp7_mcnemar.csv")
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
=== McNEMAR (continuity-corrected, df = 1) ===
Worst(10) vs All(30), logistic regression
   b=6  c=3  chi2=0.444  p=0.5050
RBF-SVM vs logistic regression, all 30 features
   b=6  c=4  chi2=0.100  p=0.7518

written: results/exp7_mcnemar.csv
````

**Interpretation.** `The two continuity-corrected McNemar p-values are 0.5050 and 0.7518.` A nonsignificant result is not an equivalence or non-inferiority finding, particularly with few discordant pairs. Overlapping training sets also limit simple independence interpretations.

## 13. WDBC experiment 7B: paired sensitivity bootstrap

The target is $\Delta=\operatorname{Sens}_{\rm Worst}-\operatorname{Sens}_{\rm All}$. Each bootstrap replicate resamples the same patient indices for both models, preserving pairing. The 2.5th and 97.5th percentiles of 2,000 replicates give a conditional interval; the models and OOF splits are not refitted inside the bootstrap.


<!-- Notebook code cell 41; execution count 14 -->

````python
# --- paired bootstrap of the sensitivity difference (Worst - All) -----------
rng = np.random.default_rng(SEED)

def sens(pred, yt):
    tp = ((pred == 1) & (yt == 1)).sum()
    fn = ((pred == 0) & (yt == 1)).sum()
    return tp / (tp + fn) if (tp + fn) else np.nan

diffs = []
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    diffs.append(sens(p_worst[idx], y.values[idx]) - sens(p_all[idx], y.values[idx]))
diffs = np.array(diffs)

point = sens(p_worst, y.values) - sens(p_all, y.values)
lo, hi = np.percentile(diffs, [2.5, 97.5])

print(f"sensitivity(Worst) = {sens(p_worst, y.values):.4f}")
print(f"sensitivity(All)   = {sens(p_all,   y.values):.4f}")
print(f"difference         = {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"fraction of bootstrap replicates with difference > 0: {(diffs > 0).mean():.3f}")

pd.DataFrame([{"comparison": "Worst - All (LR)", "point": point,
               "ci_low": lo, "ci_high": hi,
               "frac_bootstrap_gt0": float((diffs > 0).mean())}]).to_csv(
    RES / "exp7_bootstrap_diff.csv", index=False)
print("\nwritten:", RES / "exp7_bootstrap_diff.csv")
````


**Output (stdout):**

````text
sensitivity(Worst) = 0.9623
sensitivity(All)   = 0.9434
difference         = +0.0189   95% CI [-0.0045, +0.0421]
fraction of bootstrap replicates with difference > 0: 0.925

written: results/exp7_bootstrap_diff.csv
````

**Interpretation.** `The paired sensitivity difference is +0.0189, with conditional 95% interval [-0.0045, +0.0421].` An interval containing zero is compatible with performance differences in either direction and is not proof of equivalence. Clinical non-inferiority would require a prespecified margin and a suitable validation design.

## 14. WDBC figures: blocks and model zoo

Sensitivity by feature block and by learner is displayed using the repeated-CV means; the error bars in the model-zoo plot are fold SDs. Separate PDF and PNG files are saved using the same underlying `exp1` and `exp2` CSVs. These are descriptive plots, not independent comparative tests.


<!-- Notebook code cell 44; execution count 15 -->

````python
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

fg = pd.read_csv(RES / "exp1_feature_blocks.csv")
piv = fg.pivot(index="model", columns="feature_block", values="sensitivity_mean")
piv = piv[["Error", "Mean", "Worst", "All"]]

fig, ax = plt.subplots(figsize=(6.0, 3.4))
piv.plot(kind="bar", ax=ax, width=.78, rot=0)
ax.set_ylabel("sensitivity"); ax.set_ylim(0.6, 1.0); ax.set_xlabel("")
ax.legend(title="feature block", fontsize=7, title_fontsize=7, ncol=4,
          loc="lower center")
img = savefig2(fig, "fig_blocks"); print("saved PDF+PNG:", img)

mz = pd.read_csv(RES / "exp2_model_zoo.csv").sort_values("sensitivity")
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.barh(mz.model, mz.sensitivity, xerr=mz.sensitivity_sd, capsize=2.5,
        color="#4169a8", alpha=.85, ecolor="#333")
ax.set_xlim(0.82, 1.0)
ax.set_xlabel("sensitivity (mean $\\pm$ sd over 15 folds)")
for i, s in enumerate(mz.sensitivity):
    ax.text(0.825, i, f"{s:.3f}", va="center", fontsize=7)
img = savefig2(fig, "fig_models"); print("saved PDF+PNG:", img)

# compact journal variant (used in the manuscript)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.barh(mz.model, mz.sensitivity, xerr=mz.sensitivity_sd, capsize=2, lw=.7,
        color="#2c5f8a", alpha=.9, ecolor="#222",
        error_kw={"elinewidth": .8})
ax.set_xlim(.85, 1.0)
ax.set_xlabel("sensitivity (mean $\\pm$ sd, 15 folds)")
ax.tick_params(axis="y", length=0)
img = savefig2(fig, "fig_models2"); print("saved PDF+PNG:", img)
````


**Output (stdout):**

````text
saved PDF+PNG: figures/fig_blocks.png
````


**Output (stdout):**

````text
saved PDF+PNG: figures/fig_models.png
````


**Output (stdout):**

````text
saved PDF+PNG: figures/fig_models2.png
````

**Interpretation.** Visual differences and overlapping error bars do not establish statistical significance. The compact and expanded model-zoo variants display the same tabulated estimates.

## 15. WDBC figures: threshold trade-offs

For each $\tau$ on the prespecified grid, sensitivity, specificity and precision follow from thresholding the same OOF probabilities; FN and FP are plotted as counts. Sweeping $\tau$ illustrates the trade-off but does not provide an unbiased assessment of a threshold selected on those same scores. The nested table is the relevant internal estimate.


<!-- Notebook code cell 47; execution count 16 -->

````python
th = pd.read_csv(RES / "exp4a_threshold_curve.csv")

fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(th.threshold, th.sensitivity, "-o", ms=2.5, label="Sensitivity (malignant)")
ax.plot(th.threshold, th.precision,   "-s", ms=2.5, label="Precision")
ax.plot(th.threshold, th.specificity, "-^", ms=2.5, label="Specificity (benign)")
ax.axhline(0.95, ls="--", c="gray", lw=1, label="0.95 operating point")
ax.set_xlabel("decision threshold $\\tau$"); ax.set_ylabel("metric value")
ax.set_xlim(0, 0.95); ax.set_ylim(0.4, 1.02)
ax.legend(fontsize=7, loc="lower left")
img = savefig2(fig, "fig_threshold"); print("saved PDF+PNG:", img)

# two-panel journal variant
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
axes[0].plot(th.threshold, th.sensitivity, "-", lw=1.4, color="#1f4e79", label="sensitivity")
axes[0].plot(th.threshold, th.precision,   "-", lw=1.1, color="#c00000", label="precision")
axes[0].plot(th.threshold, th.specificity, "-", lw=1.1, color="#2e7d32", label="specificity")
axes[0].axhline(.95, ls=":", lw=.9, c="k", label="0.95")
axes[0].axvline(.45, ls="--", lw=.8, c="gray")
axes[0].set_xlabel("threshold $\\tau$"); axes[0].set_ylabel("metric")
axes[0].set_xlim(0, .95); axes[0].set_ylim(.4, 1.02)
axes[0].legend(loc="lower left", ncol=2)
axes[1].plot(th.threshold, th.fn, "-", lw=1.3, color="#7b1fa2", label="false negatives")
axes[1].plot(th.threshold, th.fp, "-", lw=1.3, color="#ef6c00", label="false positives")
axes[1].set_xlabel("threshold $\\tau$"); axes[1].set_ylabel("count")
axes[1].set_xlim(0, .95); axes[1].legend()
img = savefig2(fig, "fig_threshold2"); print("saved PDF+PNG:", img)
````


**Output (stdout):**

````text
saved PDF+PNG: figures/fig_threshold.png
````


**Output (stdout):**

````text
saved PDF+PNG: figures/fig_threshold2.png
````

**Interpretation.** A lower threshold generally favors sensitivity at the expense of more false positives. The marked guide is illustrative; clinical operating points require independently justified preferences and validation.

## 16. WDBC figure: reliability diagram

For each nonempty probability bin $b$, plot $(\overline p_b,\overline y_b)$; perfect calibration lies on $\overline p_b=\overline y_b$. The curves use OOF predictions for both probability variants. Binned points are descriptive and may be noisy when bin counts are small.


<!-- Notebook code cell 50; execution count 17 -->

````python
rel = pd.read_csv(RES / "exp5b_reliability.csv")

fig, ax = plt.subplots(figsize=(3.4, 3.4))
for v, sty in [("uncalibrated", "o-"), ("Platt-calibrated", "s--")]:
    d = rel[rel.variant == v]
    ax.plot(d.mean_predicted, d.observed_rate, sty, ms=3.5, lw=1, label=v)
ax.plot([0, 1], [0, 1], "k:", lw=.8, label="perfect")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed malignant fraction")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7)
img = savefig2(fig, "fig_reliability"); print("saved PDF+PNG:", img)
print(rel.groupby("variant").size())
````


**Output (stdout):**

````text
saved PDF+PNG: figures/fig_reliability.png
variant
Platt-calibrated    10
uncalibrated        10
dtype: int64
````

**Interpretation.** Distance from the diagonal and the Brier/log-loss values should be considered together; the bin count is not a confidence interval. Prevalence shifts may invalidate apparent calibration in another setting.

## 17. WDBC figure: fidelity-kernel matrix

A reproducibly sampled collection of 14 angle vectors is mapped to statevectors. The Gram matrix $K_{ij}=|\langle\psi_i|\psi_j\rangle|^2$ must be symmetric with a unit diagonal and entries in $[0,1]$. This is a computed kernel matrix rather than a conceptual illustration. Heatmap cells and the colorbar are vector objects in the exported PDF.


<!-- Notebook code cell 53; execution count 18 -->

````python
RY = lambda t: np.array([[np.cos(t/2), -np.sin(t/2)],
                         [np.sin(t/2),  np.cos(t/2)]])

def cnot(state, control, target, n=3):
    out = np.zeros_like(state)
    for idx, amp in enumerate(state):
        bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
        if bits[control]:
            bits[target] ^= 1
        j = 0
        for b in bits:
            j = (j << 1) | b
        out[j] += amp
    return out

def qstate(x, n=3):
    psi = np.array([1.0])
    for xi in x:
        psi = np.kron(psi, RY(xi) @ np.array([1.0, 0.0]))
    for c, t in [(0, 1), (1, 2)]:
        psi = cnot(psi, c, t, n=n)
    return psi

rng = np.random.default_rng(SEED)
ang = np.linspace(0.15, np.pi - 0.15, 14)
pts = np.array(np.meshgrid(ang, ang, ang)).reshape(3, -1).T
sub = pts[rng.choice(len(pts), 14, replace=False)]
A = np.array([qstate(p) for p in sub])
K = np.abs(A.conj() @ A.T) ** 2

print("state norm        :", np.linalg.norm(A[0]))
print("diagonal == 1     :", np.allclose(np.diag(K), 1))
print("symmetric         :", np.allclose(K, K.T))
print("range within [0,1]:", bool((K >= 0).all() and (K <= 1 + 1e-9).all()))

fig, ax = plt.subplots(figsize=(3.6, 3.2))
im = ax.pcolormesh(np.arange(K.shape[1]+1)-0.5, np.arange(K.shape[0]+1)-0.5, K, cmap="viridis", vmin=0, vmax=1, shading="flat", rasterized=False)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title("fidelity kernel $k(x,x')=|\\langle\\psi(x)|\\psi(x')\\rangle|^2$", fontsize=7)
ax.set_xlabel("sample index"); ax.set_ylabel("sample index"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=.85)
cb.solids.set_rasterized(False)
img = savefig2(fig, "fig_kernel"); print("saved PDF+PNG:", img)
````


**Output (stdout):**

````text
state norm        : 1.0
diagonal == 1     : True
symmetric         : True
range within [0,1]: True
````


**Output (stdout):**

````text
saved PDF+PNG: figures/fig_kernel.png
````

**Interpretation.** Unit diagonal and symmetry are necessary checks for this fidelity construction; they do not imply that the kernel is more useful than a classical kernel. The constant entangler's cancellation is independently tested below.

## 18. WDBC experiment 8: learning curve

Training and held-out sensitivity are plotted against six training-set sizes with a five-fold stratified split. The central curves are mean recall and bands are the population SD across folds as returned by the reference calculation. Because the same dataset and overlapping training sets are reused, bands are descriptive rather than formal confidence intervals.

**Sampling limitation.** The reference uses `shuffle=False` when taking training-size prefixes. The 68-row prefixes are approximately 63–71% malignant versus approximately 37% in the complete training folds; the shape is therefore not a reliable learning trend. An executable prefix audit appears in experiment 21.


<!-- Notebook code cell 56; execution count 19 -->

````python
sizes, tr_s, te_s = learning_curve(
    Pipeline([("s", StandardScaler()),
              ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
    X[worst], y,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    train_sizes=np.linspace(0.15, 1.0, 6),
    scoring="recall", n_jobs=-1)

lc = pd.DataFrame({"n_train": sizes,
                   "train_mean": tr_s.mean(1), "train_sd": tr_s.std(1),
                   "val_mean": te_s.mean(1),   "val_sd": te_s.std(1)})
lc.to_csv(RES / "exp8_learning_curve.csv", index=False)
print(lc.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.fill_between(sizes, tr_s.mean(1) - tr_s.std(1), tr_s.mean(1) + tr_s.std(1),
                alpha=.18, color="#2b7a4b")
ax.plot(sizes, tr_s.mean(1), "-o", ms=3, color="#2b7a4b", label="training")
ax.fill_between(sizes, te_s.mean(1) - te_s.std(1), te_s.mean(1) + te_s.std(1),
                alpha=.18, color="#a8412f")
ax.plot(sizes, te_s.mean(1), "-s", ms=3, color="#a8412f", label="validation")
ax.set_xlabel("training-set size"); ax.set_ylabel("sensitivity")
ax.set_ylim(0.85, 1.01); ax.legend(fontsize=7)
img = savefig2(fig, "fig_learning"); print("saved PDF+PNG:", img)
````


**Output (stdout):**

````text
 n_train  train_mean  train_sd  val_mean  val_sd
      68      0.9826    0.0087    0.9859  0.0187
     145      0.9730    0.0098    0.9671  0.0349
     222      0.9717    0.0070    0.9625  0.0431
     300      0.9684    0.0077    0.9578  0.0518
     377      0.9668    0.0085    0.9578  0.0518
     455      0.9658    0.0102    0.9625  0.0431
````


**Output (stdout):**

````text
saved PDF+PNG: figures/fig_learning.png
````

**Interpretation.** `Validation sensitivity is 0.9859 at 68 training rows and 0.9625 at 455 training rows.` Learning-curve shapes depend on sampling, class composition and model choice; they do not by themselves predict external cohort performance.

## 19. Numerical regression checks for WDBC

The 13 WDBC CSV outputs are compared with committed numerical reference fixtures. Categorical fields, shapes and numeric columns are tested; measured runtime columns are intentionally excluded. The tolerance is $10^{-8}$ relative and $10^{-10}$ absolute, allowing negligible floating-point variation without treating wall time as a reproducible scientific quantity.


<!-- Notebook code cell 59; execution count 20 -->

````python
# Numeric and categorical regression checks exclude wall-clock timings by design.
reference_dir = Path('data/reference_wdbc')
reference_audit_rows = []
for ref_path in sorted(reference_dir.glob('exp*.csv')):
    actual_path = RESULTS / ref_path.name
    assert actual_path.is_file(), f'Missing WDBC experiment: {actual_path}'
    expected = pd.read_csv(ref_path)
    measured = pd.read_csv(actual_path)
    assert set(expected.columns).issubset(measured.columns)
    measured = measured[expected.columns]
    assert expected.shape == measured.shape
    numeric_errors = []
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[col]):
            a, b = expected[col].to_numpy(dtype=float), measured[col].to_numpy(dtype=float)
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True,
                                       err_msg=f'{ref_path.name}: {col}')
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                numeric_errors.append(float(np.max(np.abs(a[finite] - b[finite]))))
        else:
            assert expected[col].fillna('<not-applicable>').astype(str).tolist() == measured[col].fillna('<not-applicable>').astype(str).tolist(), (ref_path.name, col)
    reference_audit_rows.append({'experiment_csv': ref_path.name, 'rows': len(expected),
                                 'columns_compared': len(expected.columns),
                                 'max_abs_numeric_difference': max(numeric_errors, default=0.0),
                                 'status': 'PASS'})
reference_audit = pd.DataFrame(reference_audit_rows)
assert len(reference_audit) == 13 and (reference_audit.status == 'PASS').all()
reference_audit.to_csv(RESULTS / 'wdbc_numeric_reproducibility.csv', index=False)
print('All WDBC reference fixture checks passed (timings excluded; floating tolerance 1e-8 relative).')
display(reference_audit)
````


**Output (stdout):**

````text
All WDBC reference fixture checks passed (timings excluded; floating tolerance 1e-8 relative).
````


**Displayed output:**

````text
                   experiment_csv  rows  columns_compared  max_abs_numeric_difference status
0         exp1_feature_blocks.csv    12                12                      0.0000   PASS
1          exp2_architectures.csv    11                 4                      0.0000   PASS
2              exp2_model_zoo.csv    11                13                      0.0000   PASS
3      exp3_feature_selection.csv     4                 7                      0.0000   PASS
4       exp4a_threshold_curve.csv    91                10                      0.0000   PASS
5      exp4b_nested_threshold.csv     5                15                      0.0000   PASS
6           exp5a_calibration.csv     2                 4                      0.0000   PASS
7           exp5b_reliability.csv    20                 5                      0.0000   PASS
8          exp5c_bootstrap_ci.csv     3                 5                      0.0000   PASS
9   exp6_quantum_vs_classical.csv     3                12                      0.0000   PASS
10        exp7_bootstrap_diff.csv     1                 5                      0.0000   PASS
11               exp7_mcnemar.csv     2                 5                      0.0000   PASS
12        exp8_learning_curve.csv     6                 5                      0.0000   PASS
````

**Interpretation.** `All 13 numeric fixture tables pass; the maximum absolute compared numeric deviation is 0, with timing columns excluded.` Passing these checks supports numerical regression for the specified software and data, not universal bit-for-bit reproducibility across platforms.

## 20. Four real datasets, label polarity, blocks and SHA-256

Raw UCI bytes are checksum-verified before use; cleaned CSV exports are fingerprinted in `data/SHA256SUMS`. Raw WDBC and WPBC IDs are joined solely to audit cohort overlap, and all 30 shared-feature values are compared. WPBC removes identifier and follow-up time, retains 32 predictors and all 198 records, and leaves four lymph-node values missing until median imputation inside each training pipeline. Coimbra maps original classes $1\mapsto0$ and $2\mapsto1$. Parkinson's 22 acoustic features form disjoint 5/11/6 blocks; identifiers are metadata and never predictors. WDBC class and block assertions remain in force.


<!-- Notebook code cell 62; execution count 21 -->

````python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import hashlib

DATA = Path('data')
RAW = DATA / 'raw'
DATA.mkdir(exist_ok=True)
# These hashes identify the exact UCI source bytes independently of the cleaned copies.
raw_sha256 = {
    'wdbc.data': 'd606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a',
    'wpbc.data': 'f2bd5ccdb6086fa832370dc043319907430cf6bfcfd3b96a3961a357be345c8e',
    'coimbra.csv': '68c56c7e7a38dfff1046a8b35e5b3afc17229eff7fb9d8dca7053442b1c37e7e',
    'parkinsons.data': 'a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5',
    'wpbc.names': 'c1ef99d8ed4a6bc212bab59b6df3fab870f6b3ffd8a24d7ba93e53cae14f3461',
    'parkinsons.names': 'ac4f44495100d468feb566d1272b2070000507ec6a02ea64ff50b847784d7455',
}
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

for name, expected in raw_sha256.items():
    assert sha256_file(RAW / name) == expected, f'Invalid source checksum: {name}'

# The reference WDBC matrix is taken directly from scikit-learn, with positive = malignant.
wdbc_X = X_raw.copy().reset_index(drop=True)
wdbc_y = (load_breast_cancer(as_frame=True).target == 0).astype(int).reset_index(drop=True)
assert wdbc_X.shape == (569, 30) and wdbc_y.sum() == 212 and (1 - wdbc_y).sum() == 357
wdbc_groups = {key: list(value) for key, value in GROUPS.items()}

# WPBC: the source fields are ID, outcome, time, 30 nuclear features, size, nodes.
wpbc_names = (['id', 'outcome', 'time'] + ['mean ' + s for s in STEMS]
              + [s + ' error' for s in STEMS] + ['worst ' + s for s in STEMS]
              + ['tumor_size', 'lymph_node_status'])
wpbc = pd.read_csv(RAW / 'wpbc.data', header=None, names=wpbc_names, na_values=['?'])
assert wpbc.shape == (198, 35)
assert wpbc['outcome'].value_counts().to_dict() == {'N': 151, 'R': 47}
assert wpbc['lymph_node_status'].isna().sum() == 4
wpbc_y = wpbc['outcome'].map({'N': 0, 'R': 1}).astype(int)
wpbc_X = wpbc.drop(columns=['id', 'outcome', 'time']).astype(float)
assert wpbc_X.shape == (198, 32) and wpbc_X.isna().sum().sum() == 4
wpbc_groups = {'Mean': [c for c in wpbc_X if c.startswith('mean ')],
               'Error': [c for c in wpbc_X if c.endswith(' error')],
               'Worst': [c for c in wpbc_X if c.startswith('worst ')],
               'All': list(wpbc_X.columns)}
assert all(len(wpbc_groups[k]) == 10 for k in ('Mean', 'Error', 'Worst'))
assert set(wpbc_X.columns) - set().union(*(set(wpbc_groups[k]) for k in ('Mean', 'Error', 'Worst'))) == {'tumor_size', 'lymph_node_status'}

# Cross-dataset identity audit: identifiers are never used by a predictive model.
wdbc_source = pd.read_csv(RAW / 'wdbc.data', header=None)
wdbc_source.columns = ['id', 'diagnosis'] + ['w_' + str(i) for i in range(30)]
assert np.array_equal(wdbc_source.iloc[:, 2:].to_numpy(dtype=float), wdbc_X.to_numpy(dtype=float))
assert np.array_equal((wdbc_source.diagnosis == 'M').astype(int).to_numpy(), wdbc_y.to_numpy())
assert not wdbc_source.id.duplicated().any() and not wpbc.id.duplicated().any()
wpbc_audit = wpbc[['id'] + list(wpbc_X.columns[:30])].copy()
wpbc_audit.columns = ['id'] + ['p_' + str(i) for i in range(30)]
matched = wdbc_source.merge(wpbc_audit, on='id', how='inner', validate='one_to_one')
exact_vectors = np.all(matched[[f'w_{i}' for i in range(30)]].to_numpy() ==
                       matched[[f'p_{i}' for i in range(30)]].to_numpy(), axis=1)
assert len(matched) == 139 and int(exact_vectors.sum()) == 126
assert (matched.diagnosis == 'M').all()
overlap_audit = pd.DataFrame([{'dataset_a': 'WDBC', 'dataset_b': 'WPBC',
                               'shared_identifier_values': len(matched),
                               'shared_identifiers_identical_30_features': int(exact_vectors.sum()),
                               'shared_identifiers_wdbc_malignant': int((matched.diagnosis == 'M').sum()),
                               'interpretation': 'cohorts_overlap_not_independent'}])
overlap_audit.to_csv(RESULTS / 'dataset_overlap_audit.csv', index=False)
print('Cross-dataset identifier and nuclear-feature overlap:'); display(overlap_audit)

# Coimbra: source classes 1=healthy, 2=patient; 1 is the clinical positive here.
coimbra = pd.read_csv(RAW / 'coimbra.csv')
coimbra_X = coimbra.drop(columns='Classification').astype(float)
coimbra_y = coimbra['Classification'].map({1: 0, 2: 1}).astype(int)
assert coimbra_X.shape == (116, 9) and coimbra_y.sum() == 64 and (1-coimbra_y).sum() == 52
assert list(coimbra_X.columns) == ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']

# Oxford Parkinsons: each name identifies a recording, not a distinct person.
pd_source = pd.read_csv(RAW / 'parkinsons.data')
subject_stem = pd_source['name'].str.rsplit('_', n=1).str[0]
pd_y = pd_source['status'].astype(int)
pd_X = pd_source.drop(columns=['name', 'status']).astype(float)
assert pd_X.shape == (195, 22) and pd_y.sum() == 147 and (1-pd_y).sum() == 48
assert (pd.DataFrame({'group': subject_stem, 'y': pd_y}).groupby('group').y.nunique() == 1).all()
pd_groups = {
    'Vocal_Freq_Tone': ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'NHR', 'HNR'],
    'Perturbation_Error': ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                           'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA'],
    'Nonlinear_Extremes': ['RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'],
    'All': list(pd_X.columns),
}
assert [len(pd_groups[k]) for k in ('Vocal_Freq_Tone', 'Perturbation_Error', 'Nonlinear_Extremes')] == [5, 11, 6]
assert set().union(*(set(pd_groups[k]) for k in ('Vocal_Freq_Tone', 'Perturbation_Error', 'Nonlinear_Extremes'))) == set(pd_X.columns)
assert sum(len(pd_groups[k]) for k in ('Vocal_Freq_Tone', 'Perturbation_Error', 'Nonlinear_Extremes')) == 22

DATASETS = {
    'WDBC': {'X': wdbc_X, 'y': wdbc_y, 'blocks': wdbc_groups, 'groups': None, 'unit': 'patient'},
    'WPBC': {'X': wpbc_X, 'y': wpbc_y, 'blocks': wpbc_groups, 'groups': None, 'unit': 'patient'},
    'Coimbra': {'X': coimbra_X, 'y': coimbra_y, 'blocks': {'All': list(coimbra_X.columns)}, 'groups': None, 'unit': 'patient'},
    'Parkinsons': {'X': pd_X, 'y': pd_y, 'blocks': pd_groups, 'groups': subject_stem, 'unit': 'recording'},
}

# The cleaned export retains missing WPBC nodes; median values are learned only on training folds.
manifest_rows = []
for name, dset in DATASETS.items():
    out = dset['X'].copy()
    if dset['groups'] is not None:
        out.insert(0, 'subject_stem', dset['groups'].to_numpy())
    out['target'] = dset['y'].to_numpy()
    path = DATA / f'{name.lower()}_clean.csv'
    out.to_csv(path, index=False, float_format='%.17g')
    manifest_rows.append({'dataset': name, 'file': str(path), 'n_rows': len(out), 'n_features': dset['X'].shape[1],
                          'positive': int(dset['y'].sum()), 'negative': int((1-dset['y']).sum()),
                          'missing_features': int(dset['X'].isna().sum().sum()),
                          'sha256': sha256_file(path), 'sampling_unit': dset['unit']})
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(DATA / 'datasets_manifest.csv', index=False)
checksum_paths = sorted(list(RAW.glob('*')) + [DATA / f'{name.lower()}_clean.csv' for name in DATASETS])
(DATA / 'SHA256SUMS').write_text(''.join(f'{sha256_file(p)}  {p.as_posix()}\n' for p in checksum_paths), encoding='utf-8')
print('Dataset dimensions, class polarity, missingness and SHA-256:')
display(manifest)
print(f"Parkinson recording-name stems: {subject_stem.nunique()} (source description says 31 people; stems require independent verification).")
print('Feature blocks:', {name: {k: len(v) for k, v in d['blocks'].items()} for name, d in DATASETS.items()})
````


**Output (stdout):**

````text
Cross-dataset identifier and nuclear-feature overlap:
````


**Displayed output:**

````text
  dataset_a dataset_b  shared_identifier_values  shared_identifiers_identical_30_features  shared_identifiers_wdbc_malignant                   interpretation
0      WDBC      WPBC                       139                                       126                                139  cohorts_overlap_not_independent
````


**Output (stdout):**

````text
Dataset dimensions, class polarity, missingness and SHA-256:
````


**Displayed output:**

````text
      dataset                       file  n_rows  n_features  positive  negative  missing_features                                                            sha256 sampling_unit
0        WDBC        data/wdbc_clean.csv     569          30       212       357                 0  e6617bbe083649eeca027c4dc52515a98dda67959d583d7c4ad589ff7a8972f7       patient
1        WPBC        data/wpbc_clean.csv     198          32        47       151                 4  9a34e38aea464062144a7191cf81b9cf451d8dea2bb805a171a55fae38efaf48       patient
2     Coimbra     data/coimbra_clean.csv     116           9        64        52                 0  eb7dfb4781460863a37ab6875a7bacd33176292964fc55657a2706af2e75e114       patient
3  Parkinsons  data/parkinsons_clean.csv     195          22       147        48                 0  d97e0a55a0fd1885613a79799f8ca1b7c6db3aef33b1c653f1d50b3fc3dc4248     recording
````


**Output (stdout):**

````text
Parkinson recording-name stems: 32 (source description says 31 people; stems require independent verification).
Feature blocks: {'WDBC': {'Mean': 10, 'Error': 10, 'Worst': 10, 'All': 30}, 'WPBC': {'Mean': 10, 'Error': 10, 'Worst': 10, 'All': 32}, 'Coimbra': {'All': 9}, 'Parkinsons': {'Vocal_Freq_Tone': 5, 'Perturbation_Error': 11, 'Nonlinear_Extremes': 6, 'All': 22}}
````

**Interpretation.** `The four datasets contain 569, 198, 116 and 195 rows, respectively. The cohort audit found 139 shared WDBC/WPBC identifiers, including 126 exact 30-feature matches.` WDBC and WPBC share source identifiers and matching measurements, so their cohorts are not independent. The 195 Parkinson rows are recordings, not independent patients; 32 observed identifier stems conflict with the source's description of 31 people and require source clarification. WPBC outcome is not a horizon-specific survival endpoint; four missing covariates are an intentional, documented exception in the clean data, not missing result values.

## 21. Experiment 9: multi-dataset structural blocks and selection stability

Each structural block and two ten-feature selectors are evaluated with logistic regression on the same 15 repeated stratified folds. Median imputation, supervised `SelectKBest`, scaling and fitting occur within each training fold. Mean pairwise Jaccard stability is $\binom{15}{2}^{-1}\sum_{i<j}|S_i\cap S_j|/|S_i\cup S_j|$; a fixed block has stability one by construction, not by empirical discovery. Coimbra has no analogous 10-feature structural decomposition and is not included in this block experiment.


<!-- Notebook code cell 65; execution count 22 -->

````python
from itertools import combinations
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Repeated folds are for overall performance summaries; each test row occurs once per repeat.
CV_REPEATED = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
CV_OOF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def lr_pipeline(selector=None):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if selector is not None:
        steps.append(('selector', selector))
    steps.extend([('scaler', StandardScaler()),
                  ('model', LogisticRegression(penalty='l2', max_iter=2000, random_state=SEED))])
    return Pipeline(steps)

def summarize_fold_metrics(fold_frame):
    keys = ['sensitivity', 'specificity', 'precision', 'f1', 'balanced_accuracy',
            'mcc', 'roc_auc', 'pr_auc', 'fn', 'fp']
    return {f'{key}_{stat}': float(getattr(fold_frame[key], stat)())
            for key in keys for stat in ['mean', 'std']}

def average_pairwise_jaccard(sets):
    vals = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    return float(np.mean(vals)), float(np.min(vals)), len(vals)

block_rows, block_fold_rows, pick_rows = [], [], []
for dataset in ['WDBC', 'WPBC', 'Parkinsons']:
    dset = DATASETS[dataset]; Xd, yd = dset['X'], dset['y']
    configurations = [(name, cols, None) for name, cols in dset['blocks'].items()]
    configurations += [('SelectKBest_f_classif_10', list(Xd.columns), SelectKBest(f_classif, k=10)),
                       ('SelectKBest_mutual_info_10', list(Xd.columns), SelectKBest(_mi_score, k=10))]
    for label, cols, selector in configurations:
        Xsub = Xd.loc[:, cols]
        scores, selected_sets = [], []
        for fold, (tr, te) in enumerate(CV_REPEATED.split(Xsub, yd), start=1):
            pipe = lr_pipeline(clone(selector) if selector is not None else None)
            pipe.fit(Xsub.iloc[tr], yd.iloc[tr])
            pred = pipe.predict(Xsub.iloc[te]); score = pipe.predict_proba(Xsub.iloc[te])[:, 1]
            met = metrics_from_preds(yd.iloc[te].to_numpy(), pred, score)
            scores.append(met)
            block_fold_rows.append({'dataset': dataset, 'configuration': label, 'fold': fold, **met})
            if selector is None:
                selected = set(cols)
            else:
                selected = set(np.asarray(cols)[pipe.named_steps['selector'].get_support()])
            selected_sets.append(selected)
            for feature in sorted(selected):
                pick_rows.append({'dataset': dataset, 'configuration': label, 'fold': fold, 'feature': feature})
        fm = pd.DataFrame(scores)
        stability, minimum, pairs = average_pairwise_jaccard(selected_sets)
        block_rows.append({'dataset': dataset, 'configuration': label,
                           'n_selected': len(selected_sets[0]), 'n_folds': len(fm),
                           'jaccard_pairwise_mean': stability, 'jaccard_pairwise_min': minimum,
                           'jaccard_n_pairs': pairs, **summarize_fold_metrics(fm)})
        print(f"{dataset:10s} | {label:28s} | sensitivity {fm.sensitivity.mean():.4f} ± {fm.sensitivity.std(ddof=1):.4f} | Jaccard {stability:.4f}")

multi_blocks = pd.DataFrame(block_rows)
multi_blocks.to_csv(RESULTS / 'exp9_multi_dataset_blocks.csv', index=False)
pd.DataFrame(block_fold_rows).to_csv(RESULTS / 'exp9b_block_fold_metrics.csv', index=False)
pd.DataFrame(pick_rows).to_csv(RESULTS / 'exp9c_selected_features.csv', index=False)
assert (multi_blocks.n_folds == 15).all() and (multi_blocks.jaccard_n_pairs == 105).all()
display(multi_blocks)
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Mean                         | sensitivity 0.8931 ± 0.0538 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Error                        | sensitivity 0.7547 ± 0.0593 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Worst                        | sensitivity 0.9575 ± 0.0379 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | All                          | sensitivity 0.9512 ± 0.0371 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | SelectKBest_f_classif_10     | sensitivity 0.9261 ± 0.0550 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | SelectKBest_mutual_info_10   | sensitivity 0.9244 ± 0.0438 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Mean                         | sensitivity 0.0659 ± 0.0815 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Error                        | sensitivity 0.0578 ± 0.0700 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Worst                        | sensitivity 0.1007 ± 0.1051 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | All                          | sensitivity 0.1919 ± 0.1561 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | SelectKBest_f_classif_10     | sensitivity 0.1059 ± 0.0886 | Jaccard 0.7416
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | SelectKBest_mutual_info_10   | sensitivity 0.0844 ± 0.0912 | Jaccard 0.3712
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Vocal_Freq_Tone              | sensitivity 0.9273 ± 0.0564 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Perturbation_Error           | sensitivity 0.9366 ± 0.0423 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Nonlinear_Extremes           | sensitivity 0.9276 ± 0.0354 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | All                          | sensitivity 0.9320 ± 0.0385 | Jaccard 1.0000
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | SelectKBest_f_classif_10     | sensitivity 0.9276 ± 0.0399 | Jaccard 0.6338
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | SelectKBest_mutual_info_10   | sensitivity 0.9276 ± 0.0247 | Jaccard 0.6992
````


**Displayed output:**

````text
       dataset               configuration  n_selected  n_folds  jaccard_pairwise_mean  jaccard_pairwise_min  jaccard_n_pairs  sensitivity_mean  sensitivity_std  specificity_mean  specificity_std  precision_mean  precision_std  f1_mean  f1_std  balanced_accuracy_mean  balanced_accuracy_std  mcc_mean  mcc_std  roc_auc_mean  roc_auc_std  pr_auc_mean  pr_auc_std  fn_mean  fn_std  fp_mean  fp_std
0         WDBC                        Mean          10       15                 1.0000                1.0000              105            0.8931           0.0538            0.9664           0.0138          0.9417         0.0207   0.9154  0.0225                  0.9298                 0.0215    0.8701   0.0289        0.9858       0.0050       0.9804      0.0062   4.5333  2.2949   2.4000  0.9856
1         WDBC                       Error          10       15                 1.0000                1.0000              105            0.7547           0.0593            0.9421           0.0204          0.8868         0.0338   0.8139  0.0374                  0.8484                 0.0281    0.7246   0.0476        0.9426       0.0129       0.9235      0.0157  10.4000  2.5014   4.1333  1.4573
2         WDBC                       Worst          10       15                 1.0000                1.0000              105            0.9575           0.0379            0.9888           0.0121          0.9813         0.0198   0.9687  0.0182                  0.9732                 0.0172    0.9516   0.0273        0.9925       0.0077       0.9925      0.0071   1.8000  1.6125   0.8000  0.8619
3         WDBC                         All          30       15                 1.0000                1.0000              105            0.9512           0.0371            0.9907           0.0087          0.9841         0.0145   0.9669  0.0188                  0.9709                 0.0176    0.9490   0.0275        0.9946       0.0049       0.9933      0.0054   2.0667  1.5796   0.6667  0.6172
4         WDBC    SelectKBest_f_classif_10          10       15                 1.0000                1.0000              105            0.9261           0.0550            0.9664           0.0165          0.9432         0.0266   0.9335  0.0299                  0.9463                 0.0269    0.8965   0.0449        0.9886       0.0085       0.9858      0.0094   3.1333  2.3563   2.4000  1.1832
5         WDBC  SelectKBest_mutual_info_10          10       15                 1.0000                1.0000              105            0.9244           0.0438            0.9730           0.0213          0.9541         0.0346   0.9382  0.0278                  0.9487                 0.0235    0.9040   0.0431        0.9909       0.0061       0.9875      0.0078   3.2000  1.8593   1.9333  1.5337
6         WPBC                        Mean          10       15                 1.0000                1.0000              105            0.0659           0.0815            0.9889           0.0349          0.4444         0.4987   0.1119  0.1332                  0.5274                 0.0457    0.1336   0.1825        0.5893       0.1313       0.4284      0.1314   8.8000  1.0823   0.3333  1.0465
7         WPBC                       Error          10       15                 1.0000                1.0000              105            0.0578           0.0700            0.9690           0.0320          0.2822         0.3612   0.0908  0.1039                  0.5134                 0.0315    0.0525   0.1357        0.5532       0.0869       0.3500      0.0912   8.8667  0.9155   0.9333  0.9612
8         WPBC                       Worst          10       15                 1.0000                1.0000              105            0.1007           0.1051            0.9622           0.0353          0.3944         0.3759   0.1540  0.1514                  0.5315                 0.0564    0.1092   0.1871        0.5797       0.1259       0.3789      0.1094   8.4667  1.1872   1.1333  1.0601
9         WPBC                         All          32       15                 1.0000                1.0000              105            0.1919           0.1561            0.9115           0.0484          0.3853         0.2612   0.2474  0.1820                  0.5517                 0.0856    0.1314   0.2155        0.6577       0.1141       0.4177      0.1366   7.6000  1.5492   2.6667  1.4475
10        WPBC    SelectKBest_f_classif_10          10       15                 0.7416                0.4286              105            0.1059           0.0886            0.9535           0.0329          0.3833         0.3099   0.1618  0.1302                  0.5297                 0.0486    0.0999   0.1630        0.6387       0.1111       0.4170      0.1097   8.4000  0.9103   1.4000  0.9856
11        WPBC  SelectKBest_mutual_info_10          10       15                 0.3712                0.1111              105            0.0844           0.0912            0.9447           0.0467          0.3100         0.3622   0.1276  0.1358                  0.5146                 0.0537    0.0520   0.1864        0.5698       0.0983       0.3652      0.0879   8.6000  0.9103   1.6667  1.3973
12  Parkinsons             Vocal_Freq_Tone           5       15                 1.0000                1.0000              105            0.9273           0.0564            0.4711           0.1997          0.8438         0.0565   0.8822  0.0444                  0.6992                 0.1036    0.4569   0.1934        0.7818       0.0934       0.9157      0.0471   2.1333  1.6417   5.1333  2.0656
13  Parkinsons          Perturbation_Error          11       15                 1.0000                1.0000              105            0.9366           0.0423            0.3941           0.1759          0.8263         0.0495   0.8770  0.0340                  0.6654                 0.0900    0.4043   0.1832        0.8279       0.0571       0.9417      0.0233   1.8667  1.2459   5.8667  1.8465
14  Parkinsons          Nonlinear_Extremes           6       15                 1.0000                1.0000              105            0.9276           0.0354            0.6244           0.1792          0.8843         0.0543   0.9041  0.0285                  0.7760                 0.0857    0.5853   0.1389        0.8970       0.0511       0.9647      0.0184   2.1333  1.0601   3.6667  1.8387
15  Parkinsons                         All          22       15                 1.0000                1.0000              105            0.9320           0.0385            0.6119           0.2101          0.8818         0.0623   0.9046  0.0353                  0.7719                 0.1034    0.5818   0.1739        0.8955       0.0556       0.9641      0.0201   2.0000  1.1339   3.8000  2.1448
16  Parkinsons    SelectKBest_f_classif_10          10       15                 0.6338                0.4286              105            0.9276           0.0399            0.6037           0.1965          0.8795         0.0582   0.9011  0.0283                  0.7656                 0.0907    0.5700   0.1419        0.8955       0.0591       0.9650      0.0206   2.1333  1.1872   3.8667  1.9952
17  Parkinsons  SelectKBest_mutual_info_10          10       15                 0.6992                0.5385              105            0.9276           0.0247            0.5630           0.2125          0.8686         0.0622   0.8956  0.0290                  0.7453                 0.1010    0.5296   0.1552        0.8966       0.0591       0.9653      0.0210   2.1333  0.7432   4.2667  2.1536
````

**Interpretation.** `Mean pairwise Jaccard for ANOVA selection is WDBC 1.0000, WPBC 0.7416, Parkinsons 0.6338; fixed-block stability equals 1 by construction.` Pairwise Jaccard measures resampling stability, not biological validity. Parkinson row-level splits are potentially optimistic because recordings from the same stem can appear on both sides; inspect the grouped audit before making subject-level claims.

## 22. Experiment 10: 11-model × 4-dataset benchmark

The same 11 estimators and hyperparameters are fitted on all available features for each dataset. Fifteen stratified test-fold measurements are summarized by means and sample SDs for sensitivity, specificity, precision, F1, balanced accuracy, MCC, ROC-AUC, average precision (PR-AUC), FN and FP. An imputer and scaler are inside each pipeline. The 44 combinations and all 660 fold-level records are exported.


<!-- Notebook code cell 68; execution count 23 -->

````python
multi_zoo_rows, zoo_fold_rows = [], []
for dataset, dset in DATASETS.items():
    Xd, yd = dset['X'], dset['y']
    for model_name, estimator in zoo.items():
        start = time.perf_counter()
        fold_metrics = []
        for fold, (tr, te) in enumerate(CV_REPEATED.split(Xd, yd), start=1):
            pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                             ('scaler', StandardScaler()), ('model', clone(estimator))])
            pipe.fit(Xd.iloc[tr], yd.iloc[tr])
            pred = pipe.predict(Xd.iloc[te]); score = pipe.predict_proba(Xd.iloc[te])[:, 1]
            met = metrics_from_preds(yd.iloc[te].to_numpy(), pred, score)
            fold_metrics.append(met)
            zoo_fold_rows.append({'dataset': dataset, 'model': model_name, 'fold': fold, **met})
        frame = pd.DataFrame(fold_metrics)
        multi_zoo_rows.append({'dataset': dataset, 'model': model_name, 'n_folds': len(frame),
                               'n_rows': len(yd), 'n_features': Xd.shape[1],
                               **summarize_fold_metrics(frame), 'wall_seconds_15fold': time.perf_counter() - start})
        print(f"{dataset:10s} | {model_name:28s} | sensitivity {frame.sensitivity.mean():.4f} ± {frame.sensitivity.std(ddof=1):.4f} | MCC {frame.mcc.mean():.4f}")

multi_zoo = pd.DataFrame(multi_zoo_rows)
multi_zoo.to_csv(RESULTS / 'exp10_multi_dataset_zoo.csv', index=False)
pd.DataFrame(zoo_fold_rows).to_csv(RESULTS / 'exp10b_multi_dataset_zoo_folds.csv', index=False)
assert multi_zoo.shape[0] == 44 and (multi_zoo.n_folds == 15).all()
display(multi_zoo)
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Logistic Regression (L2)     | sensitivity 0.9512 ± 0.0371 | MCC 0.9490
````


**Output (stdout):**

````text
WDBC       | Logistic Regression (L1)     | sensitivity 0.9450 ± 0.0376 | MCC 0.9362
````


**Output (stdout):**

````text
WDBC       | Linear Discriminant (LDA)    | sensitivity 0.8899 ± 0.0408 | MCC 0.9086
````


**Output (stdout):**

````text
WDBC       | SVM (linear kernel)          | sensitivity 0.9465 ± 0.0412 | MCC 0.9411
````


**Output (stdout):**

````text
WDBC       | SVM (RBF kernel)             | sensitivity 0.9606 ± 0.0317 | MCC 0.9488
````


**Output (stdout):**

````text
WDBC       | k-Nearest Neighbours         | sensitivity 0.9262 ± 0.0376 | MCC 0.9292
````


**Output (stdout):**

````text
WDBC       | Gaussian Naive Bayes         | sensitivity 0.8979 ± 0.0474 | MCC 0.8545
````


**Output (stdout):**

````text
WDBC       | Random Forest                | sensitivity 0.9341 ± 0.0553 | MCC 0.9146
````


**Output (stdout):**

````text
WDBC       | Extra Trees                  | sensitivity 0.9403 ± 0.0415 | MCC 0.9280
````


**Output (stdout):**

````text
WDBC       | Gradient Boosting            | sensitivity 0.9262 ± 0.0506 | MCC 0.9104
````


**Output (stdout):**

````text
WDBC       | HistGradient Boosting        | sensitivity 0.9340 ± 0.0367 | MCC 0.9226
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Logistic Regression (L2)     | sensitivity 0.1919 ± 0.1561 | MCC 0.1314
````


**Output (stdout):**

````text
WPBC       | Logistic Regression (L1)     | sensitivity 0.1778 ± 0.1470 | MCC 0.1354
````


**Output (stdout):**

````text
WPBC       | Linear Discriminant (LDA)    | sensitivity 0.2770 ± 0.1336 | MCC 0.1428
````


**Output (stdout):**

````text
WPBC       | SVM (linear kernel)          | sensitivity 0.2059 ± 0.1239 | MCC 0.1517
````


**Output (stdout):**

````text
WPBC       | SVM (RBF kernel)             | sensitivity 0.0141 ± 0.0372 | MCC -0.0028
````


**Output (stdout):**

````text
WPBC       | k-Nearest Neighbours         | sensitivity 0.2074 ± 0.1279 | MCC 0.1460
````


**Output (stdout):**

````text
WPBC       | Gaussian Naive Bayes         | sensitivity 0.4748 ± 0.1500 | MCC 0.1864
````


**Output (stdout):**

````text
WPBC       | Random Forest                | sensitivity 0.0719 ± 0.0880 | MCC 0.0703
````


**Output (stdout):**

````text
WPBC       | Extra Trees                  | sensitivity 0.1081 ± 0.0940 | MCC 0.1273
````


**Output (stdout):**

````text
WPBC       | Gradient Boosting            | sensitivity 0.1719 ± 0.1612 | MCC 0.0495
````


**Output (stdout):**

````text
WPBC       | HistGradient Boosting        | sensitivity 0.2207 ± 0.1481 | MCC 0.1622
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Coimbra    | Logistic Regression (L2)     | sensitivity 0.7560 ± 0.0865 | MCC 0.4924
````


**Output (stdout):**

````text
Coimbra    | Logistic Regression (L1)     | sensitivity 0.7303 ± 0.1150 | MCC 0.4836
````


**Output (stdout):**

````text
Coimbra    | Linear Discriminant (LDA)    | sensitivity 0.7098 ± 0.1092 | MCC 0.4568
````


**Output (stdout):**

````text
Coimbra    | SVM (linear kernel)          | sensitivity 0.7145 ± 0.1326 | MCC 0.4955
````


**Output (stdout):**

````text
Coimbra    | SVM (RBF kernel)             | sensitivity 0.7667 ± 0.0875 | MCC 0.4917
````


**Output (stdout):**

````text
Coimbra    | k-Nearest Neighbours         | sensitivity 0.7513 ± 0.1080 | MCC 0.5410
````


**Output (stdout):**

````text
Coimbra    | Gaussian Naive Bayes         | sensitivity 0.4449 ± 0.1564 | MCC 0.3449
````


**Output (stdout):**

````text
Coimbra    | Random Forest                | sensitivity 0.7671 ± 0.1108 | MCC 0.4127
````


**Output (stdout):**

````text
Coimbra    | Extra Trees                  | sensitivity 0.7761 ± 0.1096 | MCC 0.4622
````


**Output (stdout):**

````text
Coimbra    | Gradient Boosting            | sensitivity 0.7769 ± 0.1034 | MCC 0.4586
````


**Output (stdout):**

````text
Coimbra    | HistGradient Boosting        | sensitivity 0.7919 ± 0.1110 | MCC 0.5205
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Logistic Regression (L2)     | sensitivity 0.9320 ± 0.0385 | MCC 0.5818
````


**Output (stdout):**

````text
Parkinsons | Logistic Regression (L1)     | sensitivity 0.9320 ± 0.0403 | MCC 0.5713
````


**Output (stdout):**

````text
Parkinsons | Linear Discriminant (LDA)    | sensitivity 0.9503 ± 0.0357 | MCC 0.6508
````


**Output (stdout):**

````text
Parkinsons | SVM (linear kernel)          | sensitivity 0.9460 ± 0.0502 | MCC 0.5892
````


**Output (stdout):**

````text
Parkinsons | SVM (RBF kernel)             | sensitivity 0.9954 ± 0.0121 | MCC 0.6592
````


**Output (stdout):**

````text
Parkinsons | k-Nearest Neighbours         | sensitivity 0.9457 ± 0.0457 | MCC 0.7572
````


**Output (stdout):**

````text
Parkinsons | Gaussian Naive Bayes         | sensitivity 0.6305 ± 0.0842 | MCC 0.4742
````


**Output (stdout):**

````text
Parkinsons | Random Forest                | sensitivity 0.9658 ± 0.0369 | MCC 0.7271
````


**Output (stdout):**

````text
Parkinsons | Extra Trees                  | sensitivity 0.9727 ± 0.0413 | MCC 0.7852
````


**Output (stdout):**

````text
Parkinsons | Gradient Boosting            | sensitivity 0.9570 ± 0.0415 | MCC 0.7643
````


**Output (stdout):**

````text
Parkinsons | HistGradient Boosting        | sensitivity 0.9569 ± 0.0352 | MCC 0.7799
````


**Displayed output:**

````text
       dataset                      model  n_folds  n_rows  n_features  sensitivity_mean  sensitivity_std  specificity_mean  specificity_std  precision_mean  precision_std  f1_mean  f1_std  balanced_accuracy_mean  balanced_accuracy_std  mcc_mean  mcc_std  roc_auc_mean  roc_auc_std  pr_auc_mean  pr_auc_std  fn_mean  fn_std  fp_mean  fp_std  wall_seconds_15fold
0         WDBC   Logistic Regression (L2)       15     569          30            0.9512           0.0371            0.9907           0.0087          0.9841         0.0145   0.9669  0.0188                  0.9709                 0.0176    0.9490   0.0275        0.9946       0.0049       0.9933      0.0054   2.0667  1.5796   0.6667  0.6172               0.4119
1         WDBC   Logistic Regression (L1)       15     569          30            0.9450           0.0376            0.9850           0.0099          0.9743         0.0168   0.9590  0.0227                  0.9650                 0.0200    0.9362   0.0342        0.9951       0.0045       0.9936      0.0051   2.3333  1.5887   1.0667  0.7037               0.2877
2         WDBC  Linear Discriminant (LDA)       15     569          30            0.8899           0.0408            0.9963           0.0064          0.9928         0.0124   0.9382  0.0270                  0.9431                 0.0226    0.9086   0.0383        0.9920       0.0073       0.9903      0.0083   4.6667  1.7182   0.2667  0.4577               0.2610
3         WDBC        SVM (linear kernel)       15     569          30            0.9465           0.0412            0.9879           0.0090          0.9787         0.0161   0.9620  0.0262                  0.9672                 0.0231    0.9411   0.0403        0.9942       0.0064       0.9925      0.0071   2.2667  1.7512   0.8667  0.6399               0.3888
4         WDBC           SVM (RBF kernel)       15     569          30            0.9606           0.0317            0.9850           0.0124          0.9748         0.0206   0.9674  0.0195                  0.9728                 0.0171    0.9488   0.0306        0.9953       0.0049       0.9939      0.0059   1.6667  1.3452   1.0667  0.8837               0.5546
5         WDBC       k-Nearest Neighbours       15     569          30            0.9262           0.0376            0.9907           0.0101          0.9837         0.0176   0.9536  0.0202                  0.9584                 0.0184    0.9292   0.0302        0.9844       0.0101       0.9775      0.0129   3.1333  1.5976   0.6667  0.7237               0.3801
6         WDBC       Gaussian Naive Bayes       15     569          30            0.8979           0.0474            0.9514           0.0278          0.9190         0.0393   0.9069  0.0249                  0.9246                 0.0211    0.8545   0.0379        0.9872       0.0063       0.9793      0.0165   4.3333  2.0237   3.4667  1.9952               0.3759
7         WDBC              Random Forest       15     569          30            0.9341           0.0553            0.9748           0.0225          0.9581         0.0341   0.9447  0.0302                  0.9544                 0.0266    0.9146   0.0454        0.9898       0.0092       0.9882      0.0090   2.8000  2.3664   1.8000  1.6125              11.4647
8         WDBC                Extra Trees       15     569          30            0.9403           0.0415            0.9813           0.0172          0.9688         0.0265   0.9535  0.0200                  0.9608                 0.0185    0.9280   0.0304        0.9939       0.0044       0.9918      0.0055   2.5333  1.7674   1.3333  1.2344               7.4853
9         WDBC          Gradient Boosting       15     569          30            0.9262           0.0506            0.9767           0.0195          0.9604         0.0322   0.9420  0.0295                  0.9514                 0.0253    0.9104   0.0437        0.9925       0.0052       0.9901      0.0065   3.1333  2.1668   1.6667  1.3973               9.2097
10        WDBC      HistGradient Boosting       15     569          30            0.9340           0.0367            0.9813           0.0137          0.9681         0.0225   0.9502  0.0203                  0.9577                 0.0180    0.9226   0.0308        0.9913       0.0064       0.9895      0.0068   2.8000  1.5675   1.3333  0.9759               3.2724
11        WPBC   Logistic Regression (L2)       15     198          32            0.1919           0.1561            0.9115           0.0484          0.3853         0.2612   0.2474  0.1820                  0.5517                 0.0856    0.1314   0.2155        0.6577       0.1141       0.4177      0.1366   7.6000  1.5492   2.6667  1.4475               0.3187
12        WPBC   Logistic Regression (L1)       15     198          32            0.1778           0.1470            0.9203           0.0453          0.4139         0.2241   0.2384  0.1669                  0.5490                 0.0821    0.1354   0.1996        0.6412       0.1143       0.4273      0.1244   7.7333  1.4864   2.4000  1.3522               0.2788
13        WPBC  Linear Discriminant (LDA)       15     198          32            0.2770           0.1336            0.8518           0.0523          0.3687         0.1605   0.3120  0.1393                  0.5644                 0.0776    0.1428   0.1722        0.6701       0.0864       0.4069      0.0978   6.8000  1.3202   4.4667  1.5523               0.2739
14        WPBC        SVM (linear kernel)       15     198          32            0.2059           0.1239            0.9049           0.0549          0.4342         0.2369   0.2649  0.1350                  0.5554                 0.0649    0.1517   0.1706        0.6527       0.1063       0.4010      0.1022   7.4667  1.2459   2.8667  1.6417               0.4875
15        WPBC           SVM (RBF kernel)       15     198          32            0.0141           0.0372            0.9845           0.0305          0.0556         0.1500   0.0222  0.0586                  0.4993                 0.0176   -0.0028   0.0626        0.6139       0.1073       0.4029      0.1018   9.2667  0.5936   0.4667  0.9155               0.3942
16        WPBC       k-Nearest Neighbours       15     198          32            0.2074           0.1279            0.9028           0.0567          0.4170         0.2544   0.2658  0.1505                  0.5551                 0.0715    0.1460   0.1892        0.5307       0.1348       0.3105      0.1064   7.4667  1.3558   2.9333  1.7099               0.3475
17        WPBC       Gaussian Naive Bayes       15     198          32            0.4748           0.1500            0.7217           0.1108          0.3632         0.1285   0.4045  0.1259                  0.5983                 0.0997    0.1864   0.1861        0.6271       0.1188       0.4586      0.1287   4.9333  1.3870   8.4000  3.3552               0.2706
18        WPBC              Random Forest       15     198          32            0.0719           0.0880            0.9689           0.0320          0.2889         0.3698   0.1121  0.1357                  0.5204                 0.0446    0.0703   0.1684        0.5704       0.1212       0.3672      0.0950   8.7333  1.0328   0.9333  0.9612               9.2084
19        WPBC                Extra Trees       15     198          32            0.1081           0.0940            0.9624           0.0305          0.4511         0.3621   0.1675  0.1332                  0.5353                 0.0495    0.1273   0.1733        0.5929       0.1209       0.4029      0.1045   8.4000  1.1212   1.1333  0.9155               7.3897
20        WPBC          Gradient Boosting       15     198          32            0.1719           0.1612            0.8830           0.0587          0.2593         0.1830   0.1965  0.1570                  0.5274                 0.0645    0.0495   0.1542        0.5507       0.1155       0.3182      0.0717   7.8000  1.6125   3.5333  1.7674               4.6416
21        WPBC      HistGradient Boosting       15     198          32            0.2207           0.1481            0.9051           0.0529          0.4308         0.1701   0.2750  0.1293                  0.5629                 0.0678    0.1622   0.1471        0.5965       0.1048       0.4251      0.1046   7.3333  1.4475   2.8667  1.5976               1.2411
22     Coimbra   Logistic Regression (L2)       15     116           9            0.7560           0.0865            0.7297           0.1850          0.7932         0.1188   0.7671  0.0672                  0.7428                 0.0943    0.4924   0.1880        0.8051       0.1103       0.8464      0.0981   3.1333  1.1255   2.8000  1.9346               0.2920
23     Coimbra   Logistic Regression (L1)       15     116           9            0.7303           0.1150            0.7485           0.1671          0.7945         0.1069   0.7546  0.0858                  0.7394                 0.1006    0.4836   0.2008        0.8071       0.1087       0.8452      0.0995   3.4667  1.5055   2.6000  1.7238               0.3297
24     Coimbra  Linear Discriminant (LDA)       15     116           9            0.7098           0.1092            0.7436           0.1507          0.7827         0.1073   0.7393  0.0856                  0.7267                 0.0990    0.4568   0.1999        0.7895       0.1018       0.8411      0.0924   3.7333  1.4376   2.6667  1.5887               0.2855
25     Coimbra        SVM (linear kernel)       15     116           9            0.7145           0.1326            0.7758           0.1244          0.8030         0.0899   0.7495  0.0912                  0.7451                 0.0882    0.4955   0.1787        0.7998       0.1012       0.8429      0.0942   3.6667  1.7182   2.3333  1.3452               0.3183
26     Coimbra           SVM (RBF kernel)       15     116           9            0.7667           0.0875            0.7170           0.1637          0.7842         0.1026   0.7686  0.0560                  0.7418                 0.0770    0.4917   0.1532        0.8485       0.0557       0.8871      0.0407   3.0000  1.1339   2.9333  1.7099               0.3573
27     Coimbra       k-Nearest Neighbours       15     116           9            0.7513           0.1080            0.7867           0.1505          0.8229         0.1144   0.7810  0.0921                  0.7690                 0.1013    0.5410   0.2029        0.8084       0.0867       0.8193      0.0797   3.2000  1.4243   2.2000  1.5213               0.3296
28     Coimbra       Gaussian Naive Bayes       15     116           9            0.4449           0.1564            0.8709           0.0967          0.8221         0.1202   0.5594  0.1640                  0.6579                 0.0899    0.3449   0.1771        0.7728       0.0984       0.8031      0.0956   7.1333  2.0999   1.3333  0.9759               0.2793
29     Coimbra              Random Forest       15     116           9            0.7671           0.1108            0.6400           0.1723          0.7316         0.0999   0.7450  0.0928                  0.7035                 0.1134    0.4127   0.2258        0.7899       0.0958       0.8284      0.0824   3.0000  1.4639   3.7333  1.7915               8.2173
30     Coimbra                Extra Trees       15     116           9            0.7761           0.1096            0.6776           0.1893          0.7621         0.1126   0.7622  0.0836                  0.7268                 0.1007    0.4622   0.2004        0.8209       0.0964       0.8480      0.0842   2.8667  1.4075   3.3333  1.9518               6.5931
31     Coimbra          Gradient Boosting       15     116           9            0.7769           0.1034            0.6727           0.1460          0.7524         0.0924   0.7598  0.0754                  0.7248                 0.0890    0.4586   0.1788        0.8139       0.0944       0.8533      0.0827   2.8667  1.3558   3.4000  1.5492               2.0167
32     Coimbra      HistGradient Boosting       15     116           9            0.7919           0.1110            0.7236           0.1525          0.7844         0.1016   0.7852  0.0947                  0.7578                 0.1104    0.5205   0.2206        0.8106       0.1112       0.8319      0.0967   2.6667  1.4475   2.8667  1.5523               0.6471
33  Parkinsons   Logistic Regression (L2)       15     195          22            0.9320           0.0385            0.6119           0.2101          0.8818         0.0623   0.9046  0.0353                  0.7719                 0.1034    0.5818   0.1739        0.8955       0.0556       0.9641      0.0201   2.0000  1.1339   3.8000  2.1448               0.3877
34  Parkinsons   Logistic Regression (L1)       15     195          22            0.9320           0.0403            0.5978           0.2067          0.8778         0.0614   0.9026  0.0366                  0.7649                 0.1030    0.5713   0.1771        0.8879       0.0595       0.9619      0.0211   2.0000  1.1952   3.9333  2.1202               0.3121
35  Parkinsons  Linear Discriminant (LDA)       15     195          22            0.9503           0.0357            0.6511           0.1848          0.8942         0.0554   0.9202  0.0320                  0.8007                 0.0921    0.6508   0.1558        0.8818       0.0753       0.9480      0.0387   1.4667  1.0601   3.4000  1.8439               0.2667
36  Parkinsons        SVM (linear kernel)       15     195          22            0.9460           0.0502            0.5770           0.2159          0.8747         0.0620   0.9068  0.0336                  0.7615                 0.1010    0.5892   0.1606        0.8795       0.0625       0.9562      0.0240   1.6000  1.5024   4.1333  2.1996               0.3538
37  Parkinsons           SVM (RBF kernel)       15     195          22            0.9954           0.0121            0.5356           0.2170          0.8687         0.0608   0.9268  0.0366                  0.7655                 0.1108    0.6592   0.1792        0.8866       0.0675       0.9583      0.0250   0.1333  0.3519   4.5333  2.2318               0.3874
38  Parkinsons       k-Nearest Neighbours       15     195          22            0.9457           0.0457            0.7926           0.1352          0.9353         0.0396   0.9391  0.0208                  0.8692                 0.0570    0.7572   0.0786        0.9690       0.0151       0.9849      0.0096   1.6000  1.3522   2.0000  1.3093               0.2671
39  Parkinsons       Gaussian Naive Bayes       15     195          22            0.6305           0.0842            0.9156           0.0912          0.9592         0.0459   0.7580  0.0672                  0.7730                 0.0682    0.4742   0.1208        0.8560       0.0548       0.9401      0.0319  10.8667  2.4746   0.8000  0.8619               0.2640
40  Parkinsons              Random Forest       15     195          22            0.9658           0.0369            0.7074           0.2066          0.9115         0.0609   0.9366  0.0367                  0.8366                 0.1036    0.7271   0.1655        0.9617       0.0330       0.9874      0.0110   1.0000  1.0690   2.8667  2.0656               8.4327
41  Parkinsons                Extra Trees       15     195          22            0.9727           0.0413            0.7607           0.1279          0.9258         0.0390   0.9478  0.0275                  0.8667                 0.0632    0.7852   0.1054        0.9748       0.0223       0.9918      0.0075   0.8000  1.2071   2.3333  1.2910               7.7840
42  Parkinsons          Gradient Boosting       15     195          22            0.9570           0.0415            0.7800           0.1331          0.9305         0.0410   0.9429  0.0323                  0.8685                 0.0732    0.7643   0.1378        0.9612       0.0398       0.9869      0.0137   1.2667  1.2228   2.1333  1.3020               3.5015
43  Parkinsons      HistGradient Boosting       15     195          22            0.9569           0.0352            0.8015           0.1115          0.9365         0.0349   0.9460  0.0252                  0.8792                 0.0580    0.7799   0.1017        0.9645       0.0302       0.9869      0.0128   1.2667  1.0328   1.9333  1.0998               0.9439
````

**Interpretation.** `The highest descriptive sensitivity by dataset is WDBC: SVM (RBF kernel) (0.9606); WPBC: Gaussian Naive Bayes (0.4748); Coimbra: HistGradient Boosting (0.7919); Parkinsons: SVM (RBF kernel) (0.9954).` The 15 folds overlap in training observations and are not independent datasets; SD is not a standard error. The outcomes and prevalence differ across datasets, so numerical comparisons are contextual rather than a pooled claim of generalizability.

## 23. Experiment 11: apparent versus nested operating points

All four datasets use L2 logistic regression on **all available predictors**; the separate WDBC experiment 4 uses only the Worst block, so their thresholds are not interchangeable. For a threshold grid $\tau\in\{0.05,0.06,\ldots,0.95\}$, the rule maximizes precision subject to sensitivity $\ge0.95$ on inner OOF predictions; if infeasible it maximizes sensitivity, then precision. Five outer test folds produce one OOF score and selected threshold per row. The reported $[\min_k\tau_k,\max_k\tau_k]$ reveals threshold instability. A 2,000-replicate, class-stratified percentile bootstrap resamples fixed OOF pairs; Parkinson uses identifier-stem clusters. It does not repeat fitting or repair row-level training/test subject overlap.


<!-- Notebook code cell 71; execution count 24 -->

````python
# A constrained operating-point rule: maximize precision subject to sensitivity >= 0.95.
# If no grid point is feasible, maximize sensitivity, then precision, then use the smaller threshold.
def choose_high_sensitivity_threshold(yt, scores, target=0.95):
    candidates = []
    for tau in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (scores >= tau).astype(int)
        m = metrics_from_preds(np.asarray(yt), pred)
        candidates.append((float(tau), m['sensitivity'], m['precision']))
    eligible = [r for r in candidates if r[1] >= target]
    if eligible:
        chosen = max(eligible, key=lambda r: (r[2], -r[0]))
    else:
        chosen = max(candidates, key=lambda r: (r[1], r[2], -r[0]))
    return chosen[0], bool(eligible)

def bootstrap_nested_metrics(yt, pred, groups=None, n_boot=2000):
    """Stratified percentile bootstrap of fixed OOF pairs; cluster-resample Parkinson stems."""
    yt, pred = np.asarray(yt), np.asarray(pred)
    rng = np.random.default_rng(SEED)
    samples = []
    if groups is None:
        strata = [np.flatnonzero(yt == c) for c in (0, 1)]
        def sample_indices():
            return np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in strata])
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        group_rows = {g: np.flatnonzero(groups == g) for g in unique}
        by_class = {c: [g for g in unique if yt[group_rows[g][0]] == c] for c in (0, 1)}
        assert all(len(by_class[c]) > 0 for c in (0, 1))
        def sample_indices():
            picked = [g for c in (0, 1) for g in rng.choice(by_class[c], size=len(by_class[c]), replace=True)]
            return np.concatenate([group_rows[g] for g in picked])
    for _ in range(n_boot):
        ix = sample_indices()
        yb, pb = yt[ix], pred[ix]
        tp = int(np.sum((yb == 1) & (pb == 1)))
        fn = int(np.sum((yb == 1) & (pb == 0)))
        tn = int(np.sum((yb == 0) & (pb == 0)))
        fp = int(np.sum((yb == 0) & (pb == 1)))
        samples.append([tp / (tp + fn), tn / (tn + fp), tp / (tp + fp) if tp + fp else 0.0])
    return np.quantile(np.asarray(samples), [0.025, 0.975], axis=0)

nested_summary_rows, nested_fold_rows, nested_oof_rows = [], [], []
for dataset, dset in DATASETS.items():
    Xd, yd = dset['X'], dset['y']
    score_oof = np.full(len(yd), np.nan); pred_oof = np.full(len(yd), -1, dtype=int)
    tau_oof = np.full(len(yd), np.nan); fold_oof = np.full(len(yd), -1, dtype=int)
    fold_records = []
    for fold, (tr, te) in enumerate(CV_OOF.split(Xd, yd), start=1):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        inner_scores = np.full(len(tr), np.nan)
        Xt, yt = Xd.iloc[tr], yd.iloc[tr]
        for itr, ite in inner.split(Xt, yt):
            p = lr_pipeline().fit(Xt.iloc[itr], yt.iloc[itr])
            inner_scores[ite] = p.predict_proba(Xt.iloc[ite])[:, 1]
        assert np.isfinite(inner_scores).all()
        tau, feasible = choose_high_sensitivity_threshold(yt.to_numpy(), inner_scores)
        p = lr_pipeline().fit(Xt, yt)
        scores = p.predict_proba(Xd.iloc[te])[:, 1]
        predictions = (scores >= tau).astype(int)
        score_oof[te], pred_oof[te], tau_oof[te], fold_oof[te] = scores, predictions, tau, fold
        m = metrics_from_preds(yd.iloc[te].to_numpy(), predictions, scores)
        record = {'dataset': dataset, 'fold': fold, 'threshold': tau,
                  'inner_target_feasible': int(feasible), **m}
        fold_records.append(record); nested_fold_rows.append(record)
    assert np.isfinite(score_oof).all() and np.isfinite(tau_oof).all() and (pred_oof >= 0).all()
    apparent_tau, apparent_feasible = choose_high_sensitivity_threshold(yd.to_numpy(), score_oof)
    apparent = metrics_from_preds(yd.to_numpy(), (score_oof >= apparent_tau).astype(int), score_oof)
    pooled = metrics_from_preds(yd.to_numpy(), pred_oof, score_oof)
    ci = bootstrap_nested_metrics(yd.to_numpy(), pred_oof, dset['groups'], n_boot=2000)
    ff = pd.DataFrame(fold_records)
    nested_summary_rows.append({
        'dataset': dataset, 'target_sensitivity': 0.95,
        'apparent_threshold': apparent_tau, 'apparent_target_feasible': int(apparent_feasible),
        'apparent_sensitivity': apparent['sensitivity'], 'apparent_specificity': apparent['specificity'],
        'apparent_precision': apparent['precision'], 'apparent_fn': apparent['fn'], 'apparent_fp': apparent['fp'],
        'nested_sensitivity_mean': ff.sensitivity.mean(), 'nested_sensitivity_std': ff.sensitivity.std(ddof=1),
        'nested_sensitivity_pooled': pooled['sensitivity'], 'nested_specificity_pooled': pooled['specificity'],
        'nested_precision_pooled': pooled['precision'], 'nested_fn': pooled['fn'], 'nested_fp': pooled['fp'],
        'threshold_min': ff.threshold.min(), 'threshold_max': ff.threshold.max(),
        'n_inner_feasible': int(ff.inner_target_feasible.sum()),
        'sensitivity_ci_low': ci[0, 0], 'sensitivity_ci_high': ci[1, 0],
        'specificity_ci_low': ci[0, 1], 'specificity_ci_high': ci[1, 1],
        'precision_ci_low': ci[0, 2], 'precision_ci_high': ci[1, 2],
        'bootstrap_replicates': 2000,
        'bootstrap_unit': 'subject_stem' if dset['groups'] is not None else 'row',
    })
    for i in range(len(yd)):
        nested_oof_rows.append({'dataset': dataset, 'row_index': i, 'fold': int(fold_oof[i]),
                                'target': int(yd.iloc[i]), 'score': float(score_oof[i]),
                                'threshold': float(tau_oof[i]), 'prediction': int(pred_oof[i])})
    print(f"{dataset:10s} | apparent τ={apparent_tau:.2f}, sensitivity={apparent['sensitivity']:.4f} | "
          f"nested sensitivity={ff.sensitivity.mean():.4f} ± {ff.sensitivity.std(ddof=1):.4f}, "
          f"τ range=[{ff.threshold.min():.2f},{ff.threshold.max():.2f}], "
          f"pooled 95% CI=[{ci[0,0]:.4f},{ci[1,0]:.4f}]")

nested_multi = pd.DataFrame(nested_summary_rows)
nested_multi.to_csv(RESULTS / 'exp11_multi_dataset_nested_threshold.csv', index=False)
pd.DataFrame(nested_fold_rows).to_csv(RESULTS / 'exp11b_nested_threshold_folds.csv', index=False)
pd.DataFrame(nested_oof_rows).to_csv(RESULTS / 'exp11c_nested_threshold_oof.csv', index=False)
assert len(nested_multi) == 4 and len(nested_fold_rows) == 20 and all(r['bootstrap_replicates'] == 2000 for r in nested_summary_rows)
display(nested_multi)
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | apparent τ=0.42, sensitivity=0.9575 | nested sensitivity=0.9389 ± 0.0634, τ range=[0.38,0.66], pooled 95% CI=[0.9057,0.9670]
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | apparent τ=0.08, sensitivity=0.9574 | nested sensitivity=0.9556 ± 0.0994, τ range=[0.05,0.07], pooled 95% CI=[0.8936,1.0000]
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Coimbra    | apparent τ=0.24, sensitivity=0.9531 | nested sensitivity=0.9526 ± 0.0434, τ range=[0.20,0.24], pooled 95% CI=[0.8906,1.0000]
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | apparent τ=0.44, sensitivity=0.9592 | nested sensitivity=0.9660 ± 0.0339, τ range=[0.29,0.43], pooled 95% CI=[0.9041,1.0000]
````


**Displayed output:**

````text
      dataset  target_sensitivity  apparent_threshold  apparent_target_feasible  apparent_sensitivity  apparent_specificity  apparent_precision  apparent_fn  apparent_fp  nested_sensitivity_mean  nested_sensitivity_std  nested_sensitivity_pooled  nested_specificity_pooled  nested_precision_pooled  nested_fn  nested_fp  threshold_min  threshold_max  n_inner_feasible  sensitivity_ci_low  sensitivity_ci_high  specificity_ci_low  specificity_ci_high  precision_ci_low  precision_ci_high  bootstrap_replicates bootstrap_unit
0        WDBC              0.9500              0.4200                         1                0.9575                0.9860              0.9760            9            5                   0.9389                  0.0634                     0.9387                     0.9860                   0.9755         13          5         0.3800         0.6600                 5              0.9057               0.9670              0.9747               0.9972            0.9547             0.9950                  2000            row
1        WPBC              0.9500              0.0800                         1                0.9574                0.1921              0.2695            2          122                   0.9556                  0.0994                     0.9574                     0.1391                   0.2571          2        130         0.0500         0.0700                 2              0.8936               1.0000              0.0861               0.1987            0.2400             0.2744                  2000            row
2     Coimbra              0.9500              0.2400                         1                0.9531                0.1923              0.5922            3           42                   0.9526                  0.0434                     0.9531                     0.1731                   0.5865          3         43         0.2000         0.2400                 5              0.8906               1.0000              0.0769               0.2885            0.5545             0.6238                  2000            row
3  Parkinsons              0.9500              0.4400                         1                0.9592                0.6042              0.8812            6           19                   0.9660                  0.0339                     0.9660                     0.5417                   0.8659          5         22         0.2900         0.4300                 5              0.9041               1.0000              0.2292               0.8542            0.7926             0.9497                  2000   subject_stem
````

**Interpretation.** `Nested sensitivity and threshold range: WDBC 0.9389 ± 0.0634, τ 0.38–0.66, CI [0.9057, 0.9670]; WPBC 0.9556 ± 0.0994, τ 0.05–0.07, CI [0.8936, 1.0000]; Coimbra 0.9526 ± 0.0434, τ 0.20–0.24, CI [0.8906, 1.0000]; Parkinsons 0.9660 ± 0.0339, τ 0.29–0.43, CI [0.9041, 1.0000].` The target constrains the inner training-derived selection only; outer sensitivity may be lower. Conditional intervals and small-data threshold ranges should not be presented as prospective clinical guarantees.

## 24. Experiment 12: nested Platt calibration and decision curves

All four datasets use L2 logistic regression on **all available predictors**; WDBC experiment 5 instead uses the Worst block. Each outer test prediction is generated by a model fit only on outer training data. For Platt calibration, the **entire** imputer–scaler–logistic pipeline is the estimator of `CalibratedClassifierCV(method='sigmoid', cv=5)`, so internal calibration fits learn preprocessing from their own training subsets. The 10-bin ECE is $\sum_b(n_b/n)|\bar p_b-\bar y_b|$. For threshold probability $p_t$, net benefit is $TP/n-(FP/n)\,p_t/(1-p_t)$; treat-all equals prevalence minus $(1-\text{prevalence})p_t/(1-p_t)$ and treat-none is zero.


<!-- Notebook code cell 74; execution count 25 -->

````python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

def reliability_points(yt, scores, n_bins=10):
    yt, scores = np.asarray(yt), np.asarray(scores)
    bin_index = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_index == b
        if mask.any():
            rows.append({'bin': b, 'n': int(mask.sum()),
                         'mean_predicted': float(scores[mask].mean()),
                         'observed_rate': float(yt[mask].mean())})
    ece = sum(r['n'] / len(yt) * abs(r['observed_rate'] - r['mean_predicted']) for r in rows)
    return rows, float(ece)

calibration_rows, calibration_bins, cal_oof_rows, dca_rows = [], [], [], []
calibration_scores = {}
for dataset, dset in DATASETS.items():
    Xd, yd = dset['X'], dset['y']
    raw = np.full(len(yd), np.nan); calibrated = np.full(len(yd), np.nan)
    for tr, te in CV_OOF.split(Xd, yd):
        base = lr_pipeline().fit(Xd.iloc[tr], yd.iloc[tr])
        # Entire preprocessing pipeline is the base estimator, so it is refitted in each calibration split.
        platt = CalibratedClassifierCV(estimator=lr_pipeline(), method='sigmoid', cv=5)
        platt.fit(Xd.iloc[tr], yd.iloc[tr])
        raw[te] = base.predict_proba(Xd.iloc[te])[:, 1]
        calibrated[te] = platt.predict_proba(Xd.iloc[te])[:, 1]
    assert np.isfinite(raw).all() and np.isfinite(calibrated).all()
    assert (raw >= 0).all() and (raw <= 1).all() and (calibrated >= 0).all() and (calibrated <= 1).all()
    calibration_scores[dataset] = {'uncalibrated': raw, 'Platt-calibrated': calibrated}
    for variant, s in calibration_scores[dataset].items():
        bin_rows, ece = reliability_points(yd.to_numpy(), s)
        calibration_rows.append({'dataset': dataset, 'variant': variant,
                                 'brier': brier_score_loss(yd, s), 'log_loss': log_loss(yd, s, labels=[0, 1]),
                                 'ece_10_uniform_bins': ece, 'roc_auc': roc_auc_score(yd, s),
                                 'n': len(yd), 'n_nonempty_bins': len(bin_rows)})
        calibration_bins.extend([{'dataset': dataset, 'variant': variant, **r} for r in bin_rows])
        for i, score in enumerate(s):
            cal_oof_rows.append({'dataset': dataset, 'variant': variant, 'row_index': i,
                                 'target': int(yd.iloc[i]), 'score': float(score)})
    # Net benefit uses observed sample prevalence; it is descriptive in case-control data.
    yt = yd.to_numpy(); n = len(yt); prevalence = float(yt.mean())
    for pt in np.round(np.arange(0.05, 0.801, 0.01), 2):
        weight = pt / (1 - pt)
        def net_benefit(s):
            positive = s >= pt
            tp = np.sum(positive & (yt == 1)); fp = np.sum(positive & (yt == 0))
            return tp / n - (fp / n) * weight
        dca_rows.append({'dataset': dataset, 'threshold': pt,
                         'net_benefit_raw': net_benefit(raw),
                         'net_benefit_platt': net_benefit(calibrated),
                         'net_benefit_treat_all': prevalence - (1 - prevalence) * weight,
                         'net_benefit_treat_none': 0.0, 'sample_prevalence': prevalence})
    print(f"{dataset:10s} | Brier raw={brier_score_loss(yd, raw):.4f}, Platt={brier_score_loss(yd, calibrated):.4f}")

multi_calibration = pd.DataFrame(calibration_rows)
reliability_multi = pd.DataFrame(calibration_bins)
dca_multi = pd.DataFrame(dca_rows)
multi_calibration.to_csv(RESULTS / 'exp12_multi_dataset_calibration.csv', index=False)
reliability_multi.to_csv(RESULTS / 'exp12a_reliability_bins.csv', index=False)
dca_multi.to_csv(RESULTS / 'exp12b_dca_curves.csv', index=False)
pd.DataFrame(cal_oof_rows).to_csv(RESULTS / 'exp12c_calibration_oof.csv', index=False)
assert len(multi_calibration) == 8 and len(dca_multi) == 4 * 76
print('\nProbability-quality metrics:'); display(multi_calibration)
print('\nDecision-curve data at illustrative probability thresholds:')
display(dca_multi[dca_multi.threshold.isin([0.1, 0.2, 0.5, 0.8])])
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Brier raw=0.0200, Platt=0.0267
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Brier raw=0.1846, Platt=0.1763
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Coimbra    | Brier raw=0.1951, Platt=0.2074
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Brier raw=0.1024, Platt=0.1186

Probability-quality metrics:
````


**Displayed output:**

````text
      dataset           variant  brier  log_loss  ece_10_uniform_bins  roc_auc    n  n_nonempty_bins
0        WDBC      uncalibrated 0.0200    0.0764               0.0151   0.9947  569               10
1        WDBC  Platt-calibrated 0.0267    0.1100               0.0553   0.9949  569               10
2        WPBC      uncalibrated 0.1846    0.5554               0.0934   0.6606  198               10
3        WPBC  Platt-calibrated 0.1763    0.5321               0.0277   0.6428  198                7
4     Coimbra      uncalibrated 0.1951    0.6157               0.0734   0.7713  116               10
5     Coimbra  Platt-calibrated 0.2074    0.6120               0.0850   0.7578  116                9
6  Parkinsons      uncalibrated 0.1024    0.3157               0.0803   0.9001  195               10
7  Parkinsons  Platt-calibrated 0.1186    0.3845               0.1173   0.9028  195                8
````


**Output (stdout):**

````text

Decision-curve data at illustrative probability thresholds:
````


**Displayed output:**

````text
        dataset  threshold  net_benefit_raw  net_benefit_platt  net_benefit_treat_all  net_benefit_treat_none  sample_prevalence
5          WDBC     0.1000           0.3587             0.3566                 0.3029                  0.0000             0.3726
15         WDBC     0.2000           0.3554             0.3497                 0.2157                  0.0000             0.3726
45         WDBC     0.5000           0.3462             0.3445                -0.2548                  0.0000             0.3726
75         WDBC     0.8000           0.3339             0.3005                -2.1371                  0.0000             0.3726
81         WPBC     0.1000           0.1560             0.1543                 0.1526                  0.0000             0.2374
91         WPBC     0.2000           0.0556             0.0758                 0.0467                  0.0000             0.2374
121        WPBC     0.5000          -0.0152            -0.0101                -0.5253                  0.0000             0.2374
151        WPBC     0.8000          -0.0606             0.0000                -2.8131                  0.0000             0.2374
157     Coimbra     0.1000           0.5048             0.5019                 0.5019                  0.0000             0.5517
167     Coimbra     0.2000           0.4310             0.4440                 0.4397                  0.0000             0.5517
197     Coimbra     0.5000           0.2500             0.2414                 0.1034                  0.0000             0.5517
227     Coimbra     0.8000           0.0690             0.0086                -1.2414                  0.0000             0.5517
233  Parkinsons     0.1000           0.7333             0.7265                 0.7265                  0.0000             0.7538
243  Parkinsons     0.2000           0.7154             0.6923                 0.6923                  0.0000             0.7538
273  Parkinsons     0.5000           0.6205             0.6000                 0.5077                  0.0000             0.7538
303  Parkinsons     0.8000           0.3487             0.4564                -0.2308                  0.0000             0.7538
````

**Interpretation.** `Platt scaling lowers the Brier score in 1/4 datasets in this run: WDBC 0.0200→0.0267; WPBC 0.1846→0.1763; Coimbra 0.1951→0.2074; Parkinsons 0.1024→0.1186.` These OOF curves use the sample prevalence. In case-control samples, predicted probabilities and treat-all net benefit do not automatically transport to a clinical population. WPBC lacks a fixed time horizon and Parkinson's row-level splits have clustered recordings; decision curves are descriptive, not a clinical utility claim.

## 25. Experiment 13: matched PCA-3 quantum and classical benchmarks

All four models use identical repeated and single OOF partitions. The quantum and RBF PCA-3 branches share fold-fitted imputation, standardization, PCA and angular min–max scaling, which is explicitly tested. Classical controls also include full-feature RBF SVM and logistic regression. Because the CNOT chain $U$ is input independent, $\langle U\phi(x)|U\phi(x')\rangle=\langle\phi(x)|\phi(x')\rangle$ and $k(x,x')=\prod_j\cos^2[(x_j-x'_j)/2]$: this three-qubit kernel has an exact classical closed form. Exact McNemar uses discordant OOF classifications; Wilcoxon is applied to per-row absolute probability-error differences and, separately, five paired fold sensitivities. Holm adjustment is applied across four datasets within each family. Overlapping fits and Parkinson clusters limit nominal $p$-values.


<!-- Notebook code cell 77; execution count 26 -->

````python
from scipy.stats import binomtest, wilcoxon

# Fidelity sanity checks, including the analytic separable kernel induced by a fixed unitary.
qrng = np.random.default_rng(SEED)
probe_angles = qrng.uniform(0, np.pi, size=(16, 3))
probe_states = np.array([quantum_state(x) for x in probe_angles])
probe_gram = fidelity_kernel_matrix(probe_angles, probe_angles)
closed_form = np.prod(np.cos((probe_angles[:, None, :] - probe_angles[None, :, :]) / 2) ** 2, axis=2)
assert np.allclose(np.linalg.norm(probe_states, axis=1), 1, atol=1e-12)
assert np.allclose(probe_gram, probe_gram.T, atol=1e-12)
assert np.allclose(np.diag(probe_gram), 1, atol=1e-12)
assert np.linalg.eigvalsh(probe_gram).min() >= -1e-10
assert np.allclose(probe_gram, closed_form, atol=1e-12)
print('Fidelity checks: unit norm, symmetric PSD Gram, unit diagonal, and analytic closed form: PASS')

def matched_quantum_models():
    return {
        'Quantum kernel SVM (PCA-3)': Pipeline([('imputer', SimpleImputer(strategy='median')),
                                                  ('model', QuantumKernelSVM())]),
        'RBF-SVM (matched PCA-3)': Pipeline([('imputer', SimpleImputer(strategy='median')),
                                               ('scaler', StandardScaler()),
                                               ('pca', PCA(n_components=3, random_state=SEED)),
                                               ('angle', MinMaxScaler(feature_range=(0, np.pi))),
                                               ('model', SVC(kernel='rbf', probability=True, random_state=SEED))]),
        'RBF-SVM (all features)': Pipeline([('imputer', SimpleImputer(strategy='median')),
                                             ('scaler', StandardScaler()),
                                             ('model', SVC(kernel='rbf', probability=True, random_state=SEED))]),
        'Logistic regression L2 (all features)': lr_pipeline(),
    }

def signed_rank_p(diff):
    diff = np.asarray(diff, dtype=float)
    if np.allclose(diff, 0):
        return 0.0, 1.0
    result = wilcoxon(diff, zero_method='wilcox', alternative='two-sided', method='auto')
    return float(result.statistic), float(result.pvalue)

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float); order = np.argsort(p); adjusted = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted

quantum_summary_rows, quantum_fold_rows, quantum_oof_rows, paired_rows = [], [], [], []
for dataset, dset in DATASETS.items():
    Xd, yd = dset['X'], dset['y']
    repeated = list(CV_REPEATED.split(Xd, yd)); single = list(CV_OOF.split(Xd, yd))
    assert all(np.array_equal(repeated[i][0], single[i][0]) and np.array_equal(repeated[i][1], single[i][1]) for i in range(5))
    oof = {}; fold_frames = {}; front_end_examples = {}
    for model_name, estimator in matched_quantum_models().items():
        started = time.perf_counter(); records = []
        score_oof = np.full(len(yd), np.nan); pred_oof = np.full(len(yd), -1, dtype=int)
        oof_fold = np.full(len(yd), -1, dtype=int)
        train_pairs = 0; test_pairs = 0
        for fold, (tr, te) in enumerate(repeated, start=1):
            fitted = clone(estimator).fit(Xd.iloc[tr], yd.iloc[tr])
            pred = fitted.predict(Xd.iloc[te]); scores = fitted.predict_proba(Xd.iloc[te])[:, 1]
            m = metrics_from_preds(yd.iloc[te].to_numpy(), pred, scores)
            records.append(m); quantum_fold_rows.append({'dataset': dataset, 'model': model_name, 'fold': fold, **m})
            if fold <= 5:
                score_oof[te] = scores; pred_oof[te] = pred; oof_fold[te] = fold
            if model_name.startswith('Quantum'):
                train_pairs += len(tr) * len(tr); test_pairs += len(tr) * len(te)
                if fold == 15:
                    q = fitted.named_steps['model']; imputed = fitted.named_steps['imputer'].transform(Xd.iloc[te])
                    front_end_examples['q'] = q.angle_.transform(q.pca_.transform(q.scaler_.transform(imputed)))
            elif model_name == 'RBF-SVM (matched PCA-3)' and fold == 15:
                front_end_examples['c'] = fitted[:-1].transform(Xd.iloc[te])
        assert np.isfinite(score_oof).all() and (pred_oof >= 0).all()
        fm = pd.DataFrame(records); oof[model_name] = {'score': score_oof, 'pred': pred_oof, 'fold': oof_fold}
        fold_frames[model_name] = fm
        quantum_summary_rows.append({'dataset': dataset, 'model': model_name, 'n_folds': len(fm),
                                     **summarize_fold_metrics(fm),
                                     'wall_seconds_15fold': time.perf_counter() - started,
                                     'train_full_gram_pairs': train_pairs, 'test_gram_pairs_one_pass': test_pairs})
        for i in range(len(yd)):
            quantum_oof_rows.append({'dataset': dataset, 'model': model_name, 'row_index': i,
                                     'fold': int(oof_fold[i]), 'target': int(yd.iloc[i]),
                                     'prediction': int(pred_oof[i]), 'score': float(score_oof[i])})
        print(f"{dataset:10s} | {model_name:39s} | sensitivity={fm.sensitivity.mean():.4f} ± {fm.sensitivity.std(ddof=1):.4f}")
    assert np.allclose(front_end_examples['q'], front_end_examples['c'], atol=1e-10), 'PCA/angle front ends are not aligned'
    print('  Identical imputer → scaler → PCA-3 → angle scaling verified on the test fold.')
    q, c = oof['Quantum kernel SVM (PCA-3)'], oof['RBF-SVM (matched PCA-3)']
    true = yd.to_numpy()
    b = int(((q['pred'] == true) & (c['pred'] != true)).sum())
    c_discord = int(((q['pred'] != true) & (c['pred'] == true)).sum())
    exact_p = float(binomtest(min(b, c_discord), b + c_discord, 0.5, alternative='two-sided').pvalue) if b + c_discord else 1.0
    abs_loss_diff = np.abs(true - q['score']) - np.abs(true - c['score'])
    w_stat, w_p = signed_rank_p(abs_loss_diff)
    fq = fold_frames['Quantum kernel SVM (PCA-3)'].sensitivity.to_numpy()[:5]
    fc = fold_frames['RBF-SVM (matched PCA-3)'].sensitivity.to_numpy()[:5]
    fw_stat, fw_p = signed_rank_p(fq - fc)
    paired_rows.append({'dataset': dataset, 'n_rows': len(yd), 'mcnemar_b_quantum_only_correct': b,
                        'mcnemar_c_classical_only_correct': c_discord, 'mcnemar_exact_p': exact_p,
                        'wilcoxon_abs_error_stat': w_stat, 'wilcoxon_abs_error_p': w_p,
                        'mean_abs_error_difference_q_minus_c': float(abs_loss_diff.mean()),
                        'wilcoxon_5fold_sensitivity_stat': fw_stat, 'wilcoxon_5fold_sensitivity_p': fw_p})

quantum_multi = pd.DataFrame(quantum_summary_rows)
paired_multi = pd.DataFrame(paired_rows)
for col in ['mcnemar_exact_p', 'wilcoxon_abs_error_p', 'wilcoxon_5fold_sensitivity_p']:
    paired_multi[col + '_holm4'] = holm_adjust(paired_multi[col].to_numpy())
quantum_multi.to_csv(RESULTS / 'exp13_multi_dataset_quantum_benchmark.csv', index=False)
paired_multi.to_csv(RESULTS / 'exp13b_quantum_paired_tests.csv', index=False)
pd.DataFrame(quantum_fold_rows).to_csv(RESULTS / 'exp13c_quantum_fold_metrics.csv', index=False)
pd.DataFrame(quantum_oof_rows).to_csv(RESULTS / 'exp13d_quantum_oof.csv', index=False)
assert len(quantum_multi) == 16 and (quantum_multi.n_folds == 15).all() and len(paired_multi) == 4
print('\nOverall quantum/classical benchmark:'); display(quantum_multi)
print('\nPaired exploratory tests (OOF, with Holm correction within each test family):'); display(paired_multi)
````


**Output (stdout):**

````text
Fidelity checks: unit norm, symmetric PSD Gram, unit diagonal, and analytic closed form: PASS
````


**Output (stdout):**

````text
WDBC       | Quantum kernel SVM (PCA-3)              | sensitivity=0.8932 ± 0.0465
````


**Output (stdout):**

````text
WDBC       | RBF-SVM (matched PCA-3)                 | sensitivity=0.8994 ± 0.0390
````


**Output (stdout):**

````text
WDBC       | RBF-SVM (all features)                  | sensitivity=0.9606 ± 0.0317
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Logistic regression L2 (all features)   | sensitivity=0.9512 ± 0.0371
  Identical imputer → scaler → PCA-3 → angle scaling verified on the test fold.
````


**Output (stdout):**

````text
WPBC       | Quantum kernel SVM (PCA-3)              | sensitivity=0.0000 ± 0.0000
````


**Output (stdout):**

````text
WPBC       | RBF-SVM (matched PCA-3)                 | sensitivity=0.0074 ± 0.0287
````


**Output (stdout):**

````text
WPBC       | RBF-SVM (all features)                  | sensitivity=0.0141 ± 0.0372
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Logistic regression L2 (all features)   | sensitivity=0.1919 ± 0.1561
  Identical imputer → scaler → PCA-3 → angle scaling verified on the test fold.
````


**Output (stdout):**

````text
Coimbra    | Quantum kernel SVM (PCA-3)              | sensitivity=0.7551 ± 0.1777
````


**Output (stdout):**

````text
Coimbra    | RBF-SVM (matched PCA-3)                 | sensitivity=0.7821 ± 0.1793
````


**Output (stdout):**

````text
Coimbra    | RBF-SVM (all features)                  | sensitivity=0.7667 ± 0.0875
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Coimbra    | Logistic regression L2 (all features)   | sensitivity=0.7560 ± 0.0865
  Identical imputer → scaler → PCA-3 → angle scaling verified on the test fold.
````


**Output (stdout):**

````text
Parkinsons | Quantum kernel SVM (PCA-3)              | sensitivity=0.9909 ± 0.0157
````


**Output (stdout):**

````text
Parkinsons | RBF-SVM (matched PCA-3)                 | sensitivity=0.9749 ± 0.0242
````


**Output (stdout):**

````text
Parkinsons | RBF-SVM (all features)                  | sensitivity=0.9954 ± 0.0121
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Logistic regression L2 (all features)   | sensitivity=0.9320 ± 0.0385
  Identical imputer → scaler → PCA-3 → angle scaling verified on the test fold.

Overall quantum/classical benchmark:
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Displayed output:**

````text
       dataset                                  model  n_folds  sensitivity_mean  sensitivity_std  specificity_mean  specificity_std  precision_mean  precision_std  f1_mean  f1_std  balanced_accuracy_mean  balanced_accuracy_std  mcc_mean  mcc_std  roc_auc_mean  roc_auc_std  pr_auc_mean  pr_auc_std  fn_mean  fn_std  fp_mean  fp_std  wall_seconds_15fold  train_full_gram_pairs  test_gram_pairs_one_pass
0         WDBC             Quantum kernel SVM (PCA-3)       15            0.8932           0.0465            0.9795           0.0165          0.9640         0.0264   0.9262  0.0230                  0.9363                 0.0206    0.8880   0.0326        0.9905       0.0045       0.9869      0.0054   4.5333  1.9952   1.4667  1.1872               3.9183                3108108                    777024
1         WDBC                RBF-SVM (matched PCA-3)       15            0.8994           0.0390            0.9645           0.0257          0.9397         0.0378   0.9182  0.0245                  0.9320                 0.0198    0.8728   0.0379        0.9880       0.0072       0.9840      0.0080   4.2667  1.6676   2.5333  1.8465               0.4313                      0                         0
2         WDBC                 RBF-SVM (all features)       15            0.9606           0.0317            0.9850           0.0124          0.9748         0.0206   0.9674  0.0195                  0.9728                 0.0171    0.9488   0.0306        0.9953       0.0049       0.9939      0.0059   1.6667  1.3452   1.0667  0.8837               0.6011                      0                         0
3         WDBC  Logistic regression L2 (all features)       15            0.9512           0.0371            0.9907           0.0087          0.9841         0.0145   0.9669  0.0188                  0.9709                 0.0176    0.9490   0.0275        0.9946       0.0049       0.9933      0.0054   2.0667  1.5796   0.6667  0.6172               0.3297                      0                         0
4         WPBC             Quantum kernel SVM (PCA-3)       15            0.0000           0.0000            1.0000           0.0000          0.0000         0.0000   0.0000  0.0000                  0.5000                 0.0000    0.0000   0.0000        0.5756       0.1194       0.3931      0.1176   9.4000  0.5071   0.0000  0.0000               1.6311                 376362                     94086
5         WPBC                RBF-SVM (matched PCA-3)       15            0.0074           0.0287            0.9822           0.0375          0.0133         0.0516   0.0095  0.0369                  0.4948                 0.0100   -0.0224   0.0430        0.5071       0.1173       0.3525      0.1152   9.3333  0.6172   0.5333  1.1255               0.4088                      0                         0
6         WPBC                 RBF-SVM (all features)       15            0.0141           0.0372            0.9845           0.0305          0.0556         0.1500   0.0222  0.0586                  0.4993                 0.0176   -0.0028   0.0626        0.6139       0.1073       0.4029      0.1018   9.2667  0.5936   0.4667  0.9155               0.3823                      0                         0
7         WPBC  Logistic regression L2 (all features)       15            0.1919           0.1561            0.9115           0.0484          0.3853         0.2612   0.2474  0.1820                  0.5517                 0.0856    0.1314   0.2155        0.6577       0.1141       0.4177      0.1366   7.6000  1.5492   2.6667  1.4475               0.3148                      0                         0
8      Coimbra             Quantum kernel SVM (PCA-3)       15            0.7551           0.1777            0.5218           0.2534          0.6886         0.1162   0.6968  0.0651                  0.6385                 0.0632    0.3178   0.1363        0.7447       0.1201       0.8148      0.0917   3.1333  2.2949   5.0000  2.6992               1.0840                 129180                     32292
9      Coimbra                RBF-SVM (matched PCA-3)       15            0.7821           0.1793            0.5570           0.2004          0.7035         0.0984   0.7229  0.0713                  0.6695                 0.0650    0.3837   0.1484        0.7229       0.0952       0.7802      0.0902   2.8000  2.3053   4.6000  2.0633               0.3242                      0                         0
10     Coimbra                 RBF-SVM (all features)       15            0.7667           0.0875            0.7170           0.1637          0.7842         0.1026   0.7686  0.0560                  0.7418                 0.0770    0.4917   0.1532        0.8485       0.0557       0.8871      0.0407   3.0000  1.1339   2.9333  1.7099               0.2794                      0                         0
11     Coimbra  Logistic regression L2 (all features)       15            0.7560           0.0865            0.7297           0.1850          0.7932         0.1188   0.7671  0.0672                  0.7428                 0.0943    0.4924   0.1880        0.8051       0.1103       0.8464      0.0981   3.1333  1.1255   2.8000  1.9346               0.4117                      0                         0
12  Parkinsons             Quantum kernel SVM (PCA-3)       15            0.9909           0.0157            0.3807           0.1677          0.8308         0.0463   0.9032  0.0297                  0.6858                 0.0863    0.5221   0.1686        0.8251       0.1085       0.9286      0.0487   0.2667  0.4577   6.0000  1.8127               1.6132                 365040                     91260
13  Parkinsons                RBF-SVM (matched PCA-3)       15            0.9749           0.0242            0.4637           0.1855          0.8482         0.0525   0.9065  0.0363                  0.7193                 0.0987    0.5484   0.1912        0.8781       0.0803       0.9555      0.0313   0.7333  0.7037   5.2000  1.9346               0.3830                      0                         0
14  Parkinsons                 RBF-SVM (all features)       15            0.9954           0.0121            0.5356           0.2170          0.8687         0.0608   0.9268  0.0366                  0.7655                 0.1108    0.6592   0.1792        0.8866       0.0675       0.9583      0.0250   0.1333  0.3519   4.5333  2.2318               0.4037                      0                         0
15  Parkinsons  Logistic regression L2 (all features)       15            0.9320           0.0385            0.6119           0.2101          0.8818         0.0623   0.9046  0.0353                  0.7719                 0.1034    0.5818   0.1739        0.8955       0.0556       0.9641      0.0201   2.0000  1.1339   3.8000  2.1448               0.3590                      0                         0
````


**Output (stdout):**

````text

Paired exploratory tests (OOF, with Holm correction within each test family):
````


**Displayed output:**

````text
      dataset  n_rows  mcnemar_b_quantum_only_correct  mcnemar_c_classical_only_correct  mcnemar_exact_p  wilcoxon_abs_error_stat  wilcoxon_abs_error_p  mean_abs_error_difference_q_minus_c  wilcoxon_5fold_sensitivity_stat  wilcoxon_5fold_sensitivity_p  mcnemar_exact_p_holm4  wilcoxon_abs_error_p_holm4  wilcoxon_5fold_sensitivity_p_holm4
0        WDBC     569                               4                                 3           1.0000               69509.0000                0.0032                              -0.0037                           0.0000                        0.5000                 1.0000                      0.0095                              1.0000
1        WPBC     198                               2                                 0           0.5000                8504.0000                0.0953                              -0.0038                           0.0000                        1.0000                 1.0000                      0.1907                              1.0000
2     Coimbra     116                               2                                 6           0.2891                3342.0000                0.8883                               0.0072                           0.0000                        0.5000                 1.0000                      0.8883                              1.0000
3  Parkinsons     195                               1                                 3           0.6250                5520.0000                0.0000                               0.0263                           0.0000                        1.0000                 1.0000                      0.0000                              1.0000
````

**Interpretation.** `Quantum versus matched RBF mean sensitivity: WDBC 0.8932 vs 0.8994 (exact McNemar p=1.0000); WPBC 0.0000 vs 0.0074 (exact McNemar p=0.5000); Coimbra 0.7551 vs 0.7821 (exact McNemar p=0.2891); Parkinsons 0.9909 vs 0.9749 (exact McNemar p=0.6250).` A small $p$-value here would not establish clinical utility or quantum computational advantage; a large $p$-value is not equivalence. For Parkinson's, row-level paired tests are not patient-independent. Training angles are mapped into $[0,\pi]$; held-out values can fall outside that interval because clipping is not used.

## 26. Supplementary Parkinson audit: grouped holdout

`StratifiedGroupKFold(5, shuffle=True, random_state=42)` keeps all recordings with the same observed identifier stem on one side of each split. For each held-out stem, probabilities are averaged and thresholded at 0.5; pooled recording-level and stem-level metrics are exported for 11 classical models and the two PCA-3 controls. Some small folds contain a single class, so sensitivity and ROC-AUC are evaluated on pooled OOF predictions rather than averaged from undefined per-fold scores.

The internal probability-calibration step of SVC is not group-aware; this grouped outer split is an exploratory audit, not fully nested subject-independent calibration.


<!-- Notebook code cell 80; execution count 27 -->

````python
# Supplement: grouped CV uses recording-name stems as a conservative grouping proxy.
dp = DATASETS['Parkinsons']; Xp, yp = dp['X'], dp['y']; gp = np.asarray(dp['groups'])
group_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
group_splits = list(group_cv.split(Xp, yp, groups=gp))
for tr, te in group_splits:
    assert set(gp[tr]).isdisjoint(set(gp[te]))
print('Group-safe test-fold class counts (small folds can contain a single class):')
print([dict(yp.iloc[te].value_counts().sort_index()) for _, te in group_splits])

group_models = {name: Pipeline([('imputer', SimpleImputer(strategy='median')),
                                 ('scaler', StandardScaler()), ('model', clone(est))])
                for name, est in zoo.items()}
matched_models = matched_quantum_models()
group_models['Quantum kernel SVM (PCA-3)'] = matched_models['Quantum kernel SVM (PCA-3)']
group_models['RBF-SVM (matched PCA-3)'] = matched_models['RBF-SVM (matched PCA-3)']
group_summary_rows, group_oof_rows = [], []
for model_name, estimator in group_models.items():
    scores = np.full(len(yp), np.nan)
    for fold, (tr, te) in enumerate(group_splits, start=1):
        fitted = clone(estimator).fit(Xp.iloc[tr], yp.iloc[tr])
        scores[te] = fitted.predict_proba(Xp.iloc[te])[:, 1]
    assert np.isfinite(scores).all()
    # A prespecified 0.5 threshold is applied to both recording and mean stem-level probabilities.
    record_metrics = metrics_from_preds(yp.to_numpy(), (scores >= 0.5).astype(int), scores)
    group_frame = pd.DataFrame({'subject_stem': gp, 'target': yp.to_numpy(), 'score': scores})
    subject = group_frame.groupby('subject_stem', sort=True).agg(target=('target', 'first'), score=('score', 'mean')).reset_index()
    assert len(subject) == len(np.unique(gp)) and set(subject.target) == {0, 1}
    subject['prediction'] = (subject.score >= 0.5).astype(int)
    subject_metrics = metrics_from_preds(subject.target.to_numpy(), subject.prediction.to_numpy(), subject.score.to_numpy())
    group_summary_rows.append({'model': model_name, 'n_recordings': len(yp), 'n_identifier_stems': len(subject),
                               'n_positive_stems': int(subject.target.sum()), 'n_negative_stems': int((1-subject.target).sum()),
                               'recording_sensitivity_pooled': record_metrics['sensitivity'],
                               'recording_specificity_pooled': record_metrics['specificity'],
                               'recording_mcc_pooled': record_metrics['mcc'],
                               'stem_sensitivity_pooled': subject_metrics['sensitivity'],
                               'stem_specificity_pooled': subject_metrics['specificity'],
                               'stem_precision_pooled': subject_metrics['precision'],
                               'stem_f1_pooled': subject_metrics['f1'], 'stem_mcc_pooled': subject_metrics['mcc'],
                               'stem_roc_auc_pooled': subject_metrics['roc_auc'],
                               'stem_pr_auc_pooled': subject_metrics['pr_auc'],
                               'stem_fn': subject_metrics['fn'], 'stem_fp': subject_metrics['fp']})
    group_oof_rows.extend([{'model': model_name, **r} for r in subject.to_dict('records')])
    print(f"{model_name:39s} | stem sensitivity={subject_metrics['sensitivity']:.4f} "
          f"specificity={subject_metrics['specificity']:.4f} MCC={subject_metrics['mcc']:.4f}")

group_audit = pd.DataFrame(group_summary_rows)
group_audit.to_csv(RESULTS / 'exp14_parkinsons_group_aware.csv', index=False)
pd.DataFrame(group_oof_rows).to_csv(RESULTS / 'exp14b_parkinsons_group_oof.csv', index=False)
assert len(group_audit) == 13 and (group_audit.n_identifier_stems == 32).all()
display(group_audit)
````


**Output (stdout):**

````text
Group-safe test-fold class counts (small folds can contain a single class):
[{1: np.int64(38)}, {0: np.int64(24), 1: np.int64(18)}, {0: np.int64(12), 1: np.int64(25)}, {0: np.int64(6), 1: np.int64(36)}, {0: np.int64(6), 1: np.int64(30)}]
````


**Output (stdout):**

````text
Logistic Regression (L2)                | stem sensitivity=0.9583 specificity=0.5000 MCC=0.5466
````


**Output (stdout):**

````text
Logistic Regression (L1)                | stem sensitivity=0.9583 specificity=0.5000 MCC=0.5466
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Linear Discriminant (LDA)               | stem sensitivity=0.9167 specificity=0.5000 MCC=0.4623
SVM (linear kernel)                     | stem sensitivity=0.9583 specificity=0.5000 MCC=0.5466
````


**Output (stdout):**

````text
SVM (RBF kernel)                        | stem sensitivity=0.9583 specificity=0.5000 MCC=0.5466
````


**Output (stdout):**

````text
k-Nearest Neighbours                    | stem sensitivity=0.9583 specificity=0.3750 MCC=0.4364
Gaussian Naive Bayes                    | stem sensitivity=0.6667 specificity=0.8750 MCC=0.4700
````


**Output (stdout):**

````text
Random Forest                           | stem sensitivity=1.0000 specificity=0.2500 MCC=0.4472
````


**Output (stdout):**

````text
Extra Trees                             | stem sensitivity=1.0000 specificity=0.3750 MCC=0.5571
````


**Output (stdout):**

````text
Gradient Boosting                       | stem sensitivity=1.0000 specificity=0.3750 MCC=0.5571
````


**Output (stdout):**

````text
HistGradient Boosting                   | stem sensitivity=1.0000 specificity=0.3750 MCC=0.5571
````


**Output (stdout):**

````text
Quantum kernel SVM (PCA-3)              | stem sensitivity=0.9583 specificity=0.5000 MCC=0.5466
RBF-SVM (matched PCA-3)                 | stem sensitivity=0.9583 specificity=0.5000 MCC=0.5466
````


**Displayed output:**

````text
                         model  n_recordings  n_identifier_stems  n_positive_stems  n_negative_stems  recording_sensitivity_pooled  recording_specificity_pooled  recording_mcc_pooled  stem_sensitivity_pooled  stem_specificity_pooled  stem_precision_pooled  stem_f1_pooled  stem_mcc_pooled  stem_roc_auc_pooled  stem_pr_auc_pooled  stem_fn  stem_fp
0     Logistic Regression (L2)           195                  32                24                 8                        0.9116                        0.4792                0.4338                   0.9583                   0.5000                 0.8519          0.9020           0.5466               0.7292              0.8976        1        4
1     Logistic Regression (L1)           195                  32                24                 8                        0.8912                        0.5208                0.4355                   0.9583                   0.5000                 0.8519          0.9020           0.5466               0.7448              0.9039        1        4
2    Linear Discriminant (LDA)           195                  32                24                 8                        0.8844                        0.5208                0.4246                   0.9167                   0.5000                 0.8462          0.8800           0.4623               0.6823              0.8208        2        4
3          SVM (linear kernel)           195                  32                24                 8                        0.8844                        0.4583                0.3690                   0.9583                   0.5000                 0.8519          0.9020           0.5466               0.7031              0.8772        1        4
4             SVM (RBF kernel)           195                  32                24                 8                        0.8844                        0.4792                0.3878                   0.9583                   0.5000                 0.8519          0.9020           0.5466               0.6406              0.8069        1        4
5         k-Nearest Neighbours           195                  32                24                 8                        0.8912                        0.3333                0.2611                   0.9583                   0.3750                 0.8214          0.8846           0.4364               0.7422              0.8704        1        5
6         Gaussian Naive Bayes           195                  32                24                 8                        0.6395                        0.7917                0.3723                   0.6667                   0.8750                 0.9412          0.7805           0.4700               0.7891              0.8787        8        1
7                Random Forest           195                  32                24                 8                        0.9388                        0.3542                0.3712                   1.0000                   0.2500                 0.8000          0.8889           0.4472               0.7786              0.9206        0        6
8                  Extra Trees           195                  32                24                 8                        0.9524                        0.4167                0.4603                   1.0000                   0.3750                 0.8276          0.9057           0.5571               0.7552              0.9109        0        5
9            Gradient Boosting           195                  32                24                 8                        0.9252                        0.3125                0.3012                   1.0000                   0.3750                 0.8276          0.9057           0.5571               0.6354              0.8426        0        5
10       HistGradient Boosting           195                  32                24                 8                        0.9728                        0.3750                0.4735                   1.0000                   0.3750                 0.8276          0.9057           0.5571               0.8073              0.9279        0        5
11  Quantum kernel SVM (PCA-3)           195                  32                24                 8                        0.9388                        0.4583                0.4678                   0.9583                   0.5000                 0.8519          0.9020           0.5466               0.7135              0.8546        1        4
12     RBF-SVM (matched PCA-3)           195                  32                24                 8                        0.9252                        0.4167                0.4027                   0.9583                   0.5000                 0.8519          0.9020           0.5466               0.6771              0.8129        1        4
````

**Interpretation.** `For logistic regression, pooled stem-level sensitivity is 0.9583, specificity is 0.5000, and MCC is 0.5466 across 32 observed stems.` Identifier stems are a conservative grouping proxy, not a verified patient registry. The 32 observed stems and the source's description of 31 people are inconsistent; actual participant links must be resolved before publication-level subject-independent claims.

Grouped metrics use probabilities thresholded at 0.5; in SVC this can differ from `predict()`, which the row-wise zoo reports. Hence the difference combines grouping and decision-rule changes.

## 26A. Experiment 15: the original unmatched PCA-3 control on all four datasets

This extends the **original** `ClassicalPCA3SVM` (`StandardScaler → PCA(3) → RBF-SVC`, without angular scaling). A training-fold median imputer is added in front because WPBC has missing predictors; on the three complete datasets it is a no-op. Repeated CV uses 15 folds and records every fold; its first five folds form the single OOF partition. For WDBC all first-five-fold numeric metrics and every OOF score are asserted against the original reference. This is distinct from experiment 13's **matched** PCA-3 comparator and should not be interpreted as a matched quantum comparison.


<!-- Notebook code cell 83; execution count 28 -->

````python
# Re-run the original unmatched classical PCA-3 comparator on every dataset.
# The median imputer is a no-op for WDBC, Coimbra and Parkinson; WPBC needs it.
legacy_pca_summary, legacy_pca_folds, legacy_pca_oof = [], [], []
for dataset, dset in DATASETS.items():
    Xd, yd = dset['X'], dset['y']
    repeated = list(CV_REPEATED.split(Xd, yd))
    single = list(CV_OOF.split(Xd, yd))
    assert all(np.array_equal(repeated[i][0], single[i][0]) and
               np.array_equal(repeated[i][1], single[i][1]) for i in range(5))
    scores_oof = np.full(len(yd), np.nan)
    preds_oof = np.full(len(yd), -1, dtype=int)
    folds_oof = np.full(len(yd), -1, dtype=int)
    records = []
    for fold, (tr, te) in enumerate(repeated, start=1):
        legacy = Pipeline([('imputer', SimpleImputer(strategy='median')),
                           ('legacy_model', ClassicalPCA3SVM(C=1.0))])
        legacy.fit(Xd.iloc[tr], yd.iloc[tr])
        pred = legacy.predict(Xd.iloc[te]); score = legacy.predict_proba(Xd.iloc[te])[:, 1]
        metrics = metrics_from_preds(yd.iloc[te].to_numpy(), pred, score)
        records.append(metrics)
        legacy_pca_folds.append({'dataset': dataset, 'fold': fold, **metrics})
        if fold <= 5:
            scores_oof[te], preds_oof[te], folds_oof[te] = score, pred, fold
    assert np.isfinite(scores_oof).all() and (preds_oof >= 0).all() and (folds_oof >= 1).all()
    fm = pd.DataFrame(records)
    legacy_pca_summary.append({'dataset': dataset, 'model': 'Original RBF-SVM PCA-3 (no angle scaling)',
                               'n_folds': len(fm), 'n_rows': len(yd),
                               'n_features': Xd.shape[1], **summarize_fold_metrics(fm)})
    legacy_pca_oof.extend([{'dataset': dataset, 'row_index': i, 'fold': int(folds_oof[i]),
                            'target': int(yd.iloc[i]), 'prediction': int(preds_oof[i]),
                            'score': float(scores_oof[i])} for i in range(len(yd))])
    if dataset == 'WDBC':
        # The first repeat is the original shuffled five-fold split. On WDBC the
        # added imputer has no missing values to fill. Verify every numeric score
        # from the exact original control, rather than merely matching its name.
        ref = pd.read_csv(RESULTS / 'exp6_quantum_vs_classical.csv')
        ref = ref.loc[ref.model == 'Classical RBF-SVM (same PCA-3 front end)'].iloc[0]
        for ref_col, fold_col in [('sensitivity', 'sensitivity'), ('precision', 'precision'),
                                   ('specificity', 'specificity'), ('f1', 'f1'),
                                   ('balanced_acc', 'balanced_accuracy'), ('mcc', 'mcc'),
                                   ('roc_auc', 'roc_auc'), ('pr_auc', 'pr_auc'),
                                   ('fn', 'fn'), ('fp', 'fp')]:
            np.testing.assert_allclose(fm[fold_col].iloc[:5].mean(), ref[ref_col],
                                       rtol=1e-8, atol=1e-10, err_msg=ref_col)
        np.testing.assert_allclose(fm.sensitivity.iloc[:5].std(ddof=1), ref.sensitivity_sd,
                                   rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(scores_oof, quantum_curves['Classical RBF-SVM (same PCA-3 front end)'],
                                   rtol=1e-8, atol=1e-10)
        print('WDBC: all original five-fold PCA-3 metrics and every OOF score reproduced: PASS')
    print(f"{dataset:10s} | legacy PCA-3: sensitivity {fm.sensitivity.mean():.4f} ± "
          f"{fm.sensitivity.std(ddof=1):.4f}, specificity {fm.specificity.mean():.4f}, "
          f"ROC-AUC {fm.roc_auc.mean():.4f}")
legacy_pca = pd.DataFrame(legacy_pca_summary)
legacy_pca.to_csv(RESULTS / 'exp15_legacy_pca3_all_datasets.csv', index=False)
pd.DataFrame(legacy_pca_folds).to_csv(RESULTS / 'exp15b_legacy_pca3_fold_metrics.csv', index=False)
pd.DataFrame(legacy_pca_oof).to_csv(RESULTS / 'exp15c_legacy_pca3_oof.csv', index=False)
assert len(legacy_pca) == 4 and len(legacy_pca_folds) == 60 and len(legacy_pca_oof) == 1078
display(legacy_pca)
````


**Output (stdout):**

````text
WDBC: all original five-fold PCA-3 metrics and every OOF score reproduced: PASS
WDBC       | legacy PCA-3: sensitivity 0.8915 ± 0.0450, specificity 0.9636, ROC-AUC 0.9875
````


**Output (stdout):**

````text
WPBC       | legacy PCA-3: sensitivity 0.0074 ± 0.0287, specificity 0.9822, ROC-AUC 0.5296
````


**Output (stdout):**

````text
Coimbra    | legacy PCA-3: sensitivity 0.7509 ± 0.1414, specificity 0.5952, ROC-AUC 0.7226
````


**Output (stdout):**

````text
Parkinsons | legacy PCA-3: sensitivity 0.9841 ± 0.0176, specificity 0.4289, ROC-AUC 0.8153
````


**Displayed output:**

````text
      dataset                                      model  n_folds  n_rows  n_features  sensitivity_mean  sensitivity_std  specificity_mean  specificity_std  precision_mean  precision_std  f1_mean  f1_std  balanced_accuracy_mean  balanced_accuracy_std  mcc_mean  mcc_std  roc_auc_mean  roc_auc_std  pr_auc_mean  pr_auc_std  fn_mean  fn_std  fp_mean  fp_std
0        WDBC  Original RBF-SVM PCA-3 (no angle scaling)       15     569          30            0.8915           0.0450            0.9636           0.0222          0.9371         0.0338   0.9128  0.0278                  0.9276                 0.0232    0.8650   0.0421        0.9875       0.0083       0.9839      0.0082   4.6000  1.9198   2.6000  1.5946
1        WPBC  Original RBF-SVM PCA-3 (no angle scaling)       15     198          32            0.0074           0.0287            0.9822           0.0375          0.0133         0.0516   0.0095  0.0369                  0.4948                 0.0100   -0.0224   0.0430        0.5296       0.1059       0.3661      0.1132   9.3333  0.6172   0.5333  1.1255
2     Coimbra  Original RBF-SVM PCA-3 (no angle scaling)       15     116           9            0.7509           0.1414            0.5952           0.1984          0.7132         0.1122   0.7191  0.0757                  0.6730                 0.0912    0.3659   0.1881        0.7226       0.1029       0.7825      0.1002   3.2000  1.8205   4.2000  2.0424
3  Parkinsons  Original RBF-SVM PCA-3 (no angle scaling)       15     195          22            0.9841           0.0176            0.4289           0.1754          0.8412         0.0479   0.9064  0.0301                  0.7065                 0.0900    0.5437   0.1666        0.8153       0.1091       0.9122      0.0612   0.4667  0.5164   5.5333  1.8465
````

**Interpretation.** These are real fits and held-out scores, not estimates inferred from experiment 13. Each dataset has its own predictors and endpoint; Parkinson remains recording-wise. The median imputer is a necessary addition for WPBC, so that dataset is not a byte-identical preprocessing pipeline.

## 26B. Experiment 16: the original-style calibration architecture across datasets

The original WDBC pipeline fits scaling on the outer training fold before the sigmoid calibrator's internal five-way split. We preserve that architecture here for a historical comparison, adding a median imputer before the scaler for missing data. We apply it to all available features in all four datasets and additionally reproduce the original WDBC Worst-10 condition. Five-fold OOF Brier, log loss, ROC-AUC and thresholded metrics are exported; all WDBC Worst raw and calibrated OOF probabilities are checked against the original. **Prefer experiment 12:** its entire imputer–scaler–LR pipeline is nested within calibration folds. Outer test rows remain untouched in both designs, but this legacy design can leak preprocessing information across internal calibration splits.


<!-- Notebook code cell 86; execution count 29 -->

````python
# Reproduce the reference Platt architecture on all features of all datasets.
# WARNING: as in the original, preprocessing is fitted before the calibrator's
# internal splits (the outer held-out fold remains untouched). Prefer exp12 for
# the properly nested alternative. WDBC Worst is additionally run as an exact
# reference check; the imputer is necessary only for missing WPBC predictors.
legacy_cal_summary, legacy_cal_predictions = [], []
cal_cases = [(dataset, 'All features', dset['X'], dset['y']) for dataset, dset in DATASETS.items()]
cal_cases.append(('WDBC', 'Worst 10 reference', DATASETS['WDBC']['X'].loc[:, worst_cols], DATASETS['WDBC']['y']))
for dataset, feature_set, Xd, yd in cal_cases:
    raw = np.full(len(yd), np.nan); calibrated = np.full(len(yd), np.nan)
    folds = np.full(len(yd), -1, dtype=int)
    for fold, (tr, te) in enumerate(CV_OOF.split(Xd, yd), start=1):
        uncal = lr_pipeline().fit(Xd.iloc[tr], yd.iloc[tr])
        original_platt = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('cal', CalibratedClassifierCV(estimator=LogisticRegression(max_iter=2000, random_state=SEED),
                                           method='sigmoid', cv=5)),
        ])
        original_platt.fit(Xd.iloc[tr], yd.iloc[tr])
        raw[te] = uncal.predict_proba(Xd.iloc[te])[:, 1]
        calibrated[te] = original_platt.predict_proba(Xd.iloc[te])[:, 1]
        folds[te] = fold
    assert np.isfinite(raw).all() and np.isfinite(calibrated).all() and (folds >= 1).all()
    if dataset == 'WDBC' and feature_set == 'Worst 10 reference':
        np.testing.assert_allclose(raw, score_raw, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(calibrated, score_cal, rtol=1e-8, atol=1e-10)
        print('WDBC Worst: every original raw and Platt OOF score reproduced: PASS')
    for variant, scores in [('Uncalibrated', raw), ('Legacy Platt', calibrated)]:
        pred = (scores >= 0.5).astype(int)
        m = metrics_from_preds(yd.to_numpy(), pred, scores)
        legacy_cal_summary.append({'dataset': dataset, 'feature_set': feature_set,
                                   'variant': variant, 'n': len(yd), 'n_features': Xd.shape[1],
                                   'brier': brier_score_loss(yd, scores),
                                   'log_loss': log_loss(yd, scores, labels=[0, 1]),
                                   'roc_auc': m['roc_auc'], 'sensitivity_pooled_at_0_5': m['sensitivity'],
                                   'specificity_pooled_at_0_5': m['specificity'], 'mcc_pooled_at_0_5': m['mcc']})
        legacy_cal_predictions.extend([{'dataset': dataset, 'feature_set': feature_set,
                                       'variant': variant, 'row_index': i, 'fold': int(folds[i]),
                                       'target': int(yd.iloc[i]), 'score': float(scores[i]),
                                       'prediction_at_0_5': int(pred[i])} for i in range(len(yd))])
    print(f"{dataset:10s} | {feature_set:18s} | Brier raw={brier_score_loss(yd, raw):.4f}; "
          f"legacy Platt={brier_score_loss(yd, calibrated):.4f}")
legacy_calibration = pd.DataFrame(legacy_cal_summary)
legacy_calibration.to_csv(RESULTS / 'exp16_legacy_calibration.csv', index=False)
pd.DataFrame(legacy_cal_predictions).to_csv(RESULTS / 'exp16b_legacy_calibration_oof.csv', index=False)
assert len(legacy_calibration) == 10 and len(legacy_cal_predictions) == 3294
display(legacy_calibration)
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | All features       | Brier raw=0.0200; legacy Platt=0.0266
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | All features       | Brier raw=0.1846; legacy Platt=0.1764
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Coimbra    | All features       | Brier raw=0.1951; legacy Platt=0.2066
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | All features       | Brier raw=0.1024; legacy Platt=0.1185
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC Worst: every original raw and Platt OOF score reproduced: PASS
WDBC       | Worst 10 reference | Brier raw=0.0201; legacy Platt=0.0264
````


**Displayed output:**

````text
      dataset         feature_set       variant    n  n_features  brier  log_loss  roc_auc  sensitivity_pooled_at_0_5  specificity_pooled_at_0_5  mcc_pooled_at_0_5
0        WDBC        All features  Uncalibrated  569          30 0.0200    0.0764   0.9947                     0.9434                     0.9916             0.9437
1        WDBC        All features  Legacy Platt  569          30 0.0266    0.1098   0.9949                     0.9292                     0.9972             0.9404
2        WPBC        All features  Uncalibrated  198          32 0.1846    0.5554   0.6606                     0.2340                     0.9073             0.1810
3        WPBC        All features  Legacy Platt  198          32 0.1764    0.5324   0.6418                     0.0000                     0.9801            -0.0692
4     Coimbra        All features  Uncalibrated  116           9 0.1951    0.6157   0.7713                     0.7500                     0.6346             0.3872
5     Coimbra        All features  Legacy Platt  116           9 0.2066    0.6097   0.7584                     0.6562                     0.7500             0.4046
6  Parkinsons        All features  Uncalibrated  195          22 0.1024    0.3157   0.9001                     0.9388                     0.6458             0.6237
7  Parkinsons        All features  Legacy Platt  195          22 0.1185    0.3842   0.9026                     0.9796                     0.4375             0.5469
8        WDBC  Worst 10 reference  Uncalibrated  569          10 0.0201    0.0891   0.9917                     0.9623                     0.9888             0.9548
9        WDBC  Worst 10 reference  Legacy Platt  569          10 0.0264    0.1149   0.9920                     0.9434                     0.9944             0.9475
````

**Interpretation.** The legacy calibration run is for reproducing the source, not a recommended leakage-control protocol. Results for all-feature LR and WDBC Worst-10 have different inputs and should not be conflated. Parkinson's outer folds remain recording-wise.

## 26C. Experiment 17: the original nested threshold fallback across datasets

The original `choose_tau` maximizes precision subject to inner sensitivity ≥0.95 and, when no grid point qualifies, falls back to **minimum classification error**. We run that exact rule with fold-fitted imputation/scaling and L2 logistic regression on all available features for all four datasets. We additionally repeat original WDBC Worst-10 and assert every fold metric against the original. Experiment 11 instead uses a sensitivity-first fallback and can therefore yield different predictions; these are experimental policies for the same classifier, not new independent model architectures.


<!-- Notebook code cell 89; execution count 30 -->

````python
# Apply the original threshold rule to all features of every dataset; preserve
# its original fallback (minimum classification error if sensitivity >= 0.95
# is impossible). This differs from the revised fallback in exp11. WDBC Worst
# is also re-run to verify the exact reference experiment.
legacy_threshold_summary, legacy_threshold_folds, legacy_threshold_oof = [], [], []
threshold_cases = [(dataset, 'All features', dset['X'], dset['y']) for dataset, dset in DATASETS.items()]
threshold_cases.append(('WDBC', 'Worst 10 reference', DATASETS['WDBC']['X'].loc[:, worst_cols], DATASETS['WDBC']['y']))
for dataset, feature_set, Xd, yd in threshold_cases:
    scores_oof = np.full(len(yd), np.nan); preds_oof = np.full(len(yd), -1, dtype=int)
    taus_oof = np.full(len(yd), np.nan); folds_oof = np.full(len(yd), -1, dtype=int)
    rows_case = []
    for fold, (tr, te) in enumerate(CV_OOF.split(Xd, yd), start=1):
        Xt, yt = Xd.iloc[tr], yd.iloc[tr]
        inner_scores = np.full(len(tr), np.nan)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for itr, ite in inner.split(Xt, yt):
            p = lr_pipeline().fit(Xt.iloc[itr], yt.iloc[itr])
            inner_scores[ite] = p.predict_proba(Xt.iloc[ite])[:, 1]
        assert np.isfinite(inner_scores).all()
        original_grid = np.arange(0.05, 0.96, 0.01)
        feasible = any(np.mean(inner_scores[yt.to_numpy() == 1] >= t) >= 0.95 for t in original_grid)
        tau = choose_tau(yt.to_numpy(), inner_scores)
        fitted = lr_pipeline().fit(Xt, yt)
        score = fitted.predict_proba(Xd.iloc[te])[:, 1]
        pred = (score >= tau).astype(int)
        scores_oof[te], preds_oof[te], taus_oof[te], folds_oof[te] = score, pred, tau, fold
        metrics = metrics_from_preds(yd.iloc[te].to_numpy(), pred, score)
        row = {'dataset': dataset, 'feature_set': feature_set, 'fold': fold,
               'tau': tau, 'inner_target_feasible': int(feasible), **metrics}
        rows_case.append(row); legacy_threshold_folds.append(row)
    assert np.isfinite(scores_oof).all() and np.isfinite(taus_oof).all() and (preds_oof >= 0).all()
    ff = pd.DataFrame(rows_case)
    pooled = metrics_from_preds(yd.to_numpy(), preds_oof, scores_oof)
    legacy_threshold_summary.append({'dataset': dataset, 'feature_set': feature_set,
                                     'n_rows': len(yd), 'n_features': Xd.shape[1],
                                     'sensitivity_mean': ff.sensitivity.mean(),
                                     'sensitivity_std': ff.sensitivity.std(ddof=1),
                                     'sensitivity_pooled': pooled['sensitivity'],
                                     'specificity_pooled': pooled['specificity'],
                                     'precision_pooled': pooled['precision'], 'mcc_pooled': pooled['mcc'],
                                     'tp': pooled['tp'], 'tn': pooled['tn'], 'fp': pooled['fp'], 'fn': pooled['fn'],
                                     'tau_min': ff.tau.min(), 'tau_max': ff.tau.max(),
                                     'n_inner_feasible': int(ff.inner_target_feasible.sum())})
    legacy_threshold_oof.extend([{'dataset': dataset, 'feature_set': feature_set, 'row_index': i,
                                  'fold': int(folds_oof[i]), 'target': int(yd.iloc[i]),
                                  'score': float(scores_oof[i]), 'tau': float(taus_oof[i]),
                                  'prediction': int(preds_oof[i])} for i in range(len(yd))])
    if dataset == 'WDBC' and feature_set == 'Worst 10 reference':
        ref = pd.read_csv(RESULTS / 'exp4b_nested_threshold.csv').sort_values('fold')
        ff = ff.sort_values('fold')
        for col in ref.select_dtypes(include=[np.number]).columns:
            np.testing.assert_allclose(ff[col].to_numpy(), ref[col].to_numpy(),
                                       rtol=1e-8, atol=1e-10, err_msg=col)
        print('WDBC Worst: all original nested threshold fold metrics reproduced: PASS')
    print(f"{dataset:10s} | {feature_set:18s} | sensitivity={pooled['sensitivity']:.4f}, "
          f"specificity={pooled['specificity']:.4f}, feasible inner folds={int(ff.inner_target_feasible.sum())}/5")
legacy_threshold = pd.DataFrame(legacy_threshold_summary)
legacy_threshold.to_csv(RESULTS / 'exp17_legacy_threshold.csv', index=False)
pd.DataFrame(legacy_threshold_folds).to_csv(RESULTS / 'exp17b_legacy_threshold_folds.csv', index=False)
pd.DataFrame(legacy_threshold_oof).to_csv(RESULTS / 'exp17c_legacy_threshold_oof.csv', index=False)
assert len(legacy_threshold) == 5 and len(legacy_threshold_folds) == 25 and len(legacy_threshold_oof) == 1647
display(legacy_threshold)
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | All features       | sensitivity=0.9387, specificity=0.9860, feasible inner folds=5/5
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | All features       | sensitivity=0.3830, specificity=0.6291, feasible inner folds=2/5
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Coimbra    | All features       | sensitivity=0.9531, specificity=0.1731, feasible inner folds=5/5
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | All features       | sensitivity=0.9660, specificity=0.5417, feasible inner folds=5/5
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC Worst: all original nested threshold fold metrics reproduced: PASS
WDBC       | Worst 10 reference | sensitivity=0.9528, specificity=0.9860, feasible inner folds=5/5
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Displayed output:**

````text
      dataset         feature_set  n_rows  n_features  sensitivity_mean  sensitivity_std  sensitivity_pooled  specificity_pooled  precision_pooled  mcc_pooled   tp   tn  fp  fn  tau_min  tau_max  n_inner_feasible
0        WDBC        All features     569          30            0.9389           0.0634              0.9387              0.9860            0.9755      0.9322  199  352   5  13   0.3800   0.6600                 5
1        WPBC        All features     198          32            0.3956           0.4590              0.3830              0.6291            0.2432      0.0107   18   95  56  29   0.0600   0.7700                 2
2     Coimbra        All features     116           9            0.9526           0.0434              0.9531              0.1731            0.5865      0.2061   61    9  43   3   0.2000   0.2400                 5
3  Parkinsons        All features     195          22            0.9660           0.0339              0.9660              0.5417            0.8659      0.5981  142   26  22   5   0.2900   0.4300                 5
4        WDBC  Worst 10 reference     569          10            0.9530           0.0594              0.9528              0.9860            0.9758      0.9435  202  352   5  10   0.4100   0.5900                 5
````

**Interpretation.** The 0.95 constraint is set using training-fold predictions and is not a guarantee on held-out folds. Legacy fallback can sharply reduce sensitivity when the constraint is infeasible; compare experiment 11 and consult the number of feasible inner folds. Parkinson remains recording-wise and WPBC is not censoring-adjusted.

## 26D. Experiment 18: original three model families across available feature blocks

The original structural analysis crosses logistic L2, RBF-SVM and gradient boosting with Mean, Error, Worst and All on WDBC. We repeat and assert all twelve WDBC results, and run the three classifiers on the corresponding available blocks in WPBC and the three **different acoustic feature groups** plus All in Parkinson. Coimbra has nine predictors and no comparable nuclear 10-feature blocks, so only All is applicable there. This is 39 model–dataset–block combinations (585 fold fits), with median imputation, scaling and model training inside each fold. Parkinson's row-wise CV remains clustered.


<!-- Notebook code cell 92; execution count 31 -->

````python
# Extend the original three-model feature-block experiment where a block is
# defined. WDBC/WPBC nuclear Mean, Error, Worst correspond by naming; Parkinson
# blocks instead partition acoustic features and are NOT biological equivalents.
# Coimbra has only nine predictors and no analogous nuclear blocks, so only its
# full-feature condition is run. All preprocessing is fitted inside each fold.
block3_summary, block3_fold_rows = [], []
for dataset, dset in DATASETS.items():
    Xd, yd = dset['X'], dset['y']
    for block_name, cols in dset['blocks'].items():
        Xsub = Xd.loc[:, cols]
        for model_name, estimator in probe_models.items():
            records = []
            for fold, (tr, te) in enumerate(CV_REPEATED.split(Xsub, yd), start=1):
                pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                 ('scaler', StandardScaler()), ('model', clone(estimator))])
                pipe.fit(Xsub.iloc[tr], yd.iloc[tr])
                pred = pipe.predict(Xsub.iloc[te]); score = pipe.predict_proba(Xsub.iloc[te])[:, 1]
                met = metrics_from_preds(yd.iloc[te].to_numpy(), pred, score)
                records.append(met)
                block3_fold_rows.append({'dataset': dataset, 'feature_block': block_name,
                                         'model': model_name, 'fold': fold, **met})
            fm = pd.DataFrame(records)
            block3_summary.append({'dataset': dataset, 'feature_block': block_name,
                                   'model': model_name, 'n_features': len(cols), 'n_folds': len(fm),
                                   **summarize_fold_metrics(fm)})
            print(f"{dataset:10s} | {block_name:19s} | {model_name:24s} | "
                  f"sens={fm.sensitivity.mean():.4f}, spec={fm.specificity.mean():.4f}")
block3_multi = pd.DataFrame(block3_summary)
# All twelve WDBC block/model records must match the original reference table.
original_blocks = pd.read_csv(RESULTS / 'exp1_feature_blocks.csv')
column_map = {'sensitivity_mean': 'sensitivity_mean', 'sensitivity_sd': 'sensitivity_std',
              'precision_mean': 'precision_mean', 'specificity_mean': 'specificity_mean',
              'f1_mean': 'f1_mean', 'mcc_mean': 'mcc_mean', 'roc_auc_mean': 'roc_auc_mean',
              'fn_mean': 'fn_mean', 'fp_mean': 'fp_mean'}
for _, ref in original_blocks.iterrows():
    actual = block3_multi[(block3_multi.dataset == 'WDBC') &
                          (block3_multi.feature_block == ref.feature_block) &
                          (block3_multi.model == ref.model)]
    assert len(actual) == 1 and int(actual.n_features.iloc[0]) == int(ref.n_features)
    for old_col, new_col in column_map.items():
        np.testing.assert_allclose(actual[new_col].iloc[0], ref[old_col], rtol=1e-8, atol=1e-10,
                                   err_msg=f'{ref.feature_block}/{ref.model}/{old_col}')
print('WDBC: all 12 original three-model block experiments reproduced: PASS')
block3_multi.to_csv(RESULTS / 'exp18_3model_block_cross.csv', index=False)
pd.DataFrame(block3_fold_rows).to_csv(RESULTS / 'exp18b_3model_block_folds.csv', index=False)
assert len(block3_multi) == 39 and len(block3_fold_rows) == 585
display(block3_multi)
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Mean                | Logistic Regression (L2) | sens=0.8931, spec=0.9664
````


**Output (stdout):**

````text
WDBC       | Mean                | SVM (RBF)                | sens=0.8807, spec=0.9804
````


**Output (stdout):**

````text
WDBC       | Mean                | Gradient Boosting        | sens=0.9152, spec=0.9579
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Error               | Logistic Regression (L2) | sens=0.7547, spec=0.9421
````


**Output (stdout):**

````text
WDBC       | Error               | SVM (RBF)                | sens=0.7264, spec=0.9403
````


**Output (stdout):**

````text
WDBC       | Error               | Gradient Boosting        | sens=0.8162, spec=0.9272
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | Worst               | Logistic Regression (L2) | sens=0.9575, spec=0.9888
````


**Output (stdout):**

````text
WDBC       | Worst               | SVM (RBF)                | sens=0.9498, spec=0.9813
````


**Output (stdout):**

````text
WDBC       | Worst               | Gradient Boosting        | sens=0.9277, spec=0.9720
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WDBC       | All                 | Logistic Regression (L2) | sens=0.9512, spec=0.9907
````


**Output (stdout):**

````text
WDBC       | All                 | SVM (RBF)                | sens=0.9606, spec=0.9850
````


**Output (stdout):**

````text
WDBC       | All                 | Gradient Boosting        | sens=0.9262, spec=0.9767
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Mean                | Logistic Regression (L2) | sens=0.0659, spec=0.9889
````


**Output (stdout):**

````text
WPBC       | Mean                | SVM (RBF)                | sens=0.0644, spec=0.9867
````


**Output (stdout):**

````text
WPBC       | Mean                | Gradient Boosting        | sens=0.1785, spec=0.9006
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Error               | Logistic Regression (L2) | sens=0.0578, spec=0.9690
````


**Output (stdout):**

````text
WPBC       | Error               | SVM (RBF)                | sens=0.0074, spec=0.9867
````


**Output (stdout):**

````text
WPBC       | Error               | Gradient Boosting        | sens=0.1696, spec=0.8831
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | Worst               | Logistic Regression (L2) | sens=0.1007, spec=0.9622
````


**Output (stdout):**

````text
WPBC       | Worst               | SVM (RBF)                | sens=0.0074, spec=0.9844
````


**Output (stdout):**

````text
WPBC       | Worst               | Gradient Boosting        | sens=0.1148, spec=0.8786
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
WPBC       | All                 | Logistic Regression (L2) | sens=0.1919, spec=0.9115
````


**Output (stdout):**

````text
WPBC       | All                 | SVM (RBF)                | sens=0.0141, spec=0.9845
````


**Output (stdout):**

````text
WPBC       | All                 | Gradient Boosting        | sens=0.1719, spec=0.8830
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Coimbra    | All                 | Logistic Regression (L2) | sens=0.7560, spec=0.7297
````


**Output (stdout):**

````text
Coimbra    | All                 | SVM (RBF)                | sens=0.7667, spec=0.7170
````


**Output (stdout):**

````text
Coimbra    | All                 | Gradient Boosting        | sens=0.7769, spec=0.6727
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Vocal_Freq_Tone     | Logistic Regression (L2) | sens=0.9273, spec=0.4711
````


**Output (stdout):**

````text
Parkinsons | Vocal_Freq_Tone     | SVM (RBF)                | sens=0.9933, spec=0.3889
````


**Output (stdout):**

````text
Parkinsons | Vocal_Freq_Tone     | Gradient Boosting        | sens=0.9254, spec=0.6496
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Perturbation_Error  | Logistic Regression (L2) | sens=0.9366, spec=0.3941
````


**Output (stdout):**

````text
Parkinsons | Perturbation_Error  | SVM (RBF)                | sens=1.0000, spec=0.0000
````


**Output (stdout):**

````text
Parkinsons | Perturbation_Error  | Gradient Boosting        | sens=0.9070, spec=0.6237
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | Nonlinear_Extremes  | Logistic Regression (L2) | sens=0.9276, spec=0.6244
````


**Output (stdout):**

````text
Parkinsons | Nonlinear_Extremes  | SVM (RBF)                | sens=0.9661, spec=0.6437
````


**Output (stdout):**

````text
Parkinsons | Nonlinear_Extremes  | Gradient Boosting        | sens=0.9548, spec=0.7126
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stderr):**

````text
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
/usr/local/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
  opt_res = optimize.minimize(
````


**Output (stdout):**

````text
Parkinsons | All                 | Logistic Regression (L2) | sens=0.9320, spec=0.6119
````


**Output (stdout):**

````text
Parkinsons | All                 | SVM (RBF)                | sens=0.9954, spec=0.5356
````


**Output (stdout):**

````text
Parkinsons | All                 | Gradient Boosting        | sens=0.9570, spec=0.7800
WDBC: all 12 original three-model block experiments reproduced: PASS
````


**Displayed output:**

````text
       dataset       feature_block                     model  n_features  n_folds  sensitivity_mean  sensitivity_std  specificity_mean  specificity_std  precision_mean  precision_std  f1_mean  f1_std  balanced_accuracy_mean  balanced_accuracy_std  mcc_mean  mcc_std  roc_auc_mean  roc_auc_std  pr_auc_mean  pr_auc_std  fn_mean  fn_std  fp_mean  fp_std
0         WDBC                Mean  Logistic Regression (L2)          10       15            0.8931           0.0538            0.9664           0.0138          0.9417         0.0207   0.9154  0.0225                  0.9298                 0.0215    0.8701   0.0289        0.9858       0.0050       0.9804      0.0062   4.5333  2.2949   2.4000  0.9856
1         WDBC                Mean                 SVM (RBF)          10       15            0.8807           0.0559            0.9804           0.0188          0.9655         0.0319   0.9197  0.0287                  0.9306                 0.0254    0.8797   0.0400        0.9867       0.0074       0.9833      0.0073   5.0667  2.4044   1.4000  1.3522
2         WDBC                Mean         Gradient Boosting          10       15            0.9152           0.0536            0.9579           0.0300          0.9312         0.0445   0.9215  0.0288                  0.9366                 0.0243    0.8776   0.0436        0.9856       0.0051       0.9807      0.0056   3.6000  2.2928   3.0000  2.1381
3         WDBC               Error  Logistic Regression (L2)          10       15            0.7547           0.0593            0.9421           0.0204          0.8868         0.0338   0.8139  0.0374                  0.8484                 0.0281    0.7246   0.0476        0.9426       0.0129       0.9235      0.0157  10.4000  2.5014   4.1333  1.4573
4         WDBC               Error                 SVM (RBF)          10       15            0.7264           0.0462            0.9403           0.0312          0.8809         0.0538   0.7950  0.0385                  0.8334                 0.0280    0.6994   0.0579        0.9388       0.0127       0.9169      0.0185  11.6000  1.9567   4.2667  2.2509
5         WDBC               Error         Gradient Boosting          10       15            0.8162           0.0403            0.9272           0.0321          0.8717         0.0513   0.8420  0.0341                  0.8717                 0.0263    0.7548   0.0553        0.9495       0.0152       0.9354      0.0165   7.8000  1.7403   5.2000  2.3053
6         WDBC               Worst  Logistic Regression (L2)          10       15            0.9575           0.0379            0.9888           0.0121          0.9813         0.0198   0.9687  0.0182                  0.9732                 0.0172    0.9516   0.0273        0.9925       0.0077       0.9925      0.0071   1.8000  1.6125   0.8000  0.8619
7         WDBC               Worst                 SVM (RBF)          10       15            0.9498           0.0364            0.9813           0.0102          0.9684         0.0168   0.9585  0.0185                  0.9655                 0.0174    0.9351   0.0285        0.9948       0.0057       0.9936      0.0057   2.1333  1.5523   1.3333  0.7237
8         WDBC               Worst         Gradient Boosting          10       15            0.9277           0.0400            0.9720           0.0183          0.9526         0.0302   0.9393  0.0238                  0.9499                 0.0204    0.9052   0.0366        0.9899       0.0074       0.9876      0.0077   3.0667  1.7099   2.0000  1.3093
9         WDBC                 All  Logistic Regression (L2)          30       15            0.9512           0.0371            0.9907           0.0087          0.9841         0.0145   0.9669  0.0188                  0.9709                 0.0176    0.9490   0.0275        0.9946       0.0049       0.9933      0.0054   2.0667  1.5796   0.6667  0.6172
10        WDBC                 All                 SVM (RBF)          30       15            0.9606           0.0317            0.9850           0.0124          0.9748         0.0206   0.9674  0.0195                  0.9728                 0.0171    0.9488   0.0306        0.9953       0.0049       0.9939      0.0059   1.6667  1.3452   1.0667  0.8837
11        WDBC                 All         Gradient Boosting          30       15            0.9262           0.0506            0.9767           0.0195          0.9604         0.0322   0.9420  0.0295                  0.9514                 0.0253    0.9104   0.0437        0.9925       0.0052       0.9901      0.0065   3.1333  2.1668   1.6667  1.3973
12        WPBC                Mean  Logistic Regression (L2)          10       15            0.0659           0.0815            0.9889           0.0349          0.4444         0.4987   0.1119  0.1332                  0.5274                 0.0457    0.1336   0.1825        0.5893       0.1313       0.4284      0.1314   8.8000  1.0823   0.3333  1.0465
13        WPBC                Mean                 SVM (RBF)          10       15            0.0644           0.0689            0.9867           0.0211          0.4000         0.4355   0.1077  0.1114                  0.5256                 0.0315    0.1140   0.1425        0.5795       0.1166       0.4496      0.1161   8.8000  0.8619   0.4000  0.6325
14        WPBC                Mean         Gradient Boosting          10       15            0.1785           0.0874            0.9006           0.0468          0.3945         0.2301   0.2356  0.1063                  0.5396                 0.0522    0.1125   0.1453        0.5230       0.0957       0.3498      0.0854   7.7333  1.0328   3.0000  1.4142
15        WPBC               Error  Logistic Regression (L2)          10       15            0.0578           0.0700            0.9690           0.0320          0.2822         0.3612   0.0908  0.1039                  0.5134                 0.0315    0.0525   0.1357        0.5532       0.0869       0.3500      0.0912   8.8667  0.9155   0.9333  0.9612
16        WPBC               Error                 SVM (RBF)          10       15            0.0074           0.0287            0.9867           0.0303          0.0222         0.0861   0.0111  0.0430                  0.4970                 0.0150   -0.0118   0.0505        0.5220       0.1553       0.3085      0.1022   9.3333  0.6172   0.4000  0.9103
17        WPBC               Error         Gradient Boosting          10       15            0.1696           0.0874            0.8831           0.0465          0.3262         0.1869   0.2190  0.1114                  0.5264                 0.0574    0.0721   0.1514        0.5372       0.1118       0.3232      0.0906   7.8000  0.8619   3.5333  1.4075
18        WPBC               Worst  Logistic Regression (L2)          10       15            0.1007           0.1051            0.9622           0.0353          0.3944         0.3759   0.1540  0.1514                  0.5315                 0.0564    0.1092   0.1871        0.5797       0.1259       0.3789      0.1094   8.4667  1.1872   1.1333  1.0601
19        WPBC               Worst                 SVM (RBF)          10       15            0.0074           0.0287            0.9844           0.0375          0.0133         0.0516   0.0095  0.0369                  0.4959                 0.0095   -0.0163   0.0387        0.4291       0.1059       0.2723      0.0718   9.3333  0.6172   0.4667  1.1255
20        WPBC               Worst         Gradient Boosting          10       15            0.1148           0.1212            0.8786           0.0788          0.2661         0.3132   0.1466  0.1475                  0.4967                 0.0734    0.0021   0.2037        0.5723       0.1083       0.3252      0.1073   8.3333  1.2910   3.6667  2.3805
21        WPBC                 All  Logistic Regression (L2)          32       15            0.1919           0.1561            0.9115           0.0484          0.3853         0.2612   0.2474  0.1820                  0.5517                 0.0856    0.1314   0.2155        0.6577       0.1141       0.4177      0.1366   7.6000  1.5492   2.6667  1.4475
22        WPBC                 All                 SVM (RBF)          32       15            0.0141           0.0372            0.9845           0.0305          0.0556         0.1500   0.0222  0.0586                  0.4993                 0.0176   -0.0028   0.0626        0.6139       0.1073       0.4029      0.1018   9.2667  0.5936   0.4667  0.9155
23        WPBC                 All         Gradient Boosting          32       15            0.1719           0.1612            0.8830           0.0587          0.2593         0.1830   0.1965  0.1570                  0.5274                 0.0645    0.0495   0.1542        0.5507       0.1155       0.3182      0.0717   7.8000  1.6125   3.5333  1.7674
24     Coimbra                 All  Logistic Regression (L2)           9       15            0.7560           0.0865            0.7297           0.1850          0.7932         0.1188   0.7671  0.0672                  0.7428                 0.0943    0.4924   0.1880        0.8051       0.1103       0.8464      0.0981   3.1333  1.1255   2.8000  1.9346
25     Coimbra                 All                 SVM (RBF)           9       15            0.7667           0.0875            0.7170           0.1637          0.7842         0.1026   0.7686  0.0560                  0.7418                 0.0770    0.4917   0.1532        0.8485       0.0557       0.8871      0.0407   3.0000  1.1339   2.9333  1.7099
26     Coimbra                 All         Gradient Boosting           9       15            0.7769           0.1034            0.6727           0.1460          0.7524         0.0924   0.7598  0.0754                  0.7248                 0.0890    0.4586   0.1788        0.8139       0.0944       0.8533      0.0827   2.8667  1.3558   3.4000  1.5492
27  Parkinsons     Vocal_Freq_Tone  Logistic Regression (L2)           5       15            0.9273           0.0564            0.4711           0.1997          0.8438         0.0565   0.8822  0.0444                  0.6992                 0.1036    0.4569   0.1934        0.7818       0.0934       0.9157      0.0471   2.1333  1.6417   5.1333  2.0656
28  Parkinsons     Vocal_Freq_Tone                 SVM (RBF)           5       15            0.9933           0.0140            0.3889           0.1913          0.8334         0.0504   0.9055  0.0297                  0.6911                 0.0951    0.5346   0.1663        0.8233       0.0784       0.9327      0.0313   0.2000  0.4140   5.9333  2.0517
29  Parkinsons     Vocal_Freq_Tone         Gradient Boosting           5       15            0.9254           0.0696            0.6496           0.1458          0.8903         0.0438   0.9059  0.0424                  0.7875                 0.0771    0.6126   0.1539        0.9068       0.0626       0.9632      0.0305   2.2000  2.0424   3.4000  1.5024
30  Parkinsons  Perturbation_Error  Logistic Regression (L2)          11       15            0.9366           0.0423            0.3941           0.1759          0.8263         0.0495   0.8770  0.0340                  0.6654                 0.0900    0.4043   0.1832        0.8279       0.0571       0.9417      0.0233   1.8667  1.2459   5.8667  1.8465
31  Parkinsons  Perturbation_Error                 SVM (RBF)          11       15            1.0000           0.0000            0.0000           0.0000          0.7538         0.0130   0.8596  0.0084                  0.5000                 0.0000    0.0000   0.0000        0.7994       0.0717       0.9149      0.0414   0.0000  0.0000   9.6000  0.5071
32  Parkinsons  Perturbation_Error         Gradient Boosting          11       15            0.9070           0.0568            0.6237           0.1914          0.8813         0.0593   0.8926  0.0467                  0.7653                 0.1042    0.5501   0.1942        0.8634       0.0669       0.9448      0.0355   2.7333  1.6676   3.6667  1.9518
33  Parkinsons  Nonlinear_Extremes  Logistic Regression (L2)           6       15            0.9276           0.0354            0.6244           0.1792          0.8843         0.0543   0.9041  0.0285                  0.7760                 0.0857    0.5853   0.1389        0.8970       0.0511       0.9647      0.0184   2.1333  1.0601   3.6667  1.8387
34  Parkinsons  Nonlinear_Extremes                 SVM (RBF)           6       15            0.9661           0.0338            0.6437           0.1588          0.8931         0.0459   0.9272  0.0277                  0.8049                 0.0779    0.6801   0.1246        0.9280       0.0440       0.9732      0.0183   1.0000  1.0000   3.4667  1.6417
35  Parkinsons  Nonlinear_Extremes         Gradient Boosting           6       15            0.9548           0.0275            0.7126           0.1707          0.9120         0.0490   0.9318  0.0237                  0.8337                 0.0796    0.7079   0.1082        0.9223       0.0496       0.9671      0.0271   1.3333  0.8165   2.8000  1.7403
36  Parkinsons                 All  Logistic Regression (L2)          22       15            0.9320           0.0385            0.6119           0.2101          0.8818         0.0623   0.9046  0.0353                  0.7719                 0.1034    0.5818   0.1739        0.8955       0.0556       0.9641      0.0201   2.0000  1.1339   3.8000  2.1448
37  Parkinsons                 All                 SVM (RBF)          22       15            0.9954           0.0121            0.5356           0.2170          0.8687         0.0608   0.9268  0.0366                  0.7655                 0.1108    0.6592   0.1792        0.8866       0.0675       0.9583      0.0250   0.1333  0.3519   4.5333  2.2318
38  Parkinsons                 All         Gradient Boosting          22       15            0.9570           0.0415            0.7800           0.1331          0.9305         0.0410   0.9429  0.0323                  0.8685                 0.0732    0.7643   0.1378        0.9612       0.0398       0.9869      0.0137   1.2667  1.2228   2.1333  1.3020
````

**Interpretation.** Shared names do not make Parkinson acoustic groups biologically equivalent to WDBC/WPBC morphology blocks. Coimbra cannot support the original `k=10` selector because it has only nine predictors. Do not claim that every WDBC-specific ten-feature experiment exists identically on all datasets.

## 26E. Experiment 19: audit actual Parkinson row-wise overlap

On the exact 15 repeated row-wise splits used for the model zoo, count test-recording appearances whose recording-name stem also occurs in training. Each stem is only a proxy for a participant; it is not a verified identifier. Each recording is tested three times, so summed appearances do not represent distinct people.


<!-- Notebook code cell 95; execution count 32 -->

````python
# Quantify actual training/test identifier-stem overlap in the 15-fold
# row-wise Parkinson benchmark. Stems are unverified grouping proxies; counts
# are repeated test APPEARANCES, not a count of distinct individuals.
parkinson_data = DATASETS['Parkinsons']
X_overlap, y_overlap = parkinson_data['X'], parkinson_data['y']
overlap_stems = np.asarray(parkinson_data['groups'])
parkinson_overlap_rows = []
for fold, (tr, te) in enumerate(CV_REPEATED.split(X_overlap, y_overlap), start=1):
    train_stems, test_stems = set(overlap_stems[tr]), set(overlap_stems[te])
    parkinson_overlap_rows.append({'fold': fold, 'repeat': (fold - 1) // 5 + 1,
                                   'n_test_recordings': len(te),
                                   'n_unique_test_stems': len(test_stems),
                                   'n_shared_stems': len(train_stems & test_stems),
                                   'n_test_recordings_with_stem_in_training':
                                       int(sum(s in train_stems for s in overlap_stems[te]))})
parkinson_overlap = pd.DataFrame(parkinson_overlap_rows)
parkinson_overlap.to_csv(RESULTS / 'exp19_parkinsons_split_overlap.csv', index=False)
assert len(parkinson_overlap) == 15
assert (parkinson_overlap.n_test_recordings == 39).all()
assert (parkinson_overlap.n_test_recordings_with_stem_in_training == 39).all()
assert parkinson_overlap.n_test_recordings_with_stem_in_training.sum() == 585
print('Training/test identifier-stem overlap, 15 repeated folds:')
display(parkinson_overlap)
print('585 / 585 repeated test-recording appearances share a name stem with training;')
print('this establishes proxy overlap, not verified person identity or its quantitative effect on metrics.')
````


**Output (stdout):**

````text
Training/test identifier-stem overlap, 15 repeated folds:
````


**Displayed output:**

````text
    fold  repeat  n_test_recordings  n_unique_test_stems  n_shared_stems  n_test_recordings_with_stem_in_training
0      1       1                 39                   25              25                                       39
1      2       1                 39                   26              26                                       39
2      3       1                 39                   23              23                                       39
3      4       1                 39                   26              26                                       39
4      5       1                 39                   23              23                                       39
5      6       2                 39                   22              22                                       39
6      7       2                 39                   22              22                                       39
7      8       2                 39                   25              25                                       39
8      9       2                 39                   23              23                                       39
9     10       2                 39                   21              21                                       39
10    11       3                 39                   26              26                                       39
11    12       3                 39                   23              23                                       39
12    13       3                 39                   21              21                                       39
13    14       3                 39                   24              24                                       39
14    15       3                 39                   22              22                                       39
````


**Output (stdout):**

````text
585 / 585 repeated test-recording appearances share a name stem with training;
this establishes proxy overlap, not verified person identity or its quantitative effect on metrics.
````

**Interpretation.** All 585 repeated test appearances have a matching name stem in the respective training fold. This documents overlap by the proxy, not its quantitative inflation or an independently verified number of individuals. The outer group-held-out audit is reported separately above.

## 26F. Experiment 20: machine-audited source and model coverage

The included user-supplied original notebook is checked against its SHA-256. Abstract syntax trees of all 11 zoo estimator definitions, both quantum/PCA-3 classifier classes, the quantum functions, and original calibration and threshold functions are compared with the master notebook. The outputs are then checked for all 11 original classical models, the simulated quantum model, the original unmatched PCA-3 comparator, the original-style Platt architecture and the additional matched comparator on every dataset. This is model and specified variant coverage; it does not imply that every WDBC-specific analysis can be identically applied to all four heterogeneous datasets.


<!-- Notebook code cell 98; execution count 33 -->

````python
# Machine-check that the source supplied for this project is the exact original,
# that its estimator definitions survived unchanged, and that every model or
# comparator has measured outputs for all four datasets.
import ast
import nbformat

original_file = Path('original/PAPER_FULL_REPRODUCTION.ipynb')
assert sha256_file(original_file) == '22ed184c75799a1e0436b661c276e0a7c392eefa34cf0ae2da360c090c8e1ad2'
original_nb = nbformat.read(original_file, as_version=4)
master_nb = nbformat.read(Path('MASTER_REPRODUCTION.ipynb'), as_version=4)

def find_source_ast(nb, name, kind):
    found = []
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        for node in tree.body:
            if kind == 'assignment' and isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    found.append(node.value)
            elif kind == 'class' and isinstance(node, ast.ClassDef) and node.name == name:
                found.append(node)
            elif kind == 'function' and isinstance(node, ast.FunctionDef) and node.name == name:
                found.append(node)
    if name == 'cnot' and kind == 'function':
        # The original later redefines cnot in the figure cell; verify both.
        assert len(found) == 2, (name, kind, len(found))
        return ast.Module(body=found, type_ignores=[])
    assert len(found) == 1, (name, kind, len(found))
    return found[0]

original_zoo_ast = find_source_ast(original_nb, 'zoo', 'assignment')
master_zoo_ast = find_source_ast(master_nb, 'zoo', 'assignment')
assert ast.dump(original_zoo_ast, include_attributes=False) == ast.dump(master_zoo_ast, include_attributes=False)
assert {ast.literal_eval(k) for k in original_zoo_ast.keys} == set(zoo)
for name in ['QuantumKernelSVM', 'ClassicalPCA3SVM']:
    assert ast.dump(find_source_ast(original_nb, name, 'class'), include_attributes=False) == \
           ast.dump(find_source_ast(master_nb, name, 'class'), include_attributes=False), name
for name in ['cnot', 'quantum_state', 'fidelity_kernel_matrix', 'choose_tau', 'oof_scores']:
    assert ast.dump(find_source_ast(original_nb, name, 'function'), include_attributes=False) == \
           ast.dump(find_source_ast(master_nb, name, 'function'), include_attributes=False), name

zoo_folds_df = pd.DataFrame(zoo_fold_rows)
quantum_folds_df = pd.DataFrame(quantum_fold_rows)
legacy_folds_df = pd.DataFrame(legacy_pca_folds)
cal_oof_df = pd.DataFrame(legacy_cal_predictions)
coverage_rows = []
for dataset, dset in DATASETS.items():
    for name in zoo:
        f = zoo_folds_df[(zoo_folds_df.dataset == dataset) & (zoo_folds_df.model == name)]
        assert sorted(f.fold.tolist()) == list(range(1, 16))
        coverage_rows.append({'dataset': dataset, 'model_or_variant': name,
                              'origin': 'original 11-model zoo', 'n_folds': 15,
                              'results_file': 'exp10_multi_dataset_zoo.csv', 'status': 'PASS'})
    for name, origin in [('Quantum kernel SVM (PCA-3)', 'original quantum estimator'),
                         ('RBF-SVM (matched PCA-3)', 'additional matched comparator')]:
        f = quantum_folds_df[(quantum_folds_df.dataset == dataset) & (quantum_folds_df.model == name)]
        assert sorted(f.fold.tolist()) == list(range(1, 16))
        coverage_rows.append({'dataset': dataset, 'model_or_variant': name, 'origin': origin,
                              'n_folds': 15, 'results_file': 'exp13_multi_dataset_quantum_benchmark.csv',
                              'status': 'PASS'})
    f = legacy_folds_df[legacy_folds_df.dataset == dataset]
    assert sorted(f.fold.tolist()) == list(range(1, 16))
    coverage_rows.append({'dataset': dataset, 'model_or_variant': 'Original RBF-SVM PCA-3 (no angle scaling)',
                          'origin': 'original unmatched PCA-3 comparator', 'n_folds': 15,
                          'results_file': 'exp15_legacy_pca3_all_datasets.csv', 'status': 'PASS'})
    f = cal_oof_df[(cal_oof_df.dataset == dataset) & (cal_oof_df.feature_set == 'All features') &
                   (cal_oof_df.variant == 'Legacy Platt')]
    assert len(f) == len(dset['y']) and not f.row_index.duplicated().any()
    assert sorted(f.fold.unique().tolist()) == list(range(1, 6))
    coverage_rows.append({'dataset': dataset, 'model_or_variant': 'Original-style Platt-calibrated L2 LR',
                          'origin': 'original calibration architecture (all features)', 'n_folds': 5,
                          'results_file': 'exp16_legacy_calibration.csv', 'status': 'PASS'})
coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(RESULTS / 'exp20_model_coverage_audit.csv', index=False)
assert len(coverage) == 60 and coverage.groupby('dataset').size().eq(15).all()
assert (coverage.status == 'PASS').all()
print('Original source hash, 11 estimator AST definitions, quantum/PCA-3 classes,')
print('quantum functions, original threshold and calibration functions: PASS')
print('All four datasets have 11 original estimators, original quantum, original')
print('unmatched PCA-3 comparator, legacy Platt architecture, and matched comparator: PASS')
display(coverage.pivot(index='model_or_variant', columns='dataset', values='status'))
````


**Output (stdout):**

````text
Original source hash, 11 estimator AST definitions, quantum/PCA-3 classes,
quantum functions, original threshold and calibration functions: PASS
All four datasets have 11 original estimators, original quantum, original
unmatched PCA-3 comparator, legacy Platt architecture, and matched comparator: PASS
````


**Displayed output:**

````text
dataset                                   Coimbra Parkinsons  WDBC  WPBC
model_or_variant                                                        
Extra Trees                                  PASS       PASS  PASS  PASS
Gaussian Naive Bayes                         PASS       PASS  PASS  PASS
Gradient Boosting                            PASS       PASS  PASS  PASS
HistGradient Boosting                        PASS       PASS  PASS  PASS
Linear Discriminant (LDA)                    PASS       PASS  PASS  PASS
Logistic Regression (L1)                     PASS       PASS  PASS  PASS
Logistic Regression (L2)                     PASS       PASS  PASS  PASS
Original RBF-SVM PCA-3 (no angle scaling)    PASS       PASS  PASS  PASS
Original-style Platt-calibrated L2 LR        PASS       PASS  PASS  PASS
Quantum kernel SVM (PCA-3)                   PASS       PASS  PASS  PASS
RBF-SVM (matched PCA-3)                      PASS       PASS  PASS  PASS
Random Forest                                PASS       PASS  PASS  PASS
SVM (RBF kernel)                             PASS       PASS  PASS  PASS
SVM (linear kernel)                          PASS       PASS  PASS  PASS
k-Nearest Neighbours                         PASS       PASS  PASS  PASS
````

**Interpretation.** Coverage means executable definitions and held-out records exist. It does not guarantee clinical performance, statistical equivalence, patient independence or successful external validation. The original reference uses WDBC alone; the expanded four-dataset work is a separate extension.

## 26G. Experiment 21: learning-curve training-prefix audit

Reconstruct the class prevalence of the reference learning-curve training prefixes without fitting additional models. The outer split is stratified, but the `learning_curve` defaults do not shuffle each training prefix. The resulting source-faithful curve is retained as a reproducibility artifact; these class-composition measurements show why its apparent trend needs caution.


<!-- Notebook code cell 101; execution count 34 -->

````python
# Audit the original learning_curve training prefixes. Its default shuffle=False
# takes prefixes of each outer training index array; stratification of the outer
# split does not make those prefixes stratified.
learning_audit_rows = []
Xd, yd = DATASETS['WDBC']['X'].loc[:, worst_cols], DATASETS['WDBC']['y']
for size in pd.read_csv(RESULTS / 'exp8_learning_curve.csv').n_train.astype(int):
    for fold, (tr, te) in enumerate(CV_OOF.split(Xd, yd), start=1):
        prefix = tr[:size]
        learning_audit_rows.append({'n_train': int(size), 'fold': fold,
                                    'positive_count': int(yd.iloc[prefix].sum()),
                                    'positive_fraction': float(yd.iloc[prefix].mean()),
                                    'full_train_positive_fraction': float(yd.iloc[tr].mean())})
learning_audit = pd.DataFrame(learning_audit_rows)
learning_audit.to_csv(RESULTS / 'exp21_learning_curve_sampling_audit.csv', index=False)
smallest = learning_audit[learning_audit.n_train == learning_audit.n_train.min()]
assert len(learning_audit) == 30 and len(smallest) == 5
assert smallest.positive_fraction.min() > 0.63 and smallest.positive_fraction.max() < 0.71
print('Original unshuffled prefixes have a distorted malignant prevalence at n=68:')
display(smallest)
print('The reference learning curve is kept for reproducibility, not as a trustworthy learning trend.')
````


**Output (stdout):**

````text
Original unshuffled prefixes have a distorted malignant prevalence at n=68:
````


**Displayed output:**

````text
   n_train  fold  positive_count  positive_fraction  full_train_positive_fraction
0       68     1              47             0.6912                        0.3714
1       68     2              48             0.7059                        0.3714
2       68     3              47             0.6912                        0.3736
3       68     4              47             0.6912                        0.3736
4       68     5              43             0.6324                        0.3728
````


**Output (stdout):**

````text
The reference learning curve is kept for reproducibility, not as a trustworthy learning trend.
````

**Interpretation.** The 68-row training prefixes contain approximately 63–71% malignant cases, compared with approximately 37% in the corresponding full training folds. A new stratified/shuffled subsampling protocol would be required for a more interpretable learning curve.

## 27. Journal figures across datasets

The heatmap displays two distinct quantities on their own scales: 15-fold mean sensitivity and MCC. The grouped bar chart shows sensitivity $\pm$ fold SD; reliability diagrams use the same OOF predictions as the calibration table; the DCA panels use the computed clinical net-benefit curves. The DCA display clips values below $-0.25$ for readability; the CSV retains the full numerical range. All figures are saved as vector PDF and 300-dpi PNG.


<!-- Notebook code cell 104; execution count 35 -->

````python
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Two metric-specific heatmaps; colors are comparable only within a metric.
fig, axes = plt.subplots(1, 2, figsize=(12.3, 6.7), constrained_layout=True)
for ax, metric, title, lo, hi, cmap in [
    (axes[0], 'sensitivity_mean', 'Sensitivity', 0, 1, 'Blues'),
    (axes[1], 'mcc_mean', 'Matthews correlation coefficient', -1, 1, 'RdBu')]:
    matrix = multi_zoo.pivot(index='model', columns='dataset', values=metric).loc[list(zoo), list(DATASETS)]
    im = ax.pcolormesh(np.arange(matrix.shape[1] + 1) - 0.5,
                       np.arange(matrix.shape[0] + 1) - 0.5, matrix.to_numpy(),
                       vmin=lo, vmax=hi, cmap=cmap, shading='flat', rasterized=False)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index if ax is axes[0] else [])
    ax.set_title(title, fontweight='bold'); ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if (metric == 'sensitivity_mean' and val > 0.65) or (metric == 'mcc_mean' and abs(val) > 0.65) else '#17202a')
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.solids.set_rasterized(False)
savefig2(fig, 'fig_multi_zoo_heatmap')

fig, ax = plt.subplots(figsize=(9.3, 4.9), constrained_layout=True)
models = list(matched_quantum_models())
colors = ['#5a4d9c', '#e18d33', '#2d7b8a', '#788c43']
x = np.arange(len(DATASETS)); width = .19
for j, (model, color) in enumerate(zip(models, colors)):
    rows = quantum_multi.set_index(['dataset', 'model']).loc[[(name, model) for name in DATASETS]]
    ax.bar(x + (j - 1.5) * width, rows.sensitivity_mean, width,
           yerr=rows.sensitivity_std, capsize=2.3, color=color, edgecolor='white', linewidth=.5,
           error_kw={'elinewidth': .8, 'ecolor': '#333333'}, label=model)
ax.set_xticks(x, list(DATASETS)); ax.set_ylim(0, 1.18)
ax.set_ylabel('Sensitivity (mean ± SD; 15 folds)'); ax.set_title('Quantum and classical models on identical folds', fontweight='bold')
ax.legend(fontsize=7.7, ncol=2, loc='upper center', frameon=False)
ax.spines[['top', 'right']].set_visible(False); ax.yaxis.grid(True, alpha=.18); ax.set_axisbelow(True)
savefig2(fig, 'fig_multi_quantum_benchmark')

fig, axes = plt.subplots(2, 2, figsize=(8.3, 7.5), constrained_layout=True)
for ax, dataset in zip(axes.flat, DATASETS):
    for variant, color, marker in [('uncalibrated', '#2475a6', 'o'), ('Platt-calibrated', '#d05a42', 's')]:
        b = reliability_multi[(reliability_multi.dataset == dataset) & (reliability_multi.variant == variant)]
        score = multi_calibration.query('dataset == @dataset and variant == @variant').brier.iloc[0]
        ax.plot(b.mean_predicted, b.observed_rate, marker=marker, ms=3.2, lw=1.2, color=color,
                label=f'{variant} (Brier {score:.3f})')
    ax.plot([0, 1], [0, 1], color='#777', linestyle=':', lw=.9, label='Ideal')
    ax.set_title(dataset, fontweight='bold'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed positive fraction')
    ax.legend(fontsize=7, frameon=False, loc='upper left'); ax.grid(alpha=.16)
savefig2(fig, 'fig_multi_calibration')

fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.4), constrained_layout=True)
for ax, dataset in zip(axes.flat, DATASETS):
    d = dca_multi[dca_multi.dataset == dataset]
    ax.plot(d.threshold, d.net_benefit_raw, color='#2475a6', lw=1.5, label='LR: raw')
    ax.plot(d.threshold, d.net_benefit_platt, color='#d05a42', lw=1.3, ls='--', label='LR: Platt')
    ax.plot(d.threshold, d.net_benefit_treat_all, color='#666666', lw=1.0, ls=':', label='Treat all')
    ax.plot(d.threshold, d.net_benefit_treat_none, color='#1f1f1f', lw=.8, label='Treat none')
    ax.set_title(dataset, fontweight='bold'); ax.set_xlim(.05, .80)
    ax.set_ylim(-.25, max(.25, d[['net_benefit_raw', 'net_benefit_platt', 'net_benefit_treat_all']].to_numpy().max() + .06))
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
    ax.legend(fontsize=7, frameon=False, loc='lower left'); ax.grid(alpha=.16)
fig.suptitle('Descriptive DCA: sample prevalence; y < −0.25 clipped (full curves in CSV)', fontsize=9)
savefig2(fig, 'fig_multi_dca')
print('Wrote four multi-dataset figures as vector PDF and 300-dpi PNG.')
````


**Output (stdout):**

````text
Wrote four multi-dataset figures as vector PDF and 300-dpi PNG.
````

**Interpretation.** The graphics are descriptive summaries of internally validated estimates. Error bars are fold SDs, not confidence intervals, and the DCA curves should not be treated as population clinical recommendations.

## 28. Inline figure gallery

Each named figure is displayed directly from its 300-dpi PNG; its vector counterpart has the same stem in `figures/`. The eight WDBC variants reproduce the named reference views, followed by four multi-dataset panels. This cell contains no statistical fitting.


<!-- Notebook code cell 107; execution count 36 -->

````python
WDBC_FIGURES = ['fig_blocks', 'fig_models', 'fig_models2', 'fig_threshold', 'fig_threshold2', 'fig_reliability', 'fig_kernel', 'fig_learning']
MULTI_FIGURES = ['fig_multi_zoo_heatmap', 'fig_multi_quantum_benchmark', 'fig_multi_calibration', 'fig_multi_dca']
for name in WDBC_FIGURES + MULTI_FIGURES:
    print('FIGURE:', name)
    display(Image(filename=str(FIG / (name + '.png')), width=750))
````


**Output (stdout):**

````text
FIGURE: fig_blocks
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 1](report_assets/cell_107_output_001.png)


**Output (stdout):**

````text
FIGURE: fig_models
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 3](report_assets/cell_107_output_003.png)


**Output (stdout):**

````text
FIGURE: fig_models2
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 5](report_assets/cell_107_output_005.png)


**Output (stdout):**

````text
FIGURE: fig_threshold
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 7](report_assets/cell_107_output_007.png)


**Output (stdout):**

````text
FIGURE: fig_threshold2
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 9](report_assets/cell_107_output_009.png)


**Output (stdout):**

````text
FIGURE: fig_reliability
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 11](report_assets/cell_107_output_011.png)


**Output (stdout):**

````text
FIGURE: fig_kernel
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 13](report_assets/cell_107_output_013.png)


**Output (stdout):**

````text
FIGURE: fig_learning
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 15](report_assets/cell_107_output_015.png)


**Output (stdout):**

````text
FIGURE: fig_multi_zoo_heatmap
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 17](report_assets/cell_107_output_017.png)


**Output (stdout):**

````text
FIGURE: fig_multi_quantum_benchmark
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 19](report_assets/cell_107_output_019.png)


**Output (stdout):**

````text
FIGURE: fig_multi_calibration
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 21](report_assets/cell_107_output_021.png)


**Output (stdout):**

````text
FIGURE: fig_multi_dca
````


**Displayed output:**

````text
<IPython.core.display.Image object>
````


![Notebook figure, cell 107 output 23](report_assets/cell_107_output_023.png)

**Interpretation.** Each plot is generated from the run's model outputs or checked kernel. Inspect the accompanying source CSVs and the stated axis limits before extracting a manuscript claim.

## 29. Complete primary tables at full precision

The following cell prints every row and column of the main experiment CSVs, including all threshold and decision-curve rows. Numeric values are formatted with up to 17 significant digits; companion fold-level, prediction-level and selected-feature files are available in `results/` for deeper auditing. The printed tables are descriptive outputs of the same execution, not substituted or invented values.


<!-- Notebook code cell 110; execution count 37 -->

````python
PRIMARY_TABLES = [
    'exp1_feature_blocks.csv', 'exp2_architectures.csv', 'exp2_model_zoo.csv', 'exp3_feature_selection.csv',
    'exp4a_threshold_curve.csv', 'exp4b_nested_threshold.csv', 'exp5a_calibration.csv', 'exp5b_reliability.csv',
    'exp5c_bootstrap_ci.csv', 'exp6_quantum_vs_classical.csv', 'exp7_mcnemar.csv', 'exp7_bootstrap_diff.csv',
    'exp8_learning_curve.csv', 'wdbc_numeric_reproducibility.csv', 'dataset_overlap_audit.csv', 'exp9_multi_dataset_blocks.csv',
    'exp10_multi_dataset_zoo.csv', 'exp11_multi_dataset_nested_threshold.csv',
    'exp12_multi_dataset_calibration.csv', 'exp12b_dca_curves.csv',
    'exp13_multi_dataset_quantum_benchmark.csv', 'exp13b_quantum_paired_tests.csv',
    'exp14_parkinsons_group_aware.csv',
    'exp15_legacy_pca3_all_datasets.csv',
    'exp16_legacy_calibration.csv',
    'exp17_legacy_threshold.csv',
    'exp18_3model_block_cross.csv',
    'exp19_parkinsons_split_overlap.csv',
    'exp20_model_coverage_audit.csv',
    'exp21_learning_curve_sampling_audit.csv']
for filename in PRIMARY_TABLES:
    frame = pd.read_csv(RESULTS / filename)
    print('\n' + '='*24 + ' ' + filename + ' ' + '='*24)
    print(frame.to_string(index=False, float_format=lambda x: format(x, '.17g')))
````


**Output (stdout):**

````text

======================== exp1_feature_blocks.csv ========================
feature_block  n_features                    model    sensitivity_mean       sensitivity_sd      precision_mean    specificity_mean             f1_mean            mcc_mean        roc_auc_mean            fn_mean             fp_mean         fit_seconds
          All          30        Gradient Boosting 0.92617201919527525 0.050593910252650899 0.96043975803193038  0.9766562336984872 0.94196590587780682 0.91040936177645837 0.99248000203344677 3.1333333333333333  1.6666666666666667  8.9017530619998979
          All          30 Logistic Regression (L2)  0.9512366186784792   0.0371365961105583 0.98410313169530417 0.99066249347939483 0.96690804519479623  0.9489582370080204  0.9946430015637896 2.0666666666666669 0.66666666666666663 0.26969597199968121
          All          30                SVM (RBF)  0.9606496862310816 0.031739607939135502 0.97483977861175819  0.9850417318727176 0.96736029742258001 0.94881580906387042 0.99534834617813095 1.6666666666666667  1.0666666666666669  0.4569607590001396
        Error          10        Gradient Boosting 0.81616832779623494 0.040283513293818102 0.87171683894897933 0.92724308815858125 0.84202320601798986 0.75483885461178202 0.94954646772394358 7.7999999999999998  5.2000000000000002  4.0946585609999602
        Error          10 Logistic Regression (L2) 0.75470653377630126 0.059256933839553799 0.88683889346428058 0.94211006781429318 0.81390723630395112 0.72457647418374693 0.94264620886386363               10.4  4.1333333333333337 0.25515931600011749
        Error          10                SVM (RBF) 0.72643041712809142 0.046192937668665897 0.88089043723294203 0.94034950443401144 0.79501579913257148 0.69943066860300973 0.93883777109975397               11.6  4.2666666666666666 0.45126565599957758
         Mean          10        Gradient Boosting 0.91520856404577344 0.053586114143412703 0.93124128589585797 0.95794209702660404 0.92145453849414705 0.87763611107007733 0.98564586990793435 3.6000000000000001                   3  4.2514473499995802
         Mean          10 Logistic Regression (L2) 0.89313399778516067 0.053771389408889599 0.94173124296877275 0.96637976004173198 0.91543112172624119 0.87013579097698279 0.98582534541154498 4.5333333333333332  2.3999999999999999 0.31878472700009292
         Mean          10                SVM (RBF) 0.88069398301956447 0.055923076859490502 0.96551580167112905 0.98042514345331244  0.9196731156258624  0.8797222574184641 0.98672682087356156 5.0666666666666664  1.3999999999999999 0.40829115400038057
        Worst          10        Gradient Boosting 0.92772240679217444 0.039967982797434597 0.95260221640748244 0.97198748043818439 0.93927893647914817  0.9051689053127614  0.9899370887974992 3.0666666666666669                   2  4.2042155929998444
        Worst          10 Logistic Regression (L2) 0.95754891103728323   0.0379267674197757 0.98131748527097362 0.98879760041731857 0.96869172161764283 0.95159711640912437 0.99251973223203682                1.8 0.80000000000000004 0.26092843300011731
        Worst          10                SVM (RBF) 0.94976005906238481 0.036379963572626198 0.96838500998172017 0.98131194574856517 0.95852109216686876 0.93513157842525496 0.99484335672374757 2.1333333333333333  1.3333333333333333 0.37737394400028279

======================== exp2_architectures.csv ========================
                    model                                                   architecture                                         pipeline                                                                                                                                                                                                                                                                                                                                                                                                                                                            key_hyperparameters
 Logistic Regression (L2)                      linear, L2 penalty; p(y=1|x)=sigma(w·x+b)             StandardScaler -> LogisticRegression                                                                                                                                                                                      {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 2000, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l2', 'random_state': 42, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
 Logistic Regression (L1)                        linear, L1 penalty -> embedded sparsity             StandardScaler -> LogisticRegression                                                                                                                                                                                  {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 2000, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l1', 'random_state': 42, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
Linear Discriminant (LDA)                    Gaussian class densities, shared covariance     StandardScaler -> LinearDiscriminantAnalysis                                                                                                                                                                                                                                                                                                                             {'covariance_estimator': None, 'n_components': None, 'priors': None, 'shrinkage': None, 'solver': 'svd', 'store_covariance': False, 'tol': 0.0001}
      SVM (linear kernel)                         max-margin hyperplane, soft margin C=1                            StandardScaler -> SVC                                                                                                                                                                                        {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}
         SVM (RBF kernel)            max-margin in infinite-dim RKHS, C=1, gamma='scale'                            StandardScaler -> SVC                                                                                                                                                                                           {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}
     k-Nearest Neighbours non-parametric, 5 nearest neighbours, Euclidean, majority vote           StandardScaler -> KNeighborsClassifier                                                                                                                                                                                                                                                                                                                           {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
     Gaussian Naive Bayes    generative, per-feature Gaussians, conditional independence                     StandardScaler -> GaussianNB                                                                                                                                                                                                                                                                                                                                                                                                                                       {'priors': None, 'var_smoothing': 1e-09}
            Random Forest               bagged CART, 300 trees, sqrt(p) split candidates         StandardScaler -> RandomForestClassifier                                                    {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 300, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}
              Extra Trees                bagged CART, random split thresholds, 300 trees           StandardScaler -> ExtraTreesClassifier                                                   {'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 300, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}
        Gradient Boosting           stagewise additive CART, 100 trees, lr=0.1, log-loss     StandardScaler -> GradientBoostingClassifier                         {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_iter_no_change': None, 'random_state': 42, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
    HistGradient Boosting            histogram-binned gradient boosting (LightGBM-style) StandardScaler -> HistGradientBoostingClassifier {'categorical_features': 'from_dtype', 'class_weight': None, 'early_stopping': 'auto', 'interaction_cst': None, 'l2_regularization': 0.0, 'learning_rate': 0.1, 'loss': 'log_loss', 'max_bins': 255, 'max_depth': None, 'max_features': 1.0, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'monotonic_cst': None, 'n_iter_no_change': 10, 'random_state': 42, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

======================== exp2_model_zoo.csv ========================
                    model         sensitivity       sensitivity_sd           precision         specificity                  f1        balanced_acc                 mcc             roc_auc              pr_auc                 fn                  fp  n_folds       train_seconds
         SVM (RBF kernel)  0.9606496862310816 0.031739607939135502 0.97483977861175819  0.9850417318727176 0.96736029742258001  0.9728457090518996 0.94881580906387042 0.99534834617813095  0.9938978441554136 1.6666666666666667  1.0666666666666669       15 0.45813280399988798
 Logistic Regression (L2)  0.9512366186784792   0.0371365961105583 0.98410313169530417 0.99066249347939483 0.96690804519479623 0.97094955607893718  0.9489582370080204  0.9946430015637896 0.99329542844359464 2.0666666666666669 0.66666666666666663       15 0.26289860099996071
      SVM (linear kernel) 0.94651162790697685   0.0412163577406559 0.97874445306379498 0.98785863328116841  0.9620438352245132 0.96718513059407285 0.94114485605990883 0.99422108302990475 0.99246799546847519 2.2666666666666666  0.8666666666666667       15 0.35007376099974857
 Logistic Regression (L1) 0.94496124031007755 0.037600353199798701 0.97426915412093196  0.9850417318727176 0.95902700328700041  0.9650014860913978 0.93620596929031363 0.99510553848500116 0.99357692146788201  2.333333333333333  1.0666666666666669       15 0.26610129600021531
              Extra Trees 0.94027316352897761 0.041475929015289098  0.9687915102601744 0.98132498695878956 0.95348589396897243 0.96079907524388364 0.92800309474339882 0.99392201255191204 0.99175387487276501 2.5333333333333332  1.3333333333333333       15  7.6456946409998636
            Random Forest 0.93407161314138076 0.055349413083970699 0.95814977556924597 0.97476525821596238 0.94466904943572039 0.95441843567867157 0.91457998383505157 0.98982874782429875 0.98819411781235078 2.7999999999999998                 1.8       15  10.080266722000031
    HistGradient Boosting 0.93403469915097836 0.036732454533146199 0.96810189439051719 0.98129890453834123 0.95022538715939397 0.95766680184465958 0.92264740956688518 0.99133327094349044 0.98950992576781405 2.7999999999999998  1.3333333333333333       15  2.9379981010001761
     k-Nearest Neighbours 0.92620893318567721 0.037579768333282498 0.98366882440053161   0.990675534689619 0.95359002205804277 0.95844223393764816 0.92915526815850524 0.98440810308304039 0.97746039265063644 3.1333333333333333 0.66666666666666663       15 0.28274926399990358
        Gradient Boosting 0.92617201919527525 0.050593910252650899 0.96043975803193038  0.9766562336984872 0.94196590587780682 0.95141412644688117 0.91040936177645837 0.99248000203344677 0.99011601704339958 3.1333333333333333  1.6666666666666667       15  8.9875997990002361
     Gaussian Naive Bayes 0.89785898855666302   0.0474457516122797   0.919038876435882 0.95142149191444958 0.90694488795347739 0.92464024023555635 0.85453211307247789  0.9872272682896116 0.97931756231578637  4.333333333333333  3.4666666666666668       15  0.2309210329999587
Linear Discriminant (LDA) 0.88992248062015511 0.040810224841904302 0.99277167277167278 0.99627021387584758 0.93821281634544962 0.94309634724800118 0.90856710324357881 0.99204972874860398 0.99027320297699217  4.666666666666667 0.26666666666666661       15 0.29158767299986721
````


**Output (stdout):**

````text

======================== exp3_feature_selection.csv ========================
               configuration         sensitivity       sensitivity_sd           precision             roc_auc                 fn  fold_selection_stability_jaccard
     fixed: Worst block (10) 0.95754891103728323   0.0379267674197757 0.98131748527097362 0.99251973223203682                1.8                               NaN
             fixed: All (30)  0.9512366186784792   0.0371365961105583 0.98410313169530417  0.9946430015637896 2.0666666666666669                               NaN
  SelectKBest f_classif (10) 0.92613510520487285 0.054982341178212199 0.94320103373927144 0.98859585159536045 3.1333333333333333                                 1
SelectKBest mutual_info (10) 0.92440014765596157 0.043839029869135598 0.95412920159972126 0.99094645391730241 3.2000000000000002                                 1

======================== exp4a_threshold_curve.csv ========================
           threshold         sensitivity         specificity           precision                  f1            accuracy  fn  fp  tp  tn
0.050000000000000003 0.99056603773584917 0.80672268907563027 0.75268817204301075 0.85539714867617112 0.87521968365553604   2  69 210 288
0.059999999999999998 0.99056603773584917 0.83193277310924374 0.77777777777777779 0.87136929460580914 0.89103690685413006   2  60 210 297
0.070000000000000007 0.99056603773584917 0.84873949579831931 0.79545454545454541 0.88235294117647056 0.90158172231985945   2  54 210 303
0.080000000000000002 0.99056603773584917 0.85994397759103647 0.80769230769230771 0.88983050847457623 0.90861159929701241   2  50 210 307
0.089999999999999997 0.99056603773584917 0.87394957983193278 0.82352941176470584 0.89935760171306212 0.91739894551845358   2  45 210 312
 0.10000000000000001 0.98584905660377364 0.88235294117647056 0.83266932270916338 0.90280777537796963    0.92091388400703   3  42 209 315
                0.11 0.98584905660377364 0.88795518207282909  0.8393574297188755 0.90672451193058556 0.92442882249560643   3  40 209 317
                0.12 0.98584905660377364 0.89635854341736698 0.84959349593495936 0.91266375545851519 0.92970123022847095   3  37 209 320
                0.13 0.98584905660377364 0.90196078431372562 0.85655737704918034 0.91666666666666663  0.9332161687170476   3  35 209 322
 0.14000000000000001 0.98584905660377364 0.90756302521008403 0.86363636363636365 0.92070484581497802 0.93673110720562403   3  33 209 324
 0.14999999999999999 0.98584905660377364 0.91596638655462181 0.87447698744769875 0.92682926829268297 0.94200351493848855   3  30 209 327
                0.16 0.98584905660377364 0.92717086834733897 0.88936170212765953 0.93512304250559275 0.94903339191564162   3  26 209 331
 0.17000000000000001 0.98584905660377364 0.92997198879551823 0.89316239316239321 0.93721973094170397 0.95079086115992961   3  25 209 332
 0.17999999999999999 0.98584905660377364 0.93557422969187676 0.90086206896551724  0.9414414414414416 0.95430579964850604   3  23 209 334
                0.19 0.98113207547169801 0.94117647058823517 0.90829694323144117   0.943310657596372 0.95606326889279436   4  21 208 336
 0.20000000000000001 0.97641509433962259 0.94117647058823517  0.9078947368421052 0.94090909090909081 0.95430579964850604   5  21 207 336
 0.20999999999999999 0.97641509433962259 0.94397759103641443 0.91189427312775317 0.94305239179954437 0.95606326889279436   5  20 207 337
                0.22 0.97641509433962259  0.9467787114845938 0.91592920353982299  0.9452054794520548 0.95782073813708257   5  19 207 338
 0.23000000000000001 0.97641509433962259  0.9467787114845938 0.91592920353982299  0.9452054794520548 0.95782073813708257   5  19 207 338
 0.23999999999999999 0.97641509433962259 0.94957983193277318 0.92000000000000004 0.94736842105263164 0.95957820738137078   5  18 207 339
                0.25 0.97641509433962259 0.95238095238095244 0.92410714285714279 0.94954128440366958   0.961335676625659   5  17 207 340
 0.26000000000000001 0.97641509433962259 0.95238095238095244 0.92410714285714279 0.94954128440366958   0.961335676625659   5  17 207 340
 0.27000000000000002 0.97641509433962259 0.95518207282913159  0.9282511210762332 0.95172413793103439 0.96309314586994721   5  16 207 341
 0.28000000000000003 0.97641509433962259 0.95518207282913159  0.9282511210762332 0.95172413793103439 0.96309314586994721   5  16 207 341
 0.28999999999999998 0.97641509433962259 0.95518207282913159  0.9282511210762332 0.95172413793103439 0.96309314586994721   5  16 207 341
 0.29999999999999999 0.97641509433962259 0.95798319327731096 0.93243243243243235 0.95391705069124422 0.96485061511423564   5  15 207 342
                0.31 0.97641509433962259 0.96078431372549022 0.93665158371040724 0.95612009237875284 0.96660808435852363   5  14 207 343
 0.32000000000000001 0.97169811320754718 0.96638655462184875 0.94495412844036697 0.95813953488372094 0.96836555360281196   6  12 206 345
 0.33000000000000002 0.97169811320754718 0.96638655462184875 0.94495412844036697 0.95813953488372094 0.96836555360281196   6  12 206 345
 0.34000000000000002 0.97169811320754718 0.96918767507002801 0.94930875576036877 0.96037296037296038 0.97012302284710017   6  11 206 346
 0.34999999999999998 0.97169811320754718 0.97198879551820716 0.95370370370370361 0.96261682242990643 0.97188049209138838   6  10 206 347
 0.35999999999999999 0.97169811320754718 0.97759103641456579 0.96261682242990643 0.96713615023474175 0.97539543057996481   6   8 206 349
                0.37 0.97169811320754718 0.97759103641456579 0.96261682242990643 0.96713615023474175 0.97539543057996481   6   8 206 349
                0.38 0.97169811320754718 0.97759103641456579 0.96261682242990643 0.96713615023474175 0.97539543057996481   6   8 206 349
 0.39000000000000001 0.97169811320754718 0.98039215686274517 0.96713615023474175 0.96941176470588242 0.97715289982425324   6   7 206 350
 0.40000000000000002 0.96698113207547165 0.98319327731092443 0.97156398104265396 0.96926713947990539 0.97715289982425324   7   6 205 351
 0.40999999999999998 0.96226415094339623 0.98599439775910358 0.97607655502392343 0.96912114014251782 0.97715289982425324   8   5 204 352
 0.41999999999999998 0.96226415094339623 0.98599439775910358 0.97607655502392343 0.96912114014251782 0.97715289982425324   8   5 204 352
 0.42999999999999999 0.96226415094339623 0.98599439775910358 0.97607655502392343 0.96912114014251782 0.97715289982425324   8   5 204 352
                0.44 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
 0.45000000000000001 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
 0.46000000000000002 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
 0.46999999999999997 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
 0.47999999999999998 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
 0.48999999999999999 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
                 0.5 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
 0.51000000000000001 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
 0.52000000000000002 0.96226415094339623 0.98879551820728295 0.98076923076923084 0.97142857142857142 0.97891036906854123   8   4 204 353
 0.53000000000000003 0.95754716981132082 0.98879551820728295  0.9806763285024156 0.96897374701670635 0.97715289982425324   9   4 203 353
 0.54000000000000004 0.95754716981132082 0.98879551820728295  0.9806763285024156 0.96897374701670635 0.97715289982425324   9   4 203 353
 0.55000000000000004 0.95283018867924518 0.99159663865546221 0.98536585365853657  0.9688249400479616 0.97715289982425324  10   3 202 354
 0.56000000000000005 0.95283018867924518 0.99159663865546221 0.98536585365853657  0.9688249400479616 0.97715289982425324  10   3 202 354
 0.56999999999999995 0.94811320754716977 0.99159663865546221 0.98529411764705876 0.96634615384615397 0.97539543057996481  11   3 201 354
 0.57999999999999996 0.94339622641509435 0.99439775910364159 0.99009900990098998 0.96618357487922701 0.97539543057996481  12   2 200 355
 0.58999999999999997 0.94339622641509435 0.99439775910364159 0.99009900990098998 0.96618357487922701 0.97539543057996481  12   2 200 355
 0.59999999999999998 0.94339622641509435 0.99439775910364159 0.99009900990098998 0.96618357487922701 0.97539543057996481  12   2 200 355
 0.60999999999999999 0.94339622641509435 0.99439775910364159 0.99009900990098998 0.96618357487922701 0.97539543057996481  12   2 200 355
                0.62 0.93867924528301883 0.99719887955182085               0.995 0.96601941747572817 0.97539543057996481  13   1 199 356
                0.63 0.93867924528301883 0.99719887955182085               0.995 0.96601941747572817 0.97539543057996481  13   1 199 356
 0.64000000000000001 0.93396226415094341 0.99719887955182085 0.99497487437185916 0.96350364963503643 0.97363796133567659  14   1 198 356
 0.65000000000000002 0.93396226415094341 0.99719887955182085 0.99497487437185916 0.96350364963503643 0.97363796133567659  14   1 198 356
 0.66000000000000003   0.929245283018868 0.99719887955182085 0.99494949494949481 0.96097560975609764 0.97188049209138838  15   1 197 356
 0.67000000000000004 0.91981132075471683 0.99719887955182085 0.99489795918367363 0.95588235294117641 0.96836555360281196  17   1 195 356
 0.68000000000000005 0.91509433962264164 0.99719887955182085 0.99487179487179478 0.95331695331695321 0.96660808435852363  18   1 194 356
 0.68999999999999995 0.91509433962264164 0.99719887955182085 0.99487179487179478 0.95331695331695321 0.96660808435852363  18   1 194 356
 0.69999999999999996 0.91509433962264164 0.99719887955182085 0.99487179487179478 0.95331695331695321 0.96660808435852363  18   1 194 356
 0.70999999999999996 0.91509433962264164                   1                   1 0.95566502463054204 0.96836555360281196  18   0 194 357
 0.71999999999999997 0.91509433962264164                   1                   1 0.95566502463054204 0.96836555360281196  18   0 194 357
 0.72999999999999998 0.91509433962264164                   1                   1 0.95566502463054204 0.96836555360281196  18   0 194 357
 0.73999999999999999   0.910377358490566                   1                   1 0.95308641975308639 0.96660808435852363  19   0 193 357
                0.75 0.90094339622641517                   1                   1 0.94789081885856075 0.96309314586994721  21   0 191 357
 0.76000000000000001 0.90094339622641517                   1                   1 0.94789081885856075 0.96309314586994721  21   0 191 357
 0.77000000000000002 0.89622641509433965                   1                   1 0.94527363184079605   0.961335676625659  22   0 190 357
 0.78000000000000003 0.89150943396226412                   1                   1 0.94264339152119703 0.95957820738137078  23   0 189 357
 0.79000000000000004  0.8867924528301887                   1                   1 0.93999999999999995 0.95782073813708257  24   0 188 357
 0.80000000000000004 0.88207547169811318                   1                   1 0.93734335839599003 0.95606326889279436  25   0 187 357
 0.81000000000000005 0.88207547169811318                   1                   1 0.93734335839599003 0.95606326889279436  25   0 187 357
 0.81999999999999995 0.88207547169811318                   1                   1 0.93734335839599003 0.95606326889279436  25   0 187 357
 0.82999999999999996 0.88207547169811318                   1                   1 0.93734335839599003 0.95606326889279436  25   0 187 357
 0.83999999999999997 0.87735849056603776                   1                   1 0.93467336683417079 0.95430579964850604  26   0 186 357
 0.84999999999999998 0.86792452830188682                   1                   1 0.92929292929292917 0.95079086115992961  28   0 184 357
 0.85999999999999999 0.85849056603773588                   1                   1 0.92385786802030456 0.94727592267135319  30   0 182 357
                0.87 0.85849056603773588                   1                   1 0.92385786802030456 0.94727592267135319  30   0 182 357
                0.88 0.84905660377358494                   1                   1 0.91836734693877564 0.94376098418277676  32   0 180 357
 0.89000000000000001 0.83490566037735847                   1                   1 0.91002570694087404 0.93848857644991202  35   0 177 357
 0.90000000000000002 0.83018867924528306                   1                   1 0.90721649484536082 0.93673110720562403  36   0 176 357
 0.91000000000000003 0.83018867924528306                   1                   1 0.90721649484536082 0.93673110720562403  36   0 176 357
 0.92000000000000004 0.82075471698113212                   1                   1 0.90155440414507759  0.9332161687170476  38   0 174 357
 0.93000000000000005 0.81603773584905659                   1                   1 0.89870129870129867 0.93145869947275917  39   0 173 357
 0.93999999999999995 0.80660377358490565                   1                   1 0.89295039164490864 0.92794376098418285  41   0 171 357
 0.94999999999999996 0.78773584905660377                   1                   1  0.8812664907651715    0.92091388400703  45   0 167 357

======================== exp4b_nested_threshold.csv ========================
 tp  tn  fp  fn         sensitivity         specificity           precision                  f1            accuracy   balanced_accuracy                 mcc             roc_auc              pr_auc  fold                 tau
 43  68   3   0                   1 0.95774647887323938 0.93478260869565222 0.96629213483146081 0.97368421052631582 0.97887323943661964 0.94619488055590439                   1                   1     1 0.42999999999999999
 37  71   0   6 0.86046511627906974                   1                   1 0.92500000000000004 0.94736842105263164 0.93023255813953476 0.89073895527204949 0.98689813298395024 0.98799532232018283     2 0.54000000000000015
 39  72   0   3  0.9285714285714286                   1                   1  0.9629629629629628 0.97368421052631582 0.96428571428571441  0.9441549509633318 0.98082010582010559 0.98263305322128858     3 0.59000000000000019
 42  70   2   0                   1 0.97222222222222221 0.95454545454545459 0.97674418604651159 0.98245614035087725 0.98611111111111116 0.96334329448556555                   1                   1     4 0.42999999999999999
 41  71   0   1 0.97619047619047616                   1                   1 0.98795180722891562 0.99115044247787598 0.98809523809523803 0.98113825485110684 0.99966465459423204 0.99944629014396458     5 0.41000000000000009

======================== exp5a_calibration.csv ========================
         variant                brier             log_loss             roc_auc
    uncalibrated   0.0201430031318704 0.089083093668020599 0.99172876697848944
Platt-calibrated 0.026365943442918399  0.11492325163445601 0.99200623645684682

======================== exp5b_reliability.csv ========================
         variant     bin   n       mean_predicted         observed_rate
    uncalibrated 0.0-0.1 318 0.013068610528939401 0.0094339622641508997
    uncalibrated 0.1-0.2  23  0.14705623054664291  0.086956521739130405
    uncalibrated 0.2-0.3   6   0.2429421317962745                     0
    uncalibrated 0.3-0.4  11  0.34765268153810558    0.1818181818181818
    uncalibrated 0.4-0.5   3  0.41497266830201929   0.33333333333333331
    uncalibrated 0.5-0.6   6  0.55463316090565662   0.66666666666666663
    uncalibrated 0.6-0.7   7  0.64646810451158088    0.8571428571428571
    uncalibrated 0.7-0.8   8  0.75697628844633913                 0.875
    uncalibrated 0.8-0.9  11  0.86724522907122392                     1
    uncalibrated 0.9-1.0 176  0.99116023682487375                     1
Platt-calibrated 0.0-0.1 278 0.024940619326192601 0.0071942446043165003
Platt-calibrated 0.1-0.2  46  0.13787283306080861  0.021739130434782601
Platt-calibrated 0.2-0.3  22  0.23687294658142519  0.090909090909090898
Platt-calibrated 0.3-0.4  14  0.35813002310199438   0.21428571428571419
Platt-calibrated 0.4-0.5   7   0.4690347925595047    0.5714285714285714
Platt-calibrated 0.5-0.6   8  0.54242613295469422                  0.75
Platt-calibrated 0.6-0.7  11  0.64724245912773448                     1
Platt-calibrated 0.7-0.8  10  0.74430656307520704                     1
Platt-calibrated 0.8-0.9  27  0.85760403610008074                     1
Platt-calibrated 0.9-1.0 146  0.98006893962082442                     1

======================== exp5c_bootstrap_ci.csv ========================
 Unnamed: 0              ci_2.5              median             ci_97.5      point_estimate
sensitivity 0.92201667039606183 0.95305164319248836 0.98039215686274517 0.95283018867924518
  precision 0.95431006380105243 0.97652582159624401 0.99514563106796117 0.97584541062801922
specificity 0.97329297513448321 0.98626373626373642 0.99716713881019836 0.98599439775910358

======================== exp6_quantum_vs_classical.csv ========================
                                   model         sensitivity       sensitivity_sd           precision         specificity                  f1        balanced_acc                 mcc             roc_auc              pr_auc                 fn                  fp  wall_seconds_5fold
           Quantum kernel SVM (3 qubits) 0.89634551495016623 0.058454297452578199 0.94713832559609801    0.96924882629108 0.91960572039561606 0.93279717062062295 0.87681075451589052  0.9904328425332356 0.98662551484762484 4.4000000000000004  2.2000000000000002  1.1492094550003458
Classical RBF-SVM (same PCA-3 front end)  0.8914728682170544 0.054137705191025197 0.93873493656102358 0.96369327073552424 0.91296516031571018 0.92758306947628921 0.86584412819730461 0.98695270243337718 0.98348260754279837 4.5999999999999996  2.6000000000000001  0.1227980200001184
     Classical RBF-SVM (30 raw features) 0.95758582502768563 0.041916969838566902  0.9810472851028722 0.98877151799687002 0.96877703777520741 0.97317867151227799 0.95143348232619318 0.99452509631431996 0.99281569045437301                1.8 0.80000000000000004  0.1638623339999867

======================== exp7_mcnemar.csv ========================
                                     comparison  b  c                stat                   p
      Worst(10) vs All(30), logistic regression  6  3 0.44444444444444442 0.50498507509384571
RBF-SVM vs logistic regression, all 30 features  6  4 0.10000000000000001 0.75182963404584924

======================== exp7_bootstrap_diff.csv ========================
      comparison                point              ci_low              ci_high  frac_bootstrap_gt0
Worst - All (LR) 0.018867924528301799 -0.0045459734329598 0.042061010925365301 0.92500000000000004

======================== exp8_learning_curve.csv ========================
 n_train          train_mean              train_sd            val_mean               val_sd
      68 0.98258287976249359 0.0087422316973212004 0.98593576965670005   0.0186612543996716
     145 0.97298854969278603 0.0098401834457461992 0.96710963455149501 0.034868072919874898
     222 0.97171803304080895 0.0069810032345734999 0.96245847176079735 0.043147323146375503
     300 0.96838124897142797 0.0076785758116116997 0.95780730897009958   0.0517751581619443
     377 0.96683132930006122 0.0084629983977126001 0.95780730897009958   0.0517751581619443
     455  0.9658057779324748  0.010158121629518201 0.96245847176079735 0.043147323146375503

======================== wdbc_numeric_reproducibility.csv ========================
               experiment_csv  rows  columns_compared  max_abs_numeric_difference status
      exp1_feature_blocks.csv    12                12                           0   PASS
       exp2_architectures.csv    11                 4                           0   PASS
           exp2_model_zoo.csv    11                13                           0   PASS
   exp3_feature_selection.csv     4                 7                           0   PASS
    exp4a_threshold_curve.csv    91                10                           0   PASS
   exp4b_nested_threshold.csv     5                15                           0   PASS
        exp5a_calibration.csv     2                 4                           0   PASS
        exp5b_reliability.csv    20                 5                           0   PASS
       exp5c_bootstrap_ci.csv     3                 5                           0   PASS
exp6_quantum_vs_classical.csv     3                12                           0   PASS
      exp7_bootstrap_diff.csv     1                 5                           0   PASS
             exp7_mcnemar.csv     2                 5                           0   PASS
      exp8_learning_curve.csv     6                 5                           0   PASS

======================== dataset_overlap_audit.csv ========================
dataset_a dataset_b  shared_identifier_values  shared_identifiers_identical_30_features  shared_identifiers_wdbc_malignant                  interpretation
     WDBC      WPBC                       139                                       126                                139 cohorts_overlap_not_independent

======================== exp9_multi_dataset_blocks.csv ========================
   dataset              configuration  n_selected  n_folds  jaccard_pairwise_mean  jaccard_pairwise_min  jaccard_n_pairs     sensitivity_mean      sensitivity_std    specificity_mean       specificity_std      precision_mean        precision_std              f1_mean               f1_std  balanced_accuracy_mean  balanced_accuracy_std             mcc_mean              mcc_std        roc_auc_mean           roc_auc_std         pr_auc_mean            pr_auc_std            fn_mean              fn_std             fp_mean              fp_std
      WDBC                       Mean          10       15                      1                     1              105  0.89313399778516067 0.053771389408889599 0.96637976004173198  0.013793481062082599 0.94173124296877275   0.0207109330340467  0.91543112172624119 0.022543366972147299     0.92975687891344638     0.0214977813385685  0.87013579097698279 0.028869365090803699 0.98582534541154498 0.0049850582655259004 0.98044990344573923 0.0061647687712940002 4.5333333333333332   2.294921930407801  2.3999999999999999 0.98561076060916242
      WDBC                      Error          10       15                      1                     1              105  0.75470653377630126 0.059256933839553799 0.94211006781429318  0.020399360887737601 0.88683889346428058 0.033769481791352703  0.81390723630395112 0.037437177336970003     0.84840830079529717   0.028143837901451502  0.72457647418374693 0.047566418140126999 0.94264620886386363    0.0129492241712335 0.92353171375925303  0.015671661387104099               10.4  2.5014281634983759  4.1333333333333337  1.4573295865416045
      WDBC                      Worst          10       15                      1                     1              105  0.95754891103728323   0.0379267674197757 0.98879760041731857    0.0120588186160775 0.98131748527097362 0.019841973177695399  0.96869172161764283 0.018166273340816198     0.97317325572730085   0.017203336344188799  0.95159711640912437 0.027312675749739099 0.99251973223203682 0.0077347341159389998 0.99254230467190563 0.0071121030368422001                1.8    1.61245154965971 0.80000000000000004 0.86189160737133463
      WDBC                        All          30       15                      1                     1              105   0.9512366186784792   0.0371365961105583 0.99066249347939483 0.0086633721037924008 0.98410313169530417   0.0144841825972467  0.96690804519479623   0.0187521475892568     0.97094955607893718   0.017586910044715302   0.9489582370080204 0.027521806915929901  0.9946430015637896 0.0049305698012509997 0.99329542844359464 0.0053593453077032003 2.0666666666666669  1.5796322658258459 0.66666666666666663 0.61721339984836754
      WDBC   SelectKBest_f_classif_10          10       15                      1                     1              105  0.92613510520487285 0.054982341178212199 0.96640584246218042  0.016530183174945898 0.94320103373927144 0.026585748904739401  0.93353465242158318 0.029918657048052799      0.9462704738335268   0.026925188706204899  0.89653925121962053 0.044914300945021003 0.98859585159536045 0.0084726886092569998 0.98577394227646642 0.0093879895499911006 3.1333333333333333  2.3563490726929559  2.3999999999999999  1.1832159566199232
      WDBC SelectKBest_mutual_info_10          10       15                      1                     1              105  0.92440014765596157 0.043839029869135598 0.97299165362545659  0.021273542168184199 0.95412920159972126   0.0345595717228077  0.93821394032795558 0.027778168701221701     0.94869590064070919   0.023475142640712699  0.90396060759665819 0.043055951230329702 0.99094645391730241 0.0061413977232847002  0.9875337761028834 0.0077754746875711997 3.2000000000000002  1.8593393604027368  1.9333333333333331  1.5337473561121311
      WPBC                       Mean          10       15                      1                     1              105 0.065925925925925902 0.081483886448394494 0.98888888888888882  0.034884541231485502 0.44444444444444442  0.49867549436206371   0.1119191919191919  0.13323228494190789     0.52740740740740732   0.045738437917312101  0.13355617648740209  0.18248500148563751 0.58932218239745116   0.13131727290045631 0.42843940132039288   0.13135271668132151 8.8000000000000007  1.0823255385643318 0.33333333333333331   1.046536236944567
      WPBC                      Error          10       15                      1                     1              105 0.057777777777777699 0.070046595728787797 0.96896057347670239  0.032034212203157199 0.28222222222222221  0.36118558450611848 0.090764790764790701   0.1038976070117338     0.51336917562724016   0.031514500157612903 0.052508918727503899  0.13565939308162919 0.55321385902031062  0.086947924542102803 0.34999123220522937  0.091175430847463101 8.8666666666666671 0.91547541643412678 0.93333333333333324 0.96115010472325479
      WPBC                      Worst          10       15                      1                     1              105   0.1007407407407407  0.10511324762641761  0.9622222222222222  0.035336627579287301 0.39444444444444438  0.37594677660006448  0.15401709401709399  0.15142002358805839     0.53148148148148144   0.056408791806445403   0.1091629647282881     0.18708716900545 0.57969573874950231   0.12586642756120439  0.3788979221595361    0.1093920633434065 8.4666666666666668  1.1872336794093274  1.1333333333333333  1.0600988273786192
      WPBC                        All          32       15                      1                     1              105   0.1918518518518518   0.1560912921197162 0.91146953405017916  0.048381407186687903 0.38531746031746028   0.2612320002876769  0.24742442742442741  0.18196514668982031      0.5516606929510155   0.085588882319288406  0.13136921116791231  0.21549855069540491 0.65768140183193935   0.11413449448084741 0.41765327951911818   0.13664361937827749 7.5999999999999996  1.5491933384829666  2.6666666666666665  1.4474937289114918
      WPBC   SelectKBest_f_classif_10          10       15    0.74162979877265578   0.42857142857142849              105   0.1059259259259259 0.088577496900082098 0.95347670250896077  0.032918148345135297  0.3833333333333333  0.30988989340045608   0.1618403818403818  0.13015037446880601     0.52970131421744326   0.048615288796519503 0.099936848722186694  0.16303720884305359 0.63871445639187574    0.1110824900968836  0.4170274235545583    0.1097375770959388 8.4000000000000004 0.91025898983279963  1.3999999999999999 0.98561076060916242
      WPBC SelectKBest_mutual_info_10          10       15       0.37115889090679    0.1111111111111111              105 0.084444444444444405 0.091219447785661295  0.9446594982078852  0.046687803877969003                0.31  0.36219130260476951  0.12761238761238761  0.13576126919464179     0.51455197132616493   0.053661714650768501 0.051973162550718897  0.18640817399788359 0.56978653922739941  0.098347966918967702 0.36517095593557569  0.087854875923949693 8.5999999999999996 0.91025898983279963  1.6666666666666667  1.3972762620115438
Parkinsons            Vocal_Freq_Tone           5       15                      1                     1              105  0.92727969348659001 0.056449817179422102 0.47111111111111109   0.19968228732918669 0.84380046266273334 0.056459820064573299  0.88217997908342005   0.0444218659261005     0.69919540229885047    0.10359476947316069  0.45688930041474479  0.19336069118485669 0.78177096636866739  0.093374217832171702  0.9157454163226536  0.047102052369598399 2.1333333333333333  1.6417180315870614  5.1333333333333337  2.0655911179772888
Parkinsons         Perturbation_Error          11       15                      1                     1              105  0.93662835249042142 0.042276951090679499 0.39407407407407408   0.17587607202280381 0.82631409709871073 0.049526642590759902  0.87696151536350209 0.033997197573156099     0.66535121328224778   0.089990317438895601  0.40425739478846479  0.18323307805362921  0.8279438058748404     0.057140265989784 0.94170838334731077    0.0233461201937403 1.8666666666666667  1.2459458063579461  5.8666666666666663  1.8464895909600487
Parkinsons         Nonlinear_Extremes           6       15                      1                     1              105  0.92758620689655158 0.035417010143652901 0.62444444444444436   0.17917112705165261 0.88434227806806809 0.054345475023910898  0.90408639002903757 0.028501288574690298     0.77601532567049802   0.085699265604674904  0.58534731710079224  0.13893645260206761   0.896994465730098  0.051056045717375098 0.96466933068690797  0.018421606173845299 2.1333333333333333  1.0600988273786194  3.6666666666666665  1.8387366263150311
Parkinsons                        All          22       15                      1                     1              105  0.93203065134099639 0.038471706804146098 0.61185185185185198   0.21012579866347289 0.88175942825567133 0.062310649304040698  0.90463804960251404   0.0352891902922765     0.77194125159642391     0.1033897710728964   0.5817991827141803   0.1739150989626406 0.89549595572584084  0.055590513146063898  0.9640837212700728  0.020065410741779699                  2  1.1338934190276817  3.7999999999999998  2.1447610589527217
Parkinsons   SelectKBest_f_classif_10          10       15    0.63378843378843386   0.42857142857142849              105  0.92758620689655158   0.0399261345429546 0.60370370370370374   0.19647092162777549 0.87949275217440837 0.058185641246359297  0.90112821018583777 0.028281229795106999     0.76564495530012766   0.090725338523941407  0.57000961367440628  0.14185137668914249 0.89552149851000407  0.059123534927624399 0.96502419887068835    0.0206107769406388 2.1333333333333333  1.1872336794093272  3.8666666666666663  1.9952324127660872
Parkinsons SelectKBest_mutual_info_10          10       15     0.6992118992118993   0.53846153846153844              105   0.9275862068965518 0.024663907340725402 0.56296296296296311   0.21251275163178809 0.86857774719083036 0.062216105484182903  0.89556868329522088 0.028976193101024301     0.74527458492975718     0.1009590141761961  0.52962095456666691  0.15522743107759809 0.89660280970625805  0.059075420727934098 0.96528547944449783  0.021027995820844199 2.1333333333333333 0.74322335295720654  4.2666666666666666  2.1536237457121521

======================== exp10_multi_dataset_zoo.csv ========================
   dataset                     model  n_folds  n_rows  n_features     sensitivity_mean      sensitivity_std    specificity_mean       specificity_std       precision_mean        precision_std              f1_mean               f1_std  balanced_accuracy_mean  balanced_accuracy_std             mcc_mean              mcc_std        roc_auc_mean           roc_auc_std         pr_auc_mean            pr_auc_std             fn_mean              fn_std             fp_mean              fp_std  wall_seconds_15fold
      WDBC  Logistic Regression (L2)       15     569          30   0.9512366186784792   0.0371365961105583 0.99066249347939483 0.0086633721037924008  0.98410313169530417   0.0144841825972467  0.96690804519479623   0.0187521475892568     0.97094955607893718   0.017586910044715302   0.9489582370080204 0.027521806915929901  0.9946430015637896 0.0049305698012509997 0.99329542844359464 0.0053593453077032003  2.0666666666666669  1.5796322658258459 0.66666666666666663 0.61721339984836754   0.4118840709998039
      WDBC  Logistic Regression (L1)       15     569          30  0.94496124031007755 0.037600353199798701  0.9850417318727176 0.0098793023830350997  0.97426915412093196 0.016769193009880001  0.95902700328700041 0.022678797552248699      0.9650014860913978   0.019953865826653799  0.93620596929031363 0.034227824153687303 0.99510553848500116 0.0044785406191069996 0.99357692146788201 0.0050618048410599999   2.333333333333333  1.5886502207249786  1.0666666666666669 0.70373155054899683  0.28770660200007109
      WDBC Linear Discriminant (LDA)       15     569          30  0.88992248062015511 0.040810224841904302 0.99627021387584758 0.0064024525345681999  0.99277167277167278 0.012432235562886199  0.93821281634544962 0.027041850035119299     0.94309634724800118   0.022641025445516098  0.90856710324357881    0.038254381136313 0.99204972874860398    0.0072536360962028 0.99027320297699217 0.0083084576849846003   4.666666666666667  1.7182493859684491 0.26666666666666661  0.4577377082170635  0.26098971600004001
      WDBC       SVM (linear kernel)       15     569          30  0.94651162790697685   0.0412163577406559 0.98785863328116841    0.0089991842856598  0.97874445306379498   0.0160823883305314   0.9620438352245132     0.02622756971598     0.96718513059407285   0.023121655872756799  0.94114485605990883 0.040331553185328599 0.99422108302990475 0.0063701462057362997 0.99246799546847519 0.0070991009295302996  2.2666666666666666  1.7511900715418265  0.8666666666666667 0.63994047342218441  0.38880550800013219
      WDBC          SVM (RBF kernel)       15     569          30   0.9606496862310816 0.031739607939135502  0.9850417318727176    0.0124208982092663  0.97483977861175819 0.020641940307481201  0.96736029742258001 0.019547647301002599      0.9728457090518996   0.017141823142275098  0.94881580906387042 0.030594963028656101 0.99534834617813095 0.0049069607913871001  0.9938978441554136 0.0059131416303303003  1.6666666666666667  1.3451854182690983  1.0666666666666669 0.88371510168853684  0.55462711700010914
      WDBC      k-Nearest Neighbours       15     569          30  0.92620893318567721 0.037579768333282498   0.990675534689619  0.010084617013622101  0.98366882440053161   0.0176462838144976  0.95359002205804277   0.0201697760677867     0.95844223393764816   0.018439062407018099  0.92915526815850524 0.030211350332523802 0.98440810308304039    0.0100555178892477 0.97746039265063644    0.0129334792036922  3.1333333333333333  1.5976172734359606 0.66666666666666663 0.72374686445574588  0.38006657899995838
      WDBC      Gaussian Naive Bayes       15     569          30  0.89785898855666302   0.0474457516122797 0.95142149191444958    0.0278444781395869    0.919038876435882 0.039291932223221797  0.90694488795347739 0.024876038755242899     0.92464024023555635   0.021053751273037501  0.85453211307247789   0.0378850449794218  0.9872272682896116 0.0063381059411127998 0.97931756231578637    0.0165482221997366   4.333333333333333  2.0236694629405503  3.4666666666666668  1.9952324127660872    0.375890315000106
      WDBC             Random Forest       15     569          30  0.93407161314138076 0.055349413083970699 0.97476525821596238  0.022514184150082699  0.95814977556924597 0.034130431138110297  0.94466904943572039   0.0301601294111222     0.95441843567867157   0.026625690974032099  0.91457998383505157 0.045352881007130599 0.98982874782429875 0.0092397407371783997 0.98819411781235078 0.0089599942071904998  2.7999999999999998  2.3664319132398464                 1.8    1.61245154965971   11.464708470999994
      WDBC               Extra Trees       15     569          30  0.94027316352897761 0.041475929015289098 0.98132498695878956    0.0171908606512431   0.9687915102601744   0.0264910364628431  0.95348589396897243   0.0200179713940132     0.96079907524388364     0.0185181382404516  0.92800309474339882 0.030358673710636799 0.99392201255191204 0.0043825545135232997 0.99175387487276501     0.005492621743272  2.5333333333333332  1.7674302033770737  1.3333333333333333  1.2344267996967353   7.4852522930000305
      WDBC         Gradient Boosting       15     569          30  0.92617201919527525 0.050593910252650899  0.9766562336984872  0.019507759353705301  0.96043975803193038   0.0321815855529827  0.94196590587780682 0.029511610426649699     0.95141412644688117      0.025292588414341  0.91040936177645837 0.043715889822368501 0.99248000203344677 0.0052375752551716004 0.99011601704339958 0.0064724637250699003  3.1333333333333333  2.1668498091095505  1.6666666666666667  1.3972762620115435   9.2097461529997418
      WDBC     HistGradient Boosting       15     569          30  0.93403469915097836 0.036732454533146199 0.98129890453834123    0.0137171632125246  0.96810189439051719   0.0224864096298262  0.95022538715939397 0.020309649688604901     0.95766680184465958     0.0180186529216428  0.92264740956688518 0.030798351278841299 0.99133327094349044 0.0063927502059251996 0.98950992576781405 0.0067820948088844002  2.7999999999999998  1.5675276256394519  1.3333333333333333  0.9759000729485332   3.2723698140002853
      WPBC  Logistic Regression (L2)       15     198          32   0.1918518518518518   0.1560912921197162 0.91146953405017916  0.048381407186687903  0.38531746031746028   0.2612320002876769  0.24742442742442741  0.18196514668982031      0.5516606929510155   0.085588882319288406  0.13136921116791231  0.21549855069540491 0.65768140183193935   0.11413449448084741 0.41765327951911818   0.13664361937827749  7.5999999999999996  1.5491933384829666  2.6666666666666665  1.4474937289114918  0.31873768200011909
      WPBC  Logistic Regression (L1)       15     198          32  0.17777777777777781   0.1470461660700757 0.92028673835125463  0.045283403531471901  0.41388888888888881  0.22408691860514091  0.23839789839789829  0.16685012228915819     0.54903225806451605   0.082100728911543894   0.1354391725765931  0.19961449008505031 0.64116129032258062    0.1142928194167666 0.42734553753965682    0.1244349284259737  7.7333333333333334  1.4864467059144129  2.3999999999999999  1.3522468075656269  0.27876821700010618
      WPBC Linear Discriminant (LDA)       15     198          32  0.27703703703703708  0.13362842125710411 0.85175627240143392  0.052312000300726101   0.3686772486772486  0.16054815373146669  0.31201133804848979  0.13929758069116349     0.56439665471923528   0.077569845375489693  0.14277835184347809  0.17221901677654661 0.67008283552369563  0.086429174764957498 0.40692268030739498  0.097837901546441897  6.7999999999999998  1.3201731488169051  4.4666666666666668  1.5522640914238173  0.27390504699997109
      WPBC       SVM (linear kernel)       15     198          32   0.2059259259259259   0.1238610664656502 0.90487455197132638  0.054882165046103701   0.4342328042328043  0.23690231516618179  0.26494996231838341  0.13497413214924539     0.55540023894862611   0.064896220163291399   0.1516919529423105  0.17064129060811051 0.65270330545599364    0.1063317335146178 0.40098941555061179    0.1022339504159365  7.4666666666666668  1.2459458063579458  2.8666666666666667  1.6417180315870614  0.48745415499979572
      WPBC          SVM (RBF kernel)       15     198          32 0.014074074074073999 0.037200696610900599 0.98451612903225805  0.030472334501615701 0.055555555555555497  0.14995590180707291 0.022222222222222199    0.058644295879083     0.49929510155316609   0.017642543398358301  -0.0027778271127161 0.062570277709223104  0.6138765432098765    0.1073221880033981 0.40291949542261452    0.1017583672424274  9.2666666666666675 0.59361683970466372 0.46666666666666667 0.91547541643412678   0.3941989169998123
      WPBC      k-Nearest Neighbours       15     198          32   0.2074074074074074  0.12790998451544699 0.90279569892473122  0.056721685908594298   0.4169841269841269  0.25437517004475602  0.26580863580863578  0.15046575721260269     0.55510155316606935   0.071493628996610395  0.14598004092807601   0.1892045285541484  0.5306587017124651    0.1348413443048708  0.3105343578938079    0.1064344727186471  7.4666666666666668  1.3557637102737474  2.9333333333333331  1.7099150633319551  0.34749233699994869
      WPBC      Gaussian Naive Bayes       15     198          32   0.4748148148148148   0.1500421261637902 0.72172043010752707    0.1108497330338593  0.36321356203709138   0.1285469312565764  0.40451035214036057  0.12591705684966581     0.59826762246117082   0.099660837823070003   0.1864018557757022   0.1861017370523437 0.62708960573476702    0.1187634424098003 0.45859000161450758   0.12870684931413701  4.9333333333333336  1.3870146083619754  8.4000000000000004  3.3551665915633544  0.27062166400037307
      WPBC             Random Forest       15     198          32 0.071851851851851806 0.087978138484875595 0.96888888888888902    0.0320383368241084  0.28888888888888881   0.3698276446839619  0.11209013209013211  0.13565234445229729     0.52037037037037026     0.0445765233223821 0.070317253725430906   0.1684343318015889 0.57041696535244912   0.12116440075937859 0.36722434679966798  0.094966782546443002  8.7333333333333325  1.0327955589886444 0.93333333333333324 0.96115010472325479   9.2084228890003033
      WPBC               Extra Trees       15     198          32   0.1081481481481481 0.094043656291393302 0.96236559139784961  0.030540555019247202  0.45111111111111107  0.36206345726950562  0.16746808746808739  0.13320203989089319      0.5352568697729988   0.049503879716751097  0.12725415980992941  0.17325422719858269 0.59286061330147355    0.1208721026345817 0.40293488850542541   0.10448207174812239  8.4000000000000004  1.1212238211627761  1.1333333333333333 0.91547541643412678   7.3896885680001105
      WPBC         Gradient Boosting       15     198          32  0.17185185185185181  0.16117221975554241 0.88301075268817208  0.058673204855702599   0.2592592592592593   0.1830404908253882   0.1964814814814814   0.1569886656961656     0.52743130227001189   0.064543803683011602 0.049539944127294802   0.1542136741238494 0.55073516527279964    0.1154728913411306 0.31822575896103628  0.071669159802218996  7.7999999999999998    1.61245154965971  3.5333333333333332  1.7674302033770737   4.6416209039998648
      WPBC     HistGradient Boosting       15     198          32  0.22074074074074079  0.14805420301753719 0.90508960573476716  0.052928176316760299   0.4308080808080807  0.17007887092732371  0.27501221001220999  0.12934239392188421     0.56291517323775386   0.067845937371550605  0.16215235195848299  0.14710368421182241  0.5965145360414178   0.10477652096910731  0.4250762630660691    0.1045562228173938   7.333333333333333   1.447493728911492  2.8666666666666667  1.5976172734359606   1.2410961450000286
   Coimbra  Logistic Regression (L2)       15     116           9  0.75598290598290596 0.086494924154246297 0.72969696969696962    0.1849875284307867  0.79320781070781077  0.11877659880889339   0.7670779371848836 0.067166443582780205     0.74283993783993774   0.094309028537030604  0.49241428195405701  0.18801978164981609 0.80506604506604496    0.1103078695624501 0.84643875535908564  0.098140890037077502  3.1333333333333333  1.1254628677422756  2.7999999999999998    1.93464651625488   0.2920093399998222
   Coimbra  Logistic Regression (L1)       15     116           9   0.7303418803418803  0.11503984578736839 0.74848484848484831   0.16708723010331161  0.79448218448218455   0.1069483035668683  0.75460543384281509 0.085780537720035105     0.73941336441336436     0.1006094962207256  0.48363146445362742  0.20079114058047889 0.80705516705516689    0.1087431134861381 0.84521603635155929  0.099535072241783804  3.4666666666666668   1.505545305418162  2.6000000000000001  1.7237832147426693  0.32970331800015629
   Coimbra Linear Discriminant (LDA)       15     116           9  0.70982905982905975   0.1091752798344663 0.74363636363636365   0.15068517263312869  0.78272542272542267  0.10730505807290169  0.73931334795102899 0.085556775704530102     0.72673271173271181   0.099023144485871795  0.45678570762616638  0.19991289460147871 0.78954156954156951    0.1017929897066646  0.8411430951112544  0.092420976988913703  3.7333333333333329  1.4375905768565218  2.6666666666666665  1.5886502207249789  0.28546759399978328
   Coimbra       SVM (linear kernel)       15     116           9  0.71452991452991454    0.132635595697563 0.77575757575757565    0.1244428749362242  0.80299607799607797 0.089892506563974794  0.74952184047836212   0.0911798261376843     0.74514374514374515   0.088226595233435998  0.49546221340132762  0.17867872843889021 0.79980574980574959    0.1012120037441787 0.84292458308510687  0.094188137198237001  3.6666666666666665  1.7182493859684491   2.333333333333333  1.3451854182690983  0.31834935300003048
   Coimbra          SVM (RBF kernel)       15     116           9  0.76666666666666672 0.087484652278748895 0.71696969696969703   0.16370368985342179  0.78422894099364682   0.1026084174768728  0.76858997037907595   0.0559874340260735     0.74181818181818182   0.077038694406347197  0.49169791442235061  0.15321248155079009 0.84845376845376863    0.0557227882737499 0.88714532405614599  0.040734824513044297                   3  1.1338934190276817  2.9333333333333331  1.7099150633319551  0.35727004899990789
   Coimbra      k-Nearest Neighbours       15     116           9  0.75128205128205139   0.1080329950398574 0.78666666666666663   0.15047609084429719  0.82294372294372298   0.1144340176426821  0.78098429214371246 0.092083549617795593     0.76897435897435895     0.1012556401264884  0.54103363978718355   0.2029150106959911 0.80844599844599851  0.086702634934496994 0.81925511426150821  0.079700638116800199  3.2000000000000002  1.4242792663559449  2.2000000000000002  1.5212776585113297  0.32956568400004471
   Coimbra      Gaussian Naive Bayes       15     116           9   0.4448717948717949  0.15637797661261699 0.87090909090909108  0.096710964351193904  0.82208994708994709  0.12019633494462451  0.55941556207810073  0.16399631455572369     0.65789044289044274   0.089944699327291697  0.34490659279325508  0.17710769981229779 0.77277389277389286     0.098426445766505 0.80305432456062043  0.095641831062582297  7.1333333333333337  2.0998866182543785  1.3333333333333333  0.9759000729485332   0.2792685310000706
   Coimbra             Random Forest       15     116           9  0.76709401709401703   0.1108083835069392  0.6399999999999999   0.17227555423784729  0.73156075406075394   0.0998702480701852  0.74504591288847544 0.092810688874814501     0.70354700854700858     0.1134473159303338  0.41273047059369511  0.22577170046409209 0.78988733488733476  0.095784601335556704 0.82842604081673898  0.082443245420360994                   3  1.4638501094227998  3.7333333333333329  1.7915143899851349    8.217332524000085
   Coimbra               Extra Trees       15     116           9   0.7760683760683762   0.1095510132377811  0.6775757575757575   0.18932137137883689  0.76207257396885886   0.1126095046523847  0.76215093236857379 0.083635511163093104     0.72682206682206674     0.1006605792358541   0.4621702676990409  0.20042828663604639 0.82085470085470091  0.096359334677689396 0.84798669754165112  0.084207243263162396  2.8666666666666667  1.4074631010979934   3.333333333333333  1.9518001458970664   6.5931436959999701
   Coimbra         Gradient Boosting       15     116           9  0.77692307692307694   0.1033679579144149 0.67272727272727251   0.14602162185644549  0.75238155635214465 0.092423180682109807  0.75979703066659587 0.075421991887392903     0.72482517482517494   0.088962911406007594  0.45864435942751319   0.1788495879014694 0.81391608391608394  0.094408798965521304 0.85332552986475296  0.082689215693186602  2.8666666666666667  1.3557637102737474  3.3999999999999999  1.5491933384829666   2.0166575029998057
   Coimbra     HistGradient Boosting       15     116           9  0.79188034188034195   0.1109989336989403 0.72363636363636374    0.1524611162613756  0.78440647587706425  0.10160369118954039  0.78520315625662962 0.094694425068875807     0.75775835275835279     0.1103916985489846  0.52051757790081254   0.2206455828613777  0.8105905205905205    0.1111840016617435 0.83192389194335825  0.096659909183319404  2.6666666666666665  1.4474937289114918  2.8666666666666667  1.5522640914238175  0.64708304399982808
Parkinsons  Logistic Regression (L2)       15     195          22  0.93203065134099639 0.038471706804146098 0.61185185185185198   0.21012579866347289  0.88175942825567133 0.062310649304040698  0.90463804960251404   0.0352891902922765     0.77194125159642391     0.1033897710728964   0.5817991827141803   0.1739150989626406 0.89549595572584084  0.055590513146063898  0.9640837212700728  0.020065410741779699                   2  1.1338934190276817  3.7999999999999998  2.1447610589527217   0.3876527359998363
Parkinsons  Logistic Regression (L1)       15     195          22  0.93203065134099616 0.040344351969900903 0.59777777777777774    0.2067298077295098  0.87776592905338902   0.0613803167551214  0.90260435155773722 0.036578395822933603     0.76490421455938706     0.1030394473782634  0.57125010679998633   0.1770816762898218 0.88789272030651345    0.0594821271968149 0.96192208910558596  0.021102714111666601                   2  1.1952286093343936  3.9333333333333331  2.1201976547572388  0.31214216099988329
Parkinsons Linear Discriminant (LDA)       15     195          22  0.95026819923371642 0.035728633347266697 0.65111111111111108   0.18477308644504159  0.89417114612724091 0.055405069788955402  0.92016838593120143   0.0320108550800592     0.80068965517241375     0.0920677207137892  0.65076667078041994    0.155784884327691 0.88182205193699448  0.075279524351792707 0.94803200865199144  0.038745476930621199  1.4666666666666666  1.0600988273786194  3.3999999999999999  1.8439088914585773  0.26666943200007148
Parkinsons       SVM (linear kernel)       15     195          22  0.94597701149425295 0.050232083158421599 0.57703703703703701    0.2159212520027417  0.87474084879510849 0.061991159032299403  0.90681181742281436   0.0336253461337978     0.76150702426564498     0.1010335006317897  0.58923302845702252  0.16058923059899599  0.8794806300553426  0.062524348417645301 0.95618829474195843  0.024029922380160801  1.6000000000000001  1.5023790657297036  4.1333333333333337  2.1995670569671746  0.35380025699987527
Parkinsons          SVM (RBF kernel)       15     195          22  0.99540229885057485   0.0121333025956723 0.53555555555555545    0.2170103094942904  0.86872938350879536 0.060825354735578201   0.9268292531156096 0.036554282459911901      0.7654789272030651     0.1107574207569829  0.65917583709347349   0.1792118278054797 0.88661558109833982  0.067517248813085406 0.95834797684712003  0.025007844649407598  0.1333333333333333  0.3518657752744983  4.5333333333333332  2.2318047362958038  0.38735488500014981
Parkinsons      k-Nearest Neighbours       15     195          22   0.9457471264367816 0.045746961407727597 0.79259259259259263   0.13515075302603319  0.93526848040482458 0.039560014458372497  0.93907827393967058   0.0207905144042595     0.86916985951468717   0.057037565669394802  0.75715447993923191 0.078578437363207507 0.96902937420178803    0.0151415843114378 0.98492376663334202 0.0095725248421999008  1.6000000000000001  1.3522468075656269                   2  1.3093073414159542   0.2671494710002662
Parkinsons      Gaussian Naive Bayes       15     195          22   0.6304980842911877 0.084223339365099403 0.91555555555555579  0.091219447785661198  0.95916261824055615 0.045910401566898101  0.75801885094777488 0.067215261900787096     0.77302681992337163   0.068180404588792098  0.47422258933391798  0.12077800503432259 0.85598126862494683  0.054765858479585698 0.94013325568543282  0.031882384972667398  10.866666666666667  2.4746332099544621 0.80000000000000004 0.86189160737133474  0.26401911099992498
Parkinsons             Random Forest       15     195          22  0.96582375478927196   0.0368673740108586 0.70740740740740737   0.20657903359087909  0.91146557933480776 0.060923722812192403  0.93656637811056798 0.036730054657867203     0.83661558109833989     0.1035768440925818  0.72714433980769855  0.16553699042078421 0.96171988080034043  0.032981520567926297 0.98735365993537683  0.010972933230236801                   1  1.0690449676496976  2.8666666666666667  2.0655911179772888   8.4327423559998351
Parkinsons               Extra Trees       15     195          22  0.97272030651341002 0.041305048000123197 0.76074074074074061    0.1278548191946601  0.92578858381253204 0.038967511242376997  0.94781219289285201   0.0275300688358751     0.86673052362707537   0.063171088549201296  0.78519570989988829  0.10543284961664171  0.9747552149851002    0.0222897339511593 0.99180620550106324 0.0075096862959899997 0.80000000000000004   1.207121724244435   2.333333333333333  1.2909944487358054   7.7840150960000756
Parkinsons         Gradient Boosting       15     195          22  0.95701149425287357 0.041533647588429798 0.77999999999999992   0.13308177327277029   0.9305431259199336 0.040991503833061597  0.94294251664751616 0.032304608104220998     0.86850574712643691   0.073192390863211498  0.76427307174483894  0.13777661448594131 0.96118348233290762  0.039788881106092702 0.98689800651934678    0.0137382972442869  1.2666666666666666  1.2227992865708155  2.1333333333333333  1.3020130933435712   3.5014882250002302
Parkinsons     HistGradient Boosting       15     195          22  0.95693486590038324 0.035159926377951102 0.80148148148148157    0.1114966854931361  0.93651086785541737 0.034878311930208898  0.94603584490384096 0.025217836889635002     0.87920817369093229   0.057980270420197599  0.77986409102897014  0.10166229661879581 0.96446998722860799  0.030193981821273801 0.98691021572963444  0.012849075485728801  1.2666666666666666  1.0327955589886446  1.9333333333333331  1.0997835284835873  0.94390934800003379

======================== exp11_multi_dataset_nested_threshold.csv ========================
   dataset  target_sensitivity   apparent_threshold  apparent_target_feasible  apparent_sensitivity  apparent_specificity  apparent_precision  apparent_fn  apparent_fp  nested_sensitivity_mean  nested_sensitivity_std  nested_sensitivity_pooled  nested_specificity_pooled  nested_precision_pooled  nested_fn  nested_fp        threshold_min        threshold_max  n_inner_feasible  sensitivity_ci_low  sensitivity_ci_high   specificity_ci_low  specificity_ci_high    precision_ci_low   precision_ci_high  bootstrap_replicates bootstrap_unit
      WDBC 0.94999999999999996  0.41999999999999998                         1   0.95754716981132082   0.98599439775910358 0.97596153846153844            9            5       0.9388704318936878    0.063375715377174902        0.93867924528301883        0.98599439775910358      0.97549019607843124         13          5                 0.38  0.66000000000000003                 5 0.90566037735849059  0.96698113207547165  0.97471988795518205  0.99719887955182085  0.9547417319153908               0.995                  2000            row
      WPBC 0.94999999999999996 0.080000000000000002                         1   0.95744680851063835   0.19205298013245031 0.26946107784431139            2          122       0.9555555555555556    0.099380798999990597        0.95744680851063835        0.13907284768211919      0.25714285714285712          2        130 0.050000000000000003 0.070000000000000007                 2  0.8936170212765957                    1 0.086092715231788006  0.19867549668874171 0.23999999999999999 0.27439024390243899                  2000            row
   Coimbra 0.94999999999999996  0.23999999999999999                         1              0.953125   0.19230769230769229 0.59223300970873782            3           42       0.9525641025641024    0.043381857219714402                   0.953125        0.17307692307692299      0.58653846153846156          3         43  0.20000000000000001  0.23999999999999999                 5            0.890625                    1   0.0769230769230769  0.28846153846153838 0.55454545454545456 0.62376237623762376                  2000            row
Parkinsons 0.94999999999999996                 0.44                         1    0.9591836734693876   0.60416666666666663 0.88124999999999998            6           19      0.96597701149425275    0.033918759266065301         0.9659863945578232        0.54166666666666663      0.86585365853658536          5         22  0.28999999999999998  0.42999999999999999                 5 0.90410958904109584                    1   0.2291666666666666  0.85416666666666663 0.79255319148936165 0.94967550435176318                  2000   subject_stem

======================== exp12_multi_dataset_calibration.csv ========================
   dataset          variant                brier             log_loss  ece_10_uniform_bins             roc_auc   n  n_nonempty_bins
      WDBC     uncalibrated 0.020045313832946401 0.076418977459740203   0.0150962912892821 0.99468844141430157 569               10
      WDBC Platt-calibrated 0.026677905715619399   0.1100450047923065 0.055253512770501202  0.9949262723957506 569               10
      WPBC     uncalibrated  0.18455936705961351  0.55537812703123202 0.093398401058000402 0.66056080033817111 198               10
      WPBC Platt-calibrated  0.17632853159666531  0.53207185559913572 0.027670663652442199 0.64280681978300702 198                7
   Coimbra     uncalibrated  0.19505436829291681  0.61568456005472116 0.073416716750614203 0.77133413461538469 116               10
   Coimbra Platt-calibrated   0.2074388356740928  0.61195942528156211 0.085029716534129302 0.75781249999999989 116                9
Parkinsons     uncalibrated   0.1023534791757305   0.3156578672615758 0.080269489190595594  0.9000850340136054 195               10
Parkinsons Platt-calibrated   0.1186023188778552  0.38451683239673129   0.1172880500502744 0.90277777777777757 195                8

======================== exp12b_dca_curves.csv ========================
   dataset            threshold        net_benefit_raw       net_benefit_platt  net_benefit_treat_all  net_benefit_treat_none   sample_prevalence
      WDBC 0.050000000000000003    0.36435112385533253     0.35806123392840622    0.33956155767274071                       0 0.37258347978910361
      WDBC 0.059999999999999998      0.363571775791796     0.35657929177728748    0.33253561679691879                       0 0.37258347978910361
      WDBC 0.070000000000000007    0.35960088440387777     0.35653948636544019      0.325358580418391                       0 0.37258347978910361
      WDBC 0.080000000000000002    0.35898219607243831      0.3555436692901352    0.31802552150989521                       0 0.37258347978910361
      WDBC 0.089999999999999997    0.35877479286969621     0.35657312810212632    0.31053129647154248                       0 0.37258347978910361
      WDBC  0.10000000000000001    0.35871900019527431      0.3565709822300332    0.30287053309900408                       0 0.37258347978910361
      WDBC                 0.11    0.35903714381627538     0.35562093955490609    0.29503761774056592                       0 0.37258347978910361
      WDBC                 0.12     0.3590829205943441     0.35644671672791178     0.2870266815785269                       0 0.37258347978910361
      WDBC                 0.13    0.35696826454962322     0.35638244146819381    0.27883158596448698                       0 0.37258347978910361
      WDBC  0.14000000000000001    0.35664364245718722     0.35537662974618872    0.27044590673151592                       0 0.37258347978910361
      WDBC  0.14999999999999999    0.35604259278403799     0.35500878734622138    0.26186291739894552                       0 0.37258347978910361
      WDBC                 0.16    0.35542723240438528     0.35333500711356591    0.25307557117750429                       0 0.37258347978910361
      WDBC  0.17000000000000001    0.35551697122408787     0.34899527812480152    0.24407648167361881                       0 0.37258347978910361
      WDBC  0.17999999999999999    0.35530884307085597     0.34857902181833772    0.23485790218183369                       0 0.37258347978910361
      WDBC                 0.19     0.3555512161253227     0.34947601379938809     0.2254117034433378                       0 0.37258347978910361
      WDBC  0.20000000000000001     0.3554481546572934      0.3497363796133568     0.2157293497363795                       0 0.37258347978910361
      WDBC  0.20999999999999999    0.35538697693043531     0.35107116638117059    0.20580187315076409                       0 0.37258347978910361
      WDBC                 0.22    0.35636068676490468     0.35041232932269839     0.1956198458834662                       0 0.37258347978910361
      WDBC  0.23000000000000001    0.35644671672791178     0.35067217492525049    0.18517335037545929                       0 0.37258347978910361
      WDBC  0.23999999999999999    0.35602626954028299     0.34992137637591342     0.1744519470909259                       0 0.37258347978910361
      WDBC                 0.25    0.35618043350908019     0.34973637961335668    0.16344463971880491                       0 0.37258347978910361
      WDBC  0.26000000000000001     0.3563862632403933     0.35082886049494127    0.15213983755284269                       0 0.37258347978910361
      WDBC  0.27000000000000002     0.3566458819847364     0.35209572188651078    0.14052531477959401                       0 0.37258347978910361
      WDBC  0.28000000000000003    0.35627807068931849     0.35286076938098021      0.128588166373755                       0 0.37258347978910361
      WDBC  0.28999999999999998    0.35589989851233939      0.3551820589618554     0.1163147602663431                       0 0.37258347978910361
      WDBC  0.29999999999999999    0.35551092141601798     0.35475772031132308    0.10369068541300509                       0 0.37258347978910361
      WDBC                 0.31    0.35511066962125259     0.35432108198976081   0.090700695346526999                       0 0.37258347978910361
      WDBC  0.32000000000000001     0.3529411764705882     0.35469864571487641   0.077328646748681798                       0 0.37258347978910361
      WDBC  0.33000000000000002    0.35251685334312621     0.35427432258741443   0.063557432521050103                       0 0.37258347978910361
      WDBC  0.34000000000000002    0.35207967193907441     0.35474250412738989   0.049368908771369097                       0 0.37258347978910361
      WDBC  0.34999999999999998    0.35162903879951329     0.35257536839259151     0.0347438150601595                       0 0.37258347978910361
      WDBC  0.35999999999999999     0.3531414762741652      0.3531414762741652   0.019661687170474399                       0 0.37258347978910361
      WDBC                 0.37     0.3509917147878483     0.35378134850894077     0.0041007615700058                       0 0.37258347978910361
      WDBC                 0.38    0.34882929871307899     0.35489540223368671  -0.011962129372413401                       0 0.37258347978910361
      WDBC  0.39000000000000001    0.34841107493733608     0.35290558644731918  -0.028551672476879199                       0 0.37258347978910361
      WDBC  0.40000000000000002    0.34856473345049788     0.35266549502050382  -0.045694200351493901                       0 0.37258347978910361
      WDBC  0.40999999999999998    0.34943850347025701     0.35241726490125408  -0.063417830865925895                       0 0.37258347978910361
      WDBC  0.41999999999999998    0.35040300587843159     0.35119083691897462    -0.0817526210532694                       0 0.37258347978910361
      WDBC  0.42999999999999999    0.35013720593222952      0.3492738877069651    -0.1007307372120987                       0 0.37258347978910361
      WDBC                 0.44    0.34986191313080589      0.3504895807180517    -0.1203866432337434                       0 0.37258347978910361
      WDBC  0.45000000000000001    0.34957660968205778      0.3500559194759546   -0.14075730947435691                       0 0.37258347978910361
      WDBC  0.46000000000000002    0.34928073943891158      0.3499967454273254    -0.1618824448349932                       0 0.37258347978910361
      WDBC  0.46999999999999997     0.3454587657923533     0.34993533839572899   -0.18380475511489869                       0 0.37258347978910361
      WDBC  0.47999999999999998     0.3451399215898337      0.3463566310666486    -0.2065702311748005                       0 0.37258347978910361
      WDBC  0.48999999999999999     0.3464971225748647     0.34629036148730141   -0.23022847100175739                       0 0.37258347978910361
      WDBC                  0.5    0.34622144112478032      0.3444639718804921   -0.25483304042179261                       0 0.37258347978910361
      WDBC  0.51000000000000001    0.34783544349198381      0.3426347691976615   -0.28044187798142117                       0 0.37258347978910361
      WDBC  0.52000000000000002    0.34768599882835383     0.34270650263620389   -0.30711775043936751                       0 0.37258347978910361
      WDBC  0.53000000000000003    0.34753019481733538     0.34270650263620389   -0.33492876640616248                       0 0.37258347978910361
      WDBC  0.54000000000000004    0.34561014747459301     0.34270650263620389    -0.3639489569802094                       0 0.37258347978910361
      WDBC  0.55000000000000004    0.34544034368287441     0.34094903339191562   -0.39425893380199201                       0 0.37258347978910361
      WDBC  0.56000000000000005    0.34526282153698667      0.3374340949033392   -0.42594663684294631                       0 0.37258347978910361
      WDBC  0.56999999999999995     0.3450770425471042     0.33567662565905099     -0.459108186536968                       0 0.37258347978910361
      WDBC  0.57999999999999996     0.3448824169386559     0.33567662565905099   -0.49384885764499109                       0 0.37258347978910361
      WDBC  0.58999999999999997     0.3446782973980882     0.33216168717047451   -0.53028419563633233                       0 0.37258347978910361
      WDBC  0.59999999999999998    0.34446397188049199     0.33216168717047451   -0.56854130052724072                       0 0.37258347978910361
      WDBC  0.60999999999999999     0.3469875174620341     0.33040421792618629   -0.60876030823306748                       0 0.37258347978910361
      WDBC                 0.62    0.34686892979372858     0.32864674868189808   -0.65109610581814814                       0 0.37258347978910361
      WDBC                 0.63     0.3467439319811903     0.32864674868189808   -0.69572032489431446                       0 0.37258347978910361
      WDBC  0.64000000000000001      0.344854520601445     0.32864674868189808   -0.74282366725248994                       0 0.37258347978910361
      WDBC  0.65000000000000002     0.3447150389153904     0.32688927943760981   -0.79261862917398962                       0 0.37258347978910361
      WDBC  0.66000000000000003    0.34456735242427372      0.3251318101933216   -0.84534270650263643                       0 0.37258347978910361
      WDBC  0.67000000000000004    0.34441071523672578      0.3251318101933216   -0.90126218245726197                       0 0.37258347978910361
      WDBC  0.68000000000000005    0.34072934973637958      0.3251318101933216   -0.96067662565905121                       0 0.37258347978910361
      WDBC  0.68999999999999995    0.34270650263620389      0.3251318101933216    -1.0239242587448265                       0 0.37258347978910361
      WDBC  0.69999999999999996    0.34094903339191562     0.32161687170474518    -1.0913884007029877                       0 0.37258347978910361
      WDBC  0.70999999999999996    0.34094903339191562     0.32161687170474518    -1.1635052421065391                       0 0.37258347978910361
      WDBC  0.71999999999999997    0.34094903339191562      0.3181019332161687    -1.2407732864674867                       0 0.37258347978910361
      WDBC  0.72999999999999998    0.34094903339191562     0.31634446397188049    -1.3237648896699861                       0 0.37258347978910361
      WDBC  0.73999999999999999    0.34094903339191562     0.31634446397188049    -1.4131404623496011                       0 0.37258347978910361
      WDBC                 0.75    0.34094903339191562     0.31282952548330401    -1.5096660808435851                       0 0.37258347978910361
      WDBC  0.76000000000000001    0.33567662565905099     0.31107205623901579     -1.614235500878735                       0 0.37258347978910361
      WDBC  0.77000000000000002    0.33391915641476272     0.31107205623901579    -1.7278979139604189                       0 0.37258347978910361
      WDBC  0.78000000000000003    0.33391915641476272     0.30755711775043931    -1.8518932736858928                       0 0.37258347978910361
      WDBC  0.79000000000000004    0.33391915641476272     0.30228471001757468    -1.9876977152899831                       0 0.37258347978910361
      WDBC  0.80000000000000004    0.33391915641476272     0.30052724077328641    -2.1370826010544821                       0 0.37258347978910361
      WPBC 0.050000000000000003    0.20175438596491219     0.19723551302498671    0.19723551302498671                       0 0.23737373737373729
      WPBC 0.059999999999999998    0.18536428110896189      0.1886954652912099     0.1886954652912099                       0 0.23737373737373729
      WPBC 0.070000000000000007    0.17899424351037249     0.18035190615835769     0.1799717606169218                       0 0.23737373737373729
      WPBC 0.080000000000000002    0.17369345630215191     0.17149758454106279    0.17105841018884499                       0 0.23737373737373729
      WPBC 0.089999999999999997    0.16477966477966469     0.16294816294816289     0.1619491619491619                       0 0.23737373737373729
      WPBC  0.10000000000000001     0.1560044893378226      0.1543209876543209     0.1526374859708193                       0 0.23737373737373729
      WPBC                 0.11    0.14782658041085001     0.14561343774826921    0.14311655884689589                       0 0.23737373737373729
      WPBC                 0.12    0.14669421487603301     0.13820018365472911     0.1333792470156106                       0 0.23737373737373729
      WPBC                 0.13    0.14193660745384881     0.12010913735051661    0.12341808893533029                       0 0.23737373737373729
      WPBC  0.14000000000000001     0.1313131313131313      0.1154568945266619     0.1132252760159736                       0 0.23737373737373729
      WPBC  0.14999999999999999    0.11883541295305999      0.1105169340463457     0.1027926322043969                       0 0.23737373737373729
      WPBC                 0.16     0.1026936026936026     0.10798460798460791   0.092111592111592006                       0 0.23737373737373729
      WPBC  0.17000000000000001    0.10228793963733721    0.098089327004989596   0.081173177558719695                       0 0.23737373737373729
      WPBC  0.17999999999999999   0.091894555309189399    0.094727765459472701   0.069967972406996801                       0 0.23737373737373729
      WPBC                 0.19   0.063536600573637594    0.088913829654570398   0.058486095523132497                       0 0.23737373737373729
      WPBC  0.20000000000000001   0.055555555555555497    0.075757575757575704   0.046717171717171699                       0 0.23737373737373729
      WPBC  0.20999999999999999   0.048395345863700298    0.074095384221966501   0.034650300473085198                       0 0.23737373737373729
      WPBC                 0.22     0.0464905464905465    0.071354571354571295   0.022274022274022202                       0 0.23737373737373729
      WPBC  0.23000000000000001     0.0362062180243998      0.0433556342647251  0.0095762823035550008                       0 0.23737373737373729
      WPBC  0.23999999999999999   0.031632110579478998    0.027113237639553402 -0.0034556087187665998                       0 0.23737373737373729
      WPBC                 0.25     0.0252525252525252    0.035353535353535297    -0.0168350168350168                       0 0.23737373737373729
      WPBC  0.26000000000000001   0.025935025935025901      0.0229320229320229  -0.030576030576030599                       0 0.23737373737373729
      WPBC  0.27000000000000002   0.018264840182648401    0.014321295143212899  -0.044693510446935099                       0 0.23737373737373729
      WPBC  0.28000000000000003     0.0137485970819304    0.018237934904601499  -0.059203142536475899                       0 0.23737373737373729
      WPBC  0.28999999999999998     0.0123061601934841      0.0168587281263337  -0.074121496656707905                       0 0.23737373737373729
      WPBC  0.29999999999999999   0.014430014430014401    0.018037518037518002  -0.089466089466089499                       0 0.23737373737373729
      WPBC                 0.31     0.0125164690382081      0.0303762260283999      -0.10525545308154                       0 0.23737373737373729
      WPBC  0.32000000000000001     0.0202020202020201      0.0252525252525252    -0.1215092097445038                       0 0.23737373737373729
      WPBC  0.33000000000000002   0.026383235338459099      0.0157545605306799   -0.13824815317352629                       0 0.23737373737373729
      WPBC  0.34000000000000002   0.025558616467707299   0.0088766452402815003   -0.15549433731251919                       0 0.23737373737373729
      WPBC  0.34999999999999998   0.022144522144522099   0.0023310023310023002    -0.1732711732711732                       0 0.23737373737373729
      WPBC  0.35999999999999999   0.019255050505050501   0.0034722222222221999   -0.19160353535353539                       0 0.23737373737373729
      WPBC                 0.37     0.0167548500881834  -0.0082571749238415006   -0.21051787718454379                       0 0.23737373737373729
      WPBC                 0.38  0.0084718149234278006  -0.0099380905832517994   -0.23004235907461709                       0 0.23737373737373729
      WPBC  0.39000000000000001 -0.0049677098857425998     -0.0120880940553071   -0.25020698791190599                       0 0.23737373737373729
      WPBC  0.40000000000000002    -0.0084175084175084  -0.0033670033670033001    -0.2710437710437712                       0 0.23737373737373729
      WPBC  0.40999999999999998 -0.0084745762711864007 -0.00085601780517030005   -0.29258688580722469                       0 0.23737373737373729
      WPBC  0.41999999999999998  0.0026123301985370999  -0.0081853012887495007   -0.31487286659700447                       0 0.23737373737373729
      WPBC  0.42999999999999999 -0.0016835016835015999   -0.013999645578592901    -0.3379408116250221                       0 0.23737373737373729
      WPBC                 0.44    -0.0108225108225108  -0.0068542568542568003   -0.36183261183261178                       0 0.23737373737373729
      WPBC  0.45000000000000001  -0.013774104683195501  -0.0073461891643709001    -0.3865932047750229                       0 0.23737373737373729
      WPBC  0.46000000000000002  -0.012532734754956899  -0.0078563411896744994   -0.41227085671530111                       0 0.23737373737373729
      WPBC  0.46999999999999997    -0.0110539355822374  -0.0083857442348008009    -0.4389174766533257                       0 0.23737373737373729
      WPBC  0.47999999999999998  -0.013986013986013899  -0.0089355089355088995   -0.46658896658896659                       0 0.23737373737373729
      WPBC  0.48999999999999999  -0.017033075856605202     -0.0145573380867498   -0.49534561299267171                       0 0.23737373737373729
      WPBC                  0.5    -0.0151515151515151     -0.0101010101010101    -0.5252525252525253                       0 0.23737373737373729
      WPBC  0.51000000000000001  -0.018037518037518002     -0.0105132962275819   -0.55638012780869928                       0 0.23737373737373729
      WPBC  0.52000000000000002     -0.021043771043771     -0.0109427609427609     -0.588804713804714                       0 0.23737373737373729
      WPBC  0.53000000000000003  -0.034278959810874698     -0.0113905007522028   -0.62260906941758021                       0 0.23737373737373729
      WPBC  0.54000000000000004  -0.036671058410188799  -0.0059288537549406998   -0.65788317962231024                       0 0.23737373737373729
      WPBC  0.55000000000000004    -0.0336700336700336  -0.0061728395061728001   -0.69472502805836167                       0 0.23737373737373729
      WPBC  0.56000000000000005    -0.0303030303030303  -0.0064279155188245998   -0.73324150596877891                       0 0.23737373737373729
      WPBC  0.56999999999999995  -0.026544514916607899  -0.0066948555320647998   -0.77354944796805236                       0 0.23737373737373729
      WPBC  0.57999999999999996    -0.0293410293410293  -0.0069745069745068998    -0.8157768157768156                       0 0.23737373737373729
      WPBC  0.58999999999999997  -0.037324464153732403  -0.0072677999507266998   -0.86006405518600626                       0 0.23737373737373729
      WPBC  0.59999999999999998    -0.0429292929292929  -0.0075757575757574996   -0.90656565656565635                       0 0.23737373737373729
      WPBC  0.60999999999999999  -0.042994042994042997  -0.0078995078995079005    -0.9554519554519556                       0 0.23737373737373729
      WPBC                 0.62  -0.045720361509835099  -0.0082402977139819002    -1.0069112174375332                       0 0.23737373737373729
      WPBC                 0.63  -0.048594048594048499  -0.0085995085995085995    -1.0611520611520611                       0 0.23737373737373729
      WPBC  0.64000000000000001  -0.051627384960718198  -0.0089786756453423006    -1.1184062850729517                       0 0.23737373737373729
      WPBC  0.65000000000000002  -0.059884559884559901  -0.0093795093795093001    -1.1789321789321792                       0 0.23737373737373729
      WPBC  0.66000000000000003  -0.043672014260249498  -0.0098039215686274005    -1.2430184194890079                       0 0.23737373737373729
      WPBC  0.67000000000000004  -0.046372819100091799   -0.010254055708601099    -1.3109886746250383                       0 0.23737373737373729
      WPBC  0.68000000000000005  -0.049242424242424199   -0.010732323232323201    -1.3832070707070709                       0 0.23737373737373729
      WPBC  0.68999999999999995  -0.052297165200390898     -0.0112414467253176    -1.4600847181492338                       0 0.23737373737373729
      WPBC  0.69999999999999996  -0.055555555555555497     -0.0117845117845117    -1.5420875420875415                       0 0.23737373737373729
      WPBC  0.70999999999999996  -0.046673632880529399     -0.0123650296064089    -1.6297457331940088                       0 0.23737373737373729
      WPBC  0.71999999999999997  -0.049783549783549701   -0.012987012987012899    -1.7236652236652237                       0 0.23737373737373729
      WPBC  0.72999999999999998  -0.053123830901608597                       0    -1.8245417134306019                       0 0.23737373737373729
      WPBC  0.73999999999999999  -0.056721056721056699                       0    -1.9331779331779329                       0 0.23737373737373729
      WPBC                 0.75    -0.0303030303030303                       0    -2.0505050505050506                       0 0.23737373737373729
      WPBC  0.76000000000000001  -0.037878787878787797                       0    -2.1776094276094282                       0 0.23737373737373729
      WPBC  0.77000000000000002  -0.045674132630654299                       0    -2.3157663592446203                       0 0.23737373737373729
      WPBC  0.78000000000000003  -0.053719008264462798                       0    -2.4664830119375578                       0 0.23737373737373729
      WPBC  0.79000000000000004     -0.056998556998557                       0    -2.6315536315536323                       0 0.23737373737373729
      WPBC  0.80000000000000004  -0.060606060606060601                       0     -2.813131313131314                       0 0.23737373737373729
   Coimbra 0.050000000000000003    0.52858439201451901     0.52813067150635207    0.52813067150635207                       0 0.55172413793103448
   Coimbra 0.059999999999999998    0.52421129860601612     0.52311078503301545    0.52311078503301545                       0 0.55172413793103448
   Coimbra 0.070000000000000007    0.51928068223952539     0.51798294401186507    0.51798294401186507                       0 0.55172413793103448
   Coimbra 0.080000000000000002     0.5142428785607196     0.51274362818590702    0.51274362818590702                       0 0.55172413793103448
   Coimbra 0.089999999999999997    0.50909435392194013     0.50738916256157629    0.50738916256157629                       0 0.55172413793103448
   Coimbra  0.10000000000000001    0.50478927203065127     0.50191570881226055    0.50191570881226055                       0 0.55172413793103448
   Coimbra                 0.11    0.49089500193723368     0.49631925610228589    0.49631925610228589                       0 0.55172413793103448
   Coimbra                 0.12    0.47688087774294668     0.49059561128526641    0.49059561128526641                       0 0.55172413793103448
   Coimbra                 0.13    0.47265160523186678     0.48474038842647638    0.48474038842647638                       0 0.55172413793103448
   Coimbra  0.14000000000000001    0.46712109061748192     0.48015236567762631    0.47874899759422612                       0 0.55172413793103448
   Coimbra  0.14999999999999999    0.46298174442190659     0.47413793103448271    0.47261663286004052                       0 0.55172413793103448
   Coimbra                 0.16     0.4573070607553365     0.46798029556650239     0.4663382594417077                       0 0.55172413793103448
   Coimbra  0.17000000000000001    0.45502700457000411     0.46167428334025751    0.45990859991690902                       0 0.55172413793103448
   Coimbra  0.17999999999999999    0.44932716568544989     0.45710681244743478    0.45332211942809081                       0 0.55172413793103448
   Coimbra                 0.19    0.44348659003831409     0.45061728395061729    0.44657300979140058                       0 0.55172413793103448
   Coimbra  0.20000000000000001    0.43103448275862061     0.44396551724137928    0.43965517241379309                       0 0.55172413793103448
   Coimbra  0.20999999999999999    0.42732431252728059     0.43714535137494548    0.43256219991270189                       0 0.55172413793103448
   Coimbra                 0.22    0.42130857648099029      0.4301503094606543    0.42528735632183901                       0 0.55172413793103448
   Coimbra  0.23000000000000001    0.41513658755038058     0.41435288849081958    0.41782355575459018                       0 0.55172413793103448
   Coimbra  0.23999999999999999    0.41152450090744103      0.4097096188747732    0.41016333938294008                       0 0.55172413793103448
   Coimbra                 0.25    0.39655172413793099     0.40229885057471271     0.4022988505747126                       0 0.55172413793103448
   Coimbra  0.26000000000000001    0.39305684995340168     0.39468779123951531    0.39422180801491141                       0 0.55172413793103448
   Coimbra  0.27000000000000002    0.39289088332546052     0.38686820973075109    0.38592347661785542                       0 0.55172413793103448
   Coimbra  0.28000000000000003     0.3845785440613026     0.37883141762452111    0.37739463601532558                       0 0.55172413793103448
   Coimbra  0.28999999999999998     0.3783389995143272     0.37408936376881979    0.36862554638173872                       0 0.55172413793103448
   Coimbra  0.29999999999999999    0.37561576354679799     0.36576354679802958    0.35960591133004921                       0 0.55172413793103448
   Coimbra                 0.31    0.36444277861069457     0.35719640179910039     0.3503248375812093                       0 0.55172413793103448
   Coimbra  0.32000000000000001     0.3661257606490872     0.36054766734279919    0.34077079107505071                       0 0.55172413793103448
   Coimbra  0.33000000000000002     0.3683736489963973     0.36052496139989709     0.3309315491507977                       0 0.55172413793103448
   Coimbra  0.34000000000000002    0.35815047021943569     0.35214211076280039    0.32079414838035519                       0 0.55172413793103448
   Coimbra  0.34999999999999998    0.35676392572944288       0.343501326259947    0.31034482758620691                       0 0.55172413793103448
   Coimbra  0.35999999999999999    0.35075431034482751     0.31357758620689657    0.29956896551724138                       0 0.55172413793103448
   Coimbra                 0.37    0.33593322386425839     0.31116584564860422    0.28845101258894362                       0 0.55172413793103448
   Coimbra                 0.38    0.32619577308120129     0.32897664071190208    0.27697441601779749                       0 0.55172413793103448
   Coimbra  0.39000000000000001    0.31670435274166198     0.33224985867721868     0.2651215375918598                       0 0.55172413793103448
   Coimbra  0.40000000000000002    0.29885057471264359     0.29885057471264359    0.25287356321839077                       0 0.55172413793103448
   Coimbra  0.40999999999999998    0.29850964348334308     0.26855639976621859     0.2402104032729398                       0 0.55172413793103448
   Coimbra  0.41999999999999998    0.29221165279429251     0.27348394768133177    0.22711058263971459                       0 0.55172413793103448
   Coimbra  0.42999999999999999     0.2921960072595281     0.26845130066545669     0.2135511191772535                       0 0.55172413793103448
   Coimbra                 0.44      0.292487684729064      0.2493842364532019     0.1995073891625615                       0 0.55172413793103448
   Coimbra  0.45000000000000001    0.30015673981191221                    0.25     0.1849529780564263                       0 0.55172413793103448
   Coimbra  0.46000000000000002    0.28416347381864621     0.25830140485312891    0.16985951468710081                       0 0.55172413793103448
   Coimbra  0.46999999999999997    0.28578399479505529     0.26756668835393621    0.15419648666232921                       0 0.55172413793103448
   Coimbra  0.47999999999999998     0.2625994694960212     0.28580901856763918    0.13793103448275859                       0 0.55172413793103448
   Coimbra  0.48999999999999999     0.2564232589587559      0.2550709939148072     0.1210277214334009                       0 0.55172413793103448
   Coimbra                  0.5                   0.25     0.24137931034482751    0.10344827586206889                       0 0.55172413793103448
   Coimbra  0.51000000000000001     0.2612596762843068     0.24577762139338491   0.085151301900070295                       0 0.55172413793103448
   Coimbra  0.52000000000000002    0.25502873563218381     0.24353448275862061   0.066091954022988397                       0 0.55172413793103448
   Coimbra  0.53000000000000003     0.2399119589141599     0.22395451210564929   0.046221570066030601                       0 0.55172413793103448
   Coimbra  0.54000000000000004    0.24325337331334329     0.22376311844077959      0.025487256371814                       0 0.55172413793103448
   Coimbra  0.55000000000000004    0.23180076628352481        0.21455938697318  0.0038314176245207998                       0 0.55172413793103448
   Coimbra  0.56000000000000005     0.2280564263322884      0.2037617554858934    -0.0188087774294671                       0 0.55172413793103448
   Coimbra  0.56999999999999995     0.2221331194867682     0.20429029671210899  -0.042502004811547603                       0 0.55172413793103448
   Coimbra  0.57999999999999996    0.20730706075533659     0.18513957307060749  -0.067323481116584399                       0 0.55172413793103448
   Coimbra  0.58999999999999997     0.1959629941126998     0.17451640033641719  -0.093355761143818203                       0 0.55172413793103448
   Coimbra  0.59999999999999998    0.18965517241379309      0.1637931034482758    -0.1206896551724137                       0 0.55172413793103448
   Coimbra  0.60999999999999999     0.1744031830238727     0.16158267020335981    -0.1494252873563218                       0 0.55172413793103448
   Coimbra                 0.62     0.1955535390199637     0.15925589836660611    -0.1796733212341198                       0 0.55172413793103448
   Coimbra                 0.63    0.18685927306616959     0.14818266542404471   -0.21155638397017701                       0 0.55172413793103448
   Coimbra  0.64000000000000001    0.18773946360153251     0.11973180076628349    -0.2452107279693487                       0 0.55172413793103448
   Coimbra  0.65000000000000002     0.1982758620689655     0.10837438423645319   -0.28078817733990158                       0 0.55172413793103448
   Coimbra  0.66000000000000003    0.19320486815415819    0.088235294117646995    -0.3184584178498987                       0 0.55172413793103448
   Coimbra  0.67000000000000004     0.1967084639498432    0.094305120167189102   -0.35841170323928961                       0 0.55172413793103448
   Coimbra  0.68000000000000005     0.1745689655172413    0.092672413793103398    -0.4008620689655174                       0 0.55172413793103448
   Coimbra  0.68999999999999995      0.160734149054505    0.082313681868742994   -0.44605116796440458                       0 0.55172413793103448
   Coimbra  0.69999999999999996     0.1551724137931034    0.071839080459770097   -0.49425287356321829                       0 0.55172413793103448
   Coimbra  0.70999999999999996     0.1406064209274672    0.069857312722948803   -0.54577883472057076                       0 0.55172413793103448
   Coimbra  0.71999999999999997    0.12561576354679799    0.067733990147783196   -0.60098522167487678                       0 0.55172413793103448
   Coimbra  0.72999999999999998     0.1481481481481482    0.080140485312899107   -0.66028097062579816                       0 0.55172413793103448
   Coimbra  0.73999999999999999    0.14323607427055701    0.078912466843501297   -0.72413793103448265                       0 0.55172413793103448
   Coimbra                 0.75    0.12931034482758619    0.068965517241379296    -0.7931034482758621                       0 0.55172413793103448
   Coimbra  0.76000000000000001     0.1235632183908045    0.058908045977011499   -0.86781609195402321                       0 0.55172413793103448
   Coimbra  0.77000000000000002     0.1173163418290854    0.057346326836581703   -0.94902548725637204                       0 0.55172413793103448
   Coimbra  0.78000000000000003   0.093260188087774198      0.0384012539184952    -1.0376175548589344                       0 0.55172413793103448
   Coimbra  0.79000000000000004   0.085796387520525402    0.036535303776683001    -1.1346469622331696                       0 0.55172413793103448
   Coimbra  0.80000000000000004   0.068965517241379198   0.0086206896551723998    -1.2413793103448281                       0 0.55172413793103448
Parkinsons 0.050000000000000003    0.74224021592442646     0.74089068825910931    0.74089068825910931                       0 0.75384615384615383
Parkinsons 0.059999999999999998    0.74009819967266777     0.73813420621931258    0.73813420621931258                       0 0.75384615384615383
Parkinsons 0.070000000000000007    0.73763440860215057      0.7353184449958643     0.7353184449958643                       0 0.75384615384615383
Parkinsons 0.080000000000000002    0.73600891861761431     0.73244147157190631    0.73244147157190631                       0 0.75384615384615383
Parkinsons 0.089999999999999997    0.73457311918850376     0.72950126796280645    0.72950126796280645                       0 0.75384615384615383
Parkinsons  0.10000000000000001    0.73333333333333328     0.72649572649572647    0.72649572649572647                       0 0.75384615384615383
Parkinsons                 0.11    0.73166234514549122     0.72342264477095941    0.72342264477095941                       0 0.75384615384615383
Parkinsons                 0.12    0.72937062937062935     0.72027972027972031    0.72027972027972031                       0 0.75384615384615383
Parkinsons                 0.13    0.73009136457412316     0.71706454465075153    0.71706454465075153                       0 0.75384615384615383
Parkinsons  0.14000000000000001     0.7279666070363745     0.71377459749552774    0.71377459749552774                       0 0.75384615384615383
Parkinsons  0.14999999999999999    0.72579185520361988     0.71040723981900455    0.71040723981900455                       0 0.75384615384615383
Parkinsons                 0.16    0.71843711843711844     0.70695970695970689    0.70695970695970689                       0 0.75384615384615383
Parkinsons  0.17000000000000001    0.71615693543404391     0.70342910101946243    0.70342910101946243                       0 0.75384615384615383
Parkinsons  0.17999999999999999    0.71494684177611012     0.69981238273921198    0.69981238273921198                       0 0.75384615384615383
Parkinsons                 0.19    0.71503640392529277      0.6961063627730294     0.6961063627730294                       0 0.75384615384615383
Parkinsons  0.20000000000000001     0.7153846153846154     0.69230769230769229    0.69230769230769229                       0 0.75384615384615383
Parkinsons  0.20999999999999999     0.7081467056150601     0.68841285296981503    0.68841285296981503                       0 0.75384615384615383
Parkinsons                 0.22    0.70598290598290603     0.68441814595660744    0.68441814595660744                       0 0.75384615384615383
Parkinsons  0.23000000000000001    0.70682650682650683     0.68031968031968026    0.68031968031968026                       0 0.75384615384615383
Parkinsons  0.23999999999999999    0.70472334682861004     0.67611336032388658    0.67611336032388658                       0 0.75384615384615383
Parkinsons                 0.25    0.70256410256410262     0.67179487179487174    0.67179487179487174                       0 0.75384615384615383
Parkinsons  0.26000000000000001    0.70034650034650037     0.66735966735966734    0.66735966735966734                       0 0.75384615384615383
Parkinsons  0.27000000000000002    0.69806814190375832     0.66280295047418336    0.66280295047418336                       0 0.75384615384615383
Parkinsons  0.28000000000000003     0.6957264957264957     0.66011396011396006    0.65811965811965811                       0 0.75384615384615383
Parkinsons  0.28999999999999998    0.69331888768508487     0.65539906103286383    0.65330444203683635                       0 0.75384615384615383
Parkinsons  0.29999999999999999    0.69304029304029302     0.65054945054945057    0.64835164835164827                       0 0.75384615384615383
Parkinsons                 0.31    0.69290226681531031      0.6501672240802675    0.64325529542920845                       0 0.75384615384615383
Parkinsons  0.32000000000000001    0.69049773755656108     0.64524886877828047    0.63800904977375561                       0 0.75384615384615383
Parkinsons  0.33000000000000002    0.68802143130501336     0.64018369690011478     0.6326061997703788                       0 0.75384615384615383
Parkinsons  0.34000000000000002    0.68811188811188817     0.63496503496503487    0.62703962703962701                       0 0.75384615384615383
Parkinsons  0.34999999999999998    0.68047337278106512     0.63510848126232744    0.62130177514792895                       0 0.75384615384615383
Parkinsons  0.35999999999999999    0.67788461538461542     0.63269230769230766    0.61538461538461542                       0 0.75384615384615383
Parkinsons                 0.37    0.67521367521367526     0.63036223036223038    0.60927960927960922                       0 0.75384615384615383
Parkinsons                 0.38    0.67245657568238215     0.62812241521918943    0.60297766749379655                       0 0.75384615384615383
Parkinsons  0.39000000000000001    0.66775956284153004     0.62269861286254724    0.59646910466582592                       0 0.75384615384615383
Parkinsons  0.40000000000000002    0.65982905982905982     0.62051282051282053    0.58974358974358965                       0 0.75384615384615383
Parkinsons  0.40999999999999998    0.65180356366797043     0.62199043893959149    0.58279009126466752                       0 0.75384615384615383
Parkinsons  0.41999999999999998    0.64880636604774533     0.62758620689655176     0.5755968169761273                       0 0.75384615384615383
Parkinsons  0.42999999999999999    0.64570400359874047     0.62618083670715252    0.56815114709851555                       0 0.75384615384615383
Parkinsons                 0.44    0.64652014652014644     0.62087912087912089    0.56043956043956045                       0 0.75384615384615383
Parkinsons  0.45000000000000001    0.63729603729603734     0.61445221445221443     0.5524475524475525                       0 0.75384615384615383
Parkinsons  0.46000000000000002    0.63418803418803416      0.6089268755935423    0.54415954415954415                       0 0.75384615384615383
Parkinsons  0.46999999999999997    0.62583454281567497      0.6071601354620223    0.53555878084179964                       0 0.75384615384615383
Parkinsons  0.47999999999999998     0.6272189349112427     0.60157790927021693    0.52662721893491127                       0 0.75384615384615383
Parkinsons  0.48999999999999999    0.62393162393162394     0.60542986425339373    0.51734539969834081                       0 0.75384615384615383
Parkinsons                  0.5    0.62051282051282053     0.60000000000000009    0.50769230769230766                       0 0.75384615384615383
Parkinsons  0.51000000000000001    0.61182626896912606     0.58922030350601773     0.4976452119309262                       0 0.75384615384615383
Parkinsons  0.52000000000000002    0.60299145299145296     0.59999999999999987    0.48717948717948711                       0 0.75384615384615383
Parkinsons  0.53000000000000003    0.59399890889252593     0.59454446262956895    0.47626841243862511                       0 0.75384615384615383
Parkinsons  0.54000000000000004    0.58996655518394647     0.58885172798216268    0.46488294314381268                       0 0.75384615384615383
Parkinsons  0.55000000000000004    0.58575498575498575     0.58917378917378915    0.45299145299145288                       0 0.75384615384615383
Parkinsons  0.56000000000000005     0.5813519813519813     0.58461538461538454    0.44055944055944041                       0 0.75384615384615383
Parkinsons  0.56999999999999995    0.57674418604651168      0.5803220035778176    0.42754919499105548                       0 0.75384615384615383
Parkinsons  0.57999999999999996    0.57191697191697188     0.56923076923076921    0.41391941391941389                       0 0.75384615384615383
Parkinsons  0.58999999999999997    0.56172607879924952     0.56297686053783613    0.39962476547842402                       0 0.75384615384615383
Parkinsons  0.59999999999999998    0.54615384615384621     0.55641025641025643    0.38461538461538458                       0 0.75384615384615383
Parkinsons  0.60999999999999999     0.5405654174884944     0.54437869822485208    0.36883629191321488                       0 0.75384615384615383
Parkinsons                 0.62    0.52955465587044537     0.51659919028340084    0.35222672064777327                       0 0.75384615384615383
Parkinsons                 0.63    0.51309771309771313     0.50381150381150386    0.33471933471933468                       0 0.75384615384615383
Parkinsons  0.64000000000000001    0.50655270655270657     0.51396011396011398    0.31623931623931612                       0 0.75384615384615383
Parkinsons  0.65000000000000002    0.49963369963369958     0.50036630036630025    0.29670329670329659                       0 0.75384615384615383
Parkinsons  0.66000000000000003    0.48717948717948711     0.51251885369532424    0.27601809954751122                       0 0.75384615384615383
Parkinsons  0.67000000000000004    0.47940947940947942     0.50520590520590514    0.25407925407925391                       0 0.75384615384615383
Parkinsons  0.68000000000000005    0.46602564102564098       0.482051282051282    0.23076923076923059                       0 0.75384615384615383
Parkinsons  0.68999999999999995    0.45210918114143928      0.4646815550041356    0.20595533498759319                       0 0.75384615384615383
Parkinsons  0.69999999999999996    0.44273504273504283     0.45641025641025629    0.17948717948717949                       0 0.75384615384615383
Parkinsons  0.70999999999999996    0.43501326259946949     0.46984969053934572     0.1511936339522546                       0 0.75384615384615383
Parkinsons  0.71999999999999997    0.42271062271062271     0.48351648351648358    0.12087912087912089                       0 0.75384615384615383
Parkinsons  0.72999999999999998    0.42127255460588803     0.49268755935422598   0.088319088319088399                       0 0.75384615384615383
Parkinsons  0.73999999999999999    0.40078895463510839     0.50650887573964498     0.0532544378698225                       0 0.75384615384615383
Parkinsons                 0.75    0.38461538461538458     0.49743589743589739     0.0153846153846153                       0 0.75384615384615383
Parkinsons  0.76000000000000001    0.38888888888888878      0.4786324786324786  -0.025641025641025699                       0 0.75384615384615383
Parkinsons  0.77000000000000002    0.37168338907469339     0.44927536231884058  -0.070234113712374605                       0 0.75384615384615383
Parkinsons  0.78000000000000003    0.34825174825174821     0.46620046620046618   -0.11888111888111901                       0 0.75384615384615383
Parkinsons  0.79000000000000004    0.37240537240537241     0.45763125763125762   -0.17216117216117241                       0 0.75384615384615383
Parkinsons  0.80000000000000004    0.34871794871794859      0.4564102564102564     -0.230769230769231                       0 0.75384615384615383

======================== exp13_multi_dataset_quantum_benchmark.csv ========================
   dataset                                 model  n_folds      sensitivity_mean      sensitivity_std    specificity_mean       specificity_std       precision_mean        precision_std               f1_mean               f1_std  balanced_accuracy_mean  balanced_accuracy_std            mcc_mean              mcc_std        roc_auc_mean           roc_auc_std         pr_auc_mean            pr_auc_std             fn_mean              fn_std             fp_mean              fp_std  wall_seconds_15fold  train_full_gram_pairs  test_gram_pairs_one_pass
      WDBC            Quantum kernel SVM (PCA-3)       15   0.89320782576596536 0.046484248249960697 0.97947313510693779  0.016518438925654799  0.96399409500120004 0.026405148868300001   0.92623341510637203   0.0229924252003979     0.93634048043645157   0.020567555421364701 0.88804238131590341 0.032595943275821097 0.99052374828933398 0.0045069916336243996 0.98694728227807038 0.0054396745099478004  4.5333333333333332   1.995232412766087  1.4666666666666666  1.1872336794093272     3.91829726099968                3108108                    777024
      WDBC               RBF-SVM (matched PCA-3)       15   0.89940937615356231 0.039019103114023498 0.96452790818987999  0.025725053359169501  0.93967281660478075    0.037770502679034   0.91815242987916479 0.024468366678692299     0.93196864217172115     0.0197590514258312  0.8727942069411817 0.037853580625262402 0.98803687759632719 0.0071624670613873002  0.9840102813108158 0.0080395350708267008  4.2666666666666666  1.6676187756655838  2.5333333333333332  1.8464895909600492    0.431266327999765                      0                         0
      WDBC                RBF-SVM (all features)       15    0.9606496862310816 0.031739607939135502  0.9850417318727176    0.0124208982092663  0.97483977861175819 0.020641940307481201   0.96736029742258001 0.019547647301002599      0.9728457090518996   0.017141823142275098 0.94881580906387042 0.030594963028656101 0.99534834617813095 0.0049069607913871001  0.9938978441554136 0.0059131416303303003  1.6666666666666667  1.3451854182690983  1.0666666666666669 0.88371510168853684  0.60106845900008921                      0                         0
      WDBC Logistic regression L2 (all features)       15    0.9512366186784792   0.0371365961105583 0.99066249347939483 0.0086633721037924008  0.98410313169530417   0.0144841825972467   0.96690804519479623   0.0187521475892568     0.97094955607893718   0.017586910044715302  0.9489582370080204 0.027521806915929901  0.9946430015637896 0.0049305698012509997 0.99329542844359464 0.0053593453077032003  2.0666666666666669  1.5796322658258459 0.66666666666666663 0.61721339984836754   0.3296551049998015                      0                         0
      WPBC            Quantum kernel SVM (PCA-3)       15                     0                    0                   1                     0                    0                    0                     0                    0                     0.5                      0                   0                    0  0.5755898048586221    0.1194166369460838 0.39307716758393929   0.11763081589672369  9.4000000000000004 0.50709255283710986                   0                   0   1.6310903300000064                 376362                     94086
      WPBC               RBF-SVM (matched PCA-3)       15 0.0074074074074073999 0.028688765527462301 0.98222222222222244  0.037515428924742501 0.013333333333333299 0.051639777949432197 0.0095238095238095004 0.036885555678165802     0.49481481481481482  0.0099528577221396006 -0.0224433916028021 0.043000571367689998 0.50711429709279165    0.1173165627232748  0.3524986692063401   0.11519107476648729  9.3333333333333339 0.61721339984836765 0.53333333333333333  1.1254628677422756    0.408793207999679                      0                         0
      WPBC                RBF-SVM (all features)       15  0.014074074074073999 0.037200696610900599 0.98451612903225805  0.030472334501615701 0.055555555555555497  0.14995590180707291  0.022222222222222199    0.058644295879083     0.49929510155316609   0.017642543398358301 -0.0027778271127161 0.062570277709223104  0.6138765432098765    0.1073221880033981 0.40291949542261452    0.1017583672424274  9.2666666666666675 0.59361683970466372 0.46666666666666667 0.91547541643412678  0.38227215699998851                      0                         0
      WPBC Logistic regression L2 (all features)       15    0.1918518518518518   0.1560912921197162 0.91146953405017916  0.048381407186687903  0.38531746031746028   0.2612320002876769   0.24742442742442741  0.18196514668982031      0.5516606929510155   0.085588882319288406 0.13136921116791231  0.21549855069540491 0.65768140183193935   0.11413449448084741 0.41765327951911818   0.13664361937827749  7.5999999999999996  1.5491933384829666  2.6666666666666665  1.4474937289114918  0.31475703399974009                      0                         0
   Coimbra            Quantum kernel SVM (PCA-3)       15   0.75512820512820511    0.177715616186891 0.52181818181818174   0.25344847111169322  0.68864416285468921  0.11616382255233269   0.69676676428732087 0.065051080413487505     0.63847319347319365   0.063246448372668199 0.31775715616178368  0.13630893365236249 0.74469308469308471   0.12009961251983051 0.81475984234181065    0.0916668343234808  3.1333333333333333   2.294921930407801                   5  2.6992062325273118   1.0839771650003058                 129180                     32292
   Coimbra               RBF-SVM (matched PCA-3)       15   0.78205128205128216   0.1793399448440362 0.55696969696969689   0.20036762749303119  0.70349672257566986 0.098378913747644597   0.72290374582639627 0.071276873880775193     0.66951048951048953   0.064997589479519696 0.38365673682826329  0.14837616499755149 0.72290598290598285  0.095223940313649794 0.78016228558919232    0.0902234831491808  2.7999999999999998   2.305273457593636  4.5999999999999996  2.0632844828435211  0.32424207699978069                      0                         0
   Coimbra                RBF-SVM (all features)       15   0.76666666666666672 0.087484652278748895 0.71696969696969703   0.16370368985342179  0.78422894099364682   0.1026084174768728   0.76858997037907595   0.0559874340260735     0.74181818181818182   0.077038694406347197 0.49169791442235061  0.15321248155079009 0.84845376845376863    0.0557227882737499 0.88714532405614599  0.040734824513044297                   3  1.1338934190276817  2.9333333333333331  1.7099150633319551  0.27941566900017278                      0                         0
   Coimbra Logistic regression L2 (all features)       15   0.75598290598290596 0.086494924154246297 0.72969696969696962    0.1849875284307867  0.79320781070781077  0.11877659880889339    0.7670779371848836 0.067166443582780205     0.74283993783993774   0.094309028537030604 0.49241428195405701  0.18801978164981609 0.80506604506604496    0.1103078695624501 0.84643875535908564  0.098140890037077502  3.1333333333333333  1.1254628677422756  2.7999999999999998    1.93464651625488  0.41167706500027629                      0                         0
Parkinsons            Quantum kernel SVM (PCA-3)       15   0.99088122605364004 0.015654785811184901 0.38074074074074071    0.1677460637884528  0.83077895122012779 0.046305769484176001   0.90321432898045795   0.0297465020724191     0.68581098339719027   0.086317403679309498 0.52213832932350224  0.16864884696124771  0.8251170710940825   0.10852304943129649 0.92860908207875803    0.0486512304320132 0.26666666666666661  0.4577377082170635                   6  1.8126539343499315   1.6131566880003447                 365040                     91260
Parkinsons               RBF-SVM (matched PCA-3)       15   0.97486590038314158 0.024207681951007101 0.46370370370370367    0.1854717359713044  0.84820681742882875 0.052460115373079999   0.90654784746072403 0.036329485254658601     0.71928480204342271   0.098668931716456396 0.54836383431833191  0.19122689760833531 0.87807577692635175  0.080295321219945301 0.95554029818919883  0.031333941027622901 0.73333333333333328 0.70373155054899683  5.2000000000000002    1.93464651625488  0.38297769599967069                      0                         0
Parkinsons                RBF-SVM (all features)       15   0.99540229885057485   0.0121333025956723 0.53555555555555545    0.2170103094942904  0.86872938350879536 0.060825354735578201    0.9268292531156096 0.036554282459911901      0.7654789272030651     0.1107574207569829 0.65917583709347349   0.1792118278054797 0.88661558109833982  0.067517248813085406 0.95834797684712003  0.025007844649407598  0.1333333333333333  0.3518657752744983  4.5333333333333332  2.2318047362958038  0.40371467399972971                      0                         0
Parkinsons Logistic regression L2 (all features)       15   0.93203065134099639 0.038471706804146098 0.61185185185185198   0.21012579866347289  0.88175942825567133 0.062310649304040698   0.90463804960251404   0.0352891902922765     0.77194125159642391     0.1033897710728964  0.5817991827141803   0.1739150989626406 0.89549595572584084  0.055590513146063898  0.9640837212700728  0.020065410741779699                   2  1.1338934190276817  3.7999999999999998  2.1447610589527217   0.3589665839999725                      0                         0

======================== exp13b_quantum_paired_tests.csv ========================
   dataset  n_rows  mcnemar_b_quantum_only_correct  mcnemar_c_classical_only_correct  mcnemar_exact_p  wilcoxon_abs_error_stat   wilcoxon_abs_error_p  mean_abs_error_difference_q_minus_c  wilcoxon_5fold_sensitivity_stat  wilcoxon_5fold_sensitivity_p  mcnemar_exact_p_holm4  wilcoxon_abs_error_p_holm4  wilcoxon_5fold_sensitivity_p_holm4
      WDBC     569                               4                                 3                1                    69509  0.0031782355156578998               -0.0037474614587816998                                0                           0.5                      1       0.0095347065469739006                                   1
      WPBC     198                               2                                 0              0.5                     8504   0.095345047866095795               -0.0037804578681653002                                0                             1                      1         0.19069009573219159                                   1
   Coimbra     116                               2                                 6        0.2890625                     3342    0.88826505812265977                0.0071950048506920997                                0                           0.5                      1         0.88826505812265977                                   1
Parkinsons     195                               1                                 3            0.625                     5520 3.1629938301086868e-07                 0.026324042916104701                                0                             1                      1      1.2651975320434743e-06                                   1

======================== exp14_parkinsons_group_aware.csv ========================
                     model  n_recordings  n_identifier_stems  n_positive_stems  n_negative_stems  recording_sensitivity_pooled  recording_specificity_pooled  recording_mcc_pooled  stem_sensitivity_pooled  stem_specificity_pooled  stem_precision_pooled      stem_f1_pooled     stem_mcc_pooled  stem_roc_auc_pooled  stem_pr_auc_pooled  stem_fn  stem_fp
  Logistic Regression (L2)           195                  32                24                 8           0.91156462585034004           0.47916666666666669   0.43381809956843709      0.95833333333333337                      0.5    0.85185185185185186 0.90196078431372562 0.54659439449994851  0.72916666666666663  0.8975812573800751        1        4
  Logistic Regression (L1)           195                  32                24                 8             0.891156462585034           0.52083333333333337    0.4355253685468583      0.95833333333333337                      0.5    0.85185185185185186 0.90196078431372562 0.54659439449994851  0.74479166666666663  0.9039059367068828        1        4
 Linear Discriminant (LDA)           195                  32                24                 8           0.88435374149659862           0.52083333333333337   0.42458486470390422      0.91666666666666663                      0.5    0.84615384615384615                0.88 0.46225016352102422  0.68229166666666674  0.8207958091130596        2        4
       SVM (linear kernel)           195                  32                24                 8           0.88435374149659862           0.45833333333333331   0.36904761904761901      0.95833333333333337                      0.5    0.85185185185185186 0.90196078431372562 0.54659439449994851             0.703125 0.87721618518160505        1        4
          SVM (RBF kernel)           195                  32                24                 8           0.88435374149659862           0.47916666666666669   0.38780395923272448      0.95833333333333337                      0.5    0.85185185185185186 0.90196078431372562 0.54659439449994851             0.640625 0.80688068616479791        1        4
      k-Nearest Neighbours           195                  32                24                 8             0.891156462585034           0.33333333333333331   0.26110013707880669      0.95833333333333337                    0.375     0.8214285714285714 0.88461538461538458 0.43643578047198478            0.7421875 0.87038501148079528        1        5
      Gaussian Naive Bayes           195                  32                24                 8           0.63945578231292521           0.79166666666666663   0.37225672980659108      0.66666666666666663                    0.875    0.94117647058823517 0.78048780487804881 0.47001599194042171            0.7890625 0.87874703308237467        8        1
             Random Forest           195                  32                24                 8           0.93877551020408156           0.35416666666666669   0.37121982722584829                        1                     0.25    0.80000000000000004 0.88888888888888884 0.44721359549995793  0.77864583333333337 0.92063656536185356        0        6
               Extra Trees           195                  32                24                 8           0.95238095238095244           0.41666666666666669   0.46028324996028641                        1                    0.375    0.82758620689655171 0.90566037735849059 0.55708601453115569  0.75520833333333326 0.91091774727906505        0        5
         Gradient Boosting           195                  32                24                 8            0.9251700680272108                        0.3125   0.30117835039078261                        1                    0.375    0.82758620689655171 0.90566037735849059 0.55708601453115569  0.63541666666666663 0.84263858116319224        0        5
     HistGradient Boosting           195                  32                24                 8           0.97278911564625836                         0.375   0.47354459105259039                        1                    0.375    0.82758620689655171 0.90566037735849059 0.55708601453115569  0.80729166666666674 0.92790861471396402        0        5
Quantum kernel SVM (PCA-3)           195                  32                24                 8           0.93877551020408156           0.45833333333333331     0.467827852111877      0.95833333333333337                      0.5    0.85185185185185186 0.90196078431372562 0.54659439449994851  0.71354166666666663 0.85457914132030488        1        4
   RBF-SVM (matched PCA-3)           195                  32                24                 8            0.9251700680272108           0.41666666666666669   0.40271262644320033      0.95833333333333337                      0.5    0.85185185185185186 0.90196078431372562 0.54659439449994851  0.67708333333333337 0.81289946934171986        1        4

======================== exp15_legacy_pca3_all_datasets.csv ========================
   dataset                                     model  n_folds  n_rows  n_features      sensitivity_mean      sensitivity_std    specificity_mean      specificity_std       precision_mean        precision_std               f1_mean               f1_std  balanced_accuracy_mean  balanced_accuracy_std            mcc_mean              mcc_std        roc_auc_mean           roc_auc_std         pr_auc_mean            pr_auc_std             fn_mean              fn_std             fp_mean             fp_std
      WDBC Original RBF-SVM PCA-3 (no angle scaling)       15     569          30   0.89154669619785898 0.044956752477484603 0.96360198226395422 0.022226741578486298   0.9370644370599136 0.033843490333725301   0.91283475017656124 0.027827283025811899     0.92757433923090638      0.023218683469081   0.864960928857327 0.042125789310818403 0.98746518011312201 0.0082749357296135993 0.98392387010050875 0.0082436909310650997  4.5999999999999996  1.9198214202665529  2.6000000000000001  1.594633858557237
      WPBC Original RBF-SVM PCA-3 (no angle scaling)       15     198          32 0.0074074074074073999 0.028688765527462301 0.98222222222222244 0.037515428924742501 0.013333333333333299 0.051639777949432197 0.0095238095238095004 0.036885555678165802     0.49481481481481482  0.0099528577221396006 -0.0224433916028021 0.043000571367689998 0.52956670649143756   0.10594168562069051 0.36607984760401208    0.1131559612643261  9.3333333333333339 0.61721339984836765 0.53333333333333333 1.1254628677422756
   Coimbra Original RBF-SVM PCA-3 (no angle scaling)       15     116           9   0.75085470085470085  0.14143851207206751  0.5951515151515151  0.19839576997179079  0.71315642418583602  0.11221843251775721   0.71905911133868128 0.075712727719020795     0.67300310800310792   0.091196749039527097 0.36594019264331468   0.1881126440346903  0.7226029526029526   0.10290920344284669 0.78245688298034033    0.1001824226065593  3.2000000000000002   1.820517979665599  4.2000000000000002 2.0424075429327448
Parkinsons Original RBF-SVM PCA-3 (no angle scaling)       15     195          22   0.98413793103448277 0.017557044625412001 0.42888888888888888   0.1754208846531301  0.84123263733557851 0.047883715697150699   0.90644423229100635 0.030124964440904099     0.70651340996168577   0.089967335099171999 0.54365821366167422   0.1665591399381334 0.81533418475947206    0.1090624376955051 0.91215475680644442  0.061203944152300897 0.46666666666666667  0.5163977794943222  5.5333333333333332 1.8464895909600487

======================== exp16_legacy_calibration.csv ========================
   dataset        feature_set      variant   n  n_features                brier             log_loss             roc_auc  sensitivity_pooled_at_0_5  specificity_pooled_at_0_5     mcc_pooled_at_0_5
      WDBC       All features Uncalibrated 569          30 0.020045313832946401 0.076418977459740203 0.99468844141430157        0.94339622641509435        0.99159663865546221   0.94367353080920457
      WDBC       All features Legacy Platt 569          30 0.026629312099974701   0.1097979307965901   0.994913059563448          0.929245283018868        0.99719887955182085   0.94037643552646799
      WPBC       All features Uncalibrated 198          32  0.18455936705961351  0.55537812703123202 0.66056080033817111        0.23404255319148931        0.90728476821192039    0.1810383048712719
      WPBC       All features Legacy Platt 198          32  0.17640856346446099  0.53243550286780839 0.64182048752994225                          0        0.98013245033112584 -0.069199630739976897
   Coimbra       All features Uncalibrated 116           9  0.19505436829291681  0.61568456005472116 0.77133413461538469                       0.75        0.63461538461538458   0.38724237004096979
   Coimbra       All features Legacy Platt 116           9  0.20655121142880459   0.6096832557201286 0.75841346153846145                    0.65625                       0.75   0.40461201080127329
Parkinsons       All features Uncalibrated 195          22   0.1023534791757305   0.3156578672615758  0.9000850340136054        0.93877551020408156        0.64583333333333337     0.623661337947364
Parkinsons       All features Legacy Platt 195          22  0.11851318998902299  0.38417077027389618 0.90263605442176875        0.97959183673469397                     0.4375   0.54689921497716765
      WDBC Worst 10 reference Uncalibrated 569          10   0.0201430031318704 0.089083093668020599 0.99172876697848944        0.96226415094339623        0.98879551820728295   0.95482665734505723
      WDBC Worst 10 reference Legacy Platt 569          10 0.026365943442918399  0.11492325163445601 0.99200623645684682        0.94339622641509435        0.99439775910364159   0.94754700832940641
````


**Output (stdout):**

````text

======================== exp17_legacy_threshold.csv ========================
   dataset        feature_set  n_rows  n_features    sensitivity_mean      sensitivity_std  sensitivity_pooled  specificity_pooled    precision_pooled           mcc_pooled  tp  tn  fp  fn              tau_min             tau_max  n_inner_feasible
      WDBC       All features     569          30  0.9388704318936878 0.063375715377174902 0.93867924528301883 0.98599439775910358 0.97549019607843124  0.93224272137289321 199 352   5  13                 0.38 0.66000000000000014                 5
      WPBC       All features     198          32 0.39555555555555549  0.45897093697494001 0.38297872340425532 0.62913907284768211  0.2432432432432432 0.010656984795985701  18  95  56  29 0.059999999999999998 0.77000000000000024                 2
   Coimbra       All features     116           9  0.9525641025641024 0.043381857219714402            0.953125 0.17307692307692299 0.58653846153846156   0.2060868773976231  61   9  43   3  0.20000000000000001 0.23999999999999999                 5
Parkinsons       All features     195          22 0.96597701149425275 0.033918759266065301  0.9659863945578232 0.54166666666666663 0.86585365853658536  0.59805830344923039 142  26  22   5  0.28999999999999998 0.42999999999999999                 5
      WDBC Worst 10 reference     569          10 0.95304540420819495 0.059403803930214202 0.95283018867924518 0.98599439775910358 0.97584541062801922  0.94351114631453759 202 352   5  10  0.41000000000000009 0.59000000000000019                 5
````


**Output (stdout):**

````text

======================== exp18_3model_block_cross.csv ========================
   dataset      feature_block                    model  n_features  n_folds      sensitivity_mean      sensitivity_std    specificity_mean       specificity_std       precision_mean        precision_std               f1_mean                f1_std  balanced_accuracy_mean  balanced_accuracy_std              mcc_mean              mcc_std        roc_auc_mean           roc_auc_std         pr_auc_mean            pr_auc_std             fn_mean              fn_std             fp_mean              fp_std
      WDBC               Mean Logistic Regression (L2)          10       15   0.89313399778516067 0.053771389408889599 0.96637976004173198  0.013793481062082599  0.94173124296877275   0.0207109330340467   0.91543112172624119  0.022543366972147299     0.92975687891344638     0.0214977813385685   0.87013579097698279 0.028869365090803699 0.98582534541154498 0.0049850582655259004 0.98044990344573923 0.0061647687712940002  4.5333333333333332   2.294921930407801  2.3999999999999999 0.98561076060916242
      WDBC               Mean                SVM (RBF)          10       15   0.88069398301956447 0.055923076859490502 0.98042514345331244    0.0188233150296582  0.96551580167112905   0.0318669358130673    0.9196731156258624    0.0287240229399474     0.93055956323643862   0.025438567243909901    0.8797222574184641 0.040013852242532899 0.98672682087356156 0.0073671726925030002 0.98328372080400361 0.0073492432583305003  5.0666666666666664  2.4043611170022654  1.3999999999999999  1.3522468075656269
      WDBC               Mean        Gradient Boosting          10       15   0.91520856404577344 0.053586114143412703 0.95794209702660404  0.029998266369850798  0.93124128589585797 0.044482636269306298   0.92145453849414705  0.028845344384374898      0.9365753305361888   0.024277482791727499   0.87763611107007733 0.043638715370182803 0.98564586990793435 0.0051480061531943002 0.98067260212282159 0.0055740020542873999  3.6000000000000001   2.292846016884444                   3  2.1380899352993952
      WDBC              Error Logistic Regression (L2)          10       15   0.75470653377630126 0.059256933839553799 0.94211006781429318  0.020399360887737601  0.88683889346428058 0.033769481791352703   0.81390723630395112  0.037437177336970003     0.84840830079529717   0.028143837901451502   0.72457647418374693 0.047566418140126999 0.94264620886386363    0.0129492241712335 0.92353171375925303  0.015671661387104099                10.4  2.5014281634983759  4.1333333333333337  1.4573295865416045
      WDBC              Error                SVM (RBF)          10       15   0.72643041712809142 0.046192937668665897 0.94034950443401144  0.031153114747314001  0.88089043723294203 0.053756901906304999   0.79501579913257148  0.038513955789209402     0.83338996078105165   0.027996593317546101   0.69943066860300973 0.057864682490595599 0.93883777109975397  0.012724009719295999 0.91692989168232675  0.018470428582127101                11.6  1.9566735620873064  4.2666666666666666  2.2509257354845511
      WDBC              Error        Gradient Boosting          10       15   0.81616832779623494 0.040283513293818102 0.92724308815858125  0.032069572002949701  0.87171683894897933 0.051313447471674101   0.84202320601798986  0.034062823887391902     0.87170570797740798   0.026347910873244201   0.75483885461178202 0.055270919792930799 0.94954646772394358    0.0151919334851905 0.93540550921461485  0.016482616787318799  7.7999999999999998   1.740279123753264  5.2000000000000002   2.305273457593636
      WDBC              Worst Logistic Regression (L2)          10       15   0.95754891103728323   0.0379267674197757 0.98879760041731857    0.0120588186160775  0.98131748527097362 0.019841973177695399   0.96869172161764283  0.018166273340816198     0.97317325572730085   0.017203336344188799   0.95159711640912437 0.027312675749739099 0.99251973223203682 0.0077347341159389998 0.99254230467190563 0.0071121030368422001                 1.8    1.61245154965971 0.80000000000000004 0.86189160737133463
      WDBC              Worst                SVM (RBF)          10       15   0.94976005906238481 0.036379963572626198 0.98131194574856517    0.0101624300841529  0.96838500998172017 0.016752552603186199   0.95852109216686876  0.018455987261283301     0.96553600240547521   0.017360082256156501   0.93513157842525496   0.0284722753632306 0.99484335672374757 0.0056777323134190004 0.99361807509860123 0.0057016622064324004  2.1333333333333333   1.552264091423817  1.3333333333333333 0.72374686445574588
      WDBC              Worst        Gradient Boosting          10       15   0.92772240679217444 0.039967982797434597 0.97198748043818439  0.018271674993834499  0.95260221640748244   0.0302352768025488   0.93927893647914817  0.023779206469439099     0.94985494361517941   0.020358727365753799    0.9051689053127614 0.036584548007669901  0.9899370887974992    0.0074260750379222 0.98756264224134815 0.0076672823997906998  3.0666666666666669  1.7099150633319551                   2  1.3093073414159542
      WDBC                All Logistic Regression (L2)          30       15    0.9512366186784792   0.0371365961105583 0.99066249347939483 0.0086633721037924008  0.98410313169530417   0.0144841825972467   0.96690804519479623    0.0187521475892568     0.97094955607893718   0.017586910044715302    0.9489582370080204 0.027521806915929901  0.9946430015637896 0.0049305698012509997 0.99329542844359464 0.0053593453077032003  2.0666666666666669  1.5796322658258459 0.66666666666666663 0.61721339984836754
      WDBC                All                SVM (RBF)          30       15    0.9606496862310816 0.031739607939135502  0.9850417318727176    0.0124208982092663  0.97483977861175819 0.020641940307481201   0.96736029742258001  0.019547647301002599      0.9728457090518996   0.017141823142275098   0.94881580906387042 0.030594963028656101 0.99534834617813095 0.0049069607913871001  0.9938978441554136 0.0059131416303303003  1.6666666666666667  1.3451854182690983  1.0666666666666669 0.88371510168853684
      WDBC                All        Gradient Boosting          30       15   0.92617201919527525 0.050593910252650899  0.9766562336984872  0.019507759353705301  0.96043975803193038   0.0321815855529827   0.94196590587780682  0.029511610426649699     0.95141412644688117      0.025292588414341   0.91040936177645837 0.043715889822368501 0.99248000203344677 0.0052375752551716004 0.99011601704339958 0.0064724637250699003  3.1333333333333333  2.1668498091095505  1.6666666666666667  1.3972762620115435
      WPBC               Mean Logistic Regression (L2)          10       15  0.065925925925925902 0.081483886448394494 0.98888888888888882  0.034884541231485502  0.44444444444444442  0.49867549436206371    0.1119191919191919   0.13323228494190789     0.52740740740740732   0.045738437917312101   0.13355617648740209  0.18248500148563751 0.58932218239745116   0.13131727290045631 0.42843940132039288   0.13135271668132151  8.8000000000000007  1.0823255385643318 0.33333333333333331   1.046536236944567
      WPBC               Mean                SVM (RBF)          10       15  0.064444444444444401 0.068904248159762094  0.9866666666666668  0.021081851067789099  0.40000000000000002  0.43552559015807518   0.10767676767676761    0.1113514388977416     0.52555555555555555   0.031483037621048003    0.1139522614365618   0.1424966951555767 0.57952847471127045   0.11656991535368701 0.44960397959250309    0.1160626999563615  8.8000000000000007 0.86189160737133474 0.40000000000000002 0.63245553203367588
      WPBC               Mean        Gradient Boosting          10       15   0.17851851851851849 0.087435203185135105 0.90064516129032268    0.0468378012688749   0.3945238095238095   0.2301083004412765   0.23562342885872289     0.106285383157449     0.53958183990442055   0.052163023170362401   0.11254813653198791  0.14526550028234439 0.52303982477100752    0.0956654089720506 0.34979544653995959  0.085363074436159897  7.7333333333333334  1.0327955589886444                   3  1.4142135623730951
      WPBC              Error Logistic Regression (L2)          10       15  0.057777777777777699 0.070046595728787797 0.96896057347670239  0.032034212203157199  0.28222222222222221  0.36118558450611848  0.090764790764790701    0.1038976070117338     0.51336917562724016   0.031514500157612903  0.052508918727503899  0.13565939308162919 0.55321385902031062  0.086947924542102803 0.34999123220522937  0.091175430847463101  8.8666666666666671 0.91547541643412678 0.93333333333333324 0.96115010472325479
      WPBC              Error                SVM (RBF)          10       15 0.0074074074074073999 0.028688765527462301  0.9866666666666668  0.030341966327759901 0.022222222222222199    0.086066296582387  0.011111111111111099    0.0430331482911935       0.497037037037037     0.0149759752734486   -0.0117795585186434   0.0505340327075521 0.52199681401831932    0.1552521887671893 0.30847191303781651    0.1021605565016927  9.3333333333333339 0.61721339984836765 0.40000000000000002 0.91025898983279963
      WPBC              Error        Gradient Boosting          10       15    0.1696296296296296 0.087435203185135105 0.88308243727598579  0.046534720178910499   0.3261904761904762  0.18686196919020309   0.21904761904761899    0.1114088374292507     0.52635603345280768   0.057352569021865203  0.072098822967069306  0.15141853447647879 0.53724412584627645    0.1118045665770625 0.32320061602097783  0.090590716193923906  7.7999999999999998 0.86189160737133474  3.5333333333333332  1.4074631010979937
      WPBC              Worst Logistic Regression (L2)          10       15    0.1007407407407407  0.10511324762641761  0.9622222222222222  0.035336627579287301  0.39444444444444438  0.37594677660006448   0.15401709401709399   0.15142002358805839     0.53148148148148144   0.056408791806445403    0.1091629647282881     0.18708716900545 0.57969573874950231   0.12586642756120439  0.3788979221595361    0.1093920633434065  8.4666666666666668  1.1872336794093274  1.1333333333333333  1.0600988273786192
      WPBC              Worst                SVM (RBF)          10       15 0.0074074074074073999 0.028688765527462301 0.98444444444444446  0.037515428924742501 0.013333333333333299 0.051639777949432197 0.0095238095238095004  0.036885555678165802     0.49592592592592588  0.0094995281296219002 -0.016280056089188401   0.0386556809628332 0.42906093189964162    0.1059224641713086  0.2722654173921798  0.071816233289483805  9.3333333333333339 0.61721339984836765 0.46666666666666667  1.1254628677422756
      WPBC              Worst        Gradient Boosting          10       15    0.1148148148148148   0.1211836586752668 0.87856630824372783  0.078771335031298795  0.26611111111111108   0.3132089841062477     0.146572536510617   0.14751436353058789     0.49669056152927121   0.073440878540610197 0.0021149930428838002  0.20368413573095509 0.57225806451612893    0.1082887228752797  0.3252182318019981    0.1072826920096472  8.3333333333333339  1.2909944487358054  3.6666666666666665  2.3804761428476167
      WPBC                All Logistic Regression (L2)          32       15    0.1918518518518518   0.1560912921197162 0.91146953405017916  0.048381407186687903  0.38531746031746028   0.2612320002876769   0.24742442742442741   0.18196514668982031      0.5516606929510155   0.085588882319288406   0.13136921116791231  0.21549855069540491 0.65768140183193935   0.11413449448084741 0.41765327951911818   0.13664361937827749  7.5999999999999996  1.5491933384829666  2.6666666666666665  1.4474937289114918
      WPBC                All                SVM (RBF)          32       15  0.014074074074073999 0.037200696610900599 0.98451612903225805  0.030472334501615701 0.055555555555555497  0.14995590180707291  0.022222222222222199     0.058644295879083     0.49929510155316609   0.017642543398358301   -0.0027778271127161 0.062570277709223104  0.6138765432098765    0.1073221880033981 0.40291949542261452    0.1017583672424274  9.2666666666666675 0.59361683970466372 0.46666666666666667 0.91547541643412678
      WPBC                All        Gradient Boosting          32       15   0.17185185185185181  0.16117221975554241 0.88301075268817208  0.058673204855702599   0.2592592592592593   0.1830404908253882    0.1964814814814814    0.1569886656961656     0.52743130227001189   0.064543803683011602  0.049539944127294802   0.1542136741238494 0.55073516527279964    0.1154728913411306 0.31822575896103628  0.071669159802218996  7.7999999999999998    1.61245154965971  3.5333333333333332  1.7674302033770737
   Coimbra                All Logistic Regression (L2)           9       15   0.75598290598290596 0.086494924154246297 0.72969696969696962    0.1849875284307867  0.79320781070781077  0.11877659880889339    0.7670779371848836  0.067166443582780205     0.74283993783993774   0.094309028537030604   0.49241428195405701  0.18801978164981609 0.80506604506604496    0.1103078695624501 0.84643875535908564  0.098140890037077502  3.1333333333333333  1.1254628677422756  2.7999999999999998    1.93464651625488
   Coimbra                All                SVM (RBF)           9       15   0.76666666666666672 0.087484652278748895 0.71696969696969703   0.16370368985342179  0.78422894099364682   0.1026084174768728   0.76858997037907595    0.0559874340260735     0.74181818181818182   0.077038694406347197   0.49169791442235061  0.15321248155079009 0.84845376845376863    0.0557227882737499 0.88714532405614599  0.040734824513044297                   3  1.1338934190276817  2.9333333333333331  1.7099150633319551
   Coimbra                All        Gradient Boosting           9       15   0.77692307692307694   0.1033679579144149 0.67272727272727251   0.14602162185644549  0.75238155635214465 0.092423180682109807   0.75979703066659587  0.075421991887392903     0.72482517482517494   0.088962911406007594   0.45864435942751319   0.1788495879014694 0.81391608391608394  0.094408798965521304 0.85332552986475296  0.082689215693186602  2.8666666666666667  1.3557637102737474  3.3999999999999999  1.5491933384829666
Parkinsons    Vocal_Freq_Tone Logistic Regression (L2)           5       15   0.92727969348659001 0.056449817179422102 0.47111111111111109   0.19968228732918669  0.84380046266273334 0.056459820064573299   0.88217997908342005    0.0444218659261005     0.69919540229885047    0.10359476947316069   0.45688930041474479  0.19336069118485669 0.78177096636866739  0.093374217832171702  0.9157454163226536  0.047102052369598399  2.1333333333333333  1.6417180315870614  5.1333333333333337  2.0655911179772888
Parkinsons    Vocal_Freq_Tone                SVM (RBF)           5       15   0.99325670498084284 0.013962200114656801  0.3888888888888889   0.19125502149002499  0.83336383345671294 0.050404471794555997   0.90546853436778818  0.029709103746650201      0.6910727969348659   0.095117810123285906   0.53464007430803084    0.166261248313366 0.82326947637292458  0.078354073261082305 0.93271481689395364  0.031261767318646598 0.20000000000000001 0.41403933560541251  5.9333333333333336  2.0517124090680472
Parkinsons    Vocal_Freq_Tone        Gradient Boosting           5       15   0.92544061302682001   0.0695810878446287 0.64962962962962956   0.14583377424977431  0.89034563451126503 0.043761443230747203   0.90588258475268679    0.0423633225391815     0.78753512132822479   0.077147001863330303   0.61258486518147348   0.1538809675212596 0.90676032354193281  0.062573518343847598 0.96315448019335437  0.030544637609027599  2.2000000000000002  2.0424075429327448  3.3999999999999999  1.5023790657297038
Parkinsons Perturbation_Error Logistic Regression (L2)          11       15   0.93662835249042142 0.042276951090679499 0.39407407407407408   0.17587607202280381  0.82631409709871073 0.049526642590759902   0.87696151536350209  0.033997197573156099     0.66535121328224778   0.089990317438895601   0.40425739478846479  0.18323307805362921  0.8279438058748404     0.057140265989784 0.94170838334731077    0.0233461201937403  1.8666666666666667  1.2459458063579461  5.8666666666666663  1.8464895909600487
Parkinsons Perturbation_Error                SVM (RBF)          11       15                     1                    0                   0                     0  0.75384615384615405 0.013002373149669399   0.85959079283887474 0.0084299273489544992                     0.5                      0                     0                    0  0.7993529161345253  0.071745033076114795 0.91494503787520165  0.041367082478292003                   0                   0  9.5999999999999996 0.50709255283710986
Parkinsons Perturbation_Error        Gradient Boosting          11       15   0.90697318007662842 0.056831358159306102 0.62370370370370376   0.19138715145001989   0.8812636706185093 0.059270495508989003   0.89263947928223264  0.046683437839117298      0.7653384418901662     0.1042398475655892   0.55005619677058382   0.1942317181476477 0.86340570455512988  0.066865240042099497  0.9448123783802288  0.035465653646997497  2.7333333333333334  1.6676187756655838  3.6666666666666665  1.9518001458970664
Parkinsons Nonlinear_Extremes Logistic Regression (L2)           6       15   0.92758620689655158 0.035417010143652901 0.62444444444444436   0.17917112705165261  0.88434227806806809 0.054345475023910898   0.90408639002903757  0.028501288574690298     0.77601532567049802   0.085699265604674904   0.58534731710079224  0.13893645260206761   0.896994465730098  0.051056045717375098 0.96466933068690797  0.018421606173845299  2.1333333333333333  1.0600988273786194  3.6666666666666665  1.8387366263150311
Parkinsons Nonlinear_Extremes                SVM (RBF)           6       15    0.9660536398467432 0.033835933077494801 0.64370370370370378   0.15883541483707839  0.89313526974817292   0.0458558095484164   0.92721171538930158    0.0277484685499076     0.80487867177522332   0.077945167126356601   0.68013493536895031   0.1245995846018126 0.92797786292039164  0.043967028454845999 0.97323671591886096  0.018341036686776401                   1                   1  3.4666666666666668  1.6417180315870612
Parkinsons Nonlinear_Extremes        Gradient Boosting           6       15   0.95478927203065123 0.027499407322093099 0.71259259259259267   0.17068523928925039  0.91196283508853038 0.049006541674857899   0.93177319856974661  0.023728582872190301     0.83369093231162195   0.079628063366634694   0.70788427015971278   0.1082391426083265 0.92227330779054917  0.049629514132015197 0.96708651372353205  0.027108942097009502  1.3333333333333333 0.81649658092772603  2.7999999999999998  1.7402791237532642
Parkinsons                All Logistic Regression (L2)          22       15   0.93203065134099639 0.038471706804146098 0.61185185185185198   0.21012579866347289  0.88175942825567133 0.062310649304040698   0.90463804960251404    0.0352891902922765     0.77194125159642391     0.1033897710728964    0.5817991827141803   0.1739150989626406 0.89549595572584084  0.055590513146063898  0.9640837212700728  0.020065410741779699                   2  1.1338934190276817  3.7999999999999998  2.1447610589527217
Parkinsons                All                SVM (RBF)          22       15   0.99540229885057485   0.0121333025956723 0.53555555555555545    0.2170103094942904  0.86872938350879536 0.060825354735578201    0.9268292531156096  0.036554282459911901      0.7654789272030651     0.1107574207569829   0.65917583709347349   0.1792118278054797 0.88661558109833982  0.067517248813085406 0.95834797684712003  0.025007844649407598  0.1333333333333333  0.3518657752744983  4.5333333333333332  2.2318047362958038
Parkinsons                All        Gradient Boosting          22       15   0.95701149425287357 0.041533647588429798 0.77999999999999992   0.13308177327277029   0.9305431259199336 0.040991503833061597   0.94294251664751616  0.032304608104220998     0.86850574712643691   0.073192390863211498   0.76427307174483894  0.13777661448594131 0.96118348233290762  0.039788881106092702 0.98689800651934678    0.0137382972442869  1.2666666666666666  1.2227992865708155  2.1333333333333333  1.3020130933435712

======================== exp19_parkinsons_split_overlap.csv ========================
 fold  repeat  n_test_recordings  n_unique_test_stems  n_shared_stems  n_test_recordings_with_stem_in_training
    1       1                 39                   25              25                                       39
    2       1                 39                   26              26                                       39
    3       1                 39                   23              23                                       39
    4       1                 39                   26              26                                       39
    5       1                 39                   23              23                                       39
    6       2                 39                   22              22                                       39
    7       2                 39                   22              22                                       39
    8       2                 39                   25              25                                       39
    9       2                 39                   23              23                                       39
   10       2                 39                   21              21                                       39
   11       3                 39                   26              26                                       39
   12       3                 39                   23              23                                       39
   13       3                 39                   21              21                                       39
   14       3                 39                   24              24                                       39
   15       3                 39                   22              22                                       39

======================== exp20_model_coverage_audit.csv ========================
   dataset                          model_or_variant                                           origin  n_folds                              results_file status
      WDBC                  Logistic Regression (L2)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                  Logistic Regression (L1)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                 Linear Discriminant (LDA)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                       SVM (linear kernel)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                          SVM (RBF kernel)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                      k-Nearest Neighbours                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                      Gaussian Naive Bayes                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                             Random Forest                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                               Extra Trees                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                         Gradient Boosting                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                     HistGradient Boosting                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WDBC                Quantum kernel SVM (PCA-3)                       original quantum estimator       15 exp13_multi_dataset_quantum_benchmark.csv   PASS
      WDBC                   RBF-SVM (matched PCA-3)                    additional matched comparator       15 exp13_multi_dataset_quantum_benchmark.csv   PASS
      WDBC Original RBF-SVM PCA-3 (no angle scaling)              original unmatched PCA-3 comparator       15        exp15_legacy_pca3_all_datasets.csv   PASS
      WDBC     Original-style Platt-calibrated L2 LR original calibration architecture (all features)        5              exp16_legacy_calibration.csv   PASS
      WPBC                  Logistic Regression (L2)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                  Logistic Regression (L1)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                 Linear Discriminant (LDA)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                       SVM (linear kernel)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                          SVM (RBF kernel)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                      k-Nearest Neighbours                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                      Gaussian Naive Bayes                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                             Random Forest                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                               Extra Trees                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                         Gradient Boosting                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                     HistGradient Boosting                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
      WPBC                Quantum kernel SVM (PCA-3)                       original quantum estimator       15 exp13_multi_dataset_quantum_benchmark.csv   PASS
      WPBC                   RBF-SVM (matched PCA-3)                    additional matched comparator       15 exp13_multi_dataset_quantum_benchmark.csv   PASS
      WPBC Original RBF-SVM PCA-3 (no angle scaling)              original unmatched PCA-3 comparator       15        exp15_legacy_pca3_all_datasets.csv   PASS
      WPBC     Original-style Platt-calibrated L2 LR original calibration architecture (all features)        5              exp16_legacy_calibration.csv   PASS
   Coimbra                  Logistic Regression (L2)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                  Logistic Regression (L1)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                 Linear Discriminant (LDA)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                       SVM (linear kernel)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                          SVM (RBF kernel)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                      k-Nearest Neighbours                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                      Gaussian Naive Bayes                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                             Random Forest                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                               Extra Trees                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                         Gradient Boosting                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                     HistGradient Boosting                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
   Coimbra                Quantum kernel SVM (PCA-3)                       original quantum estimator       15 exp13_multi_dataset_quantum_benchmark.csv   PASS
   Coimbra                   RBF-SVM (matched PCA-3)                    additional matched comparator       15 exp13_multi_dataset_quantum_benchmark.csv   PASS
   Coimbra Original RBF-SVM PCA-3 (no angle scaling)              original unmatched PCA-3 comparator       15        exp15_legacy_pca3_all_datasets.csv   PASS
   Coimbra     Original-style Platt-calibrated L2 LR original calibration architecture (all features)        5              exp16_legacy_calibration.csv   PASS
Parkinsons                  Logistic Regression (L2)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                  Logistic Regression (L1)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                 Linear Discriminant (LDA)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                       SVM (linear kernel)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                          SVM (RBF kernel)                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                      k-Nearest Neighbours                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                      Gaussian Naive Bayes                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                             Random Forest                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                               Extra Trees                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                         Gradient Boosting                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                     HistGradient Boosting                            original 11-model zoo       15               exp10_multi_dataset_zoo.csv   PASS
Parkinsons                Quantum kernel SVM (PCA-3)                       original quantum estimator       15 exp13_multi_dataset_quantum_benchmark.csv   PASS
Parkinsons                   RBF-SVM (matched PCA-3)                    additional matched comparator       15 exp13_multi_dataset_quantum_benchmark.csv   PASS
Parkinsons Original RBF-SVM PCA-3 (no angle scaling)              original unmatched PCA-3 comparator       15        exp15_legacy_pca3_all_datasets.csv   PASS
Parkinsons     Original-style Platt-calibrated L2 LR original calibration architecture (all features)        5              exp16_legacy_calibration.csv   PASS

======================== exp21_learning_curve_sampling_audit.csv ========================
 n_train  fold  positive_count   positive_fraction  full_train_positive_fraction
      68     1              47 0.69117647058823528           0.37142857142857139
      68     2              48 0.70588235294117652           0.37142857142857139
      68     3              47 0.69117647058823528           0.37362637362637358
      68     4              47 0.69117647058823528           0.37362637362637358
      68     5              43 0.63235294117647056           0.37280701754385959
     145     1              76 0.52413793103448281           0.37142857142857139
     145     2              77 0.53103448275862064           0.37142857142857139
     145     3              75 0.51724137931034486           0.37362637362637358
     145     4              79 0.54482758620689653           0.37362637362637358
     145     5              68  0.4689655172413793           0.37280701754385959
     222     1             110 0.49549549549549549           0.37142857142857139
     222     2             117 0.52702702702702697           0.37142857142857139
     222     3             114 0.51351351351351349           0.37362637362637358
     222     4             114 0.51351351351351349           0.37362637362637358
     222     5             113 0.50900900900900903           0.37280701754385959
     300     1             131 0.43666666666666659           0.37142857142857139
     300     2             135 0.45000000000000001           0.37142857142857139
     300     3             135 0.45000000000000001           0.37362637362637358
     300     4             134  0.4466666666666666           0.37362637362637358
     300     5             131 0.43666666666666659           0.37280701754385959
     377     1             150 0.39787798408488062           0.37142857142857139
     377     2             149 0.39522546419098142           0.37142857142857139
     377     3             154 0.40848806366047741           0.37362637362637358
     377     4             151 0.40053050397877982           0.37362637362637358
     377     5             150 0.39787798408488062           0.37280701754385959
     455     1             169 0.37142857142857139           0.37142857142857139
     455     2             169 0.37142857142857139           0.37142857142857139
     455     3             170 0.37362637362637358           0.37362637362637358
     455     4             170 0.37362637362637358           0.37362637362637358
     455     5             170 0.37362637362637358           0.37280701754385959
````

**Interpretation.** CSV files preserve the saved precision and are the authoritative source for numerical extraction. The two blank fixed-block stability values are documented not-applicable entries; all other result values are required to be finite.

## 30. Final executable quality gate

The final cell asserts existence, nonempty contents and finite numeric values for every required result CSV; it documents the two structurally not-applicable fixed-subset stability cells. It validates the PDF signatures, PNG integrity and 300-dpi metadata, raw and cleaned SHA-256 fingerprints, class counts, fold counts, bootstrap size and WPBC's four intentional missing predictors. A failure stops execution rather than silently reporting success.


<!-- Notebook code cell 113; execution count 38 -->

````python
from PIL import Image as PILImage

EXPECTED_RESULTS = [
    'exp1_feature_blocks.csv', 'exp2_model_zoo.csv', 'exp2_architectures.csv',
    'exp3_feature_selection.csv', 'exp4a_threshold_curve.csv', 'exp4b_nested_threshold.csv',
    'exp5a_calibration.csv', 'exp5b_reliability.csv', 'exp5c_bootstrap_ci.csv',
    'exp6_quantum_vs_classical.csv', 'exp7_mcnemar.csv', 'exp7_bootstrap_diff.csv',
    'exp8_learning_curve.csv', 'wdbc_numeric_reproducibility.csv', 'dataset_overlap_audit.csv',
    'exp9_multi_dataset_blocks.csv', 'exp9b_block_fold_metrics.csv', 'exp9c_selected_features.csv',
    'exp10_multi_dataset_zoo.csv', 'exp10b_multi_dataset_zoo_folds.csv',
    'exp11_multi_dataset_nested_threshold.csv', 'exp11b_nested_threshold_folds.csv', 'exp11c_nested_threshold_oof.csv',
    'exp12_multi_dataset_calibration.csv', 'exp12a_reliability_bins.csv', 'exp12b_dca_curves.csv', 'exp12c_calibration_oof.csv',
    'exp13_multi_dataset_quantum_benchmark.csv', 'exp13b_quantum_paired_tests.csv',
    'exp13c_quantum_fold_metrics.csv', 'exp13d_quantum_oof.csv',
    'exp14_parkinsons_group_aware.csv', 'exp14b_parkinsons_group_oof.csv',
    'exp15_legacy_pca3_all_datasets.csv',
    'exp15b_legacy_pca3_fold_metrics.csv',
    'exp15c_legacy_pca3_oof.csv',
    'exp16_legacy_calibration.csv',
    'exp16b_legacy_calibration_oof.csv',
    'exp17_legacy_threshold.csv',
    'exp17b_legacy_threshold_folds.csv',
    'exp17c_legacy_threshold_oof.csv',
    'exp18_3model_block_cross.csv',
    'exp18b_3model_block_folds.csv',
    'exp19_parkinsons_split_overlap.csv',
    'exp20_model_coverage_audit.csv',
    'exp21_learning_curve_sampling_audit.csv',
]
EXPECTED_FIGURES = ['fig_blocks', 'fig_models', 'fig_models2', 'fig_threshold', 'fig_threshold2',
                    'fig_reliability', 'fig_kernel', 'fig_learning', 'fig_multi_zoo_heatmap',
                    'fig_multi_quantum_benchmark', 'fig_multi_calibration', 'fig_multi_dca']
assert len(EXPECTED_RESULTS) == len(set(EXPECTED_RESULTS)) == 46
for filename in EXPECTED_RESULTS:
    path = RESULTS / filename
    assert path.is_file() and path.stat().st_size > 0, f'Missing or empty CSV: {path}'
    df = pd.read_csv(path)
    assert not df.empty, f'Empty CSV: {path}'
    # Exactly two fixed-subset stability cells are legitimately not applicable.
    if filename == 'exp3_feature_selection.csv':
        allowed = (df.configuration.str.startswith('fixed:') & df.fold_selection_stability_jaccard.isna())
        assert allowed.sum() == 2
        checked = df.copy()
        checked.loc[allowed, 'fold_selection_stability_jaccard'] = 1.0
    else:
        checked = df
    assert not checked.isna().any().any(), f'Unexpected missing value in {path}'
    numeric = checked.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all(), f'Nonfinite numeric value in {path}'

for name in EXPECTED_FIGURES:
    pdf, png = FIG / (name + '.pdf'), FIG / (name + '.png')
    assert pdf.is_file() and pdf.stat().st_size > 500 and pdf.read_bytes()[:4] == b'%PDF', pdf
    assert b'/Subtype /Image' not in pdf.read_bytes(), f'Embedded raster in vector PDF: {pdf}'
    assert png.is_file() and png.stat().st_size > 500, png
    with PILImage.open(png) as img:
        img.verify()
    with PILImage.open(png) as img:
        assert img.width >= 500 and img.height >= 500, (png, img.size)
        dpi = img.info.get('dpi', (0, 0))
        assert abs(dpi[0] - 300) < 1 and abs(dpi[1] - 300) < 1, (png, dpi)

for line in (DATA / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    digest, filename = line.split('  ', 1)
    assert sha256_file(filename) == digest, f'Checksum mismatch: {filename}'
for dataset, dset in DATASETS.items():
    clean = pd.read_csv(DATA / f'{dataset.lower()}_clean.csv')
    if dataset == 'WPBC':
        assert clean.isna().sum().sum() == 4 and clean.lymph_node_status.isna().sum() == 4
    else:
        assert not clean.isna().any().any()
assert multi_zoo.groupby('dataset').size().eq(11).all()
assert quantum_multi.groupby('dataset').size().eq(4).all()
assert (nested_multi.bootstrap_replicates == 2000).all()
assert dca_multi.groupby('dataset').size().eq(76).all()
print(f'PASS: {len(EXPECTED_RESULTS)} CSV results; {len(EXPECTED_FIGURES)} figures × PDF and 300-dpi PNG.')
print('PASS: all result cells finite, except exactly two documented not-applicable fixed-block stability values.')
print('PASS: four clean dataset exports, source hashes, class counts, fold counts, and required bootstrap replicates.')
print('PASS: WPBC has exactly four intentional missing measurements, imputed only within training pipelines.')
assert len(coverage) == 60 and coverage.groupby('dataset').size().eq(15).all()
assert len(block3_multi) == 39 and len(legacy_pca) == 4
print('PASS: original-estimator AST identity and all-dataset coverage audit, legacy variants and split-overlap checks.')
````


**Output (stdout):**

````text
PASS: 46 CSV results; 12 figures × PDF and 300-dpi PNG.
PASS: all result cells finite, except exactly two documented not-applicable fixed-block stability values.
PASS: four clean dataset exports, source hashes, class counts, fold counts, and required bootstrap replicates.
PASS: WPBC has exactly four intentional missing measurements, imputed only within training pipelines.
PASS: original-estimator AST identity and all-dataset coverage audit, legacy variants and split-overlap checks.
````
