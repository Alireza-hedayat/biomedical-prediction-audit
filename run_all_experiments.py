#!/usr/bin/env python3
"""Run the same cells as MASTER_REPRODUCTION.ipynb from any working directory."""
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.chdir(Path(__file__).resolve().parent)


# ===== Notebook cell 2 =====
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

# ===== Notebook cell 5 =====
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

# ===== Notebook cell 8 =====
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

# ===== Notebook cell 11 =====
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

# ===== Notebook cell 14 =====
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

# ===== Notebook cell 17 =====
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
    "Extra Trees":              "non-bootstrap randomised-threshold CART ensemble (bootstrap=False), 300 trees",
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

# ===== Notebook cell 20 =====
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

# ===== Notebook cell 23 =====
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

# ===== Notebook cell 26 =====
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

# ===== Notebook cell 29 =====
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

# ===== Notebook cell 32 =====
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

# ===== Notebook cell 35 =====
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

# ===== Notebook cell 38 =====
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

# ===== Notebook cell 41 =====
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

# ===== Notebook cell 44 =====
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

# ===== Notebook cell 47 =====
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

# ===== Notebook cell 50 =====
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

# ===== Notebook cell 53 =====
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

# ===== Notebook cell 56 =====
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

# ===== Notebook cell 59 =====
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
            exp_text = expected[col].fillna('<not-applicable>').astype(str).tolist()
            got_text = measured[col].fillna('<not-applicable>').astype(str).tolist()
            if ref_path.name == 'exp2_architectures.csv' and col == 'architecture':
                # Documented metadata correction (CHANGELOG.md, 2026-11-23): the original reference
                # text 'bagged CART' for Extra Trees is inconsistent with bootstrap=False.
                # Only the Extra Trees description may differ; every other cell must match.
                for i, (e_txt, g_txt) in enumerate(zip(exp_text, got_text)):
                    if e_txt != g_txt:
                        assert 'bagged CART, random split thresholds' in e_txt and 'bootstrap=False' in g_txt, (i, e_txt, g_txt)
                        exp_text[i] = g_txt
            assert exp_text == got_text, (ref_path.name, col)
    reference_audit_rows.append({'experiment_csv': ref_path.name, 'rows': len(expected),
                                 'columns_compared': len(expected.columns),
                                 'max_abs_numeric_difference': max(numeric_errors, default=0.0),
                                 'status': 'PASS'})
reference_audit = pd.DataFrame(reference_audit_rows)
assert len(reference_audit) == 13 and (reference_audit.status == 'PASS').all()
reference_audit.to_csv(RESULTS / 'wdbc_numeric_reproducibility.csv', index=False)
print('All WDBC reference fixture checks passed (timings excluded; floating tolerance 1e-8 relative).')
display(reference_audit)

# ===== Notebook cell 62 =====
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

# ===== Notebook cell 65 =====
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

# ===== Notebook cell 68 =====
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

# ===== Notebook cell 71 =====
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

# ===== Notebook cell 74 =====
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

# ===== Notebook cell 77 =====
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

# ===== Notebook cell 80 =====
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

# ===== Notebook cell 83 =====
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


# ===== Notebook cell 86 =====
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


# ===== Notebook cell 89 =====
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


# ===== Notebook cell 92 =====
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


# ===== Notebook cell 95 =====
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


# ===== Notebook cell 98 =====
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


# ===== Notebook cell 101 =====
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


# ===== Notebook cell 104 =====
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

# ===== Notebook cell 107 =====
WDBC_FIGURES = ['fig_blocks', 'fig_models', 'fig_models2', 'fig_threshold', 'fig_threshold2', 'fig_reliability', 'fig_kernel', 'fig_learning']
MULTI_FIGURES = ['fig_multi_zoo_heatmap', 'fig_multi_quantum_benchmark', 'fig_multi_calibration', 'fig_multi_dca']
for name in WDBC_FIGURES + MULTI_FIGURES:
    print('FIGURE:', name)
    display(Image(filename=str(FIG / (name + '.png')), width=750))

# ===== Notebook cell 110 =====
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

# ===== Notebook cell 113 =====
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

