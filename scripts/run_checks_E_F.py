"""E: calibration cross-validation checks.  F: gamma=0.25 RBF control on the quantum-path inputs.
Reads data/*_clean.csv from the repository root. Writes results/E_*.csv and results/F_*.csv.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss, log_loss, matthews_corrcoef, roc_auc_score
warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def load(name):
    df = pd.read_csv(DATA / f"{name}_clean.csv")
    y = df.pop("target").astype(int).to_numpy()
    groups = df.pop("subject_stem").to_numpy() if "subject_stem" in df.columns else None
    return df.astype(float), y, groups

DS = ["wdbc", "wpbc", "coimbra", "parkinsons"]
NAMES = {"wdbc": "WDBC", "wpbc": "WPBC", "coimbra": "Coimbra", "parkinsons": "Parkinsons"}

def lr_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("model", LogisticRegression(penalty="l2", max_iter=2000, random_state=SEED))])

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def slope_intercept(y, p):
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

# ---------------------------------------------------------------- E
E_rows = []
E_check = []
for key in DS:
    X, y, _ = load(key)
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # CV_OOF in project code
    cv_platt_default = None
    raw = np.full(len(y), np.nan)
    cur = np.full(len(y), np.nan)      # Platt as in project (CalibratedClassifierCV cv=5 default)
    expl = np.full(len(y), np.nan)     # same, with explicit StratifiedKFold(5, shuffle=False)
    insample = np.full(len(y), np.nan) # variant: sigmoid fitted on in-sample scores of the full-train model
    for tr, te in cv_outer.split(X, y):
        Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
        base = lr_pipeline().fit(Xtr, ytr)
        raw[te] = base.predict_proba(Xte)[:, 1]
        cal = CalibratedClassifierCV(estimator=lr_pipeline(), method="sigmoid", cv=5).fit(Xtr, ytr)
        cur[te] = cal.predict_proba(Xte)[:, 1]
        cal_e = CalibratedClassifierCV(estimator=lr_pipeline(), method="sigmoid",
                                       cv=StratifiedKFold(n_splits=5, shuffle=False)).fit(Xtr, ytr)
        expl[te] = cal_e.predict_proba(Xte)[:, 1]
        # Variant: sigmoid (Platt) map fitted to the training fold's own base-model scores (no cross-fitting)
        s_tr = base.predict_proba(Xtr)[:, 1]
        m = LogisticRegression(C=1e6, max_iter=5000).fit(logit(s_tr).reshape(-1, 1), ytr)
        insample[te] = m.predict_proba(logit(raw[te]).reshape(-1, 1))[:, 1]
    max_diff = float(np.max(np.abs(cur - expl)))
    E_check.append({"dataset": NAMES[key], "max_abs_diff_default_vs_explicit_cv": max_diff,
                    "identical_within_1e-12": max_diff < 1e-12})
    for variant, p in [("uncalibrated", raw), ("Platt (project, internal cv=5)", cur),
                       ("Platt, in-sample mapping (variant)", insample)]:
        s, i = slope_intercept(y, p)
        E_rows.append({"dataset": NAMES[key], "n": len(y), "variant": variant,
                       "calibration_slope": round(s, 4), "calibration_intercept": round(i, 4),
                       "brier": round(brier_score_loss(y, p), 6), "log_loss": round(log_loss(y, p, labels=[0, 1]), 6)})
pd.DataFrame(E_rows).to_csv(OUT / "E_calibration_slope_variants.csv", index=False)
pd.DataFrame(E_check).to_csv(OUT / "E_calibration_cv_explicit_check.csv", index=False)
print(pd.DataFrame(E_rows).to_string(index=False)); print(pd.DataFrame(E_check).to_string(index=False))

# ---------------------------------------------------------------- F
def ang_pipeline(gamma, kernel_fn=None):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("pca", PCA(n_components=3, random_state=SEED)),
                     ("angle", MinMaxScaler(feature_range=(0, np.pi))),
                     ("svc", SVC(kernel="rbf", gamma=gamma, probability=True, random_state=SEED))])

def quantum_state(x, n=3):
    # RY encoding then CNOT(0,1), CNOT(1,2), as in the project's quantum_state
    psi = np.array([1.0 + 0j])
    for xi in x:
        psi = np.kron(psi, np.array([np.cos(xi / 2), np.sin(xi / 2)], dtype=complex))
    return psi

# Exact quantum-helper source copied from run_all_experiments.py (RY, cnot, quantum_state, fidelity_kernel_matrix)
_src = (ROOT / "run_all_experiments.py").read_text(encoding="utf-8")
_start = _src.index("RY = lambda t:")
_end = _src.index("return np.abs(inner) ** 2", _start) + len("return np.abs(inner) ** 2")
H = {"np": np}
exec(_src[_start:_end], H)

class QuantumKernelSVM:
    def __init__(self):
        self.imp = SimpleImputer(strategy="median")
    def fit(self, X, y):
        Xi = self.imp.fit_transform(X)
        self.scaler = StandardScaler().fit(Xi)
        Z = self.scaler.transform(Xi)
        self.pca = PCA(n_components=3, random_state=SEED).fit(Z)
        Zp = self.pca.transform(Z)
        self.angle = MinMaxScaler(feature_range=(0, np.pi)).fit(Zp)
        A = self.angle.transform(Zp)
        self.svc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=SEED)
        self.svc.fit(H["fidelity_kernel_matrix"](A, A), y)
        self.A = A
        return self
    def _A(self, X):
        Xi = self.imp.transform(X)
        return self.angle.transform(self.pca.transform(self.scaler.transform(Xi)))
    def predict_proba(self, X):
        return self.svc.predict_proba(H["fidelity_kernel_matrix"](self._A(X), self.A))
    def predict(self, X):
        return self.svc.predict(H["fidelity_kernel_matrix"](self._A(X), self.A))

def metrics(y, p, yhat):
    return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1]),
            "mcc": matthews_corrcoef(y, yhat), "roc_auc": roc_auc_score(y, p)}

F_rows, F_fold = [], []
for key in DS:
    X, y, groups = load(key)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    splits = list(cv.split(X, y))
    per = {"quantum": [], "rbf_gamma025": []}
    for f, (tr, te) in enumerate(splits, 1):
        Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
        qm = QuantumKernelSVM().fit(Xtr, ytr)
        rm = ang_pipeline(0.25).fit(Xtr, ytr)
        q = qm.predict_proba(Xte)[:, 1]; r = rm.predict_proba(Xte)[:, 1]
        for name, p, yh in [("quantum", q, qm.predict(Xte)), ("rbf_gamma025", r, rm.predict(Xte))]:
            m = metrics(yte, p, yh); per[name].append(m)
            F_fold.append({"dataset": NAMES[key], "fold": f, "model": name, **{k: round(v, 8) for k, v in m.items()}})
    fq = pd.DataFrame(per["quantum"]); fr = pd.DataFrame(per["rbf_gamma025"])
    d = fq - fr
    F_rows.append({"dataset": NAMES[key], "folds": len(splits),
                   "quantum_brier_mean": round(fq.brier.mean(), 5), "rbf025_brier_mean": round(fr.brier.mean(), 5),
                   "brier_diff_quantum_minus_rbf_mean": round(d.brier.mean(), 5),
                   "brier_diff_sd": round(d.brier.std(), 5), "brier_folds_quantum_better": int((d.brier < 0).sum()),
                   "logloss_diff_quantum_minus_rbf_mean": round(d.log_loss.mean(), 5),
                   "mcc_quantum_mean": round(fq.mcc.mean(), 4), "mcc_rbf025_mean": round(fr.mcc.mean(), 4),
                   "auc_quantum_mean": round(fq.roc_auc.mean(), 4), "auc_rbf025_mean": round(fr.roc_auc.mean(), 4)})
pd.DataFrame(F_fold).to_csv(OUT / "F_rbf_gamma025_fold_metrics.csv", index=False)
pd.DataFrame(F_rows).to_csv(OUT / "F_rbf_gamma025_control_summary.csv", index=False)
print(pd.DataFrame(F_rows).to_string(index=False))
