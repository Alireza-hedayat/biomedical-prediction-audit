"""Platt calibration with ensemble=False versus the project default (ensemble='auto', i.e. True).

Same outer split as scripts/run_checks_E_F.py block E (StratifiedKFold(5, shuffle=True, random_state=42)),
same lr_pipeline(). Writes results/E_calibration_ensemble_variant.csv with OOF calibration slope/intercept.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DATA = {"WDBC": "wdbc_clean.csv", "WPBC": "wpbc_clean.csv", "Coimbra": "coimbra_clean.csv", "Parkinsons": "parkinsons_clean.csv"}

def lr_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("model", LogisticRegression(penalty="l2", max_iter=2000, random_state=SEED))])

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

rows = []
for name, fn in DATA.items():
    df = pd.read_csv(ROOT / "data" / fn)
    y = df["target"].to_numpy().astype(int)
    X = df.drop(columns=[c for c in ["target", "subject_stem"] if c in df.columns])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p_def = np.full(len(y), np.nan)
    p_off = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y):
        Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
        p_def[te] = CalibratedClassifierCV(estimator=lr_pipeline(), method="sigmoid", cv=5).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        p_off[te] = CalibratedClassifierCV(estimator=lr_pipeline(), method="sigmoid", cv=5, ensemble=False).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    for variant, p in [("Platt, ensemble=True (project default)", p_def), ("Platt, ensemble=False", p_off)]:
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(logit(p).reshape(-1, 1), y)
        rows.append({"dataset": name, "n": len(y), "variant": variant,
                     "calibration_slope": round(float(lr.coef_[0][0]), 4),
                     "calibration_intercept": round(float(lr.intercept_[0]), 4),
                     "brier": round(brier_score_loss(y, p), 6),
                     "log_loss": round(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1]), 6)})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "E_calibration_ensemble_variant.csv", index=False)
print(out.to_string(index=False))
