"""Quantify the 13 WDBC/WPBC shared identifiers whose 30 shared features are not identical.

Reads data/raw/wdbc.data and data/raw/wpbc.data (no header; WDBC: id, diagnosis, 30 features;
WPBC: id, outcome, time, 32 features, of which the first 30 are the WDBC features).
Writes results/wisconsin_shared_pairs_differences.csv (one row per differing identifier).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "wisconsin_shared_pairs_differences.csv"

wdbc = pd.read_csv(RAW / "wdbc.data", header=None)
wpbc = pd.read_csv(RAW / "wpbc.data", header=None)
wd = wdbc.set_index(0)
wp = wpbc.set_index(0)
shared = wd.index.intersection(wp.index)
A = wd.loc[shared].iloc[:, 1:31].to_numpy(dtype=float)   # WDBC 30 features
B = wp.loc[shared].iloc[:, 2:32].to_numpy(dtype=float)   # WPBC first 30 feature columns (after outcome, time)
differs = ~np.all(np.isclose(A, B, rtol=0, atol=1e-12), axis=1)
rows = []
for idx, a, b, flag in zip(shared, A, B, differs):
    if not flag:
        continue
    d = np.abs(a - b)
    rows.append({
        "identifier": int(idx),
        "n_features_differing": int(np.sum(d > 1e-12)),
        "max_abs_difference": float(d.max()),
        "max_relative_difference": float(np.max(d / np.maximum(np.abs(a), 1e-12))),
        "all_differences_within_0.005": bool(np.all(d <= 0.005 + 1e-12)),
        "wdbc_diagnosis": str(wd.loc[idx, 1]),
    })
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"shared identifiers: {len(shared)}; identical on 30 features: {int((~differs).sum())}; differing: {int(differs.sum())}")
print(df.to_string(index=False))
