"""Check how often Parkinson's held-out recordings share a filename stem with training recordings.

Uses the row-wise RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42) partition
on data/parkinsons_clean.csv (195 rows). Each row is held out once per repeat, so the number of
held-out appearances is 195 x 3 = 585. Writes results/parkinsons_stem_overlap_check.csv.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
frame = pd.read_csv(ROOT / "data" / "parkinsons_clean.csv")
X = frame.drop(columns=["target", "subject_stem"])
y = frame["target"].to_numpy()
stems = frame["subject_stem"].to_numpy()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
held_out = 0
shared = 0
for train_idx, test_idx in cv.split(X, y):
    train_stems = set(stems[train_idx])
    for i in test_idx:
        held_out += 1
        shared += int(stems[i] in train_stems)

out = pd.DataFrame([{
    "held_out_appearances": held_out,
    "with_training_stem": shared,
    "distinct_stems": int(pd.Series(stems).nunique()),
    "rows": len(frame),
}])
out.to_csv(ROOT / "results" / "parkinsons_stem_overlap_check.csv", index=False)
print(out.to_string(index=False))
assert held_out == 585, held_out
