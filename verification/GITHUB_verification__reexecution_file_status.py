#!/usr/bin/env python3
"""Per-file status of a standalone re-execution.

Usage: python verification/reexecution_file_status.py RERUN_RESULTS_DIR PRE_RUN_RESULTS_DIR OUTPUT_CSV

For every CSV in RERUN_RESULTS_DIR, compares it with the same file name in PRE_RUN_RESULTS_DIR.
Non-timing numeric columns: rtol=1e-8, atol=1e-10, NaNs equal. Non-numeric columns: exact string equality.
Timing columns (names containing 'second' or 'time') are reported separately, not compared.
Status values: identical | identical_except_timing | DIFFERS_non_timing.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

new, old, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = []
for p in sorted(new.glob('*.csv')):
    q = old / p.name
    if not q.exists():
        rows.append((p.name, 'missing_in_pre_run', '', '')); continue
    a, b = pd.read_csv(p), pd.read_csv(q)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        rows.append((p.name, 'shape_or_columns_differ', '', '')); continue
    bad, timing = [], []
    for col in a:
        if 'second' in col.lower() or 'time' in col.lower():
            if not a[col].equals(b[col]):
                timing.append(col)
            continue
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            if not np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=1e-8, atol=1e-10, equal_nan=True):
                bad.append(col)
        elif not a[col].astype(str).equals(b[col].astype(str)):
            bad.append(col)
    status = 'DIFFERS_non_timing' if bad else ('identical_except_timing' if timing else 'identical')
    rows.append((p.name, status, ';'.join(bad), ';'.join(timing)))
out = pd.DataFrame(rows, columns=['file', 'status', 'non_timing_columns_differing', 'timing_columns_differing'])
out.to_csv(out_path, index=False)
print(out['status'].value_counts().to_string())
print('total CSVs:', len(out))
