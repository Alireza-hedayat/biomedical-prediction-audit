#!/usr/bin/env python3
"""Compare non-timing CSV outputs of two result directories.

Usage: python verification/compare_csv_trees.py results /path/to/other/results
Both directories must contain the same CSV filenames. Allows equal NaNs and a
floating tolerance of rtol=1e-8, atol=1e-10; timing columns are excluded.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('left', type=Path)
parser.add_argument('right', type=Path)
args = parser.parse_args()
left = {p.name: p for p in args.left.glob('*.csv')}
right = {p.name: p for p in args.right.glob('*.csv')}
assert left and set(left) == set(right), (set(left) - set(right), set(right) - set(left))
columns, maximum = 0, 0.0
for name in sorted(left):
    a, b = pd.read_csv(left[name]), pd.read_csv(right[name])
    assert a.shape == b.shape and list(a.columns) == list(b.columns), name
    for col in a:
        if 'second' in col.lower() or 'time' in col.lower():
            continue
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
            np.testing.assert_allclose(x, y, rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=name + ':' + col)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(x[finite] - y[finite]))))
        else:
            assert a[col].fillna('<NA>').astype(str).tolist() == b[col].fillna('<NA>').astype(str).tolist(), (name, col)
        columns += 1
print(f'PASS: {len(left)} CSVs, {columns} non-timing columns; maximum absolute numerical difference {maximum:.17g}.')
