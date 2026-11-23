"""Check the key numbers quoted in the manuscript against the archived CSV outputs.

Every check reads a file in results/ and compares it with the value printed in the
manuscript (rounded as in the text). Run from anywhere: python scripts/verify_key_numbers.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
checks = []


def check(label, value, expected, decimals):
    ok = round(float(value), decimals) == round(expected, decimals)
    checks.append((label, round(float(value), decimals), expected, ok))


# Nested WPBC threshold (Experiment 11): pooled sensitivity and specificity
e11 = pd.read_csv(R / "exp11_multi_dataset_nested_threshold.csv").set_index("dataset")
check("WPBC pooled sensitivity", e11.loc["WPBC", "nested_sensitivity_pooled"], 0.9574, 4)
check("WPBC pooled specificity", e11.loc["WPBC", "nested_specificity_pooled"], 0.1391, 4)
check("WDBC fold-mean sensitivity", e11.loc["WDBC", "nested_sensitivity_mean"], 0.9389, 4)

# Multi-seed WPBC (analysis A): seed 42 counts
a = pd.read_csv(R / "A_wpbc_nested_pooled_multiseed.csv").set_index("seed")
check("A seed 42 TP", a.loc[42, "pooled_tp"], 45, 0)
check("A seed 42 FP", a.loc[42, "pooled_fp"], 130, 0)
check("A seed 42 FN", a.loc[42, "pooled_fn"], 2, 0)
check("A seed 42 TN", a.loc[42, "pooled_tn"], 21, 0)

# Calibration slope (analysis B), Coimbra
b = pd.read_csv(R / "B_calibration_slope_intercept.csv")
row = b[(b.dataset == "Coimbra") & (b.variant == "uncalibrated")].iloc[0]
check("Coimbra uncalibrated slope", row.calibration_slope, 0.58, 2)
row = b[(b.dataset == "Coimbra") & (b.variant == "Platt-calibrated")].iloc[0]
check("Coimbra Platt slope", row.calibration_slope, 1.34, 2)

# Paired quantum-minus-RBF Brier difference (analysis C), WDBC
c = pd.read_csv(R / "C_quantum_vs_rbf_paired_brier_logloss.csv").set_index("dataset")
check("WDBC Brier difference", c.loc["WDBC", "brier_diff_q_minus_c"], -0.0048, 4)
check("WDBC Brier CI low", c.loc["WDBC", "brier_diff_ci_low"], -0.0091, 4)
check("WDBC Brier CI high", c.loc["WDBC", "brier_diff_ci_high"], -0.0009, 4)

# Exact McNemar p-values (exp13b)
e13 = pd.read_csv(R / "exp13b_quantum_paired_tests.csv").set_index("dataset")
check("WDBC exact McNemar p", e13.loc["WDBC", "mcnemar_exact_p"], 1.0, 4)
check("WPBC exact McNemar p", e13.loc["WPBC", "mcnemar_exact_p"], 0.5, 4)
check("Coimbra exact McNemar p", e13.loc["Coimbra", "mcnemar_exact_p"], 0.2891, 4)
check("Parkinsons exact McNemar p", e13.loc["Parkinsons", "mcnemar_exact_p"], 0.625, 4)

# Gamma = 0.25 RBF control (analysis F), WPBC Brier difference
f = pd.read_csv(R / "F_rbf_gamma025_control_summary.csv").set_index("dataset")
check("WPBC gamma 0.25 Brier difference", f.loc["WPBC", "brier_diff_quantum_minus_rbf_mean"], -0.00181, 5)

# Parkinson's stem overlap (script verify_parkinsons_stems.py output)
s = pd.read_csv(R / "parkinsons_stem_overlap_check.csv").iloc[0]
check("Held-out stem appearances", s.held_out_appearances, 585, 0)
check("Held-out with training stem", s.with_training_stem, 585, 0)

failed = [c for c in checks if not c[3]]
for label, got, want, ok in checks:
    print(f"{'OK  ' if ok else 'FAIL'} {label}: archived={got} manuscript={want}")
print(f"{len(checks) - len(failed)} of {len(checks)} checks passed")
if failed:
    raise SystemExit(1)
