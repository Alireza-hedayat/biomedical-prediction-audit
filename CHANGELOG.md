# Changelog

## Unreleased (2026-11-23), repository preparation
- Repository assembled from the reproduction working copy. Added `scripts/` with the multi-seed and calibration analyses (A–F), the Parkinson's stem check and the manuscript number check.
- Scripts write to `results/` at the repository root. Verified: analyses A–F and the stem check rerun from this repository reproduce the archived CSVs byte for byte.
- Documented metadata correction: the Extra Trees description in `results/exp2_architectures.csv` and in the reference assertion now reads "non-bootstrap randomised-threshold CART ensemble (bootstrap=False), 300 trees". The original reference text, "bagged CART, random split thresholds", was inconsistent with `bootstrap=False`. The only cell allowed to differ from the original reference is this description. The unchanged historical copy remains in `docs/FULL_BENCHMARK_REPORT.md`.
- `run_all_experiments.py` and the notebook were rerun in this repository. All non-timing values match the archived results; only timing columns differ.
- Added `original/` (user-supplied original notebook, SHA-256 checked) to satisfy the source-identity test.
- Not included: a LICENSE file (licence decision pending), a DOI, a public URL, and the manuscript itself (pending journal decision).
- Added `scripts/check_wisconsin_shared_pairs.py` (writes `results/wisconsin_shared_pairs_differences.csv`): the 13 non-identical WDBC/WPBC shared identifiers differ in 29–30 features, by up to 1,325 units, so they are not rounding differences.
- Added `scripts/run_calibration_ensemble_check.py` (writes `results/E_calibration_ensemble_variant.csv`): out-of-fold Platt slopes with ensemble=False (WDBC 1.40, WPBC 0.77, Coimbra 1.11, Parkinson's 2.17) versus the default ensemble=True.
- Added `scripts/check_quantum_refit_wpbc.py` (writes `results/quantum_refit_wpbc_check.csv`). Notebook and γ=0.25-script quantum models give identical WPBC predictions on all 15 folds (mean ROC-AUC 0.5767). The archived `results/exp13c_quantum_fold_metrics.csv` ROC-AUC column (mean 0.5756) does not match the refit on 9 of 15 folds; flagged for regeneration.
- Corrected a supplement wording that described the γ=0.25 control as not executed.
