# Changelog

## Unreleased (2026-11-23), repository preparation
- Repository assembled from the reproduction working copy. Added `scripts/` with the multi-seed and calibration analyses (A–F), the Parkinson's stem check and the manuscript number check.
- Scripts write to `results/` at the repository root. Verified: analyses A–F and the stem check rerun from this repository reproduce the archived CSVs byte for byte.
- Documented metadata correction: the Extra Trees description in `results/exp2_architectures.csv` and in the reference assertion now reads "non-bootstrap randomised-threshold CART ensemble (bootstrap=False), 300 trees". The original reference text, "bagged CART, random split thresholds", was inconsistent with `bootstrap=False`. The only cell allowed to differ from the original reference is this description. The unchanged historical copy remains in `docs/FULL_BENCHMARK_REPORT.md`.
- `run_all_experiments.py` and the notebook were rerun in this repository. All non-timing values match the archived results; only timing columns differ.
- Added `original/` (user-supplied original notebook, SHA-256 checked) to satisfy the source-identity test.
- Not included: a LICENSE file (licence decision pending), a DOI, a public URL, and the manuscript itself (pending journal decision).
