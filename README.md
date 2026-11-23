# Beyond Apparent Accuracy: A Reproducibility Audit of Biomedical Prediction Validation

Code, data and outputs for a reproducibility audit of small biomedical prediction studies on four public tabular datasets. The repository reproduces every archived result table and figure from the public data files.

> **Status: draft, not published.** The associated manuscript is under preparation. No public DOI or archived release exists yet. Items marked **[TODO]** require author input.

## 1. Scientific aim
Test how apparent (in-sample or unnested) accuracy changes under nested, sensitivity-targeted threshold selection, calibration assessment, paired probabilistic comparison and grouped validation. The four datasets are WDBC, WPBC, Breast Cancer Coimbra and Oxford Parkinson's voice recordings. The work is a methodological audit. It does not establish clinical utility or readiness for clinical use.

## 2. Methods summary
- Eleven fixed-configuration classifiers (`run_all_experiments.py`, `MASTER_REPRODUCTION.ipynb`).
- Repeated stratified cross-validation: five folds, three repeats (`RepeatedStratifiedKFold(5, 3, random_state=42)`). Out-of-fold analyses use `StratifiedKFold(5, shuffle=True, random_state=42)`.
- Platt cross-fitting uses `CalibratedClassifierCV(cv=5)`, whose internal splits are unshuffled `StratifiedKFold` (scikit-learn default). This was checked against an explicit `StratifiedKFold(5, shuffle=False)`: identical on all four datasets (`results/E_calibration_cv_explicit_check.csv`).
- Nested threshold selection: five outer and five inner stratified folds; sensitivity target 0.95; sensitivity-first rule with a legacy minimum-error fallback when the target is infeasible. Fallback use is reported.
- Calibration: Platt scaling via `CalibratedClassifierCV(method="sigmoid", cv=5)`; Brier score, log loss, 10-bin expected calibration error, and calibration slope and intercept from logistic recalibration of out-of-fold logits.
- Quantum-kernel comparison: a three-qubit angle-encoded fidelity kernel against a matched RBF-SVM on identical three-dimensional inputs. The kernel is exactly classically expressible for this circuit (see section 6).
- Paired Brier and log-loss differences with bootstrap intervals (row-level, or cluster-level by Parkinson's recording stem) and Holm adjustment across datasets.
- Multi-seed nested WPBC analysis (seeds 0–9 and 42), a γ = 0.25 RBF control, and an explicit check that the calibration cross-validation equals the project default.

## 3. Repository structure
```
run_all_experiments.py      Main reproduction script (46 CSV tables, 12 figures); asserts reference fixtures
MASTER_REPRODUCTION.ipynb   Executed notebook (38 of 38 code cells, no errors)
scripts/
  run_new_analyses.py       Analyses A–D (multi-seed WPBC, calibration slope, paired tests, input equality)
  run_checks_E_F.py         Analyses E–F (calibration CV check, γ = 0.25 RBF control)
  verify_parkinsons_stems.py  Held-out recording-stem overlap for Parkinson's (585 of 585)
  verify_key_numbers.py     Checks numbers quoted in the manuscript against archived CSVs (19 checks)
tests/                      Unit tests for artefacts, coverage, reference fixtures and overlap
data/                       Public raw and cleaned datasets, SHA-256 sums, manifest, provenance
results/                    Archived CSV outputs (46 from the main run; A–F and checks)
figures/                    Generated figures (PDF and 300-dpi PNG)
docs/                       Benchmark report, model coverage, run verification, Persian summary
verification/               CSV comparison tool and execution logs
original/                   User-supplied original notebook (SHA-256 checked by a test)
requirements.txt            Pinned versions
CITATION.cff, CHANGELOG.md, MANIFEST.sha256
```

## 4. Environment
- Tested with CPython 3.13.18 on Linux. Pinned analysis versions: numpy 2.3.5, pandas 2.2.3, scipy 1.17.1, scikit-learn 1.6.1, matplotlib 3.11.2 (full list in `requirements.txt`).
- Install: `pip install -r requirements.txt`.
- Results on other platforms, or with other library builds, may differ at floating-point precision. Comparisons use relative tolerance 1e-8.
- Upstream SciPy/scikit-learn produce `DeprecationWarning` messages about L-BFGS-B options with these pinned versions. They are not errors.

## 5. Data
Four public UCI Machine Learning Repository datasets are included under `data/` together with checksums and provenance:

| Dataset | UCI DOI |
|---|---|
| WDBC (Wisconsin Diagnostic Breast Cancer) | 10.24432/C5DW2B |
| WPBC (Wisconsin Prognostic Breast Cancer) | 10.24432/C5GK50 |
| Breast Cancer Coimbra | 10.24432/C52P59 |
| Oxford Parkinson's Disease Telemonitoring / voice dataset | 10.24432/C59C74 |

- Verify integrity from the repository root: `sha256sum -c` against `data/SHA256SUMS`, or run the tests.
- `data/DATA_PROVENANCE.md` documents every transformation, attribution and identifier overlap.
- Key data caveats: 139 identifiers are shared between WDBC and WPBC (126 with identical morphology features), so the two cohorts are not independent. Parkinson's recordings carry 32 filename stems, while the UCI documentation describes 31 subjects; stems are used only as grouping proxies, never as features.
- The datasets are described by UCI as CC BY 4.0. **[TODO: authors to confirm licence terms and attribution wording before publication.]**

## 6. Known limitations
- Small datasets; internal resampling only; no external validation cohort.
- WDBC and WPBC are not independent (section 5). The cause of the 13 non-identical identifier pairs is unresolved.
- Parkinson's stems are proxies for speaker identity. Stem-level overlap between training and test recordings is 585 of 585 held-out appearances in the row-wise design. The stem-grouped analysis uses a single partition.
- WPBC has variable follow-up and is treated as a binary label, not as a fixed-horizon survival endpoint.
- The quantum-kernel result applies only to the specific terminal-unitary circuit studied. The kernel is classically expressible, so no quantum advantage is claimed.
- The original WDBC learning-curve prefixes are unshuffled and are retained only for reproduction.
- Calibration slopes above 2 in the cross-fitted Platt variant are reported, not explained away.

## 7. Outputs
Each CSV is named after the analysis that produces it:
- `exp*_*.csv`: main experiments (produced by `run_all_experiments.py`).
- `A_*`, `B_*`, `C_*`, `D_*`: `scripts/run_new_analyses.py`.
- `E_*`, `F_*`: `scripts/run_checks_E_F.py`.
- `parkinsons_stem_overlap_check.csv`: `scripts/verify_parkinsons_stems.py`.
- `new_analyses_summary.json`: summary of analyses A–D.
- `summary.json`: summary of the main run.

## 8. Citation
Cite the software with `CITATION.cff`. **[TODO: DOI, release date and final author ORCID iDs after author approval.]** The associated article citation will be added after acceptance.

## 9. Authors
- Alireza Hedayati, Allameh Tabataba'i University, alireza_hedayati@atu.ac.ir (ORCID 0009-0006-3221-8639).
- Hazhir Homei, University of Tabriz, homei@tabrizu.ac.ir. **[TODO: ORCID iD and written approval of the final author list.]**
- The author order and corresponding-author designation are pending final author confirmation.

## 10. Licence
**[TODO: licence not yet decided by the authors.]** No LICENSE file is included. Until one is added, the code and documents are not licensed for reuse or redistribution beyond what the law allows. The datasets keep their own licence (section 5).

## 11. How to run
```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1                    # exact reproduction
python run_all_experiments.py                # writes results/ and figures/ (about 4 minutes)
python scripts/run_new_analyses.py           # analyses A–D
python scripts/run_checks_E_F.py             # analyses E–F
python scripts/verify_parkinsons_stems.py    # 585 of 585 check
python scripts/verify_key_numbers.py         # manuscript numbers vs CSVs (19 checks)
python -m unittest discover -s tests -v      # 15 tests
```
To compare a new run with the archived results: `python verification/compare_csv_trees.py results <new_results_dir>`. Timing columns are excluded.

## 12. Tests and verification
- `tests/` contains 15 tests covering data hashes and counts, reference fixtures, original estimator identity, model coverage, overlap, and figure formats. All 15 pass in the recorded environment.
- `run_all_experiments.py` asserts the expected results manifest, finiteness, the data fingerprints and the reference fixtures. The only documented deviation from the original reference is the Extra Trees description (see `CHANGELOG.md`, 2026-11-23): the original label "bagged CART" is incorrect because the model uses `bootstrap=False`.
- `MASTER_REPRODUCTION.ipynb` was executed in this repository on 2026-11-23: 38 of 38 code cells ran without error. Its saved outputs are the record for this release.
- Details and the earlier verification record: `docs/RUN_VERIFICATION.md`.

## 13. Contact, funding, ethics and data-use statements
- Contact: Alireza Hedayati, alireza_hedayati@atu.ac.ir. **[TODO: authors to confirm the corresponding author.]**
- Funding: **[TODO: authors to provide, or state "no funding".]**
- Conflicts of interest: **[TODO: authors to provide.]**
- Ethics: the study uses only public, de-identified datasets. **[TODO: authors to confirm the institutional determination.]**
- Data use: the datasets are public (section 5). No patient-level clinical records are included beyond the public files.
- Generative AI: **[TODO: authors to confirm the declaration wording required by the journal.]** AI tools are not authors.
