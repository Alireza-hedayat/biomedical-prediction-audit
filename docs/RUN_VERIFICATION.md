# Execution and artifact verification

Recorded expanded execution: CPython 3.13.18 on Linux, `SEED=42`, pinned versions in `requirements.txt`, and numerical-library threads limited to one. This is the updated notebook with the original-model coverage supplements. It is not a clinical validation.

| Check | Observed result |
|---|---|
| Executed notebook | All 38 of 38 code cells ran in order; no saved error outputs; the final assertion cell passed. The recorded run took 299.2 seconds while another run was also active, so this is not a meaningful performance benchmark. |
| Independent standalone runner | The same code cells ran to completion in a separate project copy and passed the final assertions. Logs are in `verification/`. |
| Notebook vs standalone | All 46 result CSVs matched in 528 non-timing columns; maximum observed absolute numeric difference was 0. Wall-clock columns were deliberately excluded. Per-file records: `verification/revision_vs_standalone.csv`. |
| Previous expanded execution | The original 33 result CSVs were compared with the previous executed master project: 361 non-timing columns matched, with maximum numeric difference 0. The new supplements did not alter these earlier measured results. See `verification/revision_vs_previous.csv`. |
| WDBC original reference fixtures | All 13 WDBC numerical fixture CSVs passed, including categorical fields and all numerical columns other than elapsed times. The maximum numeric difference in this environment was 0. |
| Model source and coverage | SHA-256 of the user-supplied original notebook is `22ed184c75799a1e0436b661c276e0a7c392eefa34cf0ae2da360c090c8e1ad2`. AST definitions of its 11-model zoo and its quantum/PCA-3 classes match the master; selected original functions also match. Sixty dataset-by-model/control coverage records passed; see `MODEL_COVERAGE.md` and `results/exp20_model_coverage_audit.csv`. |
| Original PCA-3 comparator | All WDBC original five-fold metrics and every OOF score matched; the unchanged comparator class was fitted on all four datasets (60 repeated fold records). WPBC requires training-fold median imputation. |
| Legacy calibration and threshold | WDBC Worst-10 OOF calibration probabilities and every nested-threshold fold metric matched the original. Both legacy policies were also run on all features of every dataset; `exp12`/`exp11` implement the distinct revised protocols. |
| Three-model feature blocks | All 12 WDBC source block/probe results matched; 39 applicable dataset/block/model combinations and 585 fold records are provided. |
| Additional tests | All 15 automated tests passed, including source identity, model coverage, reference variants, held-out prediction reconstruction, overlap, learning-prefix reconstruction, data, DCA, stability and figure formats. |
| Dataset hashes and figures | All 10 raw/cleaned data files passed `data/SHA256SUMS`. Twelve PDF files contain no embedded raster image objects; twelve PNGs have 300-dpi metadata; all 12 notebook inline PNGs match saved files byte for byte. |
| Output values | The only not-applicable result entries are the two fixed-subset Jaccard cells in the WDBC reference table. Four missing WPBC input entries are intentional and are imputed during training; all other result values passed the notebook's nonmissing/finite checks. |

The executed notebook's SHA-256 is `159a47ed7daf834efea6411e0b41d809acb63b3aa4fa942145e414b0ff1fd250` (MD5 `a0a42a6ca57636e72761cc055cd0651e`). The two runs each emitted 1,340 upstream `DeprecationWarning` notices about SciPy's L-BFGS-B `disp`/`iprint` options in the pinned SciPy 1.17.1/scikit-learn 1.6.1 combination. No convergence warning or traceback was found. These deprecation warnings are retained in the notebook and standalone log; compatibility with later versions is not guaranteed.

To compare a new independent rerun against this recorded result tree:

```bash
python verification/compare_csv_trees.py results /path/to/new/run/results
```

The comparison uses relative tolerance 1e-8 and absolute tolerance 1e-10, permits corresponding documented NaNs, requires identical rows/columns and excludes timing fields. Results across hardware or library builds need not be bitwise identical.

## Critical interpretation limits

- WDBC and WPBC have 139 common source identifiers; 126 have identical 30-feature morphology vectors. They cannot be described as independent patient cohorts or external validation of each other.
- Parkinson's rows are recordings. All 585 repeated test-recording appearances in the row-wise 15-fold zoo share an unverified name stem with a training recording. Source documentation describes 31 people, while 32 stems are observed; neither patient identity nor inflation magnitude is established. The stem-held-out analysis uses a single grouped partition, and SVC's internal probability calibration is not group-aware. It also thresholds probability rather than using SVC's `predict()`.
- WPBC is a variable-follow-up, censored prognostic cohort, not a fixed-horizon survival analysis; DCA based on sample prevalence is descriptive rather than clinical validation.
- The quantum fidelity kernel is exactly classically expressible because the fixed entangler cancels in inner products. No quantum speedup or clinical advantage is established.
- The original WDBC learning curve uses unshuffled prefixes, producing a positive-class proportion of about 63–71% at 68 training rows versus about 37% for the full training folds. The original result is retained for reproduction, not a reliable learning trend.
