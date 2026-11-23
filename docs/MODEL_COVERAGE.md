# Source-to-extension model and analysis coverage

This file distinguishes **models and explicitly tested pipeline variants** from entire experimental analyses. The original user-supplied `PAPER_FULL_REPRODUCTION.ipynb` evaluated WDBC only. It is preserved in `original/` with SHA-256 `22ed184c75799a1e0436b661c276e0a7c392eefa34cf0ae2da360c090c8e1ad2`. The master notebook is the expanded four-dataset project, not a claim that the original file already contained four datasets.

## Models and variants: all four datasets

| Original source item | Four-dataset implementation and records | Important distinction |
|---|---|---|
| 11 classical estimators | `exp10_multi_dataset_zoo.csv`: 44 records; `exp10b_multi_dataset_zoo_folds.csv`: 660 fold records | Same 11 estimator definitions/hyperparameters as the source, checked by AST; all available dataset-specific predictors; add a fold-fitted median imputer before the source scaler. Only WPBC actually has missing predictors. |
| Quantum-kernel SVM | `exp13_multi_dataset_quantum_benchmark.csv`, `exp13c_quantum_fold_metrics.csv`, `exp13d_quantum_oof.csv` | Same `QuantumKernelSVM` class and fidelity/state-preparation code as the original; median imputation added. It is a classical exact statevector simulation, not a hardware run. |
| Original PCA-3 RBF comparator | `exp15_legacy_pca3_all_datasets.csv`, `exp15b_legacy_pca3_fold_metrics.csv`, `exp15c_legacy_pca3_oof.csv` | Exactly the original `ClassicalPCA3SVM` estimator: no angular scaling. The WDBC original five-fold metrics and every OOF probability are checked; WPBC requires a new imputer. |
| Original-style Platt-calibrated logistic pipeline | `exp16_legacy_calibration.csv`, `exp16b_legacy_calibration_oof.csv` | On all predictors of each dataset plus a separate exact WDBC Worst-10 reference. Preprocessing is fitted before the calibrator's internal splits, just as in the original; the WPBC-imputer is an addition. This is a pipeline variant of L2 LR, not a distinct family. |
| Additional matched PCA-3 RBF comparator | `exp13_*` | Not the original unmatched comparator. It matches the quantum path through median imputation, standardization, PCA and angular scaling. |
| Full-feature RBF-SVM and L2 LR quantum baselines | `exp13_*` | These are additional experiment entries for the same two original estimator families, not two extra algorithms. |

`exp20_model_coverage_audit.csv` contains 60 PASS entries: 4 datasets × (11 original classifiers + quantum + original PCA-3 comparator + original-style Platt variant + additional matched PCA-3 comparator). `exp20` checks the source SHA and abstract-syntax-tree identity for the 11-estimator dictionary, quantum and original PCA-3 classifier classes, fidelity/state preparation, original threshold rule and calibration helper. It then verifies the number and identity of held-out folds or OOF records. The original imports `QuadraticDiscriminantAnalysis` but never instantiates or tests it; it is not counted as a tested model.

The 15-fold results use 5-fold stratification repeated three times, seed 42. The original-style Platt variant uses a single five-fold outer OOF partition, with five internal calibration folds. Classifier results generally use `predict()`; Platt OOF threshold results and the grouped Parkinson audit use probabilities thresholded at 0.5. With `SVC`, these decision rules can disagree.

## Experimental analyses: what does and does not extend

| Original WDBC analysis | Expanded status |
|---|---|
| Three probe models (L2 LR, RBF SVM, Gradient Boosting) × Mean/Error/Worst/All | `exp18` recomputes all 12 WDBC cells and asserts agreement with the original; extends the three models across applicable WPBC blocks, analogous **acoustic** blocks in Parkinson and all features in Coimbra: 39 combinations/585 fold records. Parkinson blocks are not biological homologues of WDBC blocks; Coimbra has no analogous nuclear blocks. |
| Worst-10/All, ANOVA-10 and MI-10 with L2 LR | Reproduced for WDBC; `exp9` applies available fixed blocks and both `k=10` selectors to WDBC, WPBC and Parkinson. Coimbra has only nine predictors, so `k=10` and WDBC-style blocks are not applicable. Pairwise Jaccard in `exp9` is a different stability summary from the source's global intersection-over-union. |
| Nested logistic threshold on Worst-10 | Exact WDBC reference is checked; the original threshold-selection rule including its minimum-error fallback is applied to all predictors of each dataset in `exp17`. The improved sensitivity-first fallback in `exp11` is a different policy; both use row-wise folds for Parkinson. |
| Raw versus Platt calibration on Worst-10 | Exact WDBC output checked; legacy architecture on all features is in `exp16`. `exp12` is the stricter, separately implemented version that fits the full preprocessing pipeline within internal calibration folds. |
| Quantum, unmatched PCA-3 control and full RBF | The source classes are unchanged and each runs on four datasets; `exp13` adds an explicitly matched comparator and paired exploratory tests. |
| Source McNemar/bootstrap comparisons | Original WDBC comparisons are reproduced and fixture-checked; the matched quantum/classical comparison is extended in `exp13`. Not every original WDBC-specific hypothesis test is repeated identically for other endpoints. |
| WDBC Worst-10 learning curve | Reproduced and fixture-checked for WDBC, **not** repeated identically for all datasets. `exp21` reconstructs and exposes its unshuffled training-prefix class distortion; a properly stratified/shuffled learning curve has not been run here. |
| Parkinson grouped holdout | A new exploratory `exp14` audit, not part of the original. It covers 11 classical classifiers plus quantum and the matched PCA-3 comparator, using 32 unverified name stems. SVC probability calibration within training is not group-aware. |

In `exp17`, the original fallback applies when the target sensitivity cannot be met on inner predictions; for WPBC that occurs in 3 of 5 outer folds. The resulting pooled sensitivity/specificity are approximately 38.3%/62.9%, versus approximately 95.7%/13.9% for the explicitly different sensitivity-first fallback in `exp11`. These are separate five-fold OOF policy evaluations, not discrepancies in the same algorithm.

## Hard interpretation boundaries

- WDBC and WPBC share 139 source identifiers, including 126 with identical 30-feature measurements. They are not independent cohorts. Their endpoints differ.
- WPBC's observed recurrence label is subject to variable follow-up/censoring; this is not fixed-horizon survival modelling.
- Parkinson row-wise CV places a matching name stem in training for every one of 585 repeated test-recording appearances (`exp19`). Stems are unverified proxies; the source says 31 people but the file yields 32 stems. This does not quantify the amount of optimism. `exp14` is an exploratory stem-held-out audit, not a fully group-nested calibration/selection protocol.
- Quantum statevectors are exactly simulated in NumPy. The fixed entangler cancels in fidelity and the kernel reduces to `∏ cos²((x_j-x'_j)/2)`. This is classically computable and does not demonstrate quantum advantage.
- Fifteen overlapping fold scores are not 15 independent datasets. Their SD is descriptive; OOF tests and confidence intervals have dependence and refitting limitations. A non-significant result does not prove equivalence.
