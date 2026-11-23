"""Reconstruct coverage and reference-extension outputs without fitting models."""
import ast
import hashlib
from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DATASETS = {'WDBC': 569, 'WPBC': 198, 'Coimbra': 116, 'Parkinsons': 195}
EXPECTED_MODELS = {
    'Logistic Regression (L2)', 'Logistic Regression (L1)', 'Linear Discriminant (LDA)',
    'SVM (linear kernel)', 'SVM (RBF kernel)', 'k-Nearest Neighbours', 'Gaussian Naive Bayes',
    'Random Forest', 'Extra Trees', 'Gradient Boosting', 'HistGradient Boosting',
}

def read(name):
    return pd.read_csv(RESULTS / name)

def source_node(notebook, name, node_type):
    found = []
    for c in notebook.cells:
        if c.cell_type != 'code':
            continue
        try:
            tree = ast.parse(c.source)
        except SyntaxError:
            continue
        for node in tree.body:
            if node_type == 'assignment' and isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    found.append(node.value)
            elif node_type == 'class' and isinstance(node, ast.ClassDef) and node.name == name:
                found.append(node)
    if len(found) != 1:
        raise AssertionError((name, node_type, len(found)))
    return ast.dump(found[0], include_attributes=False)

class TestExtendedCoverage(unittest.TestCase):
    def test_original_definition_identity(self):
        source = ROOT / 'original/PAPER_FULL_REPRODUCTION.ipynb'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         '22ed184c75799a1e0436b661c276e0a7c392eefa34cf0ae2da360c090c8e1ad2')
        a, b = nbformat.read(source, 4), nbformat.read(ROOT / 'MASTER_REPRODUCTION.ipynb', 4)
        for name, kind in [('zoo', 'assignment'), ('QuantumKernelSVM', 'class'), ('ClassicalPCA3SVM', 'class')]:
            self.assertEqual(source_node(a, name, kind), source_node(b, name, kind), name)
        code = [c for c in b.cells if c.cell_type == 'code']
        self.assertEqual(len(code), 38)
        self.assertTrue(all(c.execution_count is not None for c in code))
        self.assertFalse(any(o.output_type == 'error' for c in code for o in c.get('outputs', [])))

    def test_every_model_and_control_has_held_out_records(self):
        summary = read('exp10_multi_dataset_zoo.csv')
        folds = read('exp10b_multi_dataset_zoo_folds.csv')
        self.assertEqual(len(summary), 44)
        self.assertEqual(len(folds), 660)
        for dataset in DATASETS:
            part = summary[summary.dataset == dataset]
            self.assertEqual(set(part.model), EXPECTED_MODELS)
            self.assertEqual(len(part), 11)
            for name in EXPECTED_MODELS:
                f = folds[(folds.dataset == dataset) & (folds.model == name)]
                self.assertEqual(sorted(f.fold), list(range(1, 16)))
        q = read('exp13_multi_dataset_quantum_benchmark.csv')
        qfold = read('exp13c_quantum_fold_metrics.csv')
        qnames = {'Quantum kernel SVM (PCA-3)', 'RBF-SVM (matched PCA-3)',
                  'RBF-SVM (all features)', 'Logistic regression L2 (all features)'}
        self.assertEqual((len(q), len(qfold)), (16, 240))
        for dataset in DATASETS:
            self.assertEqual(set(q.loc[q.dataset == dataset, 'model']), qnames)
            for name in qnames:
                f = qfold[(qfold.dataset == dataset) & (qfold.model == name)]
                self.assertEqual(sorted(f.fold), list(range(1, 16)))
        audit = read('exp20_model_coverage_audit.csv')
        self.assertEqual(len(audit), 60)
        self.assertTrue(audit.groupby('dataset').size().eq(15).all())
        self.assertTrue(audit.status.eq('PASS').all())

    def test_original_pca3_control(self):
        summary = read('exp15_legacy_pca3_all_datasets.csv').set_index('dataset')
        folds = read('exp15b_legacy_pca3_fold_metrics.csv')
        oof = read('exp15c_legacy_pca3_oof.csv')
        self.assertEqual((len(summary), len(folds), len(oof)), (4, 60, 1078))
        for dataset, n in DATASETS.items():
            f = folds[folds.dataset == dataset]
            self.assertEqual(sorted(f.fold), list(range(1, 16)))
            self.assertAlmostEqual(f.sensitivity.mean(), summary.loc[dataset, 'sensitivity_mean'], places=12)
            self.assertAlmostEqual(f.specificity.mean(), summary.loc[dataset, 'specificity_mean'], places=12)
            p = oof[oof.dataset == dataset]
            self.assertEqual(len(p), n)
            self.assertFalse(p.row_index.duplicated().any())
            for fold, sub in p.groupby('fold'):
                actual = f[f.fold == fold].iloc[0]
                y, pred = sub.target.to_numpy(), sub.prediction.to_numpy()
                self.assertEqual(int(((y == 1) & (pred == 1)).sum()), actual.tp)
                self.assertEqual(int(((y == 0) & (pred == 1)).sum()), actual.fp)
        ref = read('exp6_quantum_vs_classical.csv')
        ref = ref[ref.model == 'Classical RBF-SVM (same PCA-3 front end)'].iloc[0]
        first = folds[(folds.dataset == 'WDBC') & (folds.fold <= 5)]
        for a, b in [('sensitivity', 'sensitivity'), ('precision', 'precision'), ('specificity', 'specificity'),
                     ('f1', 'f1'), ('balanced_acc', 'balanced_accuracy'), ('mcc', 'mcc'),
                     ('roc_auc', 'roc_auc'), ('pr_auc', 'pr_auc'), ('fn', 'fn'), ('fp', 'fp')]:
            self.assertAlmostEqual(ref[a], first[b].mean(), places=10, msg=a)
        self.assertAlmostEqual(ref.sensitivity_sd, first.sensitivity.std(ddof=1), places=10)

    def test_legacy_calibration_reconstructs_oof(self):
        summary = read('exp16_legacy_calibration.csv').set_index(['dataset', 'feature_set', 'variant'])
        pred = read('exp16b_legacy_calibration_oof.csv')
        self.assertEqual((len(summary), len(pred)), (10, 3294))
        for keys, rows in pred.groupby(['dataset', 'feature_set', 'variant']):
            n = DATASETS[keys[0]]
            self.assertEqual(len(rows), n)
            self.assertFalse(rows.row_index.duplicated().any())
            y, score = rows.target, rows.score
            self.assertAlmostEqual(summary.loc[keys, 'brier'], brier_score_loss(y, score), places=12)
            self.assertAlmostEqual(summary.loc[keys, 'log_loss'], log_loss(y, score, labels=[0, 1]), places=12)
            self.assertAlmostEqual(summary.loc[keys, 'roc_auc'], roc_auc_score(y, score), places=12)
        ref = read('exp5a_calibration.csv').set_index('variant')
        for variant, original in [('Uncalibrated', 'uncalibrated'), ('Legacy Platt', 'Platt-calibrated')]:
            for metric in ['brier', 'log_loss', 'roc_auc']:
                self.assertAlmostEqual(summary.loc[('WDBC', 'Worst 10 reference', variant), metric],
                                       ref.loc[original, metric], places=10)

    def test_legacy_nested_thresholds_and_original(self):
        summary = read('exp17_legacy_threshold.csv').set_index(['dataset', 'feature_set'])
        folds = read('exp17b_legacy_threshold_folds.csv')
        pred = read('exp17c_legacy_threshold_oof.csv')
        self.assertEqual((len(summary), len(folds), len(pred)), (5, 25, 1647))
        for key, rows in pred.groupby(['dataset', 'feature_set']):
            self.assertFalse(rows.row_index.duplicated().any())
            np.testing.assert_array_equal((rows.score >= rows.tau).astype(int), rows.prediction)
            y, p = rows.target.to_numpy(), rows.prediction.to_numpy()
            tp = ((y == 1) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
            tn = ((y == 0) & (p == 0)).sum(); fp = ((y == 0) & (p == 1)).sum()
            self.assertEqual((tp, tn, fp, fn), tuple(int(summary.loc[key, col]) for col in ['tp', 'tn', 'fp', 'fn']))
        ref = read('exp4b_nested_threshold.csv').sort_values('fold')
        observed = folds[(folds.dataset == 'WDBC') & (folds.feature_set == 'Worst 10 reference')].sort_values('fold')
        for col in ref.columns:
            np.testing.assert_allclose(ref[col], observed[col], rtol=1e-8, atol=1e-10, err_msg=col)

    def test_three_model_blocks_reconstruct_source(self):
        frame = read('exp18_3model_block_cross.csv')
        folds = read('exp18b_3model_block_folds.csv')
        self.assertEqual((len(frame), len(folds)), (39, 585))
        self.assertEqual(set(frame.loc[frame.dataset == 'Coimbra', 'feature_block']), {'All'})
        ref = read('exp1_feature_blocks.csv')
        mapping = {'sensitivity_mean': 'sensitivity_mean', 'sensitivity_sd': 'sensitivity_std',
                   'precision_mean': 'precision_mean', 'specificity_mean': 'specificity_mean',
                   'f1_mean': 'f1_mean', 'mcc_mean': 'mcc_mean', 'roc_auc_mean': 'roc_auc_mean',
                   'fn_mean': 'fn_mean', 'fp_mean': 'fp_mean'}
        for _, r in ref.iterrows():
            actual = frame[(frame.dataset == 'WDBC') & (frame.feature_block == r.feature_block) & (frame.model == r.model)]
            self.assertEqual(len(actual), 1)
            for old_col, new_col in mapping.items():
                self.assertAlmostEqual(float(r[old_col]), float(actual[new_col].iloc[0]), places=10)
        zoo = read('exp10_multi_dataset_zoo.csv')
        names = {'Logistic Regression (L2)': 'Logistic Regression (L2)',
                 'SVM (RBF)': 'SVM (RBF kernel)', 'Gradient Boosting': 'Gradient Boosting'}
        for dataset in DATASETS:
            for old, new in names.items():
                a = frame[(frame.dataset == dataset) & (frame.feature_block == 'All') & (frame.model == old)].iloc[0]
                b = zoo[(zoo.dataset == dataset) & (zoo.model == new)].iloc[0]
                for metric in ['sensitivity_mean', 'specificity_mean', 'roc_auc_mean', 'mcc_mean']:
                    self.assertAlmostEqual(a[metric], b[metric], places=10)

    def test_split_overlap_and_learning_prefixes(self):
        raw = pd.read_csv(ROOT / 'data/raw/parkinsons.data')
        y = raw.status.to_numpy(); X = raw.drop(columns=['name', 'status'])
        stems = raw.name.str.rsplit('_', n=1).str[0].to_numpy()
        audit = read('exp19_parkinsons_split_overlap.csv').sort_values('fold').reset_index(drop=True)
        self.assertEqual(len(audit), 15)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        for i, (tr, te) in enumerate(cv.split(X, y)):
            train, test = set(stems[tr]), set(stems[te])
            self.assertEqual(len(te), audit.loc[i, 'n_test_recordings'])
            self.assertEqual(len(train & test), audit.loc[i, 'n_shared_stems'])
            self.assertEqual(sum(g in train for g in stems[te]), audit.loc[i, 'n_test_recordings_with_stem_in_training'])
        self.assertEqual(audit.n_test_recordings_with_stem_in_training.sum(), 585)
        w = pd.read_csv(ROOT / 'data/wdbc_clean.csv')
        y = w.target
        sizes = read('exp8_learning_curve.csv').n_train.astype(int)
        learned = read('exp21_learning_curve_sampling_audit.csv')
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for size in sizes:
            for fold, (tr, _) in enumerate(cv5.split(w.drop(columns='target'), y), start=1):
                r = learned[(learned.n_train == size) & (learned.fold == fold)].iloc[0]
                self.assertEqual(int(y.iloc[tr[:size]].sum()), r.positive_count)
                self.assertAlmostEqual(y.iloc[tr[:size]].mean(), r.positive_fraction, places=12)

if __name__ == '__main__':
    unittest.main(verbosity=2)
