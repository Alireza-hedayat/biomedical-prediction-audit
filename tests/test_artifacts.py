"""Post-run scientific consistency tests; run with python -m unittest discover -s tests -v."""
from itertools import combinations
from pathlib import Path
import hashlib
import unittest

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'

class TestArtifactConsistency(unittest.TestCase):
    def test_data_sha256(self):
        for line in (DATA / 'SHA256SUMS').read_text().splitlines():
            expected, path = line.split('  ', 1)
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)

    def test_dataset_counts_and_missingness(self):
        expected = {'wdbc': (569, 30, 212, 0), 'wpbc': (198, 32, 47, 4),
                    'coimbra': (116, 9, 64, 0), 'parkinsons': (195, 22, 147, 0)}
        for name, (n, p, positives, missing) in expected.items():
            frame = pd.read_csv(DATA / f'{name}_clean.csv')
            metadata_cols = ['target'] + (['subject_stem'] if name == 'parkinsons' else [])
            self.assertEqual((len(frame), len(frame.drop(columns=metadata_cols).columns)), (n, p))
            self.assertEqual(int(frame.target.sum()), positives)
            self.assertEqual(int(frame.isna().sum().sum()), missing)

    def test_wdbc_regression_fixtures(self):
        for fixture in sorted((DATA / 'reference_wdbc').glob('exp*.csv')):
            expected = pd.read_csv(fixture)
            actual = pd.read_csv(RESULTS / fixture.name)[expected.columns]
            self.assertEqual(expected.shape, actual.shape, fixture.name)
            for col in expected:
                if pd.api.types.is_numeric_dtype(expected[col]):
                    np.testing.assert_allclose(expected[col].to_numpy(dtype=float), actual[col].to_numpy(dtype=float),
                                               rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=fixture.name + ':' + col)
                else:
                    exp_vals = expected[col].fillna('').astype(str).tolist()
                    act_vals = actual[col].fillna('').astype(str).tolist()
                    # Documented metadata correction (see CHANGELOG.md, 2026-11-23): the original
                    # reference label 'bagged CART' for Extra Trees is wrong because bootstrap=False.
                    # Only this description cell is allowed to differ.
                    known = {(fixture.name, col): 'Extra Trees'} if fixture.name == 'exp2_architectures.csv' else {}
                    if (fixture.name, col) in known:
                        for i, (e, a) in enumerate(zip(exp_vals, act_vals)):
                            if e != a:
                                self.assertIn('Extra', str(expected.iloc[i, 0]), fixture.name + ':' + col)
                                self.assertIn('bagged CART', e)
                                self.assertIn('non-bootstrap', a)
                                exp_vals[i] = a
                    self.assertEqual(exp_vals, act_vals, fixture.name + ':' + col)

    def test_wdbc_wpbc_overlap(self):
        w = pd.read_csv(DATA / 'raw/wdbc.data', header=None)
        p = pd.read_csv(DATA / 'raw/wpbc.data', header=None, na_values='?')
        w.columns = ['id', 'diagnosis'] + [f'w{i}' for i in range(30)]
        p.columns = ['id', 'outcome', 'time'] + [f'p{i}' for i in range(30)] + ['size', 'nodes']
        joined = w.merge(p, on='id', validate='one_to_one')
        equal = (joined[[f'w{i}' for i in range(30)]].to_numpy() == joined[[f'p{i}' for i in range(30)]].to_numpy()).all(axis=1)
        self.assertEqual(len(joined), 139)
        self.assertEqual(int(equal.sum()), 126)
        self.assertTrue((joined.diagnosis == 'M').all())
        out = pd.read_csv(RESULTS / 'dataset_overlap_audit.csv').iloc[0]
        self.assertEqual(int(out.shared_identifier_values), 139)
        self.assertEqual(int(out.shared_identifiers_identical_30_features), 126)

    def test_nested_oof_prediction_accounting(self):
        oof = pd.read_csv(RESULTS / 'exp11c_nested_threshold_oof.csv')
        summary = pd.read_csv(RESULTS / 'exp11_multi_dataset_nested_threshold.csv').set_index('dataset')
        for dataset, g in oof.groupby('dataset'):
            self.assertFalse(g.row_index.duplicated().any())
            np.testing.assert_array_equal((g.score >= g.threshold).astype(int), g.prediction)
            y, pred = g.target.to_numpy(), g.prediction.to_numpy()
            tp = ((y == 1) & (pred == 1)).sum(); fn = ((y == 1) & (pred == 0)).sum()
            tn = ((y == 0) & (pred == 0)).sum(); fp = ((y == 0) & (pred == 1)).sum()
            self.assertAlmostEqual(tp/(tp+fn), summary.loc[dataset, 'nested_sensitivity_pooled'])
            self.assertAlmostEqual(tn/(tn+fp), summary.loc[dataset, 'nested_specificity_pooled'])
            self.assertEqual(int(fn), int(summary.loc[dataset, 'nested_fn']))
            self.assertEqual(int(fp), int(summary.loc[dataset, 'nested_fp']))

    def test_dca_formula_against_oof(self):
        pred = pd.read_csv(RESULTS / 'exp12c_calibration_oof.csv')
        curves = pd.read_csv(RESULTS / 'exp12b_dca_curves.csv')
        for dataset, frame in curves.groupby('dataset'):
            self.assertEqual(len(frame), 76)
            raw = pred[(pred.dataset == dataset) & (pred.variant == 'uncalibrated')].sort_values('row_index')
            cal = pred[(pred.dataset == dataset) & (pred.variant == 'Platt-calibrated')].sort_values('row_index')
            self.assertFalse(raw.row_index.duplicated().any())
            np.testing.assert_array_equal(raw.target.to_numpy(), cal.target.to_numpy())
            y = raw.target.to_numpy(); n = len(y)
            for r in frame.itertuples(index=False):
                weight = r.threshold / (1-r.threshold)
                for score, actual in [(raw.score.to_numpy(), r.net_benefit_raw), (cal.score.to_numpy(), r.net_benefit_platt)]:
                    d = score >= r.threshold
                    expected = np.sum(d & (y == 1))/n - np.sum(d & (y == 0))/n * weight
                    self.assertAlmostEqual(expected, actual, places=12)
                self.assertAlmostEqual(y.mean() - (1-y.mean()) * weight, r.net_benefit_treat_all, places=12)
                self.assertEqual(r.net_benefit_treat_none, 0)

    def test_jaccard_from_feature_memberships(self):
        picks = pd.read_csv(RESULTS / 'exp9c_selected_features.csv')
        table = pd.read_csv(RESULTS / 'exp9_multi_dataset_blocks.csv').set_index(['dataset', 'configuration'])
        for (dataset, config), frame in picks.groupby(['dataset', 'configuration']):
            sets = [set(g.feature) for _, g in frame.groupby('fold')]
            self.assertEqual(len(sets), 15)
            vals = [len(a & b)/len(a | b) for a, b in combinations(sets, 2)]
            self.assertEqual(len(vals), 105)
            self.assertAlmostEqual(float(np.mean(vals)), table.loc[(dataset, config), 'jaccard_pairwise_mean'], places=12)

    def test_vector_and_raster_figure_files(self):
        expected = ['fig_blocks', 'fig_models', 'fig_models2', 'fig_threshold', 'fig_threshold2',
                    'fig_reliability', 'fig_kernel', 'fig_learning', 'fig_multi_zoo_heatmap',
                    'fig_multi_quantum_benchmark', 'fig_multi_calibration', 'fig_multi_dca']
        for name in expected:
            pdf = (FIGURES / (name + '.pdf')).read_bytes()
            self.assertTrue(pdf.startswith(b'%PDF'))
            self.assertNotIn(b'/Subtype /Image', pdf, f'Raster object in {name}.pdf')
            with Image.open(FIGURES / (name + '.png')) as im:
                self.assertAlmostEqual(im.info['dpi'][0], 300, delta=1)
                self.assertGreaterEqual(min(im.size), 500)

if __name__ == '__main__':
    unittest.main(verbosity=2)
