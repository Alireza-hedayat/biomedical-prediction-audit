"""
Supplementary analyses requested during manuscript revision (2026-11-23).

Working copy of the reproduction project: uses the same data files, the same
estimator definitions (median imputation -> StandardScaler -> L2 logistic
regression; PCA(3) -> MinMax to [0, pi] for the matched angle inputs) and the
same threshold rule as run_all_experiments.py (Experiment 11).

Outputs (CSV) are written to results/ at the repository root.

A. Multi-seed nested sensitivity-target threshold for WPBC (Experiment 11
   procedure, seeds 42 and 0-9). Seed 42 is checked against the saved
   results/exp11b_nested_threshold_folds.csv before any other seed is used.
B. Calibration slope and intercept (logistic recalibration of the logit of the
   out-of-fold score) from results/exp12c_calibration_oof.csv.
C. Paired Brier-score and log-loss differences (quantum-kernel SVM minus matched
   RBF-SVM) from results/exp13d_quantum_oof.csv. Parkinson's uses a stem-cluster
   bootstrap (groups = filename stem); other datasets use a row bootstrap.
   Holm adjustment across the four datasets. Replaces the row-wise Wilcoxon
   test on absolute probability error.
D. Quantum-vs-matched-RBF input equality across ALL 15 repeated folds
   (RepeatedStratifiedKFold(5, 3, random_state=42)) for every dataset, using
   exact-equality assertions on the transformed angle inputs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
OUT = PROJECT / "results"
OUT.mkdir(exist_ok=True)
SEED = 42
rng_seeds = [42] + list(range(10))

# ---------------------------------------------------------------- helpers
def lr_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("model", LogisticRegression(penalty="l2", max_iter=2000, random_state=SEED))])

def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1))); fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0))); fp = int(np.sum((y == 0) & (pred == 1)))
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                sensitivity=tp / (tp + fn) if tp + fn else 0.0,
                specificity=tn / (tn + fp) if tn + fp else 0.0,
                precision=tp / (tp + fp) if tp + fp else 0.0)

def choose_high_sensitivity_threshold(yt, scores, target=0.95):
    candidates = []
    for tau in np.round(np.arange(0.05, 0.951, 0.01), 2):
        m = metrics(yt, (scores >= tau).astype(int))
        candidates.append((float(tau), m["sensitivity"], m["precision"]))
    eligible = [r for r in candidates if r[1] >= target]
    if eligible:
        chosen = max(eligible, key=lambda r: (r[2], -r[0]))
    else:
        chosen = max(candidates, key=lambda r: (r[1], r[2], -r[0]))
    return chosen[0], bool(eligible)

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adj[idx] = min(1.0, running)
    return adj

# ---------------------------------------------------------------- data
def load_xy(name):
    df = pd.read_csv(DATA / f"{name}_clean.csv")
    y = df["target"].astype(int)
    X = df.drop(columns=[c for c in ["target", "subject_stem"] if c in df.columns])
    stems = df["subject_stem"].to_numpy() if "subject_stem" in df.columns else None
    return X, y, stems

# ================================================================ A
def nested_wpbc(seed, X, y):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []; yo = y.to_numpy(); pred_all = np.full(len(y), -1)
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        Xt, yt = X.iloc[tr], y.iloc[tr]
        inner_scores = np.full(len(tr), np.nan)
        for itr, ite in inner.split(Xt, yt):
            p = lr_pipeline().fit(Xt.iloc[itr], yt.iloc[itr])
            inner_scores[ite] = p.predict_proba(Xt.iloc[ite])[:, 1]
        tau, feasible = choose_high_sensitivity_threshold(yt.to_numpy(), inner_scores)
        p = lr_pipeline().fit(Xt, yt)
        scores = p.predict_proba(X.iloc[te])[:, 1]
        pred = (scores >= tau).astype(int)
        pred_all[te] = pred
        m = metrics(yo[te], pred)
        rows.append(dict(seed=seed, fold=fold, threshold=tau, inner_target_feasible=int(feasible), **m))
    pooled = metrics(yo, pred_all)
    return rows, pooled

X_w, y_w, _ = load_xy("wpbc")
seed_fold_rows, seed_pooled_rows = [], []
for s in rng_seeds:
    rows, pooled = nested_wpbc(s, X_w, y_w)
    seed_fold_rows += rows
    seed_pooled_rows.append(dict(seed=s, n_feasible_folds=sum(r["inner_target_feasible"] for r in rows),
                                 pooled_sensitivity=pooled["sensitivity"], pooled_specificity=pooled["specificity"],
                                 pooled_precision=pooled["precision"], pooled_tp=pooled["tp"], pooled_tn=pooled["tn"],
                                 pooled_fp=pooled["fp"], pooled_fn=pooled["fn"]))
folds_df = pd.DataFrame(seed_fold_rows)
pooled_df = pd.DataFrame(seed_pooled_rows)

# validation against saved Experiment 11b (seed 42)
saved = pd.read_csv(RESULTS / "exp11b_nested_threshold_folds.csv")
saved = saved[saved.dataset == "WPBC"].reset_index(drop=True)
mine = folds_df[folds_df.seed == 42].reset_index(drop=True)
validation = dict(
    feasible_flags_match=bool((saved.inner_target_feasible.values == mine.inner_target_feasible.values).all()),
    tp_tn_fp_fn_match=bool(all((saved[c].values == mine[c].values).all() for c in ["tp", "tn", "fp", "fn"])),
    max_abs_threshold_diff=float(np.max(np.abs(saved.threshold.values - mine.threshold.values))),
)
print("VALIDATION vs exp11b (seed 42):", validation)
assert validation["feasible_flags_match"] and validation["tp_tn_fp_fn_match"], "reimplementation does not match saved Experiment 11b"
folds_df.to_csv(OUT / "A_wpbc_nested_folds_multiseed.csv", index=False)
pooled_df.to_csv(OUT / "A_wpbc_nested_pooled_multiseed.csv", index=False)
summary_A = dict(
    seeds=len(rng_seeds),
    feasible_folds_min_max=[int(pooled_df.n_feasible_folds.min()), int(pooled_df.n_feasible_folds.max())],
    pooled_sensitivity_min_max=[float(pooled_df.pooled_sensitivity.min()), float(pooled_df.pooled_sensitivity.max())],
    pooled_specificity_min_max=[float(pooled_df.pooled_specificity.min()), float(pooled_df.pooled_specificity.max())],
    validation=validation,
)

# ================================================================ B
oof = pd.read_csv(RESULTS / "exp12c_calibration_oof.csv")
calib_rows = []
for (ds, var), g in oof.groupby(["dataset", "variant"]):
    p = np.clip(g.score.values, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(logit, g.target.values)  # essentially unpenalized
    calib_rows.append(dict(dataset=ds, variant=var, n=len(g),
                           calibration_slope=float(lr.coef_[0][0]),
                           calibration_intercept=float(lr.intercept_[0])))
calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv(OUT / "B_calibration_slope_intercept.csv", index=False)

# ================================================================ C
q = pd.read_csv(RESULTS / "exp13d_quantum_oof.csv")
Xs = {d: load_xy(d.lower()) for d in ["WDBC", "WPBC", "Coimbra", "Parkinsons"]}
qname = "Quantum kernel SVM (PCA-3)"; cname = "RBF-SVM (matched PCA-3)"
rng = np.random.default_rng(SEED)
pair_rows = []; boot_pvals = {}
for ds in ["WDBC", "WPBC", "Coimbra", "Parkinsons"]:
    gq = q[(q.dataset == ds) & (q.model == qname)].sort_values("row_index")
    gc = q[(q.dataset == ds) & (q.model == cname)].sort_values("row_index")
    y = gq.target.values
    bq = (gq.score.values - y) ** 2; bc = (gc.score.values - y) ** 2
    d_brier = bq - bc
    eq = np.clip(gq.score.values, 1e-12, 1 - 1e-12); ec = np.clip(gc.score.values, 1e-12, 1 - 1e-12)
    ll_q = -(y * np.log(eq) + (1 - y) * np.log(1 - eq)); ll_c = -(y * np.log(ec) + (1 - y) * np.log(1 - ec))
    d_ll = ll_q - ll_c
    stems = Xs[ds][2]
    n = len(y)
    if ds == "Parkinsons" and stems is not None:
        # align stems with row_index (exp13d rows follow the cleaned file order)
        st = pd.Series(stems)[gq.row_index.values].to_numpy()
        uniq = pd.unique(st); idx_by = {u: np.where(st == u)[0] for u in uniq}
        def resample():
            picks = rng.choice(len(uniq), size=len(uniq), replace=True)
            return np.concatenate([idx_by[uniq[k]] for k in picks])
        unit = f"subject_stem (n_clusters={len(uniq)})"
    else:
        def resample():
            return rng.integers(0, n, size=n)
        unit = "row"
    B = 2000; bs_b, bs_l = [], []
    for _ in range(B):
        ix = resample()
        bs_b.append(d_brier[ix].mean()); bs_l.append(d_ll[ix].mean())
    bs_b = np.asarray(bs_b); bs_l = np.asarray(bs_l)
    # two-sided bootstrap p-value for mean difference == 0 (centred bootstrap)
    def boot_p(samples, obs):
        centred = samples - samples.mean()
        return float(np.mean(np.abs(centred) >= abs(obs)))
    pair_rows.append(dict(dataset=ds, bootstrap_unit=unit, n_rows=n,
                          brier_quantum=float(bq.mean()), brier_matched_rbf=float(bc.mean()),
                          brier_diff_q_minus_c=float(d_brier.mean()),
                          brier_diff_ci_low=float(np.quantile(bs_b, 0.025)), brier_diff_ci_high=float(np.quantile(bs_b, 0.975)),
                          brier_boot_p=boot_p(bs_b, d_brier.mean()),
                          logloss_quantum=float(ll_q.mean()), logloss_matched_rbf=float(ll_c.mean()),
                          logloss_diff_q_minus_c=float(d_ll.mean()),
                          logloss_diff_ci_low=float(np.quantile(bs_l, 0.025)), logloss_diff_ci_high=float(np.quantile(bs_l, 0.975)),
                          logloss_boot_p=boot_p(bs_l, d_ll.mean())))
pair_df = pd.DataFrame(pair_rows)
pair_df["brier_boot_p_holm"] = holm(pair_df.brier_boot_p)
pair_df["logloss_boot_p_holm"] = holm(pair_df.logloss_boot_p)
pair_df.to_csv(OUT / "C_quantum_vs_rbf_paired_brier_logloss.csv", index=False)

# ================================================================ D
# Exact equality of the transformed angle inputs in every one of the 15 folds.
# The quantum model's internal transform (StandardScaler -> PCA(3) -> MinMax[0,pi])
# is reproduced exactly as in QuantumKernelSVM, and the matched RBF pipeline's
# angle step is taken from matched_quantum_models().
def quantum_angles_fit_transform(Xtr, Xte):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_i, Xte_i = imp.transform(Xtr), imp.transform(Xte)
    sc = StandardScaler().fit(Xtr_i)
    pca = PCA(n_components=3, random_state=SEED).fit(sc.transform(Xtr_i))
    ang = MinMaxScaler(feature_range=(0, np.pi)).fit(pca.transform(sc.transform(Xtr_i)))
    return ang.transform(pca.transform(sc.transform(Xtr_i))), ang.transform(pca.transform(sc.transform(Xte_i)))

def rbf_angles_fit_transform(Xtr, Xte):
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("pca", PCA(n_components=3, random_state=SEED)),
                     ("angle", MinMaxScaler(feature_range=(0, np.pi)))]).fit(Xtr)
    return pipe.transform(Xtr), pipe.transform(Xte)

cv15 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
eq_rows = []
for ds in ["WDBC", "WPBC", "Coimbra", "Parkinsons"]:
    X, y, _ = Xs[ds]
    for fold, (tr, te) in enumerate(cv15.split(X, y), start=1):
        qa_tr, qa_te = quantum_angles_fit_transform(X.iloc[tr], X.iloc[te])
        ra_tr, ra_te = rbf_angles_fit_transform(X.iloc[tr], X.iloc[te])
        d_tr = float(np.max(np.abs(qa_tr - ra_tr))); d_te = float(np.max(np.abs(qa_te - ra_te)))
        exact = bool(np.array_equal(qa_tr, ra_tr) and np.array_equal(qa_te, ra_te))
        eq_rows.append(dict(dataset=ds, fold=fold, max_abs_diff_train=d_tr, max_abs_diff_test=d_te, exact_equal=exact))
eq_df = pd.DataFrame(eq_rows)
eq_df.to_csv(OUT / "D_quantum_rbf_input_equality_all15folds.csv", index=False)

summary = dict(
    A_wpbc_multiseed=summary_A,
    B_calibration=calib_df.to_dict(orient="records"),
    C_paired_tests=pair_df.to_dict(orient="records"),
    D_equality=dict(folds_checked=int(len(eq_df)),
                    folds_exact_equal=int(eq_df.exact_equal.sum()),
                    max_abs_diff=float(max(eq_df.max_abs_diff_train.max(), eq_df.max_abs_diff_test.max()))),
)
(OUT / "new_analyses_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False)[:6000])
