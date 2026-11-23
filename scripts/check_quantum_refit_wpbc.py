"""Refit check for the WPBC quantum-kernel fold results.

(1) The notebook implementation (QuantumKernelSVM inside run_all_experiments.py, with median imputer)
and (2) an independent re-implementation used by scripts/run_checks_E_F.py are fitted on the same
RepeatedStratifiedKFold(5, 3, random_state=42) splits. Writes results/quantum_refit_wpbc_check.csv with
per-fold ROC-AUC from the refit, from results/exp13c_quantum_fold_metrics.csv and from results/F_rbf_gamma025_fold_metrics.csv.
"""
import re,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
ROOT=Path(__file__).resolve().parents[1]; SEED=42
src=(ROOT/'run_all_experiments.py').read_text(encoding='utf-8')
st=src.index("RY = lambda t:"); en=src.index("return np.abs(inner) ** 2",st)+len("return np.abs(inner) ** 2")
H={'np':np,'SEED':SEED}; exec(src[st:en],H); fk=H['fidelity_kernel_matrix']
m2=re.search(r"class QuantumKernelSVM\(BaseEstimator, ClassifierMixin\):.*?(?=\nclass ClassicalPCA3SVM)", src, re.S)
ns2={'np':np,'StandardScaler':StandardScaler,'PCA':PCA,'MinMaxScaler':MinMaxScaler,'SVC':SVC,'BaseEstimator':BaseEstimator,'ClassifierMixin':ClassifierMixin,'SEED':SEED,'fidelity_kernel_matrix':fk}
exec(m2.group(0), ns2); NQ=ns2['QuantumKernelSVM']
class ProjQ:
    def __init__(self): self.imp=SimpleImputer(strategy='median')
    def fit(self,X,y):
        Xi=self.imp.fit_transform(X); self.scaler=StandardScaler().fit(Xi); Z=self.scaler.transform(Xi)
        self.pca=PCA(n_components=3,random_state=SEED).fit(Z); Zp=self.pca.transform(Z)
        self.angle=MinMaxScaler(feature_range=(0,np.pi)).fit(Zp); A=self.angle.transform(Zp)
        self.svc=SVC(kernel='precomputed',C=1.0,probability=True,random_state=SEED).fit(fk(A,A),y); self.A=A; return self
    def predict_proba(self,X):
        Xi=self.imp.transform(X); A=self.angle.transform(self.pca.transform(self.scaler.transform(Xi)))
        return self.svc.predict_proba(fk(A,self.A))
df=pd.read_csv(ROOT/'data'/'wpbc_clean.csv'); y=df.pop('target').astype(int).to_numpy()
if 'subject_stem' in df.columns: df=df.drop(columns='subject_stem')
X=df.astype(float)
splits=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=SEED).split(X,y))
F=pd.read_csv(ROOT/'results'/'F_rbf_gamma025_fold_metrics.csv'); F=F[(F.dataset=='WPBC')&(F.model=='quantum')].reset_index(drop=True)
E=pd.read_csv(ROOT/'results'/'exp13c_quantum_fold_metrics.csv'); E=E[(E.dataset=='WPBC')&E.model.str.startswith('Quantum')].reset_index(drop=True)
res=[]
for f,(tr,te) in enumerate(splits[:15]):
    nb=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',NQ())]).fit(X.iloc[tr],y[tr]).predict_proba(X.iloc[te])[:,1]
    pj=ProjQ().fit(X.iloc[tr],y[tr]).predict_proba(X.iloc[te])[:,1]
    res.append((f+1, float(np.max(np.abs(nb-pj))), roc_auc_score(y[te],nb), roc_auc_score(y[te],pj), float(E.roc_auc[f]), float(F.roc_auc[f])))
r=pd.DataFrame(res,columns=['fold','maxdiff_nb_vs_proj','auc_nb_refit','auc_proj_refit','auc_exp13c','auc_F'])
pd.set_option('display.width',200)
r.round(6).to_csv(ROOT/'results'/'quantum_refit_wpbc_check.csv', index=False)
print(r.round(5).to_string(index=False))
print('means', r[['auc_nb_refit','auc_proj_refit','auc_exp13c','auc_F']].mean().round(4).to_dict())
