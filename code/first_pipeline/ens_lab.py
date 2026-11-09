import numpy as np, pickle, sys, warnings, time
warnings.filterwarnings("ignore")
import cv_kin
from cv_kin import *
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean()
K3 = pickle.load(open(f"{C}/kin3d_+1_2.5.pkl", "rb")); r3 = K3["rows"]; nan = np.full(len(K3["names"]), np.nan, np.float32)
A3 = np.stack([r3.get(c, (nan, nan))[0] for c in gt["clip"]]); B3 = np.stack([r3.get(c, (nan, nan))[1] for c in gt["clip"]])
def cv_gen(mk, Xa, Xb, seeds=(0, 1, 2)):
    ym = np.array([SW[v] for v in yk]); Pr = np.zeros((len(yk), 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
            Pr[te] += (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2 / len(seeds)
    return Pr
hg = lambda **kw: (lambda s: HGB(random_state=s, **{**dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5), **kw}))
imp = lambda est: make_pipeline(SimpleImputer(strategy="median"), est)
M = {}; t0 = time.time()
M["hgb259"] = cv_gen(hg(), X, Xm)
M["hgb_3d+foot"] = cv_gen(hg(), np.c_[A3, X[:, FOOT]], np.c_[B3, Xm[:, FOOT]])
M["hgb_old_kpose"] = cv_gen(hg(), Xo, Xom)
M["extratrees"] = cv_gen(lambda s: imp(ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s)), X, Xm)
M["randforest"] = cv_gen(lambda s: imp(RandomForestClassifier(500, min_samples_leaf=3, max_features=0.2, n_jobs=-1, random_state=s)), X, Xm)
M["mlp"] = cv_gen(lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPClassifier((128, 64), alpha=1.0, max_iter=400, early_stopping=True, random_state=s)), X, Xm)
M["logreg"] = cv_gen(lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.02, max_iter=3000)), X, Xm)
for k, P in M.items(): print(f"{k:16s} kick {acc(P):.3f}")
print(f"[{time.time() - t0:.0f}s]")
def avg(keys, w=None): return sum(M[k] * (1 if w is None else w[i]) for i, k in enumerate(keys)) / (len(keys) if w is None else sum(w))
for keys in (["hgb259", "extratrees"], ["hgb259", "randforest"], ["hgb259", "mlp"], ["hgb259", "hgb_3d+foot"], ["hgb259", "hgb_3d+foot", "extratrees"], ["hgb259", "hgb_3d+foot", "extratrees", "mlp"], ["hgb259", "hgb_3d+foot", "extratrees", "mlp", "hgb_old_kpose"], list(M)):
    print(f"avg {'+'.join(keys):70s} kick {acc(avg(keys)):.3f}")
pickle.dump(M, open(f"{C}/oof_ens_lab.pkl", "wb"))
