import numpy as np, pickle, warnings, time
warnings.filterwarnings("ignore")
import cv_kin
from cv_kin import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean()
KB = pickle.load(open(f"{C}/kinb.pkl", "rb"))["rows"]; Xb_ = np.stack([KB[c][0] for c in gt["clip"]]); Xbm = np.stack([KB[c][1] for c in gt["clip"]])
M = pickle.load(open(f"{C}/oof_ens_lab.pkl", "rb"))
def cv_gen(mk, Xa, Xb, seeds=(0, 1, 2)):
    ym = np.array([SW[v] for v in yk]); Pr = np.zeros((len(yk), 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
            Pr[te] += (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2 / len(seeds)
    return Pr
hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s))
fuse = lambda a, b: np.nanmean(np.stack([a, b]), axis=0)
t0 = time.time()
M["hgb_B"] = cv_gen(hg, Xb_, Xbm); M["et_B"] = cv_gen(et, Xb_, Xbm)
XF, XFm = fuse(X, Xb_), fuse(Xm, Xbm); M["hgb_fused"] = cv_gen(hg, XF, XFm); M["et_fused"] = cv_gen(et, XF, XFm)
for k in ("hgb259", "extratrees", "hgb_B", "et_B", "hgb_fused", "et_fused"): print(f"{k:12s} kick {acc(M[k]):.3f}")
print(f"[{time.time() - t0:.0f}s]")
avg = lambda keys: sum(M[k] for k in keys) / len(keys)
for keys in (["hgb259", "extratrees", "hgb_B", "et_B"], ["hgb_fused", "et_fused"], ["hgb259", "extratrees", "hgb_B", "et_B", "hgb_fused", "et_fused"], ["extratrees", "et_B", "et_fused"], ["hgb259", "extratrees", "hgb_B", "et_B", "hgb_fused", "et_fused", "hgb_3d+foot"]):
    print(f"avg {'+'.join(keys):80s} kick {acc(avg(keys)):.3f}")
pickle.dump(M, open(f"{C}/oof_ens_lab2.pkl", "wb"))
