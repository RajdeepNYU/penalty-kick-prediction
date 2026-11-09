"""Keeper-direction CV: new stabilised features vs old keeper features, +CNN blend. Source-grouped, mirror aug (left<->right), 3 seeds."""
import numpy as np, pickle, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yg = gt["goalkeeper_direction"].values; grp = gt["source_id"].values
SW = {"left": "right", "right": "left", "stands in the middle": "stands in the middle"}
KF = pickle.load(open(f"{C}/keepfeat.pkl", "rb")); r = KF["rows"]; names = KF["names"]; nan = np.full(len(names), np.nan, np.float32)
A = np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]); B = np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]]); print("clips with new keeper feats:", sum(c in r for c in gt["clip"]), "dim", A.shape[1])
old = pickle.load(open(f"{C}/feats.pkl", "rb")); OA = np.stack([old[c][0] for c in gt["clip"]])[:, list(range(268))]; OB = np.stack([old[c][1] for c in gt["clip"]])[:, list(range(268))]
cls = ["left", "right", "stands in the middle"]; acc = lambda P: (np.array(cls)[P.argmax(1)] == yg).mean()
def cv_gen(mk, Xa, Xb, seeds=(0, 1, 2)):
    ym = np.array([SW[v] for v in yg]); Pr = np.zeros((len(yg), 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yg, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yg[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
            P = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; Pr[te] += P[:, [c_.index(k) for k in cls]] / len(seeds)
    return Pr
hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=s))
cnn = np.load(f"{C}/oof_gk_cnn30.npy")[:, [0, 2, 1]]; old_g = np.load(f"{C}/oof_final_gk_gbdt.npy"); t0 = time.time(); M = {}
M["old_hgb"] = old_g; M["cnn"] = cnn
M["new_hgb"] = cv_gen(hg, A, B); M["new_et"] = cv_gen(et, A, B)
M["both_hgb"] = cv_gen(hg, np.c_[A, OA], np.c_[B, OB])
for k, P in M.items(): print(f"{k:10s} keeper acc {acc(P):.3f}")
print(f"[{time.time() - t0:.0f}s]")
for keys in (["new_hgb", "new_et"], ["new_hgb", "new_et", "cnn"], ["both_hgb", "cnn"], ["new_hgb", "new_et", "both_hgb", "cnn"], ["new_hgb", "new_et", "old_hgb", "cnn"], ["new_hgb", "new_et", "both_hgb", "old_hgb", "cnn"]):
    print(f"avg {'+'.join(keys):45s} {acc(sum(M[k] for k in keys) / len(keys)):.3f}")
pickle.dump(M, open(f"{C}/oof_keep_lab.pkl", "wb"))
