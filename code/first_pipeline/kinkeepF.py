"""Keeper features (kinkeep.build) from the ViTPose++ large keeper keypoints (cache/kkeepF) -> cache/keepfeatF.pkl, then keeper-direction CV of trees on them vs the RTMW-based keeper features and the final blends."""
import pickle, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import kinkeep
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}; rows = {}; names = None
for c in list(gt["clip"]) + list(pm["clip"]):
    p = f"{C}/kkeepF/{c}.pkl"
    if not os.path.exists(p): continue
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k0 = pickle.load(open(f"{C}/kkeep/{c}.pkl", "rb")); ks = pickle.load(open(p, "rb"))
    if b is None or k0 is None or ks is None or tracks[c] is None or tracks[c]["keep"] is None: continue
    kk = dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"]); kinkeep.NAMES.clear(); a = kinkeep.build(b, kk, tracks[c], meta[c], False); names = list(kinkeep.NAMES); kinkeep.NAMES.clear(); m = kinkeep.build(b, kk, tracks[c], meta[c], True)
    if a is not None and m is not None: rows[c] = (a, m)
pickle.dump(dict(rows=rows, names=names), open(f"{C}/keepfeatF.pkl", "wb")); print("keepfeatF built", len(rows))
yg = gt["goalkeeper_direction"].values; grp = gt["source_id"].values; SW = {"left": "right", "right": "left", "stands in the middle": "stands in the middle"}; ym = np.array([SW[v] for v in yg]); cg = ["left", "right", "stands in the middle"]
def load(path):
    r = pickle.load(open(path, "rb"))["rows"]; n = len(next(iter(r.values()))[0]); nan = np.full(n, np.nan, np.float32); return np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]), np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]])
def oof(mk, Xa, Xb):
    P = np.zeros((len(yg), 3))
    for s in (0, 1, 2):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yg, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yg[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in cg]] / 3
    return P
hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=s))
Fa, Fb = load(f"{C}/keepfeatF.pkl"); KEEP = ~np.isnan(Fa).all(0); np.save(f"{C}/keepfeatF_keep.npy", KEEP); Fa, Fb = Fa[:, KEEP], Fb[:, KEEP]
Ra, Rb = load(f"{C}/keepfeat.pkl"); acc = lambda P: (np.array(cg)[P.argmax(1)] == yg).mean()
Hr, Er = oof(hg, Ra, Rb), oof(et, Ra, Rb); Hf, Ef = oof(hg, Fa, Fb), oof(et, Fa, Fb); cnn = np.load(f"{C}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]
np.save(f"{C}/keep_oof_rtmw.npy", (Hr + Er) / 2); np.save(f"{C}/keep_oof_F.npy", (Hf + Ef) / 2)
print(f"keeper trees on RTMW feats: HGB {acc(Hr):.3f} ET {acc(Er):.3f} | on ViTPose-L feats ({KEEP.sum()} cols): HGB {acc(Hf):.3f} ET {acc(Ef):.3f} | CNN {acc(cnn):.3f}")
print(f"current blend (RTMW trees + CNN) {acc((Hr + Er + cnn) / 3):.3f} | + ViTPose-L trees {acc((Hr + Er + Hf + Ef + cnn) / 5):.3f} | ViTPose-L trees + CNN only {acc((Hf + Ef + cnn) / 3):.3f} | all trees (no CNN) {acc((Hr + Er + Hf + Ef) / 4):.3f}")
