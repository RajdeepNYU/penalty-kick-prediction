"""Kinematic features (kin2.build) from the ViTPose++ LARGE stream (cache/kptsM) -> cache/kinM.pkl, then CV of trees on them vs the RTMW-based features."""
import pickle, numpy as np, pandas as pd, warnings, kin2
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}; rows = {}; names = None
for c in list(gt["clip"]) + list(pm["clip"]):
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k0 = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); ks = pickle.load(open(f"{C}/kptsM/{c}.pkl", "rb"))
    if b is None or k0 is None or ks is None or tracks[c] is None or tracks[c]["keep"] is None: continue
    kb = dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"]); kin2.NAMES.clear(); a = kin2.build(b, kb, tracks[c], meta[c], False); names = list(kin2.NAMES); kin2.NAMES.clear(); m = kin2.build(b, kb, tracks[c], meta[c], True)
    if a is not None and m is not None: rows[c] = (a, m)
pickle.dump(dict(rows=rows, names=names), open(f"{C}/kinM.pkl", "wb")); print("kinM built", len(rows))
yk = gt["kick_direction"].values; grp = gt["source_id"].values; SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk]); cl = np.array(["left", "middle", "right"])
def load(p):
    r = pickle.load(open(p, "rb"))["rows"]; nan = np.full(259, np.nan, np.float32); return np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]), np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]])
def oof(mk, Xa, Xb):
    P = np.zeros((len(yk), 3))
    for s in (0, 1, 2):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in cl]] / 3
    return P
hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s))
Fa, Fb = load(f"{C}/kinM.pkl"); KEEP = ~np.isnan(Fa).all(0); np.save(f"{C}/kinM_keep.npy", KEEP); Fa, Fb = Fa[:, KEEP], Fb[:, KEEP]; print("kinM columns kept:", KEEP.sum(), "of", len(KEEP)); acc = lambda P: (cl[P.argmax(1)] == yk).mean()
Hf, Ef = oof(hg, Fa, Fb), oof(et, Fa, Fb); np.save(f"{C}/base_hgb_M.npy", Hf); np.save(f"{C}/base_et_M.npy", Ef)
Ha, Ea = np.load(f"{C}/base_hgb.npy"), np.load(f"{C}/base_et.npy")
print(f"trees on RTMW features: HGB {acc(Ha):.3f} ET {acc(Ea):.3f} avg {acc((Ha + Ea) / 2):.3f}")
print(f"trees on fused pose features: HGB {acc(Hf):.3f} ET {acc(Ef):.3f} avg {acc((Hf + Ef) / 2):.3f}")
print(f"trees, both feature sets averaged: {acc((Ha + Ea + Hf + Ef) / 4):.3f}")



