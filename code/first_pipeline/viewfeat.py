"""Camera-view features (mirror-invariant): goal-box aspect/size relative to the kicker, ball->goal offset in goal-box units, apparent goal-axis angle. Test whether they help trees (they let the model learn how lateral image motion maps to kick direction)."""
import pickle, numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier as HGB
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); yk = gt["kick_direction"].values; grp = gt["source_id"].values; cl = np.array(["left", "middle", "right"]); SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk]); acc = lambda Q: (cl[Q.argmax(1)] == yk).mean(); lr = yk != "middle"
def vf(c):
    try:
        s = pickle.load(open(f"{C}/scene/{c}.pkl", "rb")); b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
        g = s["goal_a"][0]; gw, gh = g[2] - g[0], g[3] - g[1]; gc = np.array([(g[0] + g[2]) / 2, (g[1] + g[3]) / 2]); bl = np.array(b["ball_last"], float); kh = np.median(k["box"][:, 3] - k["box"][:, 1]); d = bl - gc
        return np.array([gw / gh, gh / kh, gw / kh, np.hypot(*d) / kh, abs(d[0]) / gw, d[1] / gh, abs(np.arctan2(d[1], d[0] if d[0] >= 0 else -d[0])), g[4], b["wh"][0] / 1280.0], np.float32)
    except Exception as e: return np.full(9, np.nan, np.float32)
V = np.stack([vf(c) for c in gt["clip"]]); Vp = np.stack([vf(c) for c in pm["clip"]]); np.save(f"{C}/viewfeat_train.npy", V); np.save(f"{C}/viewfeat_private.npy", Vp); print("view feats nan frac", np.isnan(V).mean(), "private", np.isnan(Vp).mean())
tag = sys.argv[1] if len(sys.argv) > 1 else "kinF"; nan = np.full(259, np.nan, np.float32); r = pickle.load(open(f"{C}/{tag}.pkl", "rb"))["rows"]; Xa = np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]); Xb = np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]]); keep = ~np.isnan(Xa).all(0); Xa, Xb = Xa[:, keep], Xb[:, keep]
et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s)); hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
def oof(mk, A, B):
    P = np.zeros((len(yk), 3))
    for s in (0, 1, 2):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(yk, yk, grp):
            m = mk(s).fit(np.r_[A[tr], B[tr]], np.r_[yk[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = (m.predict_proba(A[te]) + m.predict_proba(B[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in cl]] / 3
    return P
for nm, (A, B) in (("base", (Xa, Xb)), ("+view", (np.c_[Xa, V], np.c_[Xb, V]))):
    Pe, Ph = oof(et, A, B), oof(hg, A, B); print(f"{tag} {nm:6s}: ET {acc(Pe):.3f} HGB {acc(Ph):.3f} avg {acc(Pe + Ph):.3f} (L/R {(cl[(Pe + Ph).argmax(1)][lr] == yk[lr]).mean():.3f})", flush=True)
