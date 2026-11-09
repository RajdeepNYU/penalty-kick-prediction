"""Data-limited regime -> does in-fold feature selection (top-k by ET importance on the training fold) help ET/HGB? ViTPose-L features (kinF) and fused (kinM)."""
import pickle, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier as HGB
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; grp = gt["source_id"].values; cl = np.array(["left", "middle", "right"]); SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk]); acc = lambda Q: (cl[Q.argmax(1)] == yk).mean(); lr = yk != "middle"
nan = np.full(259, np.nan, np.float32); r = pickle.load(open(f"{C}/{sys.argv[1]}.pkl", "rb"))["rows"]; Xa = np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]); Xb = np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]]); keep = ~np.isnan(Xa).all(0); Xa, Xb = Xa[:, keep], Xb[:, keep]; print("features", Xa.shape[1])
et = lambda s, n=500: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s))
def run(K, mk):
    P = np.zeros((len(yk), 3))
    for s in (0, 1, 2):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(yk, yk, grp):
            Xtr = np.r_[Xa[tr], Xb[tr]]; ytr = np.r_[yk[tr], ym[tr]]; cols = np.arange(Xa.shape[1])
            if K < Xa.shape[1]:
                sel = et(s, 300).fit(Xtr, ytr); imp = sel[-1].feature_importances_; cols = np.argsort(-imp)[:K]
            m = mk(s).fit(Xtr[:, cols], ytr); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = (m.predict_proba(Xa[te][:, cols]) + m.predict_proba(Xb[te][:, cols])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in cl]] / 3
    return P
hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
for K in (30, 60, 100, 150, 999):
    Pe = run(K, et); Ph = run(K, hg); print(f"top-{K if K < 999 else 'all'}: ET {acc(Pe):.3f} HGB {acc(Ph):.3f} avg {acc(Pe + Ph):.3f} (L/R {(cl[(Pe + Ph).argmax(1)][lr] == yk[lr]).mean():.3f})", flush=True)
