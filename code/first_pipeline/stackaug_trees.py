"""Measurement-stream augmentation for trees: one ET/HGB trained on rows from several pose streams (A=RTMW, F=ViTPose-L, G=ViTPose-H, E=ViTPose-b, M=fused) with the same label; test = mean over streams. Compare with per-stream trees."""
import pickle, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier as HGB
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; grp = gt["source_id"].values; cl = np.array(["left", "middle", "right"]); SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk]); acc = lambda Q: (cl[Q.argmax(1)] == yk).mean(); lr = yk != "middle"
names = {"A": "kin", "E": "kinE", "F": "kinF", "G": "kinG", "M": "kinM"}; nan = np.full(259, np.nan, np.float32); FE = {}
for k, p in names.items():
    try: r = pickle.load(open(f"{C}/{p}.pkl", "rb"))["rows"]
    except Exception as e: print("missing", k); continue
    FE[k] = (np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]), np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]]))
use = sys.argv[1] if len(sys.argv) > 1 else "AFG"; use = [u for u in use if u in FE]
# common columns: those non-all-NaN in every used stream (E/F/G lack foot keypoints) -> impute others per stream
keep = np.all([~np.isnan(FE[k][0]).all(0) for k in use], 0); print("streams", use, "common columns", keep.sum())
et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s))
hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
def oof(mk, streams):
    P = np.zeros((len(yk), 3))
    for s in (0, 1, 2):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(yk, yk, grp):
            Xtr = np.concatenate([FE[k][j][tr][:, keep] for k in streams for j in (0, 1)]); ytr = np.concatenate([[yk[tr], ym[tr]][j] for k in streams for j in (0, 1)])
            m = mk(s).fit(Xtr, ytr); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = 0
            for k in streams: p = p + (m.predict_proba(FE[k][0][te][:, keep]) + m.predict_proba(FE[k][1][te][:, keep])[:, ix]) / 2
            P[te] += (p / len(streams))[:, [c_.index(k) for k in cl]] / 3
    return P
for nm, mk in (("ET", et), ("HGB", hg)):
    Ps = oof(mk, use); np.save(f"{C}/stackaug_{nm}_{''.join(use)}.npy", Ps); print(f"{nm} stacked-stream training ({''.join(use)}): kick {acc(Ps):.3f} (L/R {(cl[Ps.argmax(1)][lr] == yk[lr]).mean():.3f})", flush=True)
