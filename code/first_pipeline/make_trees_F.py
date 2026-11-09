"""Final trees on the ViTPose-large kinematic features (kinF.pkl, non-empty columns): HGB bag (4 configs x 3 seeds) + ExtraTrees (3 seeds), mirror aug + mirror TTA -> cache/trees_private_F.npy."""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); K = pickle.load(open(f"{C}/kinF.pkl", "rb"))["rows"]; KEEP = np.load(f"{C}/kinF_keep.npy"); nan = np.full(259, np.nan, np.float32)
g = lambda cs, j: np.stack([K.get(c, (nan, nan))[j] for c in cs])[:, KEEP]; Xa, Xb, Ta, Tb = g(gt["clip"], 0), g(gt["clip"], 1), g(pm["clip"], 0), g(pm["clip"], 1)
SW = {"left": "right", "right": "left", "middle": "middle"}; y = gt["kick_direction"].values; ym = np.array([SW[v] for v in y]); ck = ["left", "middle", "right"]
base = dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5); HC = [dict(), dict(max_depth=5, max_iter=200, min_samples_leaf=40), dict(max_depth=3, min_samples_leaf=50), dict(max_depth=3)]
def run(models):
    P = 0
    for mk in models:
        m = mk.fit(np.r_[Xa, Xb], np.r_[y, ym]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = (m.predict_proba(Ta) + m.predict_proba(Tb)[:, ix]) / 2; P = P + p[:, [c_.index(k) for k in ck]]
    return P / len(models)
Ph = run([HGB(random_state=s, **{**base, **cfg}) for cfg in HC for s in range(3)]); Pe = run([make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s)) for s in range(3)])
np.save(f"{C}/trees_private_F.npy", (Ph + Pe) / 2); print("saved trees_private_F.npy; predicted:", {k: int(v) for k, v in zip(ck, np.bincount(((Ph + Pe) / 2).argmax(1), minlength=3))})
