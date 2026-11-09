"""Final components for v10: (1) trees on ViTPose-huge kinematic features -> trees_private_G.npy ; (2) keeper trees (HGB x3 seeds + ET x3 seeds) on ViTPose-large keeper features -> keeper_private_F.npy (sum of the two models' probabilities)."""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
base = dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5); HC = [dict(), dict(max_depth=5, max_iter=200, min_samples_leaf=40), dict(max_depth=3, min_samples_leaf=50), dict(max_depth=3)]
def getrows(path, keep):
    K = pickle.load(open(path, "rb"))["rows"]; n = len(next(iter(K.values()))[0]); nan = np.full(n, np.nan, np.float32); g = lambda cs, j: np.stack([K.get(c, (nan, nan))[j] for c in cs])[:, keep]
    return g(gt["clip"], 0), g(gt["clip"], 1), g(pm["clip"], 0), g(pm["clip"], 1)
def run(models, Xa, Xb, Ta, Tb, y, SW, classes):
    ym = np.array([SW[v] for v in y]); P = 0
    for mk in models:
        m = mk.fit(np.r_[Xa, Xb], np.r_[y, ym]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = (m.predict_proba(Ta) + m.predict_proba(Tb)[:, ix]) / 2; P = P + p[:, [c_.index(k) for k in classes]]
    return P / len(models)
# (1) kick trees on huge-model features
SWk = {"left": "right", "right": "left", "middle": "middle"}; ck = ["left", "middle", "right"]; Xa, Xb, Ta, Tb = getrows(f"{C}/kinG.pkl", np.load(f"{C}/kinG_keep.npy")); y = gt["kick_direction"].values
Ph = run([HGB(random_state=s, **{**base, **cfg}) for cfg in HC for s in range(3)], Xa, Xb, Ta, Tb, y, SWk, ck); Pe = run([make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s)) for s in range(3)], Xa, Xb, Ta, Tb, y, SWk, ck)
np.save(f"{C}/trees_private_G.npy", (Ph + Pe) / 2); print("saved trees_private_G.npy")
# (2) keeper trees on ViTPose-large keeper features
SWg = {"left": "right", "right": "left", "stands in the middle": "stands in the middle"}; cg = ["left", "right", "stands in the middle"]; Xa, Xb, Ta, Tb = getrows(f"{C}/keepfeatF.pkl", np.load(f"{C}/keepfeatF_keep.npy")); y = gt["goalkeeper_direction"].values
Gh = run([HGB(random_state=s, **base) for s in range(3)], Xa, Xb, Ta, Tb, y, SWg, cg); Ge = run([make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=s)) for s in range(3)], Xa, Xb, Ta, Tb, y, SWg, cg)
np.save(f"{C}/keeper_private_F.npy", Gh + Ge); print("saved keeper_private_F.npy")
