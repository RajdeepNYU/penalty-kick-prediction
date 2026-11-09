"""v4 submission: kick = bagged GBDTs on label-free kinematic features (kin.pkl, 259 dims) with mirror augmentation + mirror TTA;
keeper = same keeper models as v3 (GBDT + CNN blend, copied from submission_v3.csv); goal outcome = always 'goal'."""
import pickle, sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUTCSV = ROOT + r"\Q4_work\submission_v4.csv"
K = pickle.load(open(f"{C}/kin.pkl", "rb")); rows = K["rows"]
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
SW = {"left": "right", "right": "left", "middle": "middle"}
base = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
CFGS = [dict(max_depth=4), dict(max_depth=5, max_iter=200, min_samples_leaf=40), dict(min_samples_leaf=50), dict()]   # top configs of tune_kin.py (CV 0.631-0.641)
NS = 3
Xa = np.stack([rows[c][0] for c in gt["clip"]]); Xb = np.stack([rows[c][1] for c in gt["clip"]])
Ta = np.stack([rows[c][0] for c in pm["clip"]]); Tb = np.stack([rows[c][1] for c in pm["clip"]])
y = gt["kick_direction"].values; ym = np.array([SW[v] for v in y]); P = 0
for cfg in CFGS:
    for s in range(NS):
        m = HGB(random_state=s, **{**base, **cfg}).fit(np.r_[Xa, Xb], np.r_[y, ym]); cls = list(m.classes_); ix = [cls.index(SW[k]) for k in cls]
        P = P + (m.predict_proba(Ta) + m.predict_proba(Tb)[:, ix]) / 2
P /= len(CFGS) * NS
v3 = pd.read_csv(ROOT + r"\Q4_work\submission_v3.csv"); assert list(v3["clip"]) == list(pm["clip"])
sub = pd.DataFrame({"clip": pm["clip"], "kick_direction": np.array(cls)[P.argmax(1)], "goalkeeper_direction": v3["goalkeeper_direction"], "goal_outcome": "goal"})
tmpl = pd.read_csv(D + r"\submission_template.csv")
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900 and sub["clip"].is_unique and set(sub["kick_direction"]) <= {"left", "middle", "right"}
for k in ("kick_direction", "goalkeeper_direction", "goal_outcome"): print(k, {str(a): int(b) for a, b in sub[k].value_counts().items()})
sub.to_csv(OUTCSV, index=False); np.save(f"{C}/pk_private_v4.npy", P); print("wrote", OUTCSV, "| kick agreement with v3: %.3f" % (v3.kick_direction == sub.kick_direction).mean())
