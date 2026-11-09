"""Train per-target GBDTs on all public clips (mirror-augmented), predict private clips with mirror TTA, write the CSV."""
import pickle, sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUTCSV = sys.argv[1] if len(sys.argv) > 1 else ROOT + r"\Q4_work\submission_v1.csv"
feats = pickle.load(open(f"{C}/feats.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
SW = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}
params = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
KP = list(np.r_[0:90]) + [261]                     # kicker static pose + foot
CFG = {"kick_direction": KP, "goalkeeper_direction": list(range(268))}
NS = 5


def fit_predict(cols, target):
    Xa = np.stack([feats[c][0] for c in gt["clip"]])[:, cols]; Xb = np.stack([feats[c][1] for c in gt["clip"]])[:, cols]
    y = gt[target].values; ym = np.array([SW[v] for v in y])
    ok = [c for c in pm["clip"] if c in feats]
    Ta = np.stack([feats[c][0] for c in ok])[:, cols]; Tb = np.stack([feats[c][1] for c in ok])[:, cols]
    P = 0
    for s in range(NS):
        m = HGB(**params, random_state=s).fit(np.r_[Xa, Xb], np.r_[y, ym])
        cls = list(m.classes_); ix = [cls.index(SW[k]) for k in cls]
        P = P + (m.predict_proba(Ta) + m.predict_proba(Tb)[:, ix]) / 2
    return dict(zip(ok, np.array(cls)[P.argmax(1)]))


pred = {t: fit_predict(cols, t) for t, cols in CFG.items()}
rows = []
for c in pm["clip"]:
    rows.append(dict(clip=c, kick_direction=pred["kick_direction"].get(c, "left"),
                     goalkeeper_direction=pred["goalkeeper_direction"].get(c, "left"), goal_outcome="goal"))
sub = pd.DataFrame(rows)
tmpl = pd.read_csv(D + r"\submission_template.csv")
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900
print("missing-features fallbacks:", sum(c not in feats for c in pm["clip"]))
for k in ("kick_direction", "goalkeeper_direction", "goal_outcome"): print(sub[k].value_counts().to_dict())
sub.to_csv(OUTCSV, index=False); print("wrote", OUTCSV)
