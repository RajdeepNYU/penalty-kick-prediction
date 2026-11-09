"""Stage 15: final submission. Kick: GBDT on kicker pose + foot. Keeper: GBDT on pose/motion features blended 50/50 with the keeper CNN.
Goal outcome: always 'goal' (no model beat the majority class in source-grouped CV). Mirror augmentation + mirror TTA throughout."""
import pickle, sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUTCSV = ROOT + r"\Q4_work\submission_v2.csv"
W_CNN = 0.5
feats = pickle.load(open(f"{C}/feats.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
SW = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}
params = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
KP = list(np.r_[0:90]) + [261]
CFG = {"kick_direction": KP, "goalkeeper_direction": list(range(268))}
NS = 5


def hgb_probs(cols, target):
    Xa = np.stack([feats[c][0] for c in gt["clip"]])[:, cols]; Xb = np.stack([feats[c][1] for c in gt["clip"]])[:, cols]
    y = gt[target].values; ym = np.array([SW[v] for v in y])
    Ta = np.stack([feats[c][0] for c in pm["clip"]])[:, cols]; Tb = np.stack([feats[c][1] for c in pm["clip"]])[:, cols]
    P = 0
    for s in range(NS):
        m = HGB(**params, random_state=s).fit(np.r_[Xa, Xb], np.r_[y, ym])
        cls = list(m.classes_); ix = [cls.index(SW[k]) for k in cls]
        P = P + (m.predict_proba(Ta) + m.predict_proba(Tb)[:, ix]) / 2
    return P / NS, cls


Pk, ck_ = hgb_probs(CFG["kick_direction"], "kick_direction")
Pg, cg = hgb_probs(CFG["goalkeeper_direction"], "goalkeeper_direction")       # classes sorted: left, right, stands in the middle
cnn = np.load(f"{C}/cnn_gk_private.npy")[:, [0, 2, 1]]                            # CNN order left, stands, right -> left, right, stands
Pg_blend = (1 - W_CNN) * Pg + W_CNN * cnn
sub = pd.DataFrame({"clip": pm["clip"], "kick_direction": np.array(ck_)[Pk.argmax(1)],
                    "goalkeeper_direction": np.array(cg)[Pg_blend.argmax(1)], "goal_outcome": "goal"})
tmpl = pd.read_csv(D + r"\submission_template.csv")
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900 and sub["clip"].is_unique
assert set(sub["kick_direction"]) <= {"left", "middle", "right"} and set(sub["goalkeeper_direction"]) <= {"left", "stands in the middle", "right"}
for k in ("kick_direction", "goalkeeper_direction", "goal_outcome"): print(k, {str(a): int(b) for a, b in sub[k].value_counts().items()})
sub.to_csv(OUTCSV, index=False); print("wrote", OUTCSV)
v1 = pd.read_csv(ROOT + r"\Q4_work\submission_v1.csv")
print("agreement with v1: kick %.3f keeper %.3f" % ((v1.kick_direction == sub.kick_direction).mean(), (v1.goalkeeper_direction == sub.goalkeeper_direction).mean()))
