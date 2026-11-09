"""Stage 15: final submission (v3). Kick: bag of 3 GBDT configs on kicker pose + foot. Keeper: bag of 2 GBDT configs on all features,
blended 50/50 with the keeper CNN (30 epochs, 5 seeds). Goal outcome: always 'goal' (nothing beat the majority class in source-grouped CV).
All models use mirror augmentation on training and mirror test-time averaging."""
import pickle, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUTCSV = ROOT + r"\Q4_work\submission_v3.csv"
W_CNN, NS = 0.5, 3
feats = pickle.load(open(f"{C}/feats.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
SW = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}
base = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
KP = list(np.r_[0:90]) + [261]; ALL = list(range(268))


def hgb_probs(cfgs, cols, target):
    Xa = np.stack([feats[c][0] for c in gt["clip"]])[:, cols]; Xb = np.stack([feats[c][1] for c in gt["clip"]])[:, cols]
    y = gt[target].values; ym = np.array([SW[v] for v in y])
    Ta = np.stack([feats[c][0] for c in pm["clip"]])[:, cols]; Tb = np.stack([feats[c][1] for c in pm["clip"]])[:, cols]
    P = 0
    for cfg in cfgs:
        for s in range(NS):
            m = HGB(random_state=s, **{**base, **cfg}).fit(np.r_[Xa, Xb], np.r_[y, ym])
            cls = list(m.classes_); ix = [cls.index(SW[k]) for k in cls]
            P = P + (m.predict_proba(Ta) + m.predict_proba(Tb)[:, ix]) / 2
    return P / (len(cfgs) * NS), cls


Pk, ck_ = hgb_probs([{}, {"max_depth": 4}, {"max_iter": 500}], KP, "kick_direction")
Pg, cg = hgb_probs([{"max_depth": 4}, {"max_iter": 500}], ALL, "goalkeeper_direction")     # classes sorted: left, right, stands in the middle
cnn = np.load(f"{C}/cnn_gk_private.npy")[:, [0, 2, 1]]                                      # CNN order left, stands, right -> left, right, stands
Pg_blend = (1 - W_CNN) * Pg + W_CNN * cnn
sub = pd.DataFrame({"clip": pm["clip"], "kick_direction": np.array(ck_)[Pk.argmax(1)],
                    "goalkeeper_direction": np.array(cg)[Pg_blend.argmax(1)], "goal_outcome": "goal"})
tmpl = pd.read_csv(D + r"\submission_template.csv")
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900 and sub["clip"].is_unique
assert set(sub["kick_direction"]) <= {"left", "middle", "right"} and set(sub["goalkeeper_direction"]) <= {"left", "stands in the middle", "right"}
for k in ("kick_direction", "goalkeeper_direction", "goal_outcome"): print(k, {str(a): int(b) for a, b in sub[k].value_counts().items()})
sub.to_csv(OUTCSV, index=False); print("wrote", OUTCSV)
v2 = pd.read_csv(ROOT + r"\Q4_work\submission_v2.csv")
print("agreement with v2: kick %.3f keeper %.3f" % ((v2.kick_direction == sub.kick_direction).mean(), (v2.goalkeeper_direction == sub.goalkeeper_direction).mean()))
