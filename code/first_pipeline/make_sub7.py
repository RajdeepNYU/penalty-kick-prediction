"""v7 submission = v6 recipe on the improved detections (two-stage picker, TTA candidates, goal box, filled tracks, fused ball).
v5 recipe (kept for reference):
kick    = mean of {GBDT bag (4 configs x 3 seeds), ExtraTrees (500 trees x 3 seeds)} on 259 label-free kinematic features (kin.pkl), mirror aug + mirror TTA
keeper  = mean of {GBDT, ExtraTrees on 162 camera-stabilised keeper features (keepfeat.pkl), ResNet34 keeper CNN (cnn_gk_private.npy, 5 seeds x 30 epochs)}
outcome = always 'goal'. Clips whose features could not be built get NaN features (trees handle them natively)."""
import pickle, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUTCSV = ROOT + r"\Q4_work\submission_v7.csv"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
SWK = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}
base = dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
HCFG = [dict(), dict(max_depth=5, max_iter=200, min_samples_leaf=40), dict(max_depth=3, min_samples_leaf=50), dict(max_depth=3)]
NS = 3
hg = lambda s, cfg: HGB(random_state=s, **{**base, **cfg})
et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s))
et_keep = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=s))


def load(path, clips_tr, clips_te):
    K = pickle.load(open(path, "rb")); r = K["rows"]; nan = np.full(len(K["names"]), np.nan, np.float32)
    g = lambda cs, j: np.stack([r.get(c, (nan, nan))[j] for c in cs])
    return g(clips_tr, 0), g(clips_tr, 1), g(clips_te, 0), g(clips_te, 1), sum(c not in r for c in clips_te)


def fit_predict(models, Xa, Xb, Ta, Tb, y, classes_out):
    ym = np.array([SWK[v] for v in y]); P = 0
    for mk in models:
        m = mk.fit(np.r_[Xa, Xb], np.r_[y, ym]); c_ = list(m.classes_); ix = [c_.index(SWK[k]) for k in c_]
        p = (m.predict_proba(Ta) + m.predict_proba(Tb)[:, ix]) / 2; P = P + p[:, [c_.index(k) for k in classes_out]]
    return P / len(models)


# ---- kick
Xa, Xb, Ta, Tb, miss = load(f"{C}/kin.pkl", gt["clip"], pm["clip"]); print("private clips without kick features:", miss)
yk = gt["kick_direction"].values; ck = ["left", "middle", "right"]
Ph = fit_predict([hg(s, cfg) for cfg in HCFG for s in range(NS)], Xa, Xb, Ta, Tb, yk, ck)
Pe = fit_predict([et(s) for s in range(NS)], Xa, Xb, Ta, Tb, yk, ck)
Pk = (Ph + Pe) / 2
# ---- keeper
Ka, Kb, Ua, Ub, missk = load(f"{C}/keepfeat.pkl", gt["clip"], pm["clip"]); print("private clips without keeper features:", missk)
yg = gt["goalkeeper_direction"].values; cg = ["left", "right", "stands in the middle"]
Pgh = fit_predict([hg(s, {}) for s in range(NS)], Ka, Kb, Ua, Ub, yg, cg)
Pge = fit_predict([et_keep(s) for s in range(NS)], Ka, Kb, Ua, Ub, yg, cg)
cnn = np.load(f"{C}/cnn_gk_private.npy")[:, [0, 2, 1]]                      # CNN order left, stands, right -> left, right, stands
Pg = (Pgh + Pge + cnn) / 3
sub = pd.DataFrame({"clip": pm["clip"], "kick_direction": np.array(ck)[(0.5 * Pk + 0.5 * np.load(f"{C}/nn_private.npy")).argmax(1)], "goalkeeper_direction": np.array(cg)[Pg.argmax(1)], "goal_outcome": "goal"})
tmpl = pd.read_csv(D + r"\submission_template.csv")
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900 and sub["clip"].is_unique
assert set(sub["kick_direction"]) <= set(ck) and set(sub["goalkeeper_direction"]) <= set(cg)
for k in ("kick_direction", "goalkeeper_direction", "goal_outcome"): print(k, {str(a): int(b) for a, b in sub[k].value_counts().items()})
sub.to_csv(OUTCSV, index=False); np.save(f"{C}/pk_private_v7.npy", Pk); np.save(f"{C}/pg_private_v7.npy", Pg)
v4 = pd.read_csv(ROOT + r"\Q4_work\submission_v6.csv"); print("wrote", OUTCSV, "| agreement with v6: kick %.3f keeper %.3f" % ((v4.kick_direction == sub.kick_direction).mean(), (v4.goalkeeper_direction == sub.goalkeeper_direction).mean()))
