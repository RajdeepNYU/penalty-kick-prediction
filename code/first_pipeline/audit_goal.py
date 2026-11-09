"""Sensitivity of kick accuracy to error in the goal direction (ball->keeper proxy): rotate the goal axis by N(0, delta) deg per clip (same rotation for the mirrored sample),
rebuild the 259 kinematic features for the training clips, and re-run the identical GBDT CV."""
import pickle, re, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"; Q = ROOT + r"\Q4_work"; D = ROOT + r"\Q4data"; C = Q + r"\cache"
s = open(f"{Q}\\kin2.py", encoding="utf-8").read()
hook = "f = fv / fn if fn > 1e-6 else np.array([1.0, 0.0])"; assert hook in s
s = s.replace(hook, hook + "\n    if GOAL_JITTER_DEG:\n        rs = np.random.RandomState(int(abs(ball[0] * 1e3 + ball[1] * 7)) % (2 ** 31)); a_ = np.deg2rad(rs.randn() * GOAL_JITTER_DEG); f = np.array([np.cos(a_) * f[0] - np.sin(a_) * f[1], np.sin(a_) * f[0] + np.cos(a_) * f[1]])")
s = s.replace("NAMES = []\n", "NAMES = []\nGOAL_JITTER_DEG = 0.0\n", 1); s = s.split('if __name__ == "__main__":')[0]
ns = {}; exec(compile(s, "kin2_jit", "exec"), ns)
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import StratifiedGroupKFold
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in gt.iterrows()}
yk = gt["kick_direction"].values; grp = gt["source_id"].values; SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk])
data = {c: (pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")), pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))) for c in gt["clip"]}
for delta in [float(v) for v in sys.argv[1:]]:
    ns["GOAL_JITTER_DEG"] = delta; A, B = [], []
    for c in gt["clip"]:
        b, k = data[c]; ns["NAMES"].clear(); a = ns["build"](b, k, tracks[c], meta[c], False); ns["NAMES"].clear(); m = ns["build"](b, k, tracks[c], meta[c], True); A.append(a); B.append(m)
    A, B = np.stack(A), np.stack(B); P = np.zeros((len(yk), 3))
    for sd in (0, 1, 2):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(A, yk, grp):
            m_ = HGB(random_state=sd, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5).fit(np.r_[A[tr], B[tr]], np.r_[yk[tr], ym[tr]])
            cl = list(m_.classes_); ix = [cl.index(SW[k]) for k in cl]; p = (m_.predict_proba(A[te]) + m_.predict_proba(B[te])[:, ix]) / 2; P[te] += p[:, [cl.index(k) for k in ["left", "middle", "right"]]] / 3
    pred = np.array(["left", "middle", "right"])[P.argmax(1)]; lr = yk != "middle"
    print(f"goal-axis noise sd {delta:4.0f} deg: kick acc {(pred == yk).mean():.3f} | L/R-only {(pred[lr] == yk[lr]).mean():.3f}", flush=True)
