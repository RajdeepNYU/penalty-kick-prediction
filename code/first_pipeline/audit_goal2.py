"""(1) how far is the old goal-axis proxy (ball->keeper feet) from ball->goal-box targets? (2) kick CV using different goal-axis definitions (same model, same folds)."""
import pickle, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"; Q = ROOT + r"\Q4_work"; D = ROOT + r"\Q4data"; C = Q + r"\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); yk = gt["kick_direction"].values; grp = gt["source_id"].values
SC = {c: pickle.load(open(f"{C}/scene/{c}.pkl", "rb")) for c in gt["clip"]}; BS = {c: pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")) for c in gt["clip"]}; KP = {c: pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")) for c in gt["clip"]}


def targets(c):
    kb = tracks[c]["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]; kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]); g = SC[c]["goal_l"]
    if not len(g): return dict(keeper=kf, gbottom=kf, gcentre=kf, gx_keepy=kf, avg=kf)
    g = g[0]; gcx = (g[0] + g[2]) / 2; return dict(keeper=kf, gbottom=np.array([gcx, g[3]]), gcentre=np.array([gcx, (g[1] + g[3]) / 2]), gx_keepy=np.array([gcx, kf[1]]), avg=(kf + np.array([gcx, g[3]])) / 2)
# (1) angular difference between proxies
ang = []
for c in gt["clip"]:
    b = BS[c]; ball = b["ball_last"]
    if not np.isfinite(ball[0]): continue
    t = targets(c); a0 = np.arctan2(*(t["keeper"] - ball)[::-1]); ang.append({k: np.degrees(np.abs((np.arctan2(*(v - ball)[::-1]) - a0 + np.pi) % (2 * np.pi) - np.pi)) for k, v in t.items() if k != "keeper"})
ang = pd.DataFrame(ang); print("angle between ball->keeper-feet and ball->target (degrees), median / 75th / 90th percentile:"); [print(f"  {k:10s} {ang[k].median():5.1f} / {ang[k].quantile(.75):5.1f} / {ang[k].quantile(.9):5.1f}") for k in ang.columns]
# (2) CV with each definition
s = open(f"{Q}\\kin2.py", encoding="utf-8").read(); hook = "fv = kf - ball"; assert hook in s
s = s.replace(hook, "kf = TARGET[MODE] if (MODE != 'keeper' and TARGET) else kf; " + hook, 1).replace("NAMES = []\n", "NAMES = []\nMODE = 'keeper'; TARGET = None\n", 1).split('if __name__ == "__main__":')[0]
ns = {}; exec(compile(s, "kin2_goal", "exec"), ns)
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import StratifiedGroupKFold
meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in gt.iterrows()}; SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk])
for mode in ("keeper", "gbottom", "gx_keepy", "avg", "gcentre"):
    A, B = [], []; ns["MODE"] = mode
    for c in gt["clip"]:
        ns["TARGET"] = targets(c); ns["NAMES"].clear(); A.append(ns["build"](BS[c], KP[c], tracks[c], meta[c], False)); ns["NAMES"].clear(); B.append(ns["build"](BS[c], KP[c], tracks[c], meta[c], True))
    A, B = np.stack(A), np.stack(B); P = np.zeros((len(yk), 3))
    for sd in (0, 1, 2):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(A, yk, grp):
            m_ = HGB(random_state=sd, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5).fit(np.r_[A[tr], B[tr]], np.r_[yk[tr], ym[tr]])
            cl = list(m_.classes_); ix = [cl.index(SW[k]) for k in cl]; p = (m_.predict_proba(A[te]) + m_.predict_proba(B[te])[:, ix]) / 2; P[te] += p[:, [cl.index(k) for k in ["left", "middle", "right"]]] / 3
    pred = np.array(["left", "middle", "right"])[P.argmax(1)]; lr = yk != "middle"; print(f"goal axis = {mode:9s}: kick acc {(pred == yk).mean():.3f} | L/R-only {(pred[lr] == yk[lr]).mean():.3f}", flush=True)
