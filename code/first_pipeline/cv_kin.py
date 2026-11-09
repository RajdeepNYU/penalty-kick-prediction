import pickle, numpy as np, pandas as pd, sys, warnings, re
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, HistGradientBoostingRegressor as HGR
from sklearn.model_selection import StratifiedGroupKFold
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
K = pickle.load(open(f"{C}/kin.pkl", "rb")); names = np.array(K["names"]); rows = K["rows"]
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); gt = gt[gt["clip"].isin(rows)].reset_index(drop=True)
X = np.stack([rows[c][0] for c in gt["clip"]]); Xm = np.stack([rows[c][1] for c in gt["clip"]])
yk = gt["kick_direction"].values; grp = gt["source_id"].values; SW = {"left": "right", "right": "left", "middle": "middle"}
print("clips with kinematic features:", len(gt), "dim", X.shape[1], "| NaN share %.3f" % np.isnan(X).mean())
old = pickle.load(open(f"{C}/feats.pkl", "rb")); OLD = list(np.r_[0:90]) + [261]
Xo = np.stack([old[c][0] for c in gt["clip"]])[:, OLD]; Xom = np.stack([old[c][1] for c in gt["clip"]])[:, OLD]
blk = lambda *pre: np.array([i for i, n in enumerate(names) if any(n.startswith(p) for p in pre)])
BL = {"plant foot": blk("pl_"), "kicking foot": blk("kk_"), "hip/trunk/head": blk("hipax", "shax", "earax", "trunk", "head"), "approach": blk("hip_uv", "hip_vel"), "meta": blk("foot_sign", "meta", "kpq", "foot_travel")}
FOOT = blk("foot_sign")
P = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)


def cv_clf(Xa, Xb, seeds=(0, 1, 2)):
    cl = ["left", "middle", "right"]; Pr = np.zeros((len(yk), 3)); ym = np.array([SW[v] for v in yk])
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
            m = HGB(random_state=s, **P).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
            Pr[te] += (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2 / len(seeds)
    return Pr, cl


def cv_ord(Xa, Xb, seeds=(0, 1, 2)):
    t = np.where(yk == "left", 1.0, np.where(yk == "right", -1.0, 0.0)); S = np.zeros(len(yk))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
            m = HGR(random_state=s, **P).fit(np.r_[Xa[tr], Xb[tr]], np.r_[t[tr], -t[tr]])
            S[te] += (m.predict(Xa[te]) - m.predict(Xb[te])) / 2 / len(seeds)
    return S


def report(nm, Pr, cl):
    pred = np.array(cl)[Pr.argmax(1)]; lr = yk != "middle"
    print(f"{nm:34s} kick acc {(pred == yk).mean():.3f} | L/R-only acc {(pred[lr] == yk[lr]).mean():.3f} | middle recall {(pred[~lr] == 'middle').mean():.2f}", flush=True)


if __name__ == "__main__":
    print("--- 3-class GBDT, each block + foot sign ---")
    for nm, ix in BL.items():
        if nm == "meta": continue
        cols = np.r_[ix, FOOT]; report(nm, *cv_clf(X[:, cols], Xm[:, cols]))
    allc = np.r_[BL["plant foot"], BL["kicking foot"], BL["hip/trunk/head"], BL["approach"], BL["meta"]]
    report("ALL kinematic", *cv_clf(X[:, allc], Xm[:, allc]))
    report("old kpose+foot (v3 kick features)", *cv_clf(Xo, Xom))
    Pa, cl = cv_clf(np.c_[X[:, allc], Xo], np.c_[Xm[:, allc], Xom]); report("ALL kinematic + old kpose", Pa, cl)
    print("--- mirror-symmetric ordinal regression (middle = |score| < a) ---")
    for nm, (A, B) in {"ALL kinematic": (X[:, allc], Xm[:, allc]), "ALL kin + old": (np.c_[X[:, allc], Xo], np.c_[Xm[:, allc], Xom])}.items():
        S = cv_ord(A, B)
        for a in (0.0, 0.1, 0.2, 0.3, 0.4):
            pred = np.where(S > a, "left", np.where(S < -a, "right", "middle")); lr = yk != "middle"
            print(f"{nm:20s} a={a:.1f}: kick acc {(pred == yk).mean():.3f} | L/R-only {(pred[lr] == yk[lr]).mean():.3f} | middle recall {(pred[~lr] == 'middle').mean():.2f} | share predicted middle {(pred == 'middle').mean():.2f}")
