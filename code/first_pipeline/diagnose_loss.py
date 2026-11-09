import numpy as np, pandas as pd, pickle, sys
sys.argv = ["x"]
from cv import *
KP = list(np.r_[0:90]) + [261]
ak, oofk, ck_ = cv(X[:, KP], Xm[:, KP], "kick", seeds=(0, 1, 2))
ag, oofg, cg = cv(X, Xm, "gk", seeds=(0, 1, 2))
cnn = np.load(f"{C}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]
pg = 0.5 * oofg + 0.5 * cnn
yk = gt["kick_direction"].values; yg = gt["goalkeeper_direction"].values; yo = gt["goal_outcome"].values
pk_ = np.array(ck_)[oofk.argmax(1)]; pgk = np.array(cg)[pg.argmax(1)]
ek = pk_ != yk; eg = pgk != yg; eo = yo != "goal"
print("=== score loss budget (CV) ===")
acc = dict(kick=1 - ek.mean(), gk=1 - eg.mean(), goal=1 - eo.mean()); W = dict(kick=.6, gk=.25, goal=.15)
for k in acc: print(f"{k:5s} acc {acc[k]:.3f}  points lost {W[k] * (1 - acc[k]):.3f}  (+5pt acc = +{W[k] * .05:.4f} score)")
print("score", round(sum(W[k] * acc[k] for k in acc), 4), "| max possible 1.0")
print("\n=== kick confusion (rows true, cols predicted) ===")
print(pd.crosstab(yk, pk_))
print("recall by true class:", {c: round(float((pk_[yk == c] == c).mean()), 3) for c in ("left", "middle", "right")})
print("\n=== keeper confusion ===")
print(pd.crosstab(yg, pgk)); print("recall:", {c: round(float((pgk[yg == c] == c).mean()), 3) for c in sorted(set(yg))})

d = gt[["clip", "source_id", "preferred_foot", "gender", "resolution", "fps"]].copy()
d["ek"], d["eg"] = ek, eg
d["sideon"] = X[:, 259]; d["fy"] = X[:, 260]; d["kh"] = X[:, 257] * 720; d["gh"] = X[:, 258] * 720
d["fpsb"] = pd.cut(X[:, 263], [0, 26, 31, 55, 61], labels=["<=25", "30", "50", "60"])
d["dur"] = X[:, 264]; d["nan_kp"] = np.isnan(X[:, :90]).mean(1); d["conf"] = oofk.max(1)
d["foot"] = d["preferred_foot"]
tr = pickle.load(open(f"{C}/tracks.pkl", "rb")); ckk = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip")
import cv2
hit_k, hit_g = [], []
for c in d["clip"]:
    if c in ckk.index:
        t = tr[c]; r = ckk.loc[c]; w, h = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))["wh"]
        def hit(b, who): p = (r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h); return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
        hit_k.append(hit(t["kick_a"], "kicker")); hit_g.append(hit(t["keep_a"], "keeper"))
    else: hit_k.append(np.nan); hit_g.append(np.nan)
d["kick_loc_ok"] = hit_k; d["keep_loc_ok"] = hit_g


def seg(col, bins=None, target="ek"):
    g = pd.qcut(d[col], bins, duplicates="drop") if bins else d[col]
    t = d.groupby(g, observed=True).agg(n=(target, "size"), acc=(target, lambda s: 1 - s.mean()))
    t["acc"] = t["acc"].round(3); return t


print("\n=== KICK accuracy by segment ===")
for col, b in (("foot", None), ("gender", None), ("resolution", None), ("fpsb", None), ("sideon", 3), ("fy", 3), ("kh", 4), ("dur", 4), ("nan_kp", 4), ("conf", 4)):
    print(f"-- {col}"); print(seg(col, b).to_string())
print("-- kicker picked correctly (clicked clips only)"); print(d[d.kick_loc_ok.notna()].groupby("kick_loc_ok").agg(n=("ek", "size"), acc=("ek", lambda s: round(1 - s.mean(), 3))).to_string())
print("\n=== KEEPER accuracy by segment ===")
for col, b in (("fpsb", None), ("gh", 4), ("dur", 4)):
    print(f"-- {col}"); print(seg(col, b, "eg").to_string())
print("-- keeper picked correctly (clicked clips only)"); print(d[d.keep_loc_ok.notna()].groupby("keep_loc_ok").agg(n=("eg", "size"), acc=("eg", lambda s: round(1 - s.mean(), 3))).to_string())

print("\n=== by source: kick errors concentrated? ===")
s = d.groupby("source_id").agg(n=("ek", "size"), acc=("ek", lambda x: 1 - x.mean()), errs=("ek", "sum")).sort_values("errs", ascending=False)
s["share_of_errors"] = (s.errs / s.errs.sum()).round(3); s["acc"] = s["acc"].round(2)
print(s.head(10).to_string()); print("top-10 sources hold", round(s.head(10).share_of_errors.sum(), 3), "of kick errors;", "sources with acc<0.4 (n>=6):", int(((s.acc < .4) & (s.n >= 6)).sum()), "of", int((s.n >= 6).sum()))
print("\n=== outcome: does anything predict no_goal? (CV AUC-ish via save/out/post by predicted-keeper-correct) ===")
d["gk_right"] = (pgk == yk)  # keeper side == kick side (oracle-ish diagnostic)
print(pd.crosstab(yg == yk, gt["outcome_detail"], normalize="index").round(3))
