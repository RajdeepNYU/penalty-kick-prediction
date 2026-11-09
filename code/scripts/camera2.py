"""Stage G3: per-clip orthographic ground-plane camera fitted to (ball -> goal-line centre = 11 m) and (post separation = 7.32 m to the kicker's left).
Model (zero roll, elevation theta, azimuth phi, scale s px/m):  a_u = s (cos phi, -sin th sin phi),  a_v = s (-sin phi, -sin th cos phi),  a_z = (0, -s cos th).
Least squares over (s, th, phi) with a_u weighted 4x more than a_v; clips without posts use the public-median theta.
Ground coords of an image point X at height z (m):  g = A^-1 (X - ball_ground - a_z z),  ball_ground = ball centre moved down 0.11 m.
usage: python camera2.py [ballsrc=ball2|old]   -> cache/camera2[_old].pkl"""
import sys, pickle, numpy as np, pandas as pd
from scipy.optimize import least_squares
from common import *

SRC = sys.argv[1] if len(sys.argv) > 1 else "ball2"
GP = pickle.load(open(f"{PC}/goalposts.pkl", "rb")); B2 = pickle.load(open(f"{PC}/{SRC}.pkl", "rb")) if SRC.startswith("ball") else None      # ball2 | ball3 (mkball3.py) | old


def model(p):
    s, th, ph = p; st, ct = np.sin(th), np.cos(th)
    return np.array([s * np.cos(ph), -s * st * np.sin(ph)]), np.array([-s * np.sin(ph), -s * st * np.cos(ph)]), np.array([0.0, -s * ct])


def fit(au, av, th0):
    ph0 = np.arctan2(-au[1] / np.sin(th0), au[0]); s0 = np.hypot(au[0], au[1] / np.sin(th0))
    if av is None:
        return np.array([s0, th0, ph0]), np.nan
    def res(p):
        mu, mv, _ = model(p); return np.r_[(mu - au) / 0.5, (mv - av) / 1.5]
    best = None
    for t in (0.08, 0.16, 0.25):
        r = least_squares(res, [np.hypot(au[0], au[1] / np.sin(t)), t, np.arctan2(-au[1] / np.sin(t), au[0])], bounds=([1, np.radians(2), -2 * np.pi], [1e4, np.radians(30), 2 * np.pi]))
        if best is None or r.cost < best.cost: best = r
    mu, mv, _ = model(best.x); return best.x, float(np.linalg.norm(mv - av) / max(np.linalg.norm(av), 1e-6))


rows, raw = [], {}
for c in CLIPS:
    b = load("ballspot", c); t = tracks()[c]; k = load("kpts", c); r = GP.get(c)
    if b is None or t is None or t["keep"] is None or k is None: raw[c] = None; continue
    ball = np.asarray(B2[c]["ball"], float) if (B2 is not None and B2.get(c)) else b["ball_last"].copy()
    if not np.isfinite(ball[0]): raw[c] = None; continue
    kb = t["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]; kq = kb[vv[-1]]; kfeet = np.array([(kq[0] + kq[2]) / 2, kq[3]])
    hL = max(float(k["box"][-1][3] - k["box"][-1][1]), 20.0)
    d = dict(ball=ball, kfeet=kfeet, hL=hL, posts=0)
    if r is not None and r.get("ok"):
        Gc = r["base"].mean(0); vvec = 0.5 * ((r["top"][1] - r["top"][0]) + (r["base"][1] - r["base"][0])) if r["bar"] > 0.5 else r["base"][1] - r["base"][0]
        # goal centre sanity: keeper feet should be near the post-base line; if the base midpoint is far from the keeper, posts are suspect
        d.update(posts=1, Gc=Gc, av_meas=vvec / 7.32, bar=r["bar"], az_goal=((r["top"] - r["base"]) / 2.44).mean(0))
    else: d.update(Gc=kfeet)
    au = (d["Gc"] - ball) / 11.0
    if d["posts"] and au[0] * d["av_meas"][1] - au[1] * d["av_meas"][0] > 0: d["av_meas"] = -d["av_meas"]
    d["au_meas"] = au; raw[c] = d
TH0 = None
for pass_ in (0, 1):
    for c, d in raw.items():
        if d is None: continue
        if pass_ == 0 and not d["posts"]: continue
        th0 = np.radians(9.2) if TH0 is None else TH0
        p, err = fit(d["au_meas"], d["av_meas"] if d["posts"] else None, th0)
        if d["posts"] and err > 0.6: p, err = fit(d["au_meas"], None, th0); d["posts_rejected"] = 1
        mu, mv, mz = model(p); d.update(s=p[0], theta=np.degrees(p[1]), phi=np.degrees(p[2]), A=np.c_[mu, mv], az=mz, av_err=err)
    if pass_ == 0:
        TH0 = np.radians(np.median([raw[q]["theta"] for q in gt["clip"] if raw.get(q) and raw[q]["posts"] and not raw[q].get("posts_rejected") and "theta" in raw[q]] or [9.2]))
        print(f"public-median theta from posts: {np.degrees(TH0):.2f} deg")
for c, d in raw.items():
    if d is None: continue
    d["ball_ground"] = d["ball"] - d["az"] * 0.11; d["Ai"] = np.linalg.inv(d["A"])
    d["keeper_g"] = d["Ai"] @ (d["kfeet"] - d["ball_ground"])
    rows.append(dict(clip=c, posts=d["posts"], rej=d.get("posts_rejected", 0), theta=d["theta"], phi=d["phi"], av_err=d["av_err"], ku=d["keeper_g"][0], kv=d["keeper_g"][1]))
pickle.dump(raw, open(f"{PC}/camera2{'' if SRC == 'ball2' else '_old' if SRC == 'old' else '_' + SRC}.pkl", "wb"))
R = pd.DataFrame(rows); print(R.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).T)
print("posts used:", int(((R.posts == 1) & (R.rej == 0)).sum()), "| rejected:", int(R.rej.sum()), "| no posts:", int((R.posts == 0).sum()))
