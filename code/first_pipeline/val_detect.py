import pickle, os, numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(D + r"\train_ground_truth.csv"); pm = pd.read_csv(D + r"\private_metadata.csv")
ck = pd.read_csv(D + r"\public_position_clicks.csv").set_index("clip")
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); meta = {r["clip"]: r for _, r in pd.concat([gt, pm]).iterrows()}
PAIRS = []; R = []
for part, df in (("train", gt), ("private", pm)):
    for c in df["clip"]:
        bp, kp_ = f"{C}/ballspot/{c}.pkl", f"{C}/kpts/{c}.pkl"
        if not (os.path.exists(bp) and os.path.exists(kp_)): continue
        b = pickle.load(open(bp, "rb")); k = pickle.load(open(kp_, "rb"))
        if b is None or k is None: R.append(dict(part=part, clip=c, ok=False)); continue
        nF = b["nF"]; M = b["M"]; kp = k["kp"].copy(); sc = k["sc"]; kp[sc < 0.3] = np.nan
        P = np.einsum("fij,fkj->fki", M, np.concatenate([kp, np.ones(kp.shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]
        h = k["box"][-1][3] - k["box"][-1][1]; ball = b["ball_last"]
        feet = P[-1][[15, 16, 17, 19, 20, 22]]; d = np.hypot(*(feet - ball).T) / h if np.isfinite(ball[0]) else np.array([np.nan])
        dmin = np.nanmin(d) if np.isfinite(d).any() else np.nan
        # plant vs kicking foot from motion: foot (ankle) travel over the last 4 frames
        mv = {}
        for nm, a in (("L", 15), ("R", 16)):
            tr = P[-4:, a]; mv[nm] = np.nansum(np.hypot(*np.diff(tr, axis=0).T)) / h if np.isfinite(tr).sum() >= 4 else np.nan
        moving = "R" if mv["R"] > mv["L"] else "L"; pref = "R" if meta[c]["preferred_foot"] == "right" else "L"
        plant = "L" if pref == "R" else "R"; pa = 15 if plant == "L" else 16
        pj = np.nanmedian(np.hypot(*np.diff(P[-4:, pa], axis=0).T)) / h if np.isfinite(P[-4:, pa]).all() else np.nan
        pkc = np.mean(sc[-3:][:, [15, 16, 17, 19, 20, 22]] > 0.3)
        R.append(dict(part=part, clip=c, ok=True, support=b["support"], bconf=b["conf"], dmin=dmin, agree_pref=(moving == pref) if np.isfinite(mv["L"]) and np.isfinite(mv["R"]) else np.nan,
                      plant_jit=pj, foot_kp_ok=pkc, reran=k["reran"], nF=nF, nan_ball=not np.isfinite(ball[0])))
r = pd.DataFrame(R); t = r[r.ok]
print("clips with all stages:", len(r), "(train %d, private %d)" % ((r.part == "train").sum(), (r.part == "private").sum()))
print("BALL  support>=3 frames: %.3f | >=5: %.3f | no ball at all: %.3f" % ((t.support >= 3).mean(), (t.support >= 5).mean(), t.nan_ball.mean()))
print("BALL  consistency: nearest foot keypoint to ball in last frame: median %.2f h | within 0.5h: %.3f | within 1.0h: %.3f" % (t.dmin.median(), (t.dmin < 0.5).mean(), (t.dmin < 1.0).mean()))
for lo, hi in ((0, 3), (3, 6), (6, 20)):
    s = t[(t.support >= lo) & (t.support < hi)]; print(f"       support [{lo},{hi}): n={len(s)} median dist {s.dmin.median():.2f}h within 0.5h {(s.dmin < 0.5).mean():.2f}")
print("KEYPT foot keypoints (ank/toe/heel x2) confident in last 3 frames: mean share %.3f ; clips with >=90%%: %.3f" % (t.foot_kp_ok.mean(), (t.foot_kp_ok >= 0.9).mean()))
print("KEYPT plant-foot ankle jitter over last 4 frames (median step, in body heights): %.3f ; share of clips < 0.05h: %.3f" % (t.plant_jit.median(), (t.plant_jit < 0.05).mean()))
print("FOOT  kicking foot (more ankle travel) == preferred_foot label: %.3f (n=%d)" % (t.agree_pref.dropna().mean(), t.agree_pref.notna().sum()))
print("BOX   re-run needed for box refinement: mean %.1f of %d frames" % (t.reran.mean(), t.nF.mean()))
# kicker/keeper picks vs clicks (train clips with clicks), from tracks.pkl
hk = hg = n = 0
for c in gt["clip"]:
    if c not in ck.index or tracks[c] is None: continue
    w, hh = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))["wh"]; rr = ck.loc[c]; n += 1
    for who, bx in (("kicker", tracks[c]["kick_a"]), ("keeper", tracks[c]["keep_a"])):
        p = (rr[f"{who}_x_norm"] * w, rr[f"{who}_y_norm"] * hh); ok = bx[0] <= p[0] <= bx[2] and bx[1] <= p[1] <= bx[3]
        if who == "kicker": hk += ok
        else: hg += ok
print("PICK  (clicked train clips, n=%d, OOF by source) kicker %.3f keeper %.3f" % (n, hk / n, hg / n))
r.to_pickle(f"{C}/val_detect.pkl")
