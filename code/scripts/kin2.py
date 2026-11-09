"""Stage C (v2): LABEL-FREE kinematic features. RTMW left/right labels are unreliable at this resolution (32% of clips swap ankle identities,
18% have toe/heel groups on the 'other' ankle), so feet are re-grouped physically (ankle-anchored), tracked over the last frames, and the foot that
travels farther is the kicking foot. Hip/shoulder lines are used as axes (angle mod 180 deg). Frame: stabilised coordinates, origin = stationary ball,
u = ball->keeper (goal) direction, v = kicker's left, unit = kicker height. mirror=True negates v and flips the preferred-foot sign."""
import os  # notebook: paths from the environment
import pickle, os, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment as lsa

ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]
CT = 2.0
NAMES = []
EXTRAS = True      # v3 extras: tilt of leg segments / trunk / head relative to gravity


def slope(t, y):
    m = np.isfinite(y)
    if m.sum() < 2: return np.nan
    tt, yy = t[m] - t[m].mean(), y[m]; d = (tt ** 2).sum()
    return (tt * yy).sum() / d if d > 0 else np.nan


def match2(cand_a, cand_b, ref):
    """assign two candidates (2 points each possibly NaN) to the two reference points one-to-one; returns for ref[0], ref[1] the matched candidate index"""
    cost = np.full((2, 2), 1e6)
    for i in range(2):
        for j in range(2):
            if np.isfinite(ref[i]).all() and np.isfinite(cand_a[j]).all(): cost[i, j] = np.hypot(*(ref[i] - cand_a[j]))
    r, c = lsa(cost); out = [None, None]
    for i, j in zip(r, c):
        if cost[i, j] < 1e5: out[i] = j
    return out


def legs_at(P, S):
    """two feet (ordered by ankle label, arbitrary): dict of joints per foot; toes/heels/knees/hips assigned one-to-one by proximity to the ankle."""
    Pm = np.where((S >= CT)[:, None], P, np.nan)
    ank = Pm[[15, 16]]
    feet = [dict(), dict()]
    for i in range(2): feet[i]["ank"] = ank[i]
    for nm, ids in (("toe", [17, 20]), ("sml", [18, 21]), ("heel", [19, 22]), ("knee", [13, 14]), ("hip", [11, 12])):
        cand = Pm[ids]; m = match2(cand, None, ank) if False else None
        cost = np.full((2, 2), 1e6)
        ref = ank.copy()
        # fall back to the foot centroid when an ankle is missing
        for i in range(2):
            for j in range(2):
                if np.isfinite(ref[i]).all() and np.isfinite(cand[j]).all(): cost[i, j] = np.hypot(*(ref[i] - cand[j]))
        r, c = lsa(cost)
        for i in range(2): feet[i][nm] = np.array([np.nan, np.nan])
        for i, j in zip(r, c):
            if cost[i, j] < 1e5: feet[i][nm] = cand[j]
    return feet


def build(b, k, tr, meta, mirror):
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(vals).astype(np.float32); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; kp, sc = k["kp"], k["sc"]
    M = b["M"]; P = np.einsum("fij,fkj->fki", M, np.concatenate([kp, np.ones(kp.shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]
    hL = max(float(k["box"][-1][3] - k["box"][-1][1]), 20.0); ball = b["ball_last"].copy()
    if not np.isfinite(ball[0]):
        a = P[-1][[15, 16]]; ball = np.nanmean(a, axis=0) if np.isfinite(a).any() else np.array([np.nan, np.nan])
    kb = tr["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]
    if len(vv) == 0 or not np.isfinite(ball[0]): return None
    kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]); fv = kf - ball; fn = np.linalg.norm(fv); f = fv / fn if fn > 1e-6 else np.array([1.0, 0.0])
    Lv = np.array([f[1], -f[0]]) * (-1 if mirror else 1)
    uv = lambda X: np.stack([(X - ball) @ f / hL, (X - ball) @ Lv / hL], -1)
    n6 = min(6, nF); n4 = min(4, nF); idx6 = list(range(nF - n6, nF)); t6 = ts[idx6]
    # ---- per-frame feet, identity tracking over the last frames, kicking foot = larger travel
    L = [legs_at(P[j], sc[j]) for j in idx6]
    order = [[0, 1]]                                    # order[t][identity] -> index in L[t]
    for t in range(1, len(idx6)):
        ref = [np.nanmean(np.array([L[t - 1][order[-1][i]][q] for q in ("ank", "toe", "heel")]), axis=0) if np.isfinite(L[t - 1][order[-1][i]]["ank"]).all() else np.array([np.nan, np.nan]) for i in range(2)]
        cen = [np.nanmean(np.array([L[t][j][q] for q in ("ank", "toe", "heel")]), axis=0) if np.isfinite(L[t][j]["ank"]).all() else np.array([np.nan, np.nan]) for j in range(2)]
        m = match2(np.array(cen), None, np.array(ref))
        o = [m[0] if m[0] is not None else 0, m[1] if m[1] is not None else 1]
        if o[0] == o[1]: o = [0, 1]
        order.append(o)
    ident = [[L[t][order[t][i]] for t in range(len(idx6))] for i in range(2)]   # ident[i][t] = foot dict
    def travel(i):
        pts = np.array([ident[i][t]["ank"] for t in range(len(idx6))]); d = np.hypot(*np.diff(pts, axis=0).T); return np.nansum(d) / hL if np.isfinite(d).any() else np.nan
    tv = [travel(0), travel(1)]
    if not np.isfinite(tv).any(): kick_i = 0
    else: kick_i = int(np.nanargmax(tv))
    plant_i = 1 - kick_i; ratio = (np.nanmax(tv) / max(np.nanmin(tv), 1e-3)) if np.isfinite(tv).all() else np.nan
    legs = {"kk": ident[kick_i], "pl": ident[plant_i]}
    get = lambda who, nm, ts_: np.array([legs[who][t][nm] for t in ts_])           # frames in idx6 coordinates
    last4 = list(range(len(idx6) - n4, len(idx6)))
    for who in ("pl", "kk"):
        for nm in ("ank", "toe", "heel"): add(f"{who}_{nm}_uv", uv(get(who, nm, last4)).ravel())
        d = uv(get(who, "toe", last4)) - uv(get(who, "heel", last4)); ang = np.arctan2(d[:, 1], d[:, 0])
        add(f"{who}_footang", np.r_[np.cos(ang), np.sin(ang)]); m3 = np.nanmean(np.cos(ang[-3:])), np.nanmean(np.sin(ang[-3:])); add(f"{who}_footang_mean3", [m3[0], m3[1], np.arctan2(m3[1], m3[0])])
        add(f"{who}_footlen", np.hypot(d[-1, 0], d[-1, 1]))
        cen = np.nanmean(uv(np.stack([get(who, nm, last4[-3:]) for nm in ("ank", "toe", "heel")])), axis=(0, 1)); add(f"{who}_foot_ball_uv", cen); add(f"{who}_foot_ball_dist", np.hypot(*cen))
        for seg, (a_, b_) in (("shank", ("knee", "ank")), ("thigh", ("hip", "knee"))):
            vec = uv(get(who, b_, range(len(idx6)))) - uv(get(who, a_, range(len(idx6)))); ang6 = np.arctan2(vec[:, 1], vec[:, 0])
            full = np.full(len(ang6), np.nan); fin = np.isfinite(ang6)
            if fin.sum() > 1: full[fin] = np.unwrap(ang6[fin])
            add(f"{who}_{seg}_ang_last", [np.cos(ang6[-1]), np.sin(ang6[-1])]); add(f"{who}_{seg}_angvel", slope(t6, full))
    for nm in ("toe", "ank"):
        y = uv(get("kk", nm, range(len(idx6)))); vu, vv_ = slope(t6, y[:, 0]), slope(t6, y[:, 1])
        add(f"kk_{nm}_vel", [vu, vv_]); add(f"kk_{nm}_speed_ang", [np.hypot(vu, vv_), np.arctan2(vv_, vu) if np.isfinite(vu) else np.nan])
    d_ = uv(get("kk", "ank", last4)) - uv(get("pl", "ank", last4)); add("kk_minus_pl_ank_uv", d_[-1]); add("kk_minus_pl_ank_rate", [slope(ts[idx6][-n4:], d_[:, 0]), slope(ts[idx6][-n4:], d_[:, 1])])
    add("foot_travel", [tv[kick_i], tv[plant_i], ratio])
    # ---- hip / shoulder axes (angle modulo 180 deg => immune to left/right label swaps)
    U = uv(P)
    for nm, (a_, b_) in (("hipax", (11, 12)), ("shax", (5, 6)), ("earax", (3, 4))):
        vec = U[idx6, b_] - U[idx6, a_]; vec = np.where((sc[idx6][:, [a_, b_]] >= CT).all(1)[:, None], vec, np.nan)
        th = np.arctan2(vec[:, 1], vec[:, 0]); n = np.hypot(vec[:, 0], vec[:, 1]); a2 = 2 * th; full = np.full(len(a2), np.nan); fin = np.isfinite(a2)
        if fin.sum() > 1: full[fin] = np.unwrap(a2[fin])
        add(f"{nm}_2ang_last", [np.cos(a2[-1]), np.sin(a2[-1]), n[-1]]); add(f"{nm}_2ang_mean3", [np.nanmean(np.cos(a2[-3:])), np.nanmean(np.sin(a2[-3:]))]); add(f"{nm}_2angvel", slope(t6, full))
    hm = np.nanmean(U[:, [11, 12]], axis=1); sm = np.nanmean(U[:, [5, 6]], axis=1); trunk = sm - hm; head = U[:, 0] - sm
    add("trunk_uv_last", trunk[-1]); add("trunk_uv_rate", [slope(t6, trunk[idx6, 0]), slope(t6, trunk[idx6, 1])])
    add("head_uv_last", head[-1]); add("head_uv_rate", [slope(t6, head[idx6, 0]), slope(t6, head[idx6, 1])])
    for o in (0.0, 0.15, 0.25, 0.45, 0.8):
        j = int(np.abs(ts + o).argmin()); add(f"hip_uv_at{o}", hm[j])
    m4 = np.where(ts >= -0.3)[0]; add("hip_vel_0.3", [slope(ts[m4], hm[m4, 0]), slope(ts[m4], hm[m4, 1])])
    m8 = np.where(ts >= -0.8)[0]; add("hip_vel_0.8", [slope(ts[m8], hm[m8, 0]), slope(ts[m8], hm[m8, 1])])
    # ---- (v3 extras) tilt of leg segments / trunk relative to gravity (image vertical), expressed in the goal frame
    if EXTRAS:
        add("x_goalframe", [f[0], f[1], Lv[0], Lv[1]])
        pts = {"plsh": [("pl", "knee"), ("pl", "ank")], "plth": [("pl", "hip"), ("pl", "knee")], "kksh": [("kk", "knee"), ("kk", "ank")], "kkth": [("kk", "hip"), ("kk", "knee")]}
        segs = {}
        for nm, ((w1, j1), (w2, j2)) in pts.items():
            segs[nm] = (get(w2, j2, range(len(idx6))) - get(w1, j1, range(len(idx6)))) / hL         # image-frame vectors (stabilised), [6,2]
        Pm_ = np.where((sc >= CT)[..., None], P, np.nan)
        hipm = np.nanmean(Pm_[:, [11, 12]], axis=1)[idx6]; shm = np.nanmean(Pm_[:, [5, 6]], axis=1)[idx6]
        segs["trunk"] = (shm - hipm) / hL; segs["head"] = (Pm_[idx6, 0] - shm) / hL
        for nm, s in segs.items():
            sx, sy = s[:, 0], s[:, 1]; tau = np.arctan2(sx, -sy)             # lean from vertical in the image plane (positive = to image right)
            tauf = np.full(len(tau), np.nan); fin = np.isfinite(tau)
            if fin.sum() > 1: tauf[fin] = np.unwrap(tau[fin])
            lu, lv = sx * f[0], sx * Lv[0]                                   # image-lateral lean projected on goal axes
            add(f"x_{nm}_tilt_last", [sx[-1], sy[-1], np.sin(tau[-1]), lu[-1], lv[-1]])
            add(f"x_{nm}_tilt_mean3", [np.nanmean(sx[-3:]), np.nanmean(lu[-3:]), np.nanmean(lv[-3:])])
            add(f"x_{nm}_tilt_rate", [slope(t6, tauf), slope(t6, lu), slope(t6, lv)])
            add(f"x_{nm}_lv_t", lv)                                          # lean_v at each of the last 6 frames
    add("foot_sign", (1.0 if meta["foot"] == "right" else -1.0) * (-1 if mirror else 1))
    add("meta", [1.0 if meta["gender"] == "M" else 0.0, meta["fps"], float(ts[0]), float(b["support"]), float(b["conf"])])
    add("kpq", [float(np.nanmean(sc[-3:][:, [15, 16, 17, 19, 20, 22]])), float(np.isfinite(U[-1, [15, 16, 17, 19, 20, 22], 0]).sum())])
    return np.array(f_, np.float32)


if __name__ == "__main__":
    tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
    gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
    meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}
    rows = {}; skipped = 0; names = None
    for c in list(gt["clip"]) + list(pm["clip"]):
        bp, kp_ = f"{C}/ballspot/{c}.pkl", f"{C}/kpts/{c}.pkl"
        if not (os.path.exists(bp) and os.path.exists(kp_)): continue
        b = pickle.load(open(bp, "rb")); k = pickle.load(open(kp_, "rb"))
        if b is None or k is None or tracks[c] is None or tracks[c]["keep"] is None: skipped += 1; continue
        NAMES.clear(); a = build(b, k, tracks[c], meta[c], False); names = list(NAMES); NAMES.clear(); m = build(b, k, tracks[c], meta[c], True)
        if a is None or m is None: skipped += 1; continue
        rows[c] = (a, m)
    pickle.dump(dict(rows=rows, names=names), open(f"{C}/kin.pkl", "wb")); print("built", len(rows), "skipped", skipped, "dim", len(names))
