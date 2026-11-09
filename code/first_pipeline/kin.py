"""Stage C: kinematic features in a stabilised, ball-anchored goal frame (u = ball->goal/keeper direction, v = kicker's left), original + mirrored.
Plant foot = foot opposite to preferred_foot (PDF: preferred foot is the kicking foot)."""
import pickle, os, numpy as np, pandas as pd

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22)]
PERM = np.arange(133)
for a, b in PAIRS: PERM[a], PERM[b] = b, a
CT = 0.3
NAMES = []


def slope(t, y):
    m = np.isfinite(y)
    if m.sum() < 2: return np.nan
    tt, yy = t[m] - t[m].mean(), y[m]; d = (tt ** 2).sum()
    return (tt * yy).sum() / d if d > 0 else np.nan


def wrap(a):  # angle difference wrap to (-pi, pi]
    return (a + np.pi) % (2 * np.pi) - np.pi


def build(c, b, k, tr, meta, mirror):
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(vals).astype(np.float32); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]
    kp = k["kp"][:, PERM] if mirror else k["kp"]; sc = k["sc"][:, PERM] if mirror else k["sc"]
    kp = kp.copy(); kp[sc < CT] = np.nan
    # stabilise into last-frame coordinates
    M = b["M"]; ones = np.ones(kp.shape[:2] + (1,), np.float32)
    P = np.einsum("fij,fkj->fki", M, np.concatenate([kp, ones], -1)); P = P[..., :2] / P[..., 2:3]
    hL = float(k["box"][-1][3] - k["box"][-1][1]); hL = max(hL, 20.0)
    ball = b["ball_last"].copy()
    if not np.isfinite(ball[0]):
        a = P[-1][[15, 16]]; ball = np.nanmean(a, axis=0) if np.isfinite(a).any() else np.array([np.nan, np.nan])
    kb = tr["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]
    kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]) if len(vv) else None
    if kf is None or not np.isfinite(ball[0]): return None
    fv = kf - ball; fn = np.linalg.norm(fv); f = fv / fn if fn > 1e-6 else np.array([1.0, 0.0])
    Lv = np.array([f[1], -f[0]]) * (-1 if mirror else 1)
    uv = lambda X: np.stack([(X - ball) @ f / hL, (X - ball) @ Lv / hL], -1)     # [...,2]
    U = uv(P)                                                                    # [nF,133,2]
    kick_right = (meta["foot"] == "right") != mirror                              # True -> kicking foot is anatomical right (indices after perm)
    K = dict(ank=16, toe=20, sml=21, heel=22, knee=14, hip=12, sho=6) if kick_right else dict(ank=15, toe=17, sml=18, heel=19, knee=13, hip=11, sho=5)
    Pl = dict(ank=15, toe=17, sml=18, heel=19, knee=13, hip=11, sho=5) if kick_right else dict(ank=16, toe=20, sml=21, heel=22, knee=14, hip=12, sho=6)
    last4 = list(range(nF - 4, nF)); last6 = list(range(nF - 6, nF)); last3 = list(range(nF - 3, nF))
    t6 = ts[last6]
    # ---- plant foot placement + orientation
    for nm, J in (("pl", Pl), ("kk", K)):
        for jn in ("ank", "toe", "heel"):
            add(f"{nm}_{jn}_uv", U[last4, J[jn]].ravel())
        ang = []
        for j in last4:
            d = U[j, J["toe"]] - U[j, J["heel"]]; ang.append(np.arctan2(d[1], d[0]) if np.isfinite(d).all() else np.nan)
        ang = np.array(ang); add(f"{nm}_footang", np.r_[np.cos(ang), np.sin(ang)])
        m3 = np.nanmean(ang[-3:]) if np.isfinite(ang[-3:]).any() else np.nan
        add(f"{nm}_footang_mean3", [np.cos(m3), np.sin(m3), m3])
        cen = np.nanmean(U[last3][:, [J["ank"], J["toe"], J["heel"]]], axis=(0, 1)); add(f"{nm}_foot_ball_uv", cen); add(f"{nm}_foot_ball_dist", np.hypot(*cen))
    # ---- kicking leg kinematics (velocity of toe/ankle, shank + thigh angular velocity)
    for jn in ("toe", "ank"):
        y = U[last6, K[jn]]
        add(f"kk_{jn}_vel", [slope(t6, y[:, 0]), slope(t6, y[:, 1])])
        vu, vv_ = slope(t6, y[:, 0]), slope(t6, y[:, 1]); add(f"kk_{jn}_speed_ang", [np.hypot(vu, vv_), np.arctan2(vv_, vu) if np.isfinite(vu) else np.nan])
    for nm, J in (("kk", K), ("pl", Pl)):
        for seg, (a_, b_) in (("shank", (J["knee"], J["ank"])), ("thigh", (J["hip"], J["knee"]))):
            vec = U[last6, b_] - U[last6, a_]; ang = np.arctan2(vec[:, 1], vec[:, 0])
            ang_u = np.unwrap(ang[np.isfinite(ang)]) if np.isfinite(ang).sum() > 1 else ang
            full = np.full(len(ang), np.nan); full[np.isfinite(ang)] = ang_u
            add(f"{nm}_{seg}_ang_last", [np.cos(ang[-1]), np.sin(ang[-1])]); add(f"{nm}_{seg}_angvel", slope(t6, full))
    # ---- hips / shoulders / trunk rotation and rates
    for nm, (r_, l_) in (("hipline", (12, 11)), ("shline", (6, 5))):
        vec = U[last6, r_] - U[last6, l_]; ang = np.arctan2(vec[:, 1], vec[:, 0]); n = np.hypot(vec[:, 0], vec[:, 1])
        full = np.full(len(ang), np.nan); fin = np.isfinite(ang)
        if fin.sum() > 1: full[fin] = np.unwrap(ang[fin])
        add(f"{nm}_ang_last", [np.cos(ang[-1]), np.sin(ang[-1]), n[-1]]); add(f"{nm}_angvel", slope(t6, full)); add(f"{nm}_ang_mean3", [np.nanmean(np.cos(ang[-3:])), np.nanmean(np.sin(ang[-3:]))])
    hm = np.nanmean(U[:, [11, 12]], axis=1); sm = np.nanmean(U[:, [5, 6]], axis=1)
    trunk = sm - hm; add("trunk_uv_last", trunk[-1]); add("trunk_uv_rate", [slope(t6, trunk[last6, 0]), slope(t6, trunk[last6, 1])])
    head = U[:, 0] - sm; add("head_uv_last", head[-1]); add("head_uv_rate", [slope(t6, head[last6, 0]), slope(t6, head[last6, 1])])
    ear = U[:, 3] - U[:, 4]; add("ear_uv_last", ear[-1])
    # ---- approach: hip-mid trajectory relative to ball at several look-backs
    for o in (0.0, 0.15, 0.25, 0.45, 0.8):
        j = int(np.abs(ts + o).argmin()); add(f"hip_uv_at{o}", hm[j])
    m4 = np.where(ts >= -0.3)[0]; add("hip_vel_0.3", [slope(ts[m4], hm[m4, 0]), slope(ts[m4], hm[m4, 1])])
    m8 = np.where(ts >= -0.8)[0]; add("hip_vel_0.8", [slope(ts[m8], hm[m8, 0]), slope(ts[m8], hm[m8, 1])])
    # ---- meta / quality
    add("foot_sign", (1.0 if meta["foot"] == "right" else -1.0) * (-1 if mirror else 1))
    add("meta", [1.0 if meta["gender"] == "M" else 0.0, meta["fps"], float(ts[0]), float(b["support"]), float(b["conf"])])
    add("kpq", [float(np.nanmean(sc[last3][:, [15, 16, 17, 19, 20, 22]])), float(np.isfinite(U[-1, [15, 16, 17, 19, 20, 22], 0]).sum())])
    return np.array(f_, np.float32)


if __name__ == "__main__":
    tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
    gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
    meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}
    rows = {}; skipped = 0
    for c in list(gt["clip"]) + list(pm["clip"]):
        bp, kp_ = f"{C}/ballspot/{c}.pkl", f"{C}/kpts/{c}.pkl"
        if not (os.path.exists(bp) and os.path.exists(kp_)): continue
        b = pickle.load(open(bp, "rb")); k = pickle.load(open(kp_, "rb"))
        if b is None or k is None or tracks[c] is None or tracks[c]["keep"] is None: skipped += 1; continue
        NAMES.clear(); a = build(c, b, k, tracks[c], meta[c], False); NAMES_ = list(NAMES); NAMES.clear(); m = build(c, b, k, tracks[c], meta[c], True)
        if a is None or m is None: skipped += 1; continue
        rows[c] = (a, m)
    pickle.dump(dict(rows=rows, names=NAMES_), open(f"{C}/kin.pkl", "wb")); print("built", len(rows), "skipped", skipped, "dim", len(NAMES_))
