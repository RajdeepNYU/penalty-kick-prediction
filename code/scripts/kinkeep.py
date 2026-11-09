"""Stage C4: KEEPER features from camera-stabilised RTMW keypoints. Same ball-anchored goal frame as kin2 (u = ball->keeper direction, v = kicker's left, unit = keeper height).
Label-free (left/right keypoint identities never used) and mirror-exact. Writes cache/keepfeat.pkl."""
import os  # notebook: paths from the environment
import pickle, os, numpy as np, pandas as pd
from kin2 import slope

ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]
CT = 2.0
NAMES = []


def build(b, kk, tr, meta, mirror):
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(vals).astype(np.float32); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; sc = kk["sc"]; M = b["M"]
    P = np.einsum("fij,fkj->fki", M, np.concatenate([kk["kp"], np.ones(kk["kp"].shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]
    P = np.where((sc >= CT)[..., None], P, np.nan)
    ball = b["ball_last"]; kb = tr["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]
    if len(vv) == 0 or not np.isfinite(ball[0]): return None
    kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]); fv = kf - ball; fn = np.linalg.norm(fv); f = fv / fn if fn > 1e-6 else np.array([1.0, 0.0])
    Lv = np.array([f[1], -f[0]]) * (-1 if mirror else 1)
    hk = max(float(np.nanmedian(kk["box"][:, 3] - kk["box"][:, 1])), 20.0)
    uv = lambda X: np.stack([(X - ball) @ f / hk, (X - ball) @ Lv / hk], -1)
    U = uv(P); hm = np.nanmean(U[:, [11, 12]], axis=1); sm = np.nanmean(U[:, [5, 6]], axis=1); am = np.nanmean(U[:, [15, 16]], axis=1); nose = U[:, 0]
    km = np.nanmean(U[:, [13, 14]], axis=1); wm = np.nanmean(U[:, [9, 10]], axis=1)
    near = lambda o: int(np.abs(ts + o).argmin())
    ref = near(0.45)
    for nm, S in (("hip", hm), ("feet", am), ("sho", sm), ("head", nose)):
        add(f"k_{nm}_disp", np.concatenate([S[near(o)] - S[ref] for o in (0.0, 0.03, 0.06, 0.1, 0.15, 0.25)]))
        for w in (0.1, 0.2, 0.3, 0.5):
            m = np.where(ts >= -w)[0]; add(f"k_{nm}_vel{w}", [slope(ts[m], S[m, 0]), slope(ts[m], S[m, 1])])
        m1, m2 = np.where((ts >= -0.3) & (ts < -0.12))[0], np.where(ts >= -0.12)[0]
        add(f"k_{nm}_acc", [slope(ts[m2], S[m2, 1]) - slope(ts[m1], S[m1, 1]) if len(m1) > 1 and len(m2) > 1 else np.nan, slope(ts[m2], S[m2, 0]) - slope(ts[m1], S[m1, 0]) if len(m1) > 1 and len(m2) > 1 else np.nan])
    add("k_pos_ball", [hm[-1][0], hm[-1][1], hm[ref][0], hm[ref][1]])
    # stance / posture (label-free)
    ank = U[:, [15, 16]]; sep = np.hypot(*(ank[:, 0] - ank[:, 1]).T); add("k_ank_sep", [sep[-1], sep[near(0.1)], sep[near(0.25)], sep[-1] - sep[near(0.25)]])
    kn = U[:, [13, 14]]; ksep = np.hypot(*(kn[:, 0] - kn[:, 1]).T); add("k_knee_sep", [ksep[-1], ksep[near(0.25)]])
    for nm, (a_, b_) in (("hipax", (11, 12)), ("shax", (5, 6)), ("ankax", (15, 16)), ("wriax", (9, 10))):
        vec = U[:, b_] - U[:, a_]; th = np.arctan2(vec[:, 1], vec[:, 0]); a2 = 2 * th; n = np.hypot(vec[:, 0], vec[:, 1])
        full = np.full(len(a2), np.nan); fin = np.isfinite(a2)
        if fin.sum() > 1: full[fin] = np.unwrap(a2[fin])
        m = np.where(ts >= -0.3)[0]; add(f"k_{nm}_2ang", [np.cos(a2[-1]), np.sin(a2[-1]), n[-1], slope(ts[m], full[m])])
    add("k_arms", [(wm - hm)[-1][0], (wm - hm)[-1][1], (wm - sm)[-1][0], (wm - sm)[-1][1], (wm - hm)[near(0.25)][1]])
    add("k_com_vs_feet", [(hm - am)[-1][0], (hm - am)[-1][1], (hm - am)[near(0.1)][1], (hm - am)[near(0.25)][1]])
    add("k_trunk", (sm - hm)[-1]); add("k_trunk_rate", [slope(ts[ts >= -0.3], (sm - hm)[ts >= -0.3, 0]), slope(ts[ts >= -0.3], (sm - hm)[ts >= -0.3, 1])])
    # tilt vs gravity of trunk / legs in image (stabilised) expressed on goal axes, plus keeper box shape
    Pm = P; hmI = np.nanmean(Pm[:, [11, 12]], axis=1); smI = np.nanmean(Pm[:, [5, 6]], axis=1); amI = np.nanmean(Pm[:, [15, 16]], axis=1); kmI = np.nanmean(Pm[:, [13, 14]], axis=1)
    for nm, s in (("trunk", (smI - hmI) / hk), ("thigh", (kmI - hmI) / hk), ("shank", (amI - kmI) / hk)):
        sx, sy = s[:, 0], s[:, 1]; tau = np.arctan2(sx, -sy); lu, lv = sx * f[0], sx * Lv[0]; m = np.where(ts >= -0.3)[0]
        add(f"k_{nm}_tilt", [sx[-1], sy[-1], lu[-1], lv[-1], np.nanmean(lv[-3:]), slope(ts[m], lv[m]), slope(ts[m], tau[m])])
    bx = kk["box"]; wh = (bx[:, 2] - bx[:, 0]) / np.maximum(bx[:, 3] - bx[:, 1], 1); hh = (bx[:, 3] - bx[:, 1]) / hk
    add("k_box", [wh[-1], wh[near(0.25)], wh[-1] - wh[near(0.25)], hh[-1], hh[near(0.25)], hh[-1] - hh[near(0.25)]])
    add("k_goalframe", [f[0], f[1], Lv[0], Lv[1]])
    add("meta", [1.0 if meta["gender"] == "M" else 0.0, meta["fps"], float(ts[0]), float(b["support"])])
    return np.array(f_, np.float32)


if __name__ == "__main__":
    tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
    meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}
    rows = {}; names = None; skipped = 0
    for c in list(gt["clip"]) + list(pm["clip"]):
        kp_ = f"{C}/kkeep/{c}.pkl"
        if not os.path.exists(kp_): continue
        b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); kk = pickle.load(open(kp_, "rb"))
        if b is None or kk is None or tracks[c] is None or tracks[c]["keep"] is None: skipped += 1; continue
        NAMES.clear(); a = build(b, kk, tracks[c], meta[c], False); names = list(NAMES); NAMES.clear(); m = build(b, kk, tracks[c], meta[c], True)
        if a is None or m is None: skipped += 1; continue
        rows[c] = (a, m)
    pickle.dump(dict(rows=rows, names=names), open(f"{C}/keepfeat.pkl", "wb")); print("built", len(rows), "skipped", skipped, "dim", len(names))
