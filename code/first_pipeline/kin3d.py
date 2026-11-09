"""Stage C3: depth-aware lean features from RTMW3D. Identity of kicking/plant leg re-derived from the 3D model's own 2D output (label-free, as kin2).
3D metres: X right, Y down, Zg = sigma*z (depth away from camera). Goal direction in the ground plane G = norm(fx, -kappa*fy) from the stabilised image
direction ball->keeper; kicker's left = (-Gz, Gx). Mirror negates the lateral axis (and flips the foot sign).
usage: python kin3d.py [sigma] [kappa]  -> cache/kin3d_<sigma>_<kappa>.pkl"""
import pickle, os, sys, numpy as np, pandas as pd
import kin2
from kin2 import legs_at, match2, slope

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
CT = 0.2   # RTMW3D score scale is 0.15-0.75 (RTMW 2D is 2-5)
NAMES = []


def build(b, k, k3, tr, sigma, kappa, mirror):
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(vals).astype(np.float32); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; sc = k3["sc"]; kp2 = k3["kp2"]; M = b["M"]
    P2 = np.einsum("fij,fkj->fki", M, np.concatenate([kp2, np.ones(kp2.shape[:2] + (1,), np.float32)], -1)); P2 = P2[..., :2] / P2[..., 2:3]
    ball = b["ball_last"]; kb = tr["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]
    if len(vv) == 0 or not np.isfinite(ball[0]): return None
    kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]); fv = kf - ball; fn = np.linalg.norm(fv); f = fv / fn if fn > 1e-6 else np.array([1.0, 0.0])
    G = np.array([f[0], -kappa * f[1]]); G /= np.linalg.norm(G); Lg = np.array([-G[1], G[0]]) * (-1 if mirror else 1)
    n6 = min(6, nF); idx6 = list(range(nF - n6, nF)); t6 = ts[idx6]
    # --- scale to metres from the body extent in the model crop (per clip median), 3D coords: x,y crop px -> m
    ext = []
    for j in range(nF):
        ok = sc[j, :23] >= CT; y = k3["kp3"][j, :23, 1][ok]
        if ok.sum() > 8: ext.append(y.max() - y.min())
    s = 1.75 / max(np.median(ext), 100.0) if ext else 1.75 / 300.0
    P3 = k3["kp3"].copy(); P3[..., 0] *= s; P3[..., 1] *= s; P3[..., 2] *= sigma
    P3 = np.where((sc >= CT)[..., None], P3, np.nan)
    # --- legs identity from the 3D model's own image-plane keypoints (label-free)
    L = [legs_at(P2[j], sc[j] * 10) for j in idx6]; order = [[0, 1]]
    for t in range(1, len(idx6)):
        ref = [np.nanmean(np.array([L[t - 1][order[-1][i]][q] for q in ("ank", "toe", "heel")]), axis=0) if np.isfinite(L[t - 1][order[-1][i]]["ank"]).all() else np.array([np.nan, np.nan]) for i in range(2)]
        cen = [np.nanmean(np.array([L[t][j][q] for q in ("ank", "toe", "heel")]), axis=0) if np.isfinite(L[t][j]["ank"]).all() else np.array([np.nan, np.nan]) for j in range(2)]
        m = match2(np.array(cen), None, np.array(ref)); o = [m[0] if m[0] is not None else 0, m[1] if m[1] is not None else 1]
        order.append(o if o[0] != o[1] else [0, 1])
    # identity -> which keypoint index (ankle/knee/hip) per frame: legs_at gave points not indices, so map back by matching coordinates
    def idx_of(frame_j, pt, cand):
        if not np.isfinite(pt).all(): return None
        d = [np.hypot(*(P2[frame_j, c] - pt)) if np.isfinite(P2[frame_j, c]).all() else 1e9 for c in cand]; return cand[int(np.argmin(d))]
    ident = {0: {}, 1: {}}
    for i in range(2):
        for t, j in enumerate(idx6):
            leg = L[t][order[t][i]]
            ident[i][t] = {nm: idx_of(j, leg[nm], cand) for nm, cand in (("ank", [15, 16]), ("knee", [13, 14]), ("hip", [11, 12]), ("toe", [17, 20]), ("heel", [19, 22]))}
    def trav(i):
        pts = np.array([L[t][order[t][i]]["ank"] for t in range(len(idx6))]); d = np.hypot(*np.diff(pts, axis=0).T); return np.nansum(d) if np.isfinite(d).any() else np.nan
    tv = [trav(0), trav(1)]; kick_i = int(np.nanargmax(tv)) if np.isfinite(tv).any() else 0; plant_i = 1 - kick_i
    who = {"kk": kick_i, "pl": plant_i}
    def pt3(w, nm, t):
        ix = ident[who[w]][t][nm]; return P3[idx6[t], ix] if ix is not None else np.full(3, np.nan)
    def seg_lean(a, b_):
        """lateral lean, forward lean, vertical sag of a segment a->b at each of the last frames"""
        out = []
        for t in range(len(idx6)):
            d = b_(t) - a(t); n = np.linalg.norm(d)
            if not np.isfinite(n) or n < 1e-6: out.append([np.nan] * 3); continue
            h = np.array([d[0], d[2]]); out.append([h @ Lg / n, h @ G / n, d[1] / n])
        return np.array(out)
    segs = {"plsh": seg_lean(lambda t: pt3("pl", "knee", t), lambda t: pt3("pl", "ank", t)), "plth": seg_lean(lambda t: pt3("pl", "hip", t), lambda t: pt3("pl", "knee", t)),
            "kksh": seg_lean(lambda t: pt3("kk", "knee", t), lambda t: pt3("kk", "ank", t)), "kkth": seg_lean(lambda t: pt3("kk", "hip", t), lambda t: pt3("kk", "knee", t))}
    hipm = lambda t: np.nanmean(P3[idx6[t], [11, 12]], axis=0); shm = lambda t: np.nanmean(P3[idx6[t], [5, 6]], axis=0)
    segs["trunk"] = seg_lean(hipm, shm); segs["head"] = seg_lean(shm, lambda t: P3[idx6[t], 0])
    for nm, A in segs.items():
        add(f"d3_{nm}_last", A[-1]); add(f"d3_{nm}_mean3", np.nanmean(A[-3:], axis=0)); add(f"d3_{nm}_rate", [slope(t6, A[:, 0]), slope(t6, A[:, 1])]); add(f"d3_{nm}_lat_t", A[:, 0])
    # pelvis / shoulder yaw relative to goal axis (axis, mod 180 deg)
    for nm, (a_, b_) in (("hip", (11, 12)), ("sho", (5, 6))):
        psi = []
        for t, j in enumerate(idx6):
            v_ = P3[j, b_] - P3[j, a_]; h = np.array([v_[0], v_[2]])
            psi.append(np.arctan2(h @ Lg, h @ G) if np.isfinite(h).all() else np.nan)
        psi = np.array(psi); a2 = 2 * psi; full = np.full(len(a2), np.nan); fin = np.isfinite(a2)
        if fin.sum() > 1: full[fin] = np.unwrap(a2[fin])
        add(f"d3_{nm}yaw_last", [np.cos(a2[-1]), np.sin(a2[-1])]); add(f"d3_{nm}yaw_mean3", [np.nanmean(np.cos(a2[-3:])), np.nanmean(np.sin(a2[-3:]))]); add(f"d3_{nm}yaw_rate", slope(t6, full))
    # feet relative offsets (kicking - plant) in goal-lateral/forward metres
    dk = np.array([pt3("kk", "ank", t) - pt3("pl", "ank", t) for t in range(len(idx6))])
    add("d3_footoff_last", [dk[-1][[0, 2]] @ Lg, dk[-1][[0, 2]] @ G, dk[-1][1]]); add("d3_footoff_rate", [slope(t6, dk[:, [0, 2]] @ Lg), slope(t6, dk[:, [0, 2]] @ G)])
    return np.array(f_, np.float32)


if __name__ == "__main__":
    sigma = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; kappa = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
    rows = {}; names = None; skipped = 0
    for c in list(gt["clip"]) + list(pm["clip"]):
        p3 = f"{C}/kpts3d/{c}.pkl"
        if not os.path.exists(p3): continue
        b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); k3 = pickle.load(open(p3, "rb"))
        if b is None or k3 is None or tracks[c] is None or tracks[c]["keep"] is None: skipped += 1; continue
        NAMES.clear(); a = build(b, k, k3, tracks[c], sigma, kappa, False); names = list(NAMES); NAMES.clear(); m = build(b, k, k3, tracks[c], sigma, kappa, True)
        if a is None or m is None: skipped += 1; continue
        rows[c] = (a, m)
    pickle.dump(dict(rows=rows, names=names), open(f"{C}/kin3d_{sigma:+.0f}_{kappa}.pkl", "wb")); print("built", len(rows), "skipped", skipped, "dim", len(names))
