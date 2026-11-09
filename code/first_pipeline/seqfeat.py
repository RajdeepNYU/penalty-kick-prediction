"""Stage D1: per-frame label-free pose channels for a sequence network. T=12 last frames of F (front-padded), C channels each, original + mirrored.
Frame = stabilised, ball-anchored goal frame (u = ball->keeper, v = kicker's left; unit = kicker height). Kicking foot = foot with larger ankle travel over the last frames."""
import pickle, os, numpy as np, pandas as pd
import kin2
from kin2 import legs_at, match2

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
T = 12; CT = kin2.CT


def build(b, k, tr, meta, mirror):
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; kp, sc = k["kp"], k["sc"]; M = b["M"]
    P = np.einsum("fij,fkj->fki", M, np.concatenate([kp, np.ones(kp.shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]
    hL = max(float(k["box"][-1][3] - k["box"][-1][1]), 20.0); ball = b["ball_last"].copy()
    if not np.isfinite(ball[0]):
        a = P[-1][[15, 16]]; ball = np.nanmean(a, axis=0) if np.isfinite(a).any() else np.array([np.nan, np.nan])
    kb = tr["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]
    if len(vv) == 0 or not np.isfinite(ball[0]): return None
    kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]); fv = kf - ball; fn = np.linalg.norm(fv); f = fv / fn if fn > 1e-6 else np.array([1.0, 0.0])
    Lv = np.array([f[1], -f[0]]) * (-1 if mirror else 1)
    uv = lambda X: np.stack([(X - ball) @ f / hL, (X - ball) @ Lv / hL], -1)
    # legs per frame (ankle-anchored), identity tracked across ALL frames, kicking foot from travel over the last frames
    L = [legs_at(P[j], sc[j]) for j in range(nF)]; order = [[0, 1]]
    cen = lambda leg: np.nanmean(np.array([leg[q] for q in ("ank", "toe", "heel")]), axis=0) if np.isfinite(leg["ank"]).all() else np.array([np.nan, np.nan])
    for t in range(1, nF):
        ref = [cen(L[t - 1][order[-1][i]]) for i in range(2)]; cand = np.array([cen(L[t][j]) for j in range(2)]); m = match2(cand, None, np.array(ref))
        o = [m[0] if m[0] is not None else 0, m[1] if m[1] is not None else 1]; order.append(o if o[0] != o[1] else [0, 1])
    ident = [[L[t][order[t][i]] for t in range(nF)] for i in range(2)]
    n6 = min(6, nF)
    def travel(i):
        pts = np.array([ident[i][t]["ank"] for t in range(nF - n6, nF)]); d = np.hypot(*np.diff(pts, axis=0).T); return np.nansum(d) if np.isfinite(d).any() else 0.0
    kick_i = int(np.argmax([travel(0), travel(1)])); legs = {"kk": ident[kick_i], "pl": ident[1 - kick_i]}
    U = uv(P); Pm = np.where((sc >= CT)[..., None], P, np.nan)
    hm, sm, nose = np.nanmean(U[:, [11, 12]], axis=1), np.nanmean(U[:, [5, 6]], axis=1), U[:, 0]
    cols = []
    for w in ("kk", "pl"):
        for nm in ("ank", "toe", "heel"): cols.append(uv(np.array([legs[w][t][nm] for t in range(nF)])))
    for w in ("kk", "pl"):
        for nm in ("knee", "hip"): cols.append(uv(np.array([legs[w][t][nm] for t in range(nF)])))
    cols += [hm, sm, nose]
    for a_, b_ in ((11, 12), (5, 6), (3, 4)):
        vec = U[:, b_] - U[:, a_]; th = np.arctan2(vec[:, 1], vec[:, 0]); n = np.hypot(vec[:, 0], vec[:, 1]); ok = (sc[:, [a_, b_]] >= CT).all(1)
        cols.append(np.stack([np.where(ok, np.cos(2 * th), np.nan), np.where(ok, np.sin(2 * th), np.nan)], -1)); cols.append(np.where(ok, n, np.nan)[:, None])
    hmI, smI = np.nanmean(Pm[:, [11, 12]], axis=1), np.nanmean(Pm[:, [5, 6]], axis=1); getI = lambda w, nm: np.array([legs[w][t][nm] for t in range(nF)])
    for a_, b_ in ((hmI, smI), (getI("pl", "knee"), getI("pl", "ank")), (getI("pl", "hip"), getI("pl", "knee")), (getI("kk", "knee"), getI("kk", "ank")), (getI("kk", "hip"), getI("kk", "knee")), (smI, Pm[:, 0])):
        s = (b_ - a_) / hL; sx, sy = s[:, 0], s[:, 1]; cols.append(np.stack([sx, sy, sx * f[0], sx * Lv[0]], -1))
    X = np.concatenate([c if c.ndim == 2 else c[:, None] for c in cols], axis=1).astype(np.float32)            # [nF, C0]
    fin = np.isfinite(X).mean(1, keepdims=True).astype(np.float32); X = np.nan_to_num(X, nan=0.0)
    X = np.concatenate([X, ts[:, None].astype(np.float32), fin, np.ones((nF, 1), np.float32)], axis=1)          # + dt, finite fraction, exists
    out = np.zeros((T, X.shape[1]), np.float32); n = min(T, nF); out[-n:] = X[-n:]
    static = np.array([(1.0 if meta["foot"] == "right" else -1.0) * (-1 if mirror else 1), 1.0 if meta["gender"] == "M" else 0.0, meta["fps"] / 60.0, float(b["support"]) / 15.0], np.float32)
    return out, static


if __name__ == "__main__":
    tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
    meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}
    rows = {}; skipped = 0
    for c in list(gt["clip"]) + list(pm["clip"]):
        b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
        if b is None or k is None or tracks[c] is None or tracks[c]["keep"] is None: skipped += 1; continue
        a = build(b, k, tracks[c], meta[c], False); m = build(b, k, tracks[c], meta[c], True)
        if a is None or m is None: skipped += 1; continue
        rows[c] = (a, m)
    pickle.dump(rows, open(f"{C}/seq.pkl", "wb")); r0 = next(iter(rows.values())); print("built", len(rows), "skipped", skipped, "shape", r0[0][0].shape, "static", r0[0][1].shape)
