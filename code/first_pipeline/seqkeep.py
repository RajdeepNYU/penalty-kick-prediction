"""Stage D4: per-frame label-free KEEPER channels for the keeper temporal network. T=12 last frames of F, original + mirrored (v negated).
Stream A = RTMW keypoints (kkeep), stream B = RTMW3D's 2D keypoints (kkeep3d). Frame = stabilised, ball-anchored goal frame, unit = keeper height.
usage: python seqkeep.py   -> cache/seqk.pkl (stream A) and cache/seqkB.pkl (stream B, if kkeep3d is complete)"""
import pickle, os, numpy as np, pandas as pd

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
T = 12


def build(b, kp, sc, box, tr, meta, mirror, CT):
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; M = b["M"]
    P = np.einsum("fij,fkj->fki", M, np.concatenate([kp, np.ones(kp.shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]
    P = np.where((sc >= CT)[..., None], P, np.nan)
    ball = b["ball_last"]; kb = tr["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]
    if len(vv) == 0 or not np.isfinite(ball[0]): return None
    kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]); fv = kf - ball; fn = np.linalg.norm(fv); f = fv / fn if fn > 1e-6 else np.array([1.0, 0.0])
    Lv = np.array([f[1], -f[0]]) * (-1 if mirror else 1)
    hk = max(float(np.nanmedian(box[:, 3] - box[:, 1])), 20.0)
    uv = lambda X: np.stack([(X - ball) @ f / hk, (X - ball) @ Lv / hk], -1)
    U = uv(P); nm = np.nanmean
    hm, sm, am, km, wm, nose = nm(U[:, [11, 12]], 1), nm(U[:, [5, 6]], 1), nm(U[:, [15, 16]], 1), nm(U[:, [13, 14]], 1), nm(U[:, [9, 10]], 1), U[:, 0]
    cols = [hm, sm, am, km, wm, nose]
    first = lambda S: S - S[np.where(np.isfinite(S[:, 0]))[0][0]] if np.isfinite(S[:, 0]).any() else S * np.nan
    cols += [first(hm), first(am), first(sm), first(nose)]
    for a_, b_ in ((11, 12), (5, 6), (15, 16), (9, 10)):
        vec = U[:, b_] - U[:, a_]; th = np.arctan2(vec[:, 1], vec[:, 0]); n = np.hypot(vec[:, 0], vec[:, 1]); ok = (sc[:, [a_, b_]] >= CT).all(1)
        cols.append(np.stack([np.where(ok, np.cos(2 * th), np.nan), np.where(ok, np.sin(2 * th), np.nan)], -1)); cols.append(np.where(ok, n, np.nan)[:, None])
    hmI, smI, amI, kmI = nm(P[:, [11, 12]], 1), nm(P[:, [5, 6]], 1), nm(P[:, [15, 16]], 1), nm(P[:, [13, 14]], 1)
    for a_, b_ in ((hmI, smI), (hmI, kmI), (kmI, amI)):
        s = (b_ - a_) / hk; sx, sy = s[:, 0], s[:, 1]; cols.append(np.stack([sx, sy, sx * f[0], sx * Lv[0]], -1))
    bw = (box[:, 2] - box[:, 0]) / np.maximum(box[:, 3] - box[:, 1], 1); bh = (box[:, 3] - box[:, 1]) / hk; cols.append(np.stack([bw, bh], -1))
    X = np.concatenate([c if c.ndim == 2 else c[:, None] for c in cols], axis=1).astype(np.float32)
    fin = np.isfinite(X).mean(1, keepdims=True).astype(np.float32); X = np.nan_to_num(X, nan=0.0)
    X = np.concatenate([X, ts[:, None].astype(np.float32), fin, np.ones((nF, 1), np.float32)], axis=1)
    out = np.zeros((T, X.shape[1]), np.float32); n = min(T, nF); out[-n:] = X[-n:]
    static = np.array([0.0, 1.0 if meta["gender"] == "M" else 0.0, meta["fps"] / 60.0, float(b["support"]) / 15.0], np.float32)
    return out, static


def run(stream):
    tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
    meta = {r["clip"]: dict(gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}
    rows = {}; skipped = 0
    for c in list(gt["clip"]) + list(pm["clip"]):
        b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); kk = pickle.load(open(f"{C}/kkeep/{c}.pkl", "rb")); t = tracks[c]
        if b is None or kk is None or t is None or t["keep"] is None: skipped += 1; continue
        if stream == "A": kp, sc, CT = kk["kp"], kk["sc"], 2.0
        else:
            k3 = pickle.load(open(f"{C}/kkeep3d/{c}.pkl", "rb"))
            if k3 is None: skipped += 1; continue
            nF = b["nF"]; kp = np.zeros((nF, 133, 2), np.float32); sc = np.zeros((nF, 133), np.float32); kp[:, :23] = k3["kp2"]; sc[:, :23] = k3["sc"] * 10; CT = 2.0
        a = build(b, kp, sc, kk["box"], t, meta[c], False, CT); m = build(b, kp, sc, kk["box"], t, meta[c], True, CT)
        if a is None or m is None: skipped += 1; continue
        rows[c] = (a, m)
    pickle.dump(rows, open(f"{C}/seqk{'' if stream == 'A' else 'B'}.pkl", "wb")); r0 = next(iter(rows.values())); print(f"stream {stream}: built", len(rows), "skipped", skipped, "shape", r0[0][0].shape)


if __name__ == "__main__":
    run("A")
    if len(os.listdir(f"{C}/kkeep3d")) >= 2200: run("B")
