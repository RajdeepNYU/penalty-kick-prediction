"""Stage 4: hand-built kinematic features from tracked boxes + pose, in a goal-facing (forward u / kicker-left v) frame.
mirror=True gives the exact left/right mirrored sample (v -> -v, swap L/R joints, swap preferred foot)."""
import pickle, os, numpy as np, pandas as pd

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
K9 = [0, 5, 6, 11, 12, 13, 14, 15, 16]
FI = [12, 10, 8, 6, 4]          # 0, .06, .15, .25, .4 s before last frame
KFI = [12, 8, 6]
CT = 0.25


def clean(kp, mirror):
    kp = kp.copy(); kp[kp[..., 2] < CT, :2] = np.nan
    return kp[:, PERM] if mirror else kp


def cen(b): return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def build(d, tr, ps, meta, mirror):
    nan = np.nan
    kb, gb = tr["kick"], tr["keep"]
    if gb is None or ps is None or ps["kick"] is None: return None
    kc, gc = cen(kb), cen(gb); kh = kb[:, 3] - kb[:, 1]; gh = gb[:, 3] - gb[:, 1]
    v = np.isfinite(kc[:, 0]) & np.isfinite(gc[:, 0])
    if not v.any(): return None
    fv = (gc[v] - kc[v]).mean(0); n = np.linalg.norm(fv)
    f = fv / n if n > 1e-6 else np.array([1., 0.])
    Lv = np.array([f[1], -f[0]]) * (-1 if mirror else 1)
    uv = lambda P: np.stack([P @ f, P @ Lv], -1)
    h0 = np.nanmedian(kh); g0 = np.nanmedian(gh)
    W, H = d["wh"]
    ts = np.array(d["ts"])
    out = []

    def pose_block(kp, c, hh, frames, scale):
        o = []
        for j in frames:
            p = kp[j]
            hm = np.nanmean(p[[11, 12], :2], axis=0) if np.isfinite(p[[11, 12], 0]).any() else c[j]
            q = uv(p[K9, :2] - hm) / scale
            o += list(q.ravel())
        return o

    def lines(kp, frames):
        o = []
        for j in frames:
            p = kp[j]
            for a, b in ((12, 11), (6, 5), (14, 13)):   # r-l hips, shoulders, knees (mirror swaps names, not geometry)
                dv = uv((p[a, :2] - p[b, :2])[None])[0]; nn = np.linalg.norm(dv)
                o += [dv[0] / nn, dv[1] / nn] if nn > 0 else [nan, nan]
            sm = np.nanmean(p[[5, 6], :2], axis=0); hm = np.nanmean(p[[11, 12], :2], axis=0)
            o += list(uv((sm - hm)[None])[0]); o += list(uv((p[15, :2] - p[16, :2])[None])[0])
        return o

    # kicker
    kk = clean(ps["kick"]["kp"], mirror); gk = clean(ps["keep"]["kp"], mirror)
    out += pose_block(kk, kc, kh, FI, h0)
    out += lines(kk, FI)
    # kicker centre velocity (u,v per second per height)
    for a, b in ((12, 9), (9, 6), (6, 3)):
        dt = ts[a] - ts[b]
        dv = uv((kc[a] - kc[b])[None])[0] / h0 / dt if dt > 0 else [nan, nan]
        out += list(dv)
    # ball near kicker
    for j in (12, 10, 8):
        fr = d["frames"][j]; bb = fr["bb"]; bc = fr["bc"]; r = [nan, nan, nan]
        if len(bb) and np.isfinite(kc[j, 0]):
            bcen = cen(bb); dist = np.linalg.norm(bcen - kc[j], axis=1) / h0
            m = dist < 1.6
            if m.any():
                i = np.where(m)[0][bc[m].argmax()]
                r = list(uv((bcen[i] - kc[j])[None])[0] / h0) + [bc[i]]
                r = r[:3]
        out += r
    # keeper
    for j in (12, 10, 8, 4):
        out += list(uv((gc[j] - gc[6])[None])[0] / g0) if np.isfinite(gc[j, 0]) and np.isfinite(gc[6, 0]) else [nan, nan]
        out += [(gb[j, 2] - gb[j, 0]) / (gb[j, 3] - gb[j, 1]) if np.isfinite(gb[j, 0]) else nan, gh[j] / g0]
    out += pose_block(gk, gc, gh, KFI, g0)
    out += lines(gk, KFI)
    # relative geometry at last valid frame
    jl = np.where(v)[0].max()   # valid frame closest to the end (index 12 = last frame)
    rel = uv((gc[jl] - kc[jl])[None])[0]; dist = np.linalg.norm(rel)
    out += [rel[1] / dist, dist / h0, h0 / H, g0 / H, abs(f[0]), f[1]]
    # meta
    foot = (1.0 if meta["foot"] == "right" else -1.0) * (-1 if mirror else 1)
    out += [foot, 1.0 if meta["gender"] == "M" else 0.0, meta["fps"], d["dur"], W / H,
            float(np.isfinite(kb[:, 0]).sum()), float(np.isfinite(gb[:, 0]).sum())]
    return np.array(out, np.float32)


if __name__ == "__main__":
    tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
    gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
    meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}
    rows = {}
    for c, t in tracks.items():
        fp = f"{C}/pose/{c}.pkl"
        if t is None or not os.path.exists(fp): continue
        ps = pickle.load(open(fp, "rb")); d = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))
        a, b = build(d, t, ps, meta[c], False), build(d, t, ps, meta[c], True)
        if a is not None: rows[c] = (a, b)
    pickle.dump(rows, open(f"{C}/feats.pkl", "wb"))
    print("built", len(rows), "of", len(tracks), "dim", len(next(iter(rows.values()))[0]))
