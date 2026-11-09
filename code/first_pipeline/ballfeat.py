"""Stage 11: ball-motion features in the goal-facing frame (u forward, v kicker-left), original + mirrored."""
import pickle, os, numpy as np, pandas as pd

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
NF = 8
cen = lambda b: np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def build(bd, tr, mirror):
    kb, gb = tr["kick"], tr["keep"]
    kc, gc = cen(kb), cen(gb); v = np.isfinite(kc[:, 0]) & np.isfinite(gc[:, 0])
    if bd is None or not v.any(): return np.full(NF * 3 + 14, np.nan, np.float32)
    fv = (gc[v] - kc[v]).mean(0); f = fv / max(np.linalg.norm(fv), 1e-6)
    L = np.array([f[1], -f[0]]) * (-1 if mirror else 1)
    uv = lambda P: np.stack([P @ f, P @ L], -1)
    h0 = bd["h0"]; ts = bd["ts"]; n = len(ts)
    best = np.full((n, 3), np.nan)
    for j, cd in enumerate(bd["cands"]):
        if len(cd) == 0: continue
        rel = cd[:, :2] - bd["kc"][j]; near = np.linalg.norm(rel, axis=1) / h0 < 1.6
        if not near.any(): continue
        i = np.where(near)[0][cd[near, 2].argmax()]
        p = uv(rel[i][None])[0] / h0; best[j] = [p[0], p[1], cd[i, 2]]
    out = []
    for j in range(n - NF, n):
        out += list(best[j]) if j >= 0 else [np.nan] * 3
    ok = np.where(np.isfinite(best[:, 0]))[0]
    if len(ok) >= 2:
        a, b = ok[0], ok[-1]; dt = ts[b] - ts[a]
        d = best[b, :2] - best[a, :2]
        a2 = ok[-2]; dt2 = ts[b] - ts[a2]; d2 = (best[b, :2] - best[a2, :2]) / dt2 if dt2 > 0 else [np.nan, np.nan]
        out += [d[0], d[1], dt, len(ok), d2[0], d2[1], np.hypot(*d2), np.arctan2(d2[1], d2[0]) if np.isfinite(d2[0]) else np.nan,
                (np.arctan2(d[1], d[0]))]
    else:
        out += [np.nan] * 4 + [np.nan] * 5
    out += [float(len(ok)), float(np.isfinite(best[-1, 0])), float(np.isfinite(best[-2, 0])) if n > 1 else 0.0,
            np.nanmax(best[:, 2]) if len(ok) else np.nan, np.nanmean(best[:, 0]) if len(ok) else np.nan]
    return np.array(out, np.float32)


if __name__ == "__main__":
    tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); rows = {}
    for c, t in tracks.items():
        fp = f"{C}/ball/{c}.pkl"
        if t is None or t["keep"] is None or not os.path.exists(fp): continue
        bd = pickle.load(open(fp, "rb"))
        rows[c] = (build(bd, t, False), build(bd, t, True))
    pickle.dump(rows, open(f"{C}/ballfeats.pkl", "wb"))
    print("built", len(rows), "dim", len(next(iter(rows.values()))[0]))
