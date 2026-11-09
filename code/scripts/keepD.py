"""Idea 3 of RULES_nov6.md: keeper dip timing and "committed yet?" flags, appended to the four keeper tree members (keep_S_b2 + keepD_S, keepR_S + keepD_S).
Goal-plane decomposition of keepR.py (camera2: a_v along the goal line to the kicker's left, a_z up; metres), keeper keypoints at the F frames stabilised with M,
score >= 2.0 (RTMW kkeep = stream A, ViTPose-L kkeepF = stream F). Series relative to -0.45 s: hip mid, head (nose), feet mid.
Turning points of 'up' (hip, head) among the frames from -0.6 s: scipy find_peaks on up (tops) and -up (dip bottoms), prominence PROM metres (fixed without labels, see probe).
Per series: time since the last turning point (s before the end; NaN if none), its kind (+1 dip bottom, -1 top, 0 none), rise and lateral move since then
(lateral signed and absolute), time since the last dip bottom and its depth (highest point in the 0.3 s before it minus the bottom).
Flags: lateral hip move over the last 0.12 s > 0.10 m; feet raised > 0.10 m at the last frame; time since the hip / the feet were last >= 0.15 m (lateral) from their final
position (NaN if never in the window). Mirror negates the lateral axis.
usage: python cap.py keepD.py [probe]  -> cache/keepD_{A,F}.pkl rows {clip: (static, static_mirror)}, names"""
import os, sys, pickle, warnings, numpy as np
from common import *
warnings.filterwarnings("ignore")
from scipy.signal import find_peaks

CT, PROM = 2.0, float(os.environ.get("KD_PROM", "0.03"))                  # probe: hip 'up' jitter ~0.02 m per frame -> 0.03 m (turning point in ~1 clip in 4)
CAMF = os.environ.get("CAMF", "camera2"); CAM = pickle.load(open(f"{PC}/{CAMF}.pkl", "rb")); NAMES = []


def series(b, kk, cam, sg):
    ts = b["ts"] - b["ts"][-1]; sc = kk["sc"]; M = b["M"]
    P = np.einsum("fij,fkj->fki", M, np.concatenate([kk["kp"], np.ones(kk["kp"].shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]; P = np.where((sc >= CT)[..., None], P, np.nan)
    Bi = np.linalg.inv(np.c_[cam["A"][:, 1], cam["az"]]); GP = lambda V: (np.asarray(V, float) @ Bi.T) * np.array([sg, 1.0])
    mid = lambda ids: np.nanmean(P[:, ids], axis=1); ref = int(np.abs(ts + 0.45).argmin())
    pts = {"hip": mid([11, 12]), "head": P[:, 0], "feet": mid([15, 16])}
    return ts, {k: GP(v - v[ref]) for k, v in pts.items()}


def turning(ts, up, prom=None):
    prom = PROM if prom is None else prom; m = np.where((ts >= -0.6) & np.isfinite(up))[0]
    if len(m) < 4: return np.array([], int), np.array([], int)
    hi, _ = find_peaks(up[m], prominence=prom); lo, _ = find_peaks(-up[m], prominence=prom); return m[hi], m[lo]


def build(b, kk, cam, mirror):
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(np.asarray(vals, np.float32)).ravel(); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    ts, S = series(b, kk, cam, -1.0 if mirror else 1.0); near = lambda o: int(np.abs(ts + o).argmin())
    for nm in ("hip", "head"):
        lat, up = S[nm][:, 0], S[nm][:, 1]; hi, lo = turning(ts, up); ex = sorted([(i, -1) for i in hi] + [(i, 1) for i in lo])
        if ex:
            i, kind = ex[-1]; add(f"d_{nm}_last_turn", [-ts[i], kind, up[-1] - up[i], lat[-1] - lat[i], abs(lat[-1] - lat[i])])
        else: add(f"d_{nm}_last_turn", [np.nan, 0, np.nan, np.nan, np.nan])
        if len(lo):
            i = lo[-1]; w = np.where((ts >= ts[i] - 0.3) & (ts <= ts[i]))[0]; add(f"d_{nm}_dip", [-ts[i], np.nanmax(up[w]) - up[i]])
        else: add(f"d_{nm}_dip", [np.nan, np.nan])
    hl, fu, fl = S["hip"][:, 0], S["feet"][:, 1], S["feet"][:, 0]; j12 = near(0.12)
    mv = abs(hl[-1] - hl[j12]); add("d_commit", [float(mv > 0.10) if np.isfinite(mv) else np.nan, float(fu[-1] > 0.10) if np.isfinite(fu[-1]) else np.nan])
    for nm, L in (("hip", hl), ("feet", fl)):
        far = np.where(np.isfinite(L) & (np.abs(L - L[-1]) >= 0.15))[0]; add(f"d_{nm}_t15", -ts[far[-1]] if len(far) and np.isfinite(L[-1]) else np.nan)
    return np.array(f_, np.float32)


def one(args):
    c, S = args
    b = load("ballspot", c); k0 = load("kkeep", c); ks = k0 if S == "A" else load("kkeepF", c); cam = CAM.get(c)
    if b is None or k0 is None or ks is None or cam is None: return c, None, None
    kk = dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"]); NAMES.clear(); a = build(b, kk, cam, False); names = list(NAMES); NAMES.clear(); m = build(b, kk, cam, True)
    return c, (a, m), names


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":                       # label-free: turning points per clip for a few prominences, frame-to-frame jitter of 'up'
        rng = np.random.default_rng(0); smp = [CLIPS[i] for i in rng.choice(len(CLIPS), 200, replace=False)]
        for S in "AF":
            cnt = {p: [] for p in (0.02, 0.03, 0.04, 0.06)}; jit = []
            for c in smp:
                b = load("ballspot", c); k0 = load("kkeep", c); ks = k0 if S == "A" else load("kkeepF", c); cam = CAM.get(c)
                if b is None or k0 is None or ks is None or cam is None: continue
                ts, Sr = series(b, dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"]), cam, 1.0); up = Sr["hip"][:, 1]; m = np.where(ts >= -0.24)[0]; jit.append(np.nanmedian(np.abs(np.diff(up[m]))))
                for p in cnt: hi, lo = turning(ts, up, p); cnt[p].append(len(hi) + len(lo))
            print(f"{S}: hip 'up' median frame-to-frame change over the last 0.24 s {np.nanmedian(jit):.4f} m (p90 {np.nanpercentile(jit, 90):.4f})")
            for p, v in cnt.items(): v = np.array(v); print(f"   prominence {p:.2f} m: turning points per clip median {np.median(v):.1f}, mean {v.mean():.2f}, none {np.mean(v == 0):.2f}, >= 3 {np.mean(v >= 3):.2f}")
        sys.exit()
    from multiprocessing import Pool
    for S in "AF":
        with Pool(int(os.environ.get("NPOOL", "8"))) as p: res = p.map(one, [(c, S) for c in CLIPS], chunksize=20)
        rows = {c: r for c, r, _ in res if r is not None}; names = next(n for _, r, n in res if r is not None)
        pickle.dump(dict(rows=rows, names=names), open(f"{PC}/keepD_{S}{'' if CAMF == 'camera2' else '_' + CAMF}.pkl", "wb"))
        X = np.stack([r[0] for r in rows.values()]); print(f"keepD {S}: built {len(rows)} | dim {len(names)} | NaN frac {np.isnan(X).mean():.3f} | names {names}", flush=True)
