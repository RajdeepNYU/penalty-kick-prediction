"""Ideas 1 and 2 of RULES_nov6b.md for the kick trees; tag S_b2H = kin2_S_ball2 + kinR_S + kinH_S.
Head block (idea 1, "does the kicker look up at the keeper?"), image y down, h = kicker box height / 1.1:
  stream S head / trunk points at the F frames: neck extension (shoulder-mid y - nose y) / (hip-mid y - shoulder-mid y), nose above the ears (ear-mid y - nose y) / h,
  each as last frame, mean of the last 3, value nearest -0.5 s, last minus -0.5 s; head turn (nose x - ear-mid x) / ear distance (mirror flips it), last and last 3.
  RTMW 68-point face (stream A, for every stream): face pitch (nose-tip y - eye-mid y) / (chin y - eye-mid y), last and last 3; mean face-point score over the last 3 frames.
  dense ViTPose-L (cache/kptsD2, every native frame of the last 2 s, the same for every stream): u = (shoulder-mid y - nose y) / h smoothed over ~0.1 s; u at the end,
  max / min over the last 0.5 s, time of that max before the end, mean over [-1, -0.5] and [-0.5, 0] s and their difference, up-down swings of u in the last 1 s.
Kicking-foot block (idea 2, laces drive vs inside foot), RTMW feet and legs of stream A (kicking / plant split of kinR.legs), for every stream:
  ankle angle (knee - ankle vs big toe - ankle) of the kicking and the plant leg, shank vs foot (toe - heel) angle of the kicking leg (last, last 3);
  kicking-foot pointing on the ground (toe - heel) minus the hip approach over [-0.3, 0] s and minus the goal axis (cos, sin; last, last 3); kicking minus planted pointing (last);
  big toe - small toe distance / foot length and (toe y - heel y) / foot length in the image, kicking foot (last, last 3).
Mirror: lateral axis negated (ground sines and head turn change sign). Parameters fixed without labels (probe mode prints the label-free ranges).
usage: [CAMF=camera2f] python cap.py kinH.py [probe] -> cache/kinH_{A,F,G}{_CAMF}.pkl rows {clip: (static, static_mirror)}, names"""
import os, sys, pickle, warnings, numpy as np
from common import *
warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT + r"\Q4_work")
from scipy.signal import find_peaks

CT, SCD, SM, PROMU = 2.0, 0.3, 0.1, 0.01            # PROMU 0.02 -> 0.01 after the label-free probe (swings were almost always 0)
Z = dict(toe=0.03, heel=0.05)
NOSE_TIP, CHIN, EYES = 23 + 30, 23 + 8, list(range(23 + 36, 23 + 48))
camf = os.environ.get("CAMF", "camera2"); CAM = pickle.load(open(f"{PC}/{camf}.pkl", "rb"))


def _smooth(x, k):
    if k <= 1: return x
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def _ang(a, b):
    """unsigned angle between 2D vectors per row (rad)"""
    n = np.hypot(*a.T) * np.hypot(*b.T); return np.arccos(np.clip((a * b).sum(1) / np.where(n > 0, n, np.nan), -1, 1))


def head_F(kbS, kbA, ts, sg):
    out = []
    def add(n, v): out.append((n, np.atleast_1d(np.asarray(v, float)).ravel()))
    nF = len(ts); l3 = slice(max(0, nF - 3), nF); j5 = int(np.abs(ts + 0.5).argmin())
    kp, sc = kbS["kp"].astype(float), kbS["sc"]; Pm = np.where((sc >= CT)[..., None], kp, np.nan); h = np.maximum((kbS["box"][:, 3] - kbS["box"][:, 1]) / 1.1, 10.0)
    nose, ear, sm, hm = Pm[:, 0], np.nanmean(Pm[:, [3, 4]], 1), np.nanmean(Pm[:, [5, 6]], 1), np.nanmean(Pm[:, [11, 12]], 1)
    neck = (sm[:, 1] - nose[:, 1]) / np.where(hm[:, 1] - sm[:, 1] > 1, hm[:, 1] - sm[:, 1], np.nan); up = (ear[:, 1] - nose[:, 1]) / h
    for nm, v in (("h_neck", neck), ("h_up", up)): add(nm, [v[-1], np.nanmean(v[l3]), v[j5], v[-1] - v[j5]])
    ed = np.abs(Pm[:, 4, 0] - Pm[:, 3, 0]); turn = sg * (nose[:, 0] - ear[:, 0]) / np.where(ed > 1, ed, np.nan); add("h_turn", [turn[-1], np.nanmean(turn[l3])])
    ka, sa = kbA["kp"].astype(float), kbA["sc"]; Fm = np.where((sa >= CT)[..., None], ka, np.nan); eye = np.nanmean(Fm[:, EYES], 1)
    den = Fm[:, CHIN, 1] - eye[:, 1]; pitch = (Fm[:, NOSE_TIP, 1] - eye[:, 1]) / np.where(den > 1, den, np.nan)
    add("h_face_pitch", [pitch[-1], np.nanmean(pitch[l3])]); add("h_face_seen", float(sa[l3, 23:91].mean()))   # probe: RTMW scores face points >= 2 even from behind, so the mean score replaces the 20-point count
    return out


def head_dense(c):
    out = []
    def add(n, v): out.append((n, np.atleast_1d(np.asarray(v, float)).ravel()))
    fp = f"{PC}/kptsD2/{c}.pkl"; z = pickle.load(open(fp, "rb")) if os.path.exists(fp) else None; n = 0 if z is None else len(z["ts"])
    ok = None
    if n >= 6:
        t = z["ts"] - z["ts"][-1]; kp = z["kp"].astype(float); sc = z["sc"]; h = np.maximum((z["box"][:, 3] - z["box"][:, 1]) / 1.1, 10.0)
        ok = (sc[:, 0] >= SCD) & (sc[:, 5] >= SCD) & (sc[:, 6] >= SCD); u = (kp[:, [5, 6], 1].mean(1) - kp[:, 0, 1]) / h
    if ok is None or ok.sum() < 6:
        for nm, k in (("hd_end", 1), ("hd_maxmin05", 2), ("hd_tmax", 1), ("hd_mean", 3), ("hd_swing", 1)): add(nm, [np.nan] * k)
        return out
    i = np.arange(n); k = max(1, int(round(SM * META[c]["fps"]))) | 1; s = _smooth(np.interp(i, i[ok], u[ok]), k)
    w5, w10 = t >= -0.5, t >= -1.0; a = (t >= -1.0) & (t < -0.5)
    add("hd_end", s[-1]); add("hd_maxmin05", [s[w5].max(), s[w5].min()]); add("hd_tmax", float(-t[w5][s[w5].argmax()]))
    m1, m2 = (float(s[a].mean()) if a.any() else np.nan), float(s[w5].mean()); add("hd_mean", [m1, m2, m2 - m1])
    p, _ = find_peaks(s[w10], prominence=PROMU) if w10.sum() >= 3 else (np.array([]), None); add("hd_swing", float(len(p)))
    return out


def foot(c, b, cam, kbA, sg):
    from kinR import legs
    out = []
    def add(n, v): out.append((n, np.atleast_1d(np.asarray(v, float)).ravel()))
    nF, ts, M = b["nF"], b["ts"] - b["ts"][-1], b["M"]; l3 = slice(max(0, nF - 3), nF)
    PA = (lambda Q: Q[..., :2] / Q[..., 2:3])(np.einsum("fij,fkj->fki", M, np.concatenate([kbA["kp"], np.ones(kbA["kp"].shape[:2] + (1,), np.float32)], -1)))
    lg, _, _ = legs(PA, kbA["sc"], nF); arr = lambda w, nm: np.array([lg[w][t][nm] for t in range(nF)], float)
    Ai, bg, az = cam["Ai"], cam["ball_ground"], cam["az"]
    G = lambda X, zz: ((np.asarray(X, float) - bg - az * zz) @ Ai.T) * np.array([1.0, sg]); Gd = lambda V: (np.asarray(V, float) @ Ai.T) * np.array([1.0, sg])
    for w in ("kk", "pl"):
        an = _ang(arr(w, "knee") - arr(w, "ank"), arr(w, "toe") - arr(w, "ank")); add(f"f_{w}_ankle", [an[-1], np.nanmean(an[l3])])
    fv = arr("kk", "toe") - arr("kk", "heel"); sf = _ang(arr("kk", "knee") - arr("kk", "ank"), fv); add("f_kk_shank_foot", [sf[-1], np.nanmean(sf[l3])])
    pk = G(arr("kk", "toe"), Z["toe"]) - G(arr("kk", "heel"), Z["heel"]); ak = np.arctan2(pk[:, 1], pk[:, 0])
    pp = G(arr("pl", "toe"), Z["toe"]) - G(arr("pl", "heel"), Z["heel"]); ap = np.arctan2(pp[:, 1], pp[:, 0])
    Pm = np.where((kbA["sc"] >= CT)[..., None], PA, np.nan); hm = np.nanmean(Pm[:, [11, 12]], 1); D = Gd(hm - hm[-1]); near = lambda o: int(np.abs(ts + o).argmin())
    v = D[near(0.0)] - D[near(0.3)]; run = np.arctan2(v[1], v[0]) if np.isfinite(v).all() and np.hypot(*v) > 1e-6 else np.nan
    cs = lambda d: [np.cos(d[-1]), np.sin(d[-1]), np.nanmean(np.cos(d[l3])), np.nanmean(np.sin(d[l3]))]
    add("f_kk_vs_run", cs(ak - run)); add("f_kk_vs_goal", cs(ak)); add("f_kk_vs_pl", [np.cos(ak[-1] - ap[-1]), np.sin(ak[-1] - ap[-1])])
    fl = np.hypot(*fv.T); fl = np.where(fl > 1, fl, np.nan); wd = np.hypot(*(arr("kk", "toe") - arr("kk", "sml")).T) / fl; dr = fv[:, 1] / fl
    add("f_kk_width", [wd[-1], np.nanmean(wd[l3])]); add("f_kk_drop", [dr[-1], np.nanmean(dr[l3])])
    return out


def one(c):
    from streams import stream_kb
    b = load("ballspot", c); cam = CAM.get(c); kbA = stream_kb(c, "A")
    if b is None or cam is None or kbA is None: return c, None
    ts = b["ts"] - b["ts"][-1]; hd = head_dense(c); res = {}
    for S in "AFG":
        kbS = kbA if S == "A" else stream_kb(c, S)
        if kbS is None: continue
        r = []
        for sg in (1.0, -1.0):
            blocks = head_F(kbS, kbA, ts, sg) + hd + foot(c, b, cam, kbA, sg)
            r.append((np.concatenate([v for _, v in blocks]).astype(np.float32), [f"{n}{j}" if len(v) > 1 else n for n, v in blocks for j in range(len(v))]))
        res[S] = ((r[0][0], r[1][0]), r[0][1])
    return c, res


if __name__ == "__main__":
    from multiprocessing import Pool
    probe = len(sys.argv) > 1 and sys.argv[1] == "probe"; CL = CLIPS[::10] if probe else CLIPS
    with Pool(int(os.environ.get("NPOOL", "8"))) as p: R = p.map(one, CL, chunksize=10)
    if probe:
        names = next(r["A"][1] for c, r in R if r is not None and "A" in r); X = np.stack([r["A"][0][0] for c, r in R if r is not None and "A" in r]); Xm = np.stack([r["A"][0][1] for c, r in R if r is not None and "A" in r])
        for j, nm in enumerate(names):
            v = X[:, j]; f = np.isfinite(v); print(f"{nm:20s} NaN {1 - f.mean():.2f} | p10 / p50 / p90 {np.percentile(v[f], 10) if f.any() else np.nan:8.3f} {np.percentile(v[f], 50) if f.any() else np.nan:8.3f} {np.percentile(v[f], 90) if f.any() else np.nan:8.3f}"
                                           f" | mirror equal {np.allclose(v[f], Xm[f, j], equal_nan=True)!s:5s} negated {np.allclose(v[f], -Xm[f, j], equal_nan=True)!s}")
        sys.exit()
    sfx = "" if camf == "camera2" else "_" + camf
    for S in "AFG":
        rows = {c: r[S][0] for c, r in R if r is not None and S in r}; names = next(r[S][1] for c, r in R if r is not None and S in r)
        pickle.dump(dict(rows=rows, names=names), open(f"{PC}/kinH_{S}{sfx}.pkl", "wb")); X = np.stack([v[0] for v in rows.values()])
        print(f"kinH {S}{sfx}: built {len(rows)} of {len(CLIPS)} | dim {len(names)} | NaN frac {np.isnan(X).mean():.3f} | per feature NaN max {np.isnan(X).mean(0).max():.3f}", flush=True)
