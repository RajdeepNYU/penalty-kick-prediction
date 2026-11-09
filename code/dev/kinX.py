"""Idea 2 of RULES_nov6.md: run-up rhythm (the same for every stream) and two angle pairs (per stream) for the kick trees; tag S_b2X = kin2_S_ball2 + kinR_S + kinX_S.
Rhythm, from the dense ViTPose-L kicker keypoints on every native frame of the last 2 s (cache/kptsD2):
  step signal = distance between the two ankles in kicker heights (box height / 1.1), ankles with score >= 0.3, gaps interpolated, smoothed over ~0.1 s.
  Left / right labels are never used. Peaks (prominence >= 0.06, >= 0.15 s apart) = foot strikes, troughs = the feet passing.
  Features: number of strikes in the window and in the last 1 s, strikes per second, times of the last three strikes and of the last passing (s before the end),
  the last two step durations and their ratio, step size at the last three strikes (kicker heights and ground metres) and the last-to-previous ratios,
  irregularity of the strike intervals (max / median, spread / mean), the step signal at the last frame.
  Hip speed on the ground (hip mid, camera-stabilised with the F-frame homographies interpolated by frame index, so only from the first F frame, about 1 s
  before the end): speed over [-1.0, -0.6], [-0.6, -0.3], [-0.3, 0] s, the two ratios between them (slowing down before the kick), slowest 0.2 s / mean ("pause").
Angle pairs, per stream at the F frames (as kinR): hips vs planted foot = planted-foot pointing (toe - heel on the ground) minus the hips' facing line (hip axis + 90 deg),
  mod 180 -> cos 2d, sin 2d at the last frame and over the last 3; planted foot vs run-up line = planted-foot pointing minus the hip approach direction over [-0.8, -0.3] s
  and over [-0.3, 0] s -> cos d, sin d, with the last-frame and the last-3 pointing. The pointing comes from the RTMW feet (stream A, its plant / kicking leg split) for every
  stream: ViTPose has no toe or heel points.
Mirror: lateral axis negated (the angles change sign, rhythm magnitudes and times do not).
usage: [CAMF=camera2f] python cap.py kinX.py  -> cache/kinX_{A,F,G}{_CAMF}.pkl rows {clip: (static, static_mirror)}, names"""
import os, sys, pickle, warnings, numpy as np
from common import *
warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT + r"\Q4_work")
from scipy.signal import find_peaks

CT, SCD, PROM, DIST, SM = 2.0, 0.3, 0.06, 0.15, 0.1
Z = dict(toe=0.03, heel=0.05, ank=0.08, hip=0.95)
camf = os.environ.get("CAMF", "camera2"); CAM = pickle.load(open(f"{PC}/{camf}.pkl", "rb")); NAMES = []


def _Mat(b, fidx):
    """F-frame homographies (frame -> last-frame image) interpolated by native frame index, clamped outside [F[0], F[-1]]"""
    F = np.asarray(b["F"], float); M = np.asarray(b["M"], float); M = M / M[:, 2:3, 2:3]
    return np.stack([np.stack([np.interp(fidx, F, M[:, i, j]) for j in range(3)], -1) for i in range(3)], -2)


def _proj(Mf, X):
    h = np.einsum("fij,fj->fi", Mf, np.concatenate([X, np.ones((len(X), 1))], 1)); return h[:, :2] / h[:, 2:3]


def _smooth(x, k):
    if k <= 1: return x
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def rhythm(c, b, cam, sg):
    """(names, values) of the rhythm block; mirror only changes nothing here (sg enters ground lengths through |.|)"""
    out = []
    def add(n, v): v = np.atleast_1d(np.asarray(v, float)).ravel(); out.append((n, v))
    fp = f"{PC}/kptsD2/{c}.pkl"; z = pickle.load(open(fp, "rb")) if os.path.exists(fp) else None
    fps = META[c]["fps"]; Ai = cam["Ai"]; bg = cam["ball_ground"]; az = cam["az"]
    G = lambda X, zz: ((np.asarray(X, float) - bg - az * zz) @ Ai.T) * np.array([1.0, sg])
    Gd = lambda V: (np.asarray(V, float) @ Ai.T) * np.array([1.0, sg])
    n = 0 if z is None else len(z["ts"])
    if n < 6:
        for nm, k in (("x_nstr", 3), ("x_tstr", 4), ("x_dur", 3), ("x_amp", 4), ("x_len", 4), ("x_irr", 2), ("x_sep_end", 1), ("x_spd", 3), ("x_spd_ratio", 3)): add(nm, [np.nan] * k)
        return out
    t = z["ts"] - z["ts"][-1]; kp = z["kp"].astype(float); sc = z["sc"]; h = np.maximum((z["box"][:, 3] - z["box"][:, 1]) / 1.1, 10.0); fidx = np.asarray(z["fidx"], float)
    ok = (sc[:, 15] >= SCD) & (sc[:, 16] >= SCD); sep = np.hypot(*(kp[:, 15] - kp[:, 16]).T) / h; i = np.arange(n); f = ok & np.isfinite(sep)
    Mf = _Mat(b, fidx); A15, A16 = _proj(Mf, kp[:, 15]), _proj(Mf, kp[:, 16]); Lg = np.hypot(*Gd(A15 - A16).T)          # ground step length per frame (m)
    k = max(1, int(round(SM * fps))) | 1
    if f.sum() >= 6:
        s = _smooth(np.interp(i, i[f], sep[f]), k); dist = max(1, int(DIST * fps))
        p, _ = find_peaks(s, prominence=PROM, distance=dist); q, _ = find_peaks(-s, prominence=PROM, distance=dist)
    else: s = np.full(n, np.nan); p = q = np.array([], int)
    win = float(-t[0]); add("x_nstr", [len(p), int((t[p] >= -1.0).sum()) if len(p) else 0, len(p) / max(win, 0.2)] if f.sum() >= 6 else [np.nan] * 3)
    T3 = [float(-t[p[-j]]) if len(p) >= j else np.nan for j in (1, 2, 3)]; add("x_tstr", T3 + [float(-t[q[-1]]) if len(q) else np.nan])
    d12, d23 = T3[1] - T3[0], T3[2] - T3[1]; add("x_dur", [d12, d23, d12 / d23 if np.isfinite(d23) and d23 > 0 else np.nan])
    Am = [float(s[p[-j]]) if len(p) >= j else np.nan for j in (1, 2, 3)]; add("x_amp", Am + [Am[0] / Am[1] if np.isfinite(Am[1]) and Am[1] > 0 else np.nan])
    Lm = [float(np.nanmean(Lg[max(0, p[-j] - 1): p[-j] + 2])) if len(p) >= j else np.nan for j in (1, 2, 3)]; add("x_len", Lm + [Lm[0] / Lm[1] if np.isfinite(Lm[1]) and Lm[1] > 0 else np.nan])
    iv = np.diff(t[p]) if len(p) > 2 else np.array([]); add("x_irr", [iv.max() / np.median(iv), iv.std() / iv.mean()] if len(iv) >= 2 else [np.nan, np.nan])
    add("x_sep_end", float(s[-1]) if np.isfinite(s[-1]) else np.nan)
    # hip speed on the ground, stabilised frames only (from the first F frame)
    hp = np.nanmean(np.stack([kp[:, 11], kp[:, 12]]), 0); okh = (sc[:, 11] >= SCD) & (sc[:, 12] >= SCD) & (fidx >= float(b["F"][0]))
    Hg = np.full((n, 2), np.nan)
    if okh.sum() >= 4:
        Hs = G(_proj(Mf, hp), Z["hip"]); Hg[okh] = Hs[okh]
        for d in range(2): ff = np.isfinite(Hg[:, d]) & okh; Hg[:, d] = np.where(okh, _smooth(np.interp(i, i[ff], Hg[ff, d]), k), np.nan) if ff.sum() >= 2 else np.nan
    def pos(o): j = int(np.abs(t - o).argmin()); return Hg[j] if abs(t[j] - o) < 0.06 else np.array([np.nan, np.nan])
    sp = [float(np.hypot(*(pos(b_) - pos(a)))) / (b_ - a) for a, b_ in ((-1.0, -0.6), (-0.6, -0.3), (-0.3, 0.0))]; add("x_spd", sp)
    m = np.where(np.isfinite(Hg[:, 0]) & (t >= -1.0))[0]; low = np.nan
    if len(m) >= 4 and -t[m[0]] >= 0.6:
        tt = t[m]; w = [float(np.hypot(*(Hg[m][np.abs(tt - (a + 0.2)).argmin()] - Hg[m][np.abs(tt - a).argmin()]))) / 0.2 for a in np.arange(tt[0], -0.2 + 1e-6, 0.04)]
        mean = float(np.hypot(*(Hg[m[-1]] - Hg[m[0]]))) / max(-tt[0], 0.2); low = min(w) / mean if mean > 0 and len(w) else np.nan
    add("x_spd_ratio", [sp[1] / sp[0] if sp[0] > 0 else np.nan, sp[2] / sp[1] if sp[1] > 0 else np.nan, low])
    return out


def angles(c, S, b, cam, sg):
    from streams import stream_kb
    from kinR import legs
    out = []
    def add(n, v): out.append((n, np.atleast_1d(np.asarray(v, float)).ravel()))
    kbS = stream_kb(c, S); kbA = kbS if S == "A" else stream_kb(c, "A")
    if kbS is None or kbA is None:
        for nm in ("x_hip_plant_last", "x_hip_plant_m3", "x_plant_run08_last", "x_plant_run08_m3", "x_plant_run03_last", "x_plant_run03_m3"): add(nm, [np.nan, np.nan])
        return out
    nF, ts, M = b["nF"], b["ts"] - b["ts"][-1], b["M"]
    proj = lambda kp: (lambda Q: Q[..., :2] / Q[..., 2:3])(np.einsum("fij,fkj->fki", M, np.concatenate([kp, np.ones(kp.shape[:2] + (1,), np.float32)], -1)))
    PS = proj(kbS["kp"]); PmS = np.where((kbS["sc"] >= CT)[..., None], PS, np.nan); PA = proj(kbA["kp"]); lg, _, _ = legs(PA, kbA["sc"], nF)
    Ai, bg, az = cam["Ai"], cam["ball_ground"], cam["az"]
    G = lambda X, zz: ((np.asarray(X, float) - bg - az * zz) @ Ai.T) * np.array([1.0, sg]); Gd = lambda V: (np.asarray(V, float) @ Ai.T) * np.array([1.0, sg])
    th = G(np.array([lg["pl"][t]["toe"] for t in range(nF)]), Z["toe"]) - G(np.array([lg["pl"][t]["heel"] for t in range(nF)]), Z["heel"]); apl = np.arctan2(th[:, 1], th[:, 0])
    l3 = slice(max(0, nF - 3), nF); apm = np.arctan2(np.nanmean(np.sin(apl[l3])), np.nanmean(np.cos(apl[l3])))
    V = Gd(PmS[:, 12] - PmS[:, 11]); a2 = 2 * np.arctan2(V[:, 1], V[:, 0]); dhp = 2 * apl - (a2 + np.pi)            # doubled angle: pointing vs facing line, mod 180
    add("x_hip_plant_last", [np.cos(dhp[-1]), np.sin(dhp[-1])]); add("x_hip_plant_m3", [np.nanmean(np.cos(dhp[l3])), np.nanmean(np.sin(dhp[l3]))])
    hm = np.nanmean(PmS[:, [11, 12]], 1); D = Gd(hm - hm[-1]); near = lambda o: int(np.abs(ts + o).argmin())
    for nm, (a, b_) in (("08", (0.8, 0.3)), ("03", (0.3, 0.0))):
        v = D[near(b_)] - D[near(a)]; ang = np.arctan2(v[1], v[0]) if np.isfinite(v).all() and np.hypot(*v) > 1e-6 else np.nan
        add(f"x_plant_run{nm}_last", [np.cos(apl[-1] - ang), np.sin(apl[-1] - ang)]); add(f"x_plant_run{nm}_m3", [np.cos(apm - ang), np.sin(apm - ang)])
    return out


def one(c):
    b = load("ballspot", c); cam = CAM.get(c)
    if b is None or cam is None: return c, None, None
    res = {}
    for S in "AFG":
        r = []
        for sg in (1.0, -1.0):
            blocks = rhythm(c, b, cam, sg) + angles(c, S, b, cam, sg); r.append((np.concatenate([v for _, v in blocks]).astype(np.float32), [f"{n}{j}" if len(v) > 1 else n for n, v in blocks for j in range(len(v))]))
        res[S] = ((r[0][0], r[1][0]), r[0][1])
    return c, res, None


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPOOL", "8"))) as p: R = p.map(one, CLIPS, chunksize=20)
    sfx = "" if camf == "camera2" else "_" + camf
    for S in "AFG":
        rows = {c: r[S][0] for c, r, _ in R if r is not None}; names = next(r[S][1] for c, r, _ in R if r is not None)
        pickle.dump(dict(rows=rows, names=names), open(f"{PC}/kinX_{S}{sfx}.pkl", "wb")); X = np.stack([v[0] for v in rows.values()])
        print(f"kinX {S}{sfx}: built {len(rows)} of {len(CLIPS)} | dim {len(names)} | NaN frac {np.isnan(X).mean():.3f} | per feature NaN max {np.isnan(X).mean(0).max():.3f}", flush=True)
