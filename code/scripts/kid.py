"""Kicker identity toolkit (label-free; same rules on public and private clips).
geo(c): every detected person on the 13 timed frames in stabilised last-frame image coordinates (ballspot homographies, interpolated in time to the det
frames) and, when the NBJW calibration is usable, on the ground plane in metres, plus the projected penalty spot and ball2.
gtrack(): motion-gated tracker - constant-velocity prediction in stabilised coordinates, a speed gate that grows with the time step (a person cannot move
more than VMAX kicker heights per second), gaps interpolated instead of jumping to whoever is closest.
approach(): the user's cue - how the person moves relative to the ball: approach speed (m/s toward the spot), radial acceleration, net distance gained.
evaluate(): the harness - kicker at the anchor vs the clicks, kicker at the spot at the last frame, implausible jumps, approach of the tracked kicker."""
import numpy as np, pandas as pd
from common import *

VMAX, SLACK, MINC = 5.0, 0.3, 0.1                # kicker heights per second (about 9 m/s), gate slack in heights, det confidence floor (as track5)
ANCHOR = 6
B2 = pickle.load(open(f"{PC}/ball2.pkl", "rb")); KCK = pickle.load(open(f"{PC}/kickcheck.pkl", "rb"))
CK = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip")


def _norm(M): M = np.asarray(M, float); return M / M[2, 2]


def homs(d, b):
    """frame -> last-frame homography for each det frame: ballspot M at its own frame times, linear in time in between"""
    ts, Ms = np.asarray(b["ts"], float), [_norm(m) for m in b["M"]]; out = []
    for t in d["ts"]:
        k = int(np.searchsorted(ts, t - 1e-6))
        if k < len(ts) and abs(ts[k] - t) < 1e-3: out.append(Ms[k]); continue
        k0, k1 = max(k - 1, 0), min(k, len(ts) - 1); w = 0.0 if k1 == k0 else (t - ts[k0]) / (ts[k1] - ts[k0]); out.append(_norm((1 - w) * Ms[k0] + w * Ms[k1]))
    return out


def wbox(M, bb):
    """axis-aligned box of the homography-mapped corners"""
    bb = np.atleast_2d(np.asarray(bb, float)); q = np.stack([bb[:, [0, 1]], bb[:, [2, 1]], bb[:, [0, 3]], bb[:, [2, 3]]], 1)
    w = warp(M, q.reshape(-1, 2)).reshape(-1, 4, 2); return np.c_[w[..., 0].min(1), w[..., 1].min(1), w[..., 0].max(1), w[..., 1].max(1)]


_G = {}
def geo(c):
    if c in _G: return _G[c]
    d, b = load("det", c), load("ballspot", c)
    if d is None or b is None: _G[c] = None; return None
    Hs = homs(d, b); g = dict(d=d, H=Hs, ts=np.asarray(d["ts"], float), wh=d["wh"], spot_px=None, ball=np.asarray(B2[c]["ball"], float) if c in B2 else None, gr=None)
    g["P"] = [np.asarray(f["pb"], float)[np.asarray(f["pc"]) >= MINC] for f in d["frames"]]
    g["S"] = [wbox(Hs[j], p) if len(p) else np.zeros((0, 4)) for j, p in enumerate(g["P"])]       # stabilised boxes
    fp = f"{PC}/nbjw/{c}.pkl"; k = KCK.get(c)
    if os.path.exists(fp) and k is not None:
        r = pickle.load(open(fp, "rb"))
        if r is not None and r["rep_err"] < 10:
            Pl = np.asarray(b["M"][r["frame"]], float) @ r["P"] if r["frame"] != len(b["F"]) - 1 else r["P"]; H0 = np.linalg.inv(np.c_[Pl[:, 0], Pl[:, 1], Pl[:, 3]])
            g["gr"] = lambda X, H0=H0: (lambda q: q[..., :2] / q[..., 2:3])(np.c_[np.atleast_2d(X), np.ones(len(np.atleast_2d(X)))] @ H0.T)
            g["spot"] = np.array([41.5 * k["s"], 0.0]); q = Pl @ np.array([41.5 * k["s"], 0, 0, 1.0]); g["spot_px"] = q[:2] / q[2]; g["rep_err"] = r["rep_err"]
    _G[c] = g; return g


def foot(bb): bb = np.atleast_2d(bb); return np.c_[(bb[:, 0] + bb[:, 2]) / 2, bb[:, 3]]


def iou_many(a, B):
    B = np.atleast_2d(B); ix = np.maximum(0, np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0])); iy = np.maximum(0, np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]))
    i = ix * iy; return i / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i, 1e-6)


def augment(g, extra, conf=0.05):
    """geo with extra per-frame person boxes [(boxes, conf)] (TTA detections) added where they don't duplicate a det box"""
    g2 = dict(g); g2["P"] = []; g2["S"] = []
    for j, (bb, cf) in enumerate(extra):
        P_ = g["P"][j]; add = [b for b, s in zip(np.asarray(bb, float), cf) if s >= conf and (not len(P_) or iou_many(b, P_).max() < 0.5)]
        P2 = np.r_[P_.reshape(-1, 4), np.asarray(add).reshape(-1, 4)]; g2["P"].append(P2); g2["S"].append(wbox(g["H"][j], P2) if len(P2) else np.zeros((0, 4)))
    return g2


def calib_ok(g):
    """the spot projection is usable: calibrated (reprojection < 10 px) and on screen"""
    sp = g["spot_px"]; return sp is not None and 0 <= sp[0] <= g["wh"][0] and 0 <= sp[1] <= g["wh"][1]


def target(g, h=None):
    """the ball's stabilised position: the projected spot when the calibration is usable (ball2 leans on the tracked kicker's feet, so it follows a
    wrongly tracked kicker), else ball2"""
    return g["spot_px"] if calib_ok(g) else g["ball"]


def gtrack(g, a0, start, vmax=2.0, slack=0.6, damp=0.0, tgt=None, mono=None):
    """follow the det box `start` (frame coordinates, frame a0) forward and backward. Returns (13, 4) frame-coordinate boxes and a matched mask;
    unmatched frames are interpolated (or held at the nearest match at the ends). mono (kicker heights): the user's motion cue as a gate - before contact
    the kicker does not move away from the ball, so forward in time a match may not end farther from `tgt` than the previous one by more than mono
    (backward: not nearer by more than mono)."""
    n = len(g["P"]); S0 = wbox(g["H"][a0], start)[0]; out = np.full((n, 4), np.nan); outS = np.full((n, 4), np.nan); m = np.zeros(n, bool)
    out[a0], outS[a0], m[a0] = start, S0, True
    for step in (1, -1):
        hist = [(g["ts"][a0], S0)]; j = a0 + step
        while 0 <= j < n:
            tl, bl = hist[-1]; h = max(bl[3] - bl[1], 1.0); dt = abs(g["ts"][j] - tl)
            v = (foot(bl)[0] - foot(hist[-2][1])[0]) / max(abs(tl - hist[-2][0]), 1e-3) if len(hist) > 1 else np.zeros(2)
            pred = foot(bl)[0] + damp * v * dt                                  # damped constant-velocity prediction
            Sj = g["S"][j]
            if len(Sj):
                dist = np.linalg.norm(foot(Sj) - pred, axis=1) / h; hr = np.abs(np.log(np.maximum((Sj[:, 3] - Sj[:, 1]) / h, 1e-3)))
                ok = (dist <= vmax * dt + slack) & (hr < 0.5); cst = dist + 1.5 * hr
                if mono is not None and tgt is not None:
                    r_new = np.linalg.norm(foot(Sj) - tgt, axis=1) / h; r_prev = np.linalg.norm(foot(bl)[0] - tgt) / h
                    ok &= (r_new <= r_prev + mono) if step == 1 else (r_new >= r_prev - mono)
                if ok.any():
                    i = np.where(ok)[0][cst[ok].argmin()]; out[j], outS[j], m[j] = g["P"][j][i], Sj[i], True; hist.append((g["ts"][j], Sj[i]))
            j += step
    mi = np.where(m)[0]
    for j in np.where(~m)[0]:                                                  # interpolate in stabilised coordinates, map back to the frame
        lo, hi = mi[mi < j], mi[mi > j]
        if len(lo) and len(hi):
            a, b_ = lo[-1], hi[0]; w = (g["ts"][j] - g["ts"][a]) / (g["ts"][b_] - g["ts"][a]); s = (1 - w) * outS[a] + w * outS[b_]
        else: s = outS[lo[-1]] if len(lo) else outS[hi[0]]
        outS[j] = s; out[j] = wbox(np.linalg.inv(g["H"][j]), s)[0]
    return out, m


def stab(g, boxes): return np.array([wbox(g["H"][j], b)[0] if np.all(np.isfinite(b)) else np.full(4, np.nan) for j, b in enumerate(boxes)])


def approach(g, boxes, use=slice(0, 11)):
    """motion relative to the ball. Ground plane (m, m/s) when calibrated, else stabilised image in kicker heights. use: frames for the rates (the
    last two are the plant). Returns dict(v, a, gain, r_end, units)"""
    S = stab(g, boxes); ok = np.isfinite(S[:, 0])
    if ok.sum() < 4: return dict(v=np.nan, a=np.nan, gain=np.nan, r_end=np.nan, units="none")
    if g["gr"] is not None:
        X = np.full((len(S), 2), np.nan); X[ok] = g["gr"](foot(S[ok])); tgt = g["spot"]; units = "m"
    else:
        h = np.nanmedian(S[:, 3] - S[:, 1]); X = foot(S) / h; tgt = (g["ball"] if g["ball"] is not None else foot(S[ok][-1:])[0]) / h; units = "h"
    r = np.linalg.norm(X - tgt, axis=1); t = g["ts"]; idx = np.arange(len(r))[use]; idx = idx[np.isfinite(r[idx])]
    if len(idx) < 4: return dict(v=np.nan, a=np.nan, gain=np.nan, r_end=float(r[ok][-1]), units=units)
    v = float(-np.polyfit(t[idx], r[idx], 1)[0]); a = float(-2 * np.polyfit(t[idx], r[idx], 2)[0])     # approach speed, approach acceleration
    return dict(v=v, a=a, gain=float(r[idx[0]] - r[idx[-1]]), r_end=float(r[ok][-1]), units=units)


def inb(b, p): return b is not None and np.all(np.isfinite(b)) and b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def click_px(c, g): q = CK.loc[c]; W_, H_ = g["wh"]; return np.array([q.kicker_x_norm * W_, q.kicker_y_norm * H_])


def spot_m(g, box):
    if g is None or g["gr"] is None or not np.all(np.isfinite(box)): return np.nan
    return float(np.linalg.norm(g["gr"](foot(wbox(g["H"][-1], box)))[0] - g["spot"]))


def jumps(g, boxes):
    """largest step in stabilised foot position relative to what the speed gate allows (> 1 = physically implausible)"""
    S = stab(g, boxes); ok = np.where(np.isfinite(S[:, 0]))[0]
    if len(ok) < 2: return np.nan
    F_ = foot(S[ok]); h = np.maximum(S[ok][:, 3] - S[ok][:, 1], 1); dt = np.diff(g["ts"][ok])
    return float(np.max(np.linalg.norm(np.diff(F_, axis=0), axis=1) / h[:-1] / (VMAX * dt + SLACK)))


def evaluate(T, clips=None, name=""):
    """T: {clip: dict(kick=(13,4), kick_a=(4,))}. Per-clip table + printed summary."""
    rows = []
    for c in clips or CLIPS:
        t, g = T.get(c), geo(c)
        if t is None or g is None: continue
        kb = np.asarray(t["kick"], float); r = dict(clip=c, split=SPLIT[c])
        if c in CK.index: r["anchor_ok"] = float(inb(np.asarray(t["kick_a"], float), click_px(c, g)))
        last = kb[np.where(np.isfinite(kb[:, 0]))[0][-1]] if np.isfinite(kb[:, 0]).any() else np.full(4, np.nan)
        r["spot_m"] = spot_m(g, last); r["jump"] = jumps(g, kb); ap = approach(g, kb); r.update({f"ap_{k}": v for k, v in ap.items()})
        rows.append(r)
    E = pd.DataFrame(rows).set_index("clip"); pub = E.split == "train"
    sm = E.spot_m
    if name:
        a = E.anchor_ok.dropna() if "anchor_ok" in E else pd.Series(dtype=float)
        both = E.loc[a.index].pipe(lambda e: (e.anchor_ok == 1) & (e.spot_m < 1.5))
        print(f"{name:28s} anchor vs click {a.mean():.4f} ({int((a == 0).sum())} wrong) | spot<1.5m pub {np.mean(sm[pub] < 1.5):.4f} prv {np.mean(sm[~pub] < 1.5):.4f} | "
              f">6m pub {int((sm[pub] > 6).sum())} prv {int((sm[~pub] > 6).sum())} | clicked: anchor ok & spot ok {both.mean():.4f} | implausible jump {int((E.jump > 1).sum())}")
    return E
