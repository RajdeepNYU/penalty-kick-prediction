"""Is the tracked kicker the player at the ball? At contact - 0.02 s the kicker's feet are at the penalty spot. With the NBJW field calibration
(reprojection error < 10 px, independent of ball2 and of the tracks), measure on the ground plane the distance from the tracked kicker's last-frame box
(bottom centre) to the projected spot, and find the detected person (det cache, last F frame) nearest to the spot. Label-free audit; P4 kick accuracy
by group is printed for the public clips. -> cache/kickcheck.pkl {clip: dict(d_track, d_best, best_box, iou, s)}"""
import numpy as np
from common import *
from cvlib import cl, yk
from blend_eval import trP, bagP


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-6)


out = {}
for c in CLIPS:
    b = load("ballspot", c); dt = load("det", c); t = tracks()[c]; fp = f"{PC}/nbjw/{c}.pkl"
    if b is None or dt is None or t is None or not os.path.exists(fp): continue
    r = pickle.load(open(fp, "rb"))
    if r is None or r["rep_err"] >= 10: continue
    Pl = np.asarray(b["M"][r["frame"]], float) @ r["P"] if r["frame"] != len(b["F"]) - 1 else r["P"]; W, H = dt["wh"]
    H0 = np.linalg.inv(np.c_[Pl[:, 0], Pl[:, 1], Pl[:, 3]]); gr = lambda X: (lambda g: g[:2] / g[2])(H0 @ np.r_[X, 1.0])
    pj = lambda x, y: (lambda q: q[:2] / q[2])(Pl @ np.array([x, y, 0.0, 1.0]))
    cand = [(np.hypot(*(pj(52.5 * s, 0) - [W / 2, H / 2])), s) for s in (1.0, -1.0)]; s = min(cand)[1]; spot = np.array([41.5 * s, 0.0])
    kb = t["kick"]; vk = np.where(np.isfinite(kb[:, 0]))[0]
    if len(vk) == 0: continue
    tb = kb[vk[-1]]; d_track = float(np.hypot(*(gr([(tb[0] + tb[2]) / 2, tb[3]]) - spot)))
    pb = np.asarray(dt["frames"][-1]["pb"], float); best = (np.inf, None)
    for q in pb:
        dd = float(np.hypot(*(gr([(q[0] + q[2]) / 2, q[3]]) - spot)))
        if dd < best[0]: best = (dd, q)
    out[c] = dict(d_track=d_track, d_best=best[0], best_box=best[1], iou=iou(tb, best[1]) if best[1] is not None else 0.0, s=s, track_box=tb)
pickle.dump(out, open(f"{PC}/kickcheck.pkl", "wb"))
nT = len(gt); B = 0.5 * sum(trP(f"{S}_b2R") for S in "AFG") / 3 + 0.5 * sum(bagP(st, s) for st in ("EF", "G") for s in ("b2", "b2RM")) / 4; ok = cl[B.argmax(1)] == yk
dtr = np.array([out[c]["d_track"] if c in out else np.nan for c in CLIPS]); dbe = np.array([out[c]["d_best"] if c in out else np.nan for c in CLIPS])
io = np.array([out[c]["iou"] if c in out else np.nan for c in CLIPS])
print(f"clips with a usable NBJW calibration: {np.isfinite(dtr).sum()} of {len(CLIPS)} | tracked kicker to spot (m): median {np.nanmedian(dtr):.2f}, p90 {np.nanpercentile(dtr, 90):.2f}, p95 {np.nanpercentile(dtr, 95):.2f}")
for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, 999)):
    m = (dtr >= lo) & (dtr < hi); mp = m[:nT]
    print(f"tracked kicker {lo}-{hi} m from the spot: public {mp.sum()} (P4 kick acc {ok[mp].mean() if mp.any() else float('nan'):.3f}) | private {m[nT:].sum()}"
          f" | another detected person within 1.5 m of the spot and not the tracked one: public {int((mp & (dbe[:nT] < 1.5) & (io[:nT] < 0.3)).sum())}, private {int((m[nT:] & (dbe[nT:] < 1.5) & (io[nT:] < 0.3)).sum())}")
