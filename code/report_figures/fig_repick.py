"""Kicker check figure for the Q4 report: read-only on the videos and on kickcheck.pkl (the label-free spot check) and the NBJW calibration; public clips only.
Three public clips where the tracked "kicker" ended far from the projected penalty spot: the old track, the re-picked person at the spot, and the spot.
-> logs/fig/repick.png. usage: python fig_repick.py"""
import os, pickle, cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import *

OUT = f"{P}/logs/fig"; os.makedirs(OUT, exist_ok=True)
INK, SURF, OLD, NEW, SPOT = "#0b0b0b", "#fcfcfb", "#eb6834", "#2a78d6", "#d6246e"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 8, "text.color": INK, "figure.facecolor": SURF, "savefig.facecolor": SURF})
VID = {f: os.path.join(r, f) for r, _, fs in os.walk(D) for f in fs if f.endswith(".mp4") and "__MACOSX" not in r}
KC = pickle.load(open(f"{PC}/kickcheck.pkl", "rb")); pub = set(gt["clip"])
cand = sorted([(v["d_track"], c) for c, v in KC.items() if c in pub and v["d_track"] > 6 and v["d_best"] < 2 and os.path.exists(f"{PC}/nbjw/{c}.pkl")], reverse=True)
print("flagged public clips:", len(cand), [(c, round(float(d), 1)) for d, c in cand[:6]])
AR = 1.25                                                   # every panel shows a window of this width / height


def window(c):
    v = KC[c]; nb = pickle.load(open(f"{PC}/nbjw/{c}.pkl", "rb")); tb, bb = np.asarray(v["track_box"]), np.asarray(v["best_box"])
    foot = np.array([(bb[0] + bb[2]) / 2, bb[3]]); sp = [nb["P"] @ np.array([x, 0.0, 0.0, 1.0]) for x in (-41.5, 41.5)]; sp = min((s[:2] / s[2] for s in sp), key=lambda s: np.linalg.norm(s - foot))
    xs = np.r_[tb[[0, 2]], bb[[0, 2]], sp[0]]; ys = np.r_[tb[[1, 3]], bb[[1, 3]], sp[1]]; h = bb[3] - bb[1]
    S = max(np.ptp(xs) + 1.6 * h, (np.ptp(ys) + 1.0 * h) * AR); cx, cy = xs.mean(), ys.mean()
    return tb, bb, sp, (cx - S / 2, cx + S / 2, cy - S / AR / 2, cy + S / AR / 2), S / h


CL = [c for _, c in sorted((window(c)[4], c) for _, c in cand)[:3]]


def last_frame(clip):
    cap = cv2.VideoCapture(VID[clip]); fr = None
    while True:
        ok, f = cap.read()
        if not ok: break
        fr = f
    cap.release(); return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


fig, ax = plt.subplots(1, len(CL), figsize=(7.4, 2.35), gridspec_kw=dict(wspace=0.05))
for a, c in zip(np.atleast_1d(ax), CL):
    v = KC[c]; im = last_frame(c); tb, bb, sp, (x0, x1, y0, y1), _ = window(c)
    a.imshow(im); a.set_xlim(x0, x1); a.set_ylim(y1, y0); a.set_xticks([]); a.set_yticks([]); a.set_facecolor("#e9e8e3")
    for s in a.spines.values(): s.set_visible(False)
    for b, col, ls in ((tb, OLD, "--"), (bb, NEW, "-")): a.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=1.8, ls=ls))
    a.scatter(*sp, s=60, marker="x", color=SPOT, linewidths=2, zorder=5)
    a.set_title(f"{c.split('_')[0]}: track {v['d_track']:.0f} m from the spot", fontsize=8)
fig.legend(handles=[Line2D([], [], color=OLD, ls="--", lw=1.8, label="tracked \"kicker\" (before)"), Line2D([], [], color=NEW, lw=1.8, label="re-picked: the person at the spot"),
                    Line2D([], [], marker="x", ls="", color=SPOT, mew=2, label="penalty spot (NBJW calibration)")], loc="upper center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.1))
fig.savefig(f"{OUT}/repick.png", dpi=150, bbox_inches="tight"); print("saved", f"{OUT}/repick.png", CL)
