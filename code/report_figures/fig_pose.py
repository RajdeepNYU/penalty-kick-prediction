"""Perception figure for the Q4 report: read-only on the videos and the cached detections, tracks, poses and calibration; labelled public clips only.
Row 1: the last frame of 0449 with the tracked kicker and keeper, the goal posts, the foot-prior ball and the penalty spot projected by the NBJW calibration.
Row 2: the kicker's RTMW-x whole-body pose (with feet) at four times in the last second.
Row 3: one last frame (0886) read by four pose models; left joints blue, right joints orange: ViTPose++ large labels the legs the other way round.
-> logs/fig/pose.png. usage: python fig_pose.py"""
import os, pickle, cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import *
import streams

OUT = f"{P}/logs/fig"; os.makedirs(OUT, exist_ok=True)
INK, INK2, SURF, LEFT, RIGHT, MID = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834", "#ffffff"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 8, "text.color": INK, "figure.facecolor": SURF, "savefig.facecolor": SURF})
VID = {f: os.path.join(r, f) for r, _, fs in os.walk(D) for f in fs if f.endswith(".mp4") and "__MACOSX" not in r}
EL = [(5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (15, 17), (15, 18), (15, 19)]; ER = [(6, 8), (8, 10), (6, 12), (12, 14), (14, 16), (16, 20), (16, 21), (16, 22)]
EM = [(5, 6), (11, 12)]; JL, JR = [5, 7, 9, 11, 13, 15, 17, 18, 19], [6, 8, 10, 12, 14, 16, 20, 21, 22]


def frames(clip, want):
    cap = cv2.VideoCapture(VID[clip]); out, i = {}, 0
    while len(out) < len(want):
        ok, f = cap.read()
        if not ok: break
        if i in want: out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release(); return out


def skel(a, kp, sc, ox, oy, lw=1.6, ms=9):
    ok = lambda j: j < len(kp) and sc[j] > 0 and np.all(np.isfinite(kp[j])) and np.any(kp[j] != 0)
    for es, col in ((EL, LEFT), (ER, RIGHT), (EM, MID)):
        for i, j in es:
            if ok(i) and ok(j): a.plot([kp[i, 0] - ox, kp[j, 0] - ox], [kp[i, 1] - oy, kp[j, 1] - oy], color=col, lw=lw, solid_capstyle="round")
    for js, col in ((JL, LEFT), (JR, RIGHT)):
        p = np.array([kp[j] for j in js if ok(j)])
        if len(p): a.scatter(p[:, 0] - ox, p[:, 1] - oy, s=ms, color=col, edgecolors="white", linewidths=0.4, zorder=3)


def crop(img, box, k=1.5):
    cx, cy, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, (box[3] - box[1]) * k
    x0, y0 = int(max(cx - h / 2, 0)), int(max(cy - h / 2, 0)); x1, y1 = int(min(cx + h / 2, img.shape[1])), int(min(cy + h / 2, img.shape[0]))
    return img[y0:y1, x0:x1], x0, y0


def clean(a):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)


C1, C3 = "0449_public.mp4", "0886_public.mp4"
b1 = load_raw("ballspot", C1); F1 = list(b1["F"]); ts = np.asarray(b1["ts"]); A1 = streams._stream_kb(C1, "A")
pick_t = [int(np.argmin(np.abs(ts - (ts[-1] - d)))) for d in (0.8, 0.5, 0.25, 0.0)]
fr1 = frames(C1, set(F1[i] for i in pick_t) | {F1[-1]}); last1 = fr1[F1[-1]]
tr = pickle.load(open(f"{W}/tracks.pkl", "rb"))[C1]; ball = pickle.load(open(f"{PC}/ball2.pkl", "rb"))[C1]["ball"]
gp = pickle.load(open(f"{PC}/goalposts.pkl", "rb"))[C1]; nb = pickle.load(open(f"{PC}/nbjw/{C1}.pkl", "rb"))
spots = [nb["P"] @ np.array([x, 0.0, 0.0, 1.0]) for x in (-41.5, 41.5)]; spots = [s[:2] / s[2] for s in spots]
spot = min(spots, key=lambda s: np.linalg.norm(s - ball))

fig = plt.figure(figsize=(7.4, 8.2)); gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.95, 0.95], hspace=0.32, wspace=0.05)
a = fig.add_subplot(gs[0, :]); Y0, Y1 = 40, 470; a.imshow(last1[Y0:Y1], extent=(0, last1.shape[1], Y1, Y0)); clean(a)
for bx, col, lab in ((tr["kick"][-1], LEFT, "kicker (tracked)"), (tr["keep"][-1], RIGHT, "keeper (tracked)")):
    a.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=col, lw=1.6))
    a.text(bx[0], bx[1] - 6, lab, color="white", fontsize=7, bbox=dict(fc=col, ec="none", pad=1.2))
for t_, b_ in zip(gp["top"], gp["base"]): a.plot([t_[0], b_[0]], [t_[1], b_[1]], color="#f5d000", lw=2)
a.plot([gp["top"][0][0], gp["top"][1][0]], [gp["top"][0][1], gp["top"][1][1]], color="#f5d000", lw=2)
skel(a, A1["kp"][-1], A1["sc"][-1], 0, 0, lw=1.0, ms=3)
a.scatter(*ball, s=90, facecolors="none", edgecolors="white", linewidths=1.5, zorder=4); a.scatter(*spot, s=70, marker="x", color="#d6246e", linewidths=2, zorder=5)
a.legend(handles=[Line2D([], [], color="#f5d000", lw=2, label="goal posts (merged line segments)"),
                  Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, label="ball (YOLO11x with the foot prior)"),
                  Line2D([], [], marker="x", ls="", color="#d6246e", mew=2, label="penalty spot projected by the NBJW calibration")],
         loc="lower left", fontsize=6.5, framealpha=0.85)
a.set_title(f"{C1.split('_')[0]}, last frame (0.02 s before contact): who is who, goal, ball and penalty spot", fontsize=8.5, loc="left")
for k, i in enumerate(pick_t):
    a = fig.add_subplot(gs[1, k]); im, ox, oy = crop(fr1[F1[i]], A1["box"][i]); a.imshow(im); skel(a, A1["kp"][i], A1["sc"][i], ox, oy); clean(a)
    a.set_xlim(0, im.shape[1]); a.set_ylim(im.shape[0], 0); a.set_title(f"{ts[i] - ts[-1]:+.2f} s", fontsize=8)
row_head = lambda r, s: fig.text(0.125, fig.axes[-1].get_position().y1 + 0.032, s, fontsize=8.5)
row_head(1, f"{C1.split('_')[0]}: the kicker's RTMW-x whole-body pose (with heels and toes) over the last second")
b3 = load_raw("ballspot", C3); fr3 = frames(C3, {int(b3["F"][-1])})[int(b3["F"][-1])]
for k, (S, name) in enumerate((("A", "RTMW-x"), ("B", "RTMW3D-x (2D)"), ("F", "ViTPose++ large"), ("G", "ViTPose++ huge"))):
    kb = streams._stream_kb(C3, S); a = fig.add_subplot(gs[2, k]); im, ox, oy = crop(fr3, kb["box"][-1]); a.imshow(im)
    skel(a, kb["kp"][-1], kb["sc"][-1], ox, oy); clean(a); a.set_xlim(0, im.shape[1]); a.set_ylim(im.shape[0], 0); a.set_title(name, fontsize=8)
row_head(2, f"{C3.split('_')[0]}: one last frame, four pose models (left joints blue, right orange); ViTPose++ large swaps the legs")
fig.savefig(f"{OUT}/pose.png", dpi=150, bbox_inches="tight"); print("saved", f"{OUT}/pose.png", "| times", [round(ts[i] - ts[-1], 2) for i in pick_t], "| spot", np.round(spot, 1), "ball", np.round(ball, 1))
