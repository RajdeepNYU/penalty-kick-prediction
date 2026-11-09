"""Data figure for the Q4 report (template section 2.1). Read-only on the videos and caches; labelled public clips only, no test clip is shown or used.
Row 1: the last frame of four public clips that span the resolutions and frame rates in the data. Row 2: one clip before preprocessing (its last frame)
and after (the three stabilised, kicker-centred crops the pixel model reads, mirrored so the goal is on the right). Row 3: the public label distribution.
-> logs/fig/data.png. usage: python fig_data.py"""
import os, cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
import crops5, crops8

OUT = f"{P}/logs/fig"; os.makedirs(OUT, exist_ok=True)
INK, INK2, SURF, BAR = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF})
VID = {f: os.path.join(r, f) for r, _, fs in os.walk(D) for f in fs if f.endswith(".mp4") and "__MACOSX" not in r}


def last_frame(clip):
    cap = cv2.VideoCapture(VID[clip]); fr = None
    while True:
        ok, f = cap.read()
        if not ok: break
        fr = f
    cap.release(); return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


g = gt.copy(); g["fps_r"] = g.fps.round().astype(int)
pick = lambda m: g[m]["clip"].iloc[0]
K = g.kick_direction
ROW1 = [pick((g.resolution == "1280x720") & (g.fps_r == 25) & (K == "left")), pick((g.resolution == "1280x720") & (g.fps_r == 60) & (K == "right")),
        pick((g.resolution == "854x480") & (K == "middle")), pick(g.resolution == "640x360")]
EX = "0449_public.mp4"                                       # the right kick of the crops figure

C8, C5 = np.load(f"{PC}/crops8.npy", mmap_mode="r"), np.load(f"{PC}/crops5.npy", mmap_mode="r"); FX = np.load(f"{PC}/cropsfix_D.npz"); fx = {c: j for j, c in enumerate(FX["clips"])}
ix = {c: j for j, c in enumerate(CLIPS)}
three = lambda c: FX["crops"][fx[c]] if c in fx else np.stack([C8[ix[c], crops8.VIEWS.index("body_hm")], C8[ix[c], crops8.VIEWS.index("body_tor")], C5[ix[c], crops5.VIEWS.index("legs12")]])

fig = plt.figure(figsize=(7.4, 5.9)); gs = fig.add_gridspec(3, 4, height_ratios=[0.6, 1.0, 0.72], hspace=0.38, wspace=0.06)
row = lambda c: g[g["clip"] == c].iloc[0]
for k, c in enumerate(ROW1):
    a = fig.add_subplot(gs[0, k]); a.imshow(last_frame(c)); a.set_xticks([]); a.set_yticks([]); r = row(c)
    for s in a.spines.values(): s.set_visible(False)
    a.set_title(f"{c.split('_')[0]}: {r.resolution}, {r.fps:.0f} fps\nkick {r.kick_direction}, keeper {r.goalkeeper_direction.replace('stands in the middle', 'stands')}", fontsize=7.5, color=INK)
    print(f"row 1: {c} {r.resolution} {r.fps} kick {r.kick_direction} keeper {r.goalkeeper_direction}")
r = row(EX); a = fig.add_subplot(gs[1, 0]); a.imshow(last_frame(EX)); a.set_xticks([]); a.set_yticks([])
for s in a.spines.values(): s.set_visible(False)
a.set_title(f"before: {EX.split('_')[0]}, last frame\n{r.resolution}, {r.fps:.0f} fps, kick {r.kick_direction}", fontsize=7.5)
for v, (im, t) in enumerate(zip(three(EX), ("after: body on the hips", "after: torso-scaled body", "after: legs"))):
    a = fig.add_subplot(gs[1, v + 1]); a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=7.5)
    for s in a.spines.values(): s.set_visible(False)
for k, (col, names, lab) in enumerate((("kick_direction", ["left", "middle", "right"], "kick direction"),
                                       ("goalkeeper_direction", ["left", "stands in the middle", "right"], "keeper direction"))):
    a = fig.add_subplot(gs[2, 2 * k:2 * k + 2]); n = g[col].value_counts().reindex(names).values
    a.bar(range(3), n, color=BAR, width=0.6)
    for i, v in enumerate(n): a.text(i, v + 12, f"{v} ({v / len(g):.0%})", ha="center", fontsize=7.5, color=INK)
    a.set_xticks(range(3), [x.replace("stands in the middle", "stands") for x in names]); a.set_ylim(0, 760); a.set_title(f"{lab}, 1,300 public clips", fontsize=8)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.spines["left"].set_color("#c3c2b7"); a.spines["bottom"].set_color("#c3c2b7"); a.tick_params(length=0); a.set_yticks([])
    print(lab, dict(zip(names, n.tolist())))
fig.savefig(f"{OUT}/data.png", dpi=150, bbox_inches="tight"); print("saved", f"{OUT}/data.png")
