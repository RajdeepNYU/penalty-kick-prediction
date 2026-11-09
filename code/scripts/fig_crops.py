"""Figure for the notebook's Section 14 (read-only on the caches; labelled public clips only, no test clip is shown or used).
The three last-frame crops the pixel model reads: body on the hips at 1.4 x the median kicker height, body at 1.4 x the torso-based height, legs at 1.2 x the
height on the knees. They are stabilised to the last frame and mirrored when needed so that the goal is on the right; the kicker re-pick crops (cropsfix_D) replace
the original ones where the kicker track changed, as in the model. Rows: the pixel model's most confident correct call of each kick class, then its most confident
miss of a middle kick. Calls are out of fold (source-grouped 5-fold, partition seeds 0-2, oof_px_cat3D.npy).
-> logs/fig/crops.png. usage: python fig_crops.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from cvlib import cl, yk
import crops5, crops8

OUT = f"{P}/logs/fig"; os.makedirs(OUT, exist_ok=True)
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK, "figure.facecolor": SURF, "savefig.facecolor": SURF})

C8, C5 = np.load(f"{PC}/crops8.npy", mmap_mode="r"), np.load(f"{PC}/crops5.npy", mmap_mode="r"); FX = np.load(f"{PC}/cropsfix_D.npz"); fx = {c: j for j, c in enumerate(FX["clips"])}
ix = {c: j for j, c in enumerate(CLIPS)}; meta = np.load(f"{PC}/zoom3_meta.npz"); assert list(meta["clips"]) == CLIPS; ok = meta["ok"][:len(gt)]
three = lambda c: FX["crops"][fx[c]] if c in fx else np.stack([C8[ix[c], crops8.VIEWS.index("body_hm")], C8[ix[c], crops8.VIEWS.index("body_tor")], C5[ix[c], crops5.VIEWS.index("legs12")]])
PX = np.load(f"{PC}/oof_px_cat3D.npy"); assert len(PX) == len(gt); call = cl[PX.argmax(1)]; y = np.asarray(yk, dtype=str); clips = np.asarray(gt["clip"])

rows = []
for k, c_ in enumerate(cl):
    m = ok & (y == c_) & (call == c_)
    if m.any(): rows.append((np.where(m)[0][PX[m, k].argmax()], "most confident correct call"))
m = ok & (y == "middle") & (call != "middle")
if m.any(): rows.append((np.where(m)[0][PX[m].max(1).argmax()], "most confident miss of a middle kick"))

COLS = ("body, 1.4 x height on the hips", "body, 1.4 x torso-based height", "legs, 1.2 x height on the knees")
fig, ax = plt.subplots(len(rows), 3, figsize=(7.2, 2.62 * len(rows)), gridspec_kw=dict(wspace=0.04, hspace=0.42))
for r, (j, why) in enumerate(rows):
    im = three(clips[j]); p = PX[j]
    for v in range(3):
        a = ax[r, v]; a.imshow(im[v]); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        if r == len(rows) - 1: a.set_xlabel(COLS[v], color=INK2)
    ax[r, 0].set_title(f"{clips[j]}   true: {y[j]}   pixel call: {call[j]}   (P left {p[0]:.2f}, middle {p[1]:.2f}, right {p[2]:.2f})   - {why}", loc="left", fontsize=8.5, color=INK)
    print(f"row {r}: {clips[j]} true {y[j]} call {call[j]} p {np.round(p, 3).tolist()} ({why})")
fig.subplots_adjust(top=0.95); fig.suptitle("What the pixel model reads: three crops of the last frame (goal always on the right)", x=0.125, ha="left", fontsize=10.5, y=0.985)
fig.savefig(f"{OUT}/crops.png", dpi=120, bbox_inches="tight"); print("saved", f"{OUT}/crops.png")
