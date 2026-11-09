"""Figures for the notebook's Section 17 (read-only; out-of-fold predictions of the 1,300 labelled public clips, no test clip is used).
N6 out of fold on partition seeds 0-2 (source-grouped 5-fold), built exactly as final_assess.py builds its set "012" (the 12-clip camera correction is left out, as there).
-> logs/fig/confusion.png: kick and keeper confusion matrices (rows = true label, colour = share of the row).
-> logs/fig/confidence.png: kick accuracy by confidence fifth, and where the lost metric points go (0.7 per wrong kick, 0.3 per wrong keeper).
usage: python fig_results.py"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import *
from cvlib import cl, yk
from ovf_lib import oth, Pk, cfg

OUT = f"{P}/logs/fig"; os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, AXIS, SURF, BLUE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#2a78d6"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
                     "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

# ---------------- N6 out of fold, partition seeds 0-2 (final_assess.py, set 012) ----------------
norm_ = lambda Q: Q / Q.sum(1, keepdims=True); geo = lambda x, y: norm_(np.exp(.5 * np.log(x + 1e-4) + .5 * np.log(y + 1e-4)))
tree1 = lambda S: (np.load(f"{PC}/oof_hgb_{S}_b2R.npy") + np.load(f"{PC}/oof_et_{S}_b2R.npy")) / 2
NN = sum(np.load(f"{PC}/oofP_{e}_{h}_st{st}_{q}.npy") for e, h in cfg for st in ("EF", "G") for q in ("b2", "b2RM")) / 20
BASE = np.where(oth[:, None], Pk, .5 * sum(tree1(S) for S in "AFG") / 3 + .5 * NN); PX = np.load(f"{PC}/oof_px_cat3D.npy"); QK = geo(BASE, PX)
cnn = np.load(f"{W}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]; nnk = np.load(f"{PC}/keep_oof_NNk.npy"); km = lambda S, b: np.load(f"{PC}/nov6/keep_oof_{S}_{b}D_012.npy")
QG = (2 * km("A", "b2") + 2 * km("F", "b2") + km("A", "G") + km("F", "G") + cnn + nnk) / 8
cg = np.array(["left", "right", "stands in the middle"]); y = np.asarray(yk, dtype=str); yg = np.asarray(gt["goalkeeper_direction"].values, dtype=str)
pk, pg = cl[QK.argmax(1)], cg[QG.argmax(1)]; okk, okg = pk == y, pg == yg; N = len(y)
print(f"N6 out of fold, partition seeds 0-2: kick {okk.sum()} / {N} ({okk.mean():.4f}), keeper {okg.sum()} / {N} ({okg.mean():.4f}), metric {(.7 * okk + .3 * okg).mean():.4f}")


# ---------------- confusion matrices ----------------
def confusion(a, yt, yp, classes, names, title):
    M = np.array([[np.sum((yt == t) & (yp == p)) for p in classes] for t in classes]); S = M / M.sum(1, keepdims=True)
    a.pcolormesh(S, cmap=SEQ, vmin=0, vmax=1, edgecolors=SURF, linewidth=2); a.invert_yaxis(); a.set_aspect("equal")
    for i in range(len(classes)):
        for j in range(len(classes)):
            a.text(j + .5, i + .5, f"{M[i, j]}\n{S[i, j]:.0%}", ha="center", va="center", fontsize=9, color="#ffffff" if S[i, j] > .5 else INK)
    a.set_xticks(np.arange(len(classes)) + .5, names); a.set_yticks(np.arange(len(classes)) + .5, [f"{n} (n {int(M[i].sum())})" for i, n in enumerate(names)])
    a.set_xlabel("predicted"); a.set_ylabel("true label"); a.tick_params(length=0)
    for s in a.spines.values(): s.set_visible(False)
    a.set_title(title, loc="left", fontsize=10)
    return M


fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw=dict(wspace=0.55))
Mk = confusion(ax[0], y, pk, cl, ["left", "middle", "right"], f"Kick direction: {okk.sum()} of {N} right ({okk.mean():.1%})")
Mg = confusion(ax[1], yg, pg, cg, ["left", "right", "stands"], f"Keeper direction: {okg.sum()} of {N} right ({okg.mean():.1%})")
fig.suptitle("N6 out of fold (source-grouped 5-fold, partition seeds 0-2): rows are true labels, colour is the share of the row", x=0.06, ha="left", fontsize=10.5, y=1.0)
fig.savefig(f"{OUT}/confusion.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("kick confusion (rows true left/middle/right, cols predicted):", Mk.tolist(), "| keeper (left/right/stands):", Mg.tolist())

# ---------------- accuracy by confidence, and where the points go ----------------
conf = QK.max(1); q = np.quantile(conf, [0, .2, .4, .6, .8, 1]); b = np.clip(np.searchsorted(q, conf, side="right") - 1, 0, 4)
acc = np.array([okk[b == i].mean() for i in range(5)]); lab = [f"{q[i]:.2f}-{q[i + 1]:.2f}" for i in range(5)]
agree = cl[BASE.argmax(1)] == cl[PX.argmax(1)]; side = y != "middle"
lost = {"missed middle kick": .7 * np.sum(~side & (pk != "middle")),
        "wrong side,\npose and pixel agree": .7 * np.sum(side & (pk != "middle") & ~okk & agree),
        "wrong side,\npose and pixel disagree": .7 * np.sum(side & (pk != "middle") & ~okk & ~agree),
        "wrong keeper call": .3 * np.sum(~okg),
        "false middle call": .7 * np.sum(side & (pk == "middle"))}
tot = sum(lost.values()); sh = {k: v / tot for k, v in sorted(lost.items(), key=lambda t: -t[1])}

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.1]))
a = ax[0]; xs = np.arange(5); a.bar(xs, acc, width=.62, color=BLUE, zorder=2)
for i, v in enumerate(acc): a.text(i, v + .015, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
a.axhline(okk.mean(), color=MUTED, linewidth=1, zorder=1); a.text(-.42, okk.mean() + .012, f"all clips {okk.mean():.2f}", ha="left", va="bottom", fontsize=8.5, color=INK2)
a.set_xticks(xs, lab, fontsize=8.5); a.set_xlabel("kick confidence (max probability), fifths of the 1,300 clips"); a.set_ylabel("kick accuracy"); a.set_ylim(0, 1.05)
a.yaxis.grid(True, color=GRID, linewidth=.8, zorder=0); a.set_axisbelow(True); a.tick_params(length=0)
for s in ("top", "right", "left"): a.spines[s].set_visible(False)
a.set_title("Confidence carries information", loc="left", fontsize=10)
a = ax[1]; ks = list(sh)[::-1]; vs = [sh[k] for k in ks]; a.barh(np.arange(len(ks)), vs, height=.6, color=BLUE, zorder=2)
for i, v in enumerate(vs): a.text(v + .008, i, f"{v:.0%}", va="center", fontsize=9, color=INK)
a.set_yticks(np.arange(len(ks)), ks, fontsize=8.8); a.set_xlim(0, max(vs) * 1.18); a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
a.xaxis.grid(True, color=GRID, linewidth=.8, zorder=0); a.set_axisbelow(True); a.tick_params(length=0); a.set_xlabel(f"share of the {tot:.0f} lost metric points (0.7 per wrong kick, 0.3 per wrong keeper)")
for s in ("top", "right", "bottom"): a.spines[s].set_visible(False)
a.set_title("Where the points are lost", loc="left", fontsize=10)
fig.savefig(f"{OUT}/confidence.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("kick accuracy by confidence fifth:", " | ".join(f"{l} {v:.3f}" for l, v in zip(lab, acc)))
print("lost points:", " | ".join(f"{k.replace(chr(10), ' ')} {v:.1%}" for k, v in sh.items()), f"| total {tot:.1f} of {N}")
print("saved", f"{OUT}/confusion.png", f"{OUT}/confidence.png")
