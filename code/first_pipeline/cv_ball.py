import numpy as np, pickle, sys
sys.argv = ["x"]

from cv import *
bf = pickle.load(open(f"{C}/ballfeats.pkl", "rb"))
Bx = np.stack([bf[c][0] for c in gt["clip"]]); Bm = np.stack([bf[c][1] for c in gt["clip"]])
foot = [261]; KP = list(np.r_[0:90])
print("ball feats: frac clips with any ball in last frames:", np.mean(Bx[:, -5] > 0).round(2), " with >=2 detections:", np.mean(Bx[:, 27] >= 2).round(2))
cfgs = {
    "foot only": (X[:, foot], Xm[:, foot]),
    "ball + foot": (np.c_[Bx, X[:, foot]], np.c_[Bm, Xm[:, foot]]),
    "ball + kpose + foot": (np.c_[Bx, X[:, KP + foot]], np.c_[Bm, Xm[:, KP + foot]]),
    "ball + all feats": (np.c_[Bx, X], np.c_[Bm, Xm]),
}
for nm, (a, b) in cfgs.items():
    k, oof, cls = cv(a, b, "kick", seeds=(0, 1))
    print(f"{nm:24s} kick {k:.3f}", flush=True)
    if nm == "ball + kpose + foot":
        y = gt["kick_direction"].values; pred = np.array(cls)[oof.argmax(1)]; has = Bx[:, 27] >= 2
        print("   by ball availability: with>=2 detections n=%d acc %.3f | without n=%d acc %.3f" % (has.sum(), (pred[has] == y[has]).mean(), (~has).sum(), (pred[~has] == y[~has]).mean()))
        conf = oof.max(1)
        for lo, hi in ((0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)):
            m = (conf >= lo) & (conf < hi); print(f"   confidence [{lo},{hi}) n={m.sum()} acc {(pred[m] == y[m]).mean() if m.any() else float('nan'):.3f}")
