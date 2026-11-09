import numpy as np, sys
sys.argv = ["x"]
from cv import *
cs = list(np.r_[0:90]) + [261]
a, oof, cls = cv(X[:, cs], Xm[:, cs], "kick", seeds=(0, 1, 2))
a0, oof0, _ = cv(X[:, [261]], Xm[:, [261]], "kick", seeds=(0, 1, 2))
y = gt["kick_direction"].values; pred = np.array(cls)[oof.argmax(1)]; pred0 = np.array(cls)[oof0.argmax(1)]
fx = X[:, 259]; fy = X[:, 260]
print("overall pose %.3f  foot-only %.3f" % (a, a0))
for nm, q in (("|fx| (side-on-ness)", fx), ("fy (goal further(-)/nearer(+))", fy)):
    qs = np.quantile(q[np.isfinite(q)], [0, 1 / 3, 2 / 3, 1])
    for i in range(3):
        m = (q >= qs[i]) & (q <= qs[i + 1])
        print(f"{nm:32s} tercile {i} [{qs[i]:+.2f},{qs[i + 1]:+.2f}] n={m.sum()}  pose {np.mean(pred[m] == y[m]):.3f}  foot-only {np.mean(pred0[m] == y[m]):.3f}  majority-left {np.mean(y[m] == 'left'):.3f}")
conf = oof.max(1)
for lo, hi in ((0, .5), (.5, .6), (.6, .7), (.7, 1)):
    m = (conf >= lo) & (conf < hi); print(f"model confidence [{lo},{hi}) n={m.sum()} acc {np.mean(pred[m] == y[m]) if m.any() else float('nan'):.3f}")
