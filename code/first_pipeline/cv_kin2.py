import numpy as np, warnings
warnings.filterwarnings("ignore")
from cv_kin import *
tilt = blk("x_"); base = np.setdiff1d(np.arange(X.shape[1]), tilt); print("tilt feats", len(tilt), "| base feats", len(base))
def rep(nm, cols, seeds=(0, 1, 2)):
    Pr, cl = cv_clf(X[:, cols], Xm[:, cols], seeds); pred = np.array(cl)[Pr.argmax(1)]; lr = yk != "middle"
    print(f"{nm:38s} kick {(pred == yk).mean():.3f} | L/R-only {(pred[lr] == yk[lr]).mean():.3f} | middle recall {(pred[~lr] == 'middle').mean():.2f}", flush=True); return Pr
rep("base kinematic (153)", base)
rep("tilt only + foot sign", np.r_[tilt, FOOT])
Pall = rep("base + tilt (all 259)", np.arange(X.shape[1]))
np.save(f"{C}/oof_kin_all.npy", Pall)
