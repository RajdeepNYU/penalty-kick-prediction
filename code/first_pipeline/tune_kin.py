import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import cv_kin
from cv_kin import *
t0 = time.time(); allc = np.arange(X.shape[1]); res = {}
base = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
cfgs = {"d3 it250 (base)": {}, "d2 it400": dict(max_depth=2, max_iter=400), "d4 it250": dict(max_depth=4), "d3 it500": dict(max_iter=500), "d3 it150": dict(max_iter=150),
        "d3 leaf50": dict(min_samples_leaf=50), "d3 mf0.3": dict(max_features=0.3), "d3 l2=10": dict(l2_regularization=10.0), "d5 it200 leaf40": dict(max_depth=5, max_iter=200, min_samples_leaf=40),
        "d3 it400 lr.02 mf.3": dict(max_iter=400, learning_rate=0.02, max_features=0.3)}
for nm, cfg in cfgs.items():
    cv_kin.P = {**base, **cfg}
    Pr, cl = cv_clf(X, Xm, seeds=(0, 1, 2)); pred = np.array(cl)[Pr.argmax(1)]; lr = yk != "middle"; res[nm] = Pr
    print(f"{nm:24s} kick {(pred == yk).mean():.3f} | L/R-only {(pred[lr] == yk[lr]).mean():.3f}   [{time.time() - t0:.0f}s]", flush=True)
top = sorted(res, key=lambda k: -(np.array(["left", "middle", "right"])[res[k].argmax(1)] == yk).mean())[:3]
Pb = sum(res[k] for k in top) / 3; pred = np.array(["left", "middle", "right"])[Pb.argmax(1)]
print("bag of top-3 configs", top, "kick %.3f" % (pred == yk).mean())
np.save(f"{C}/oof_kin_bag.npy", Pb)
