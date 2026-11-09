"""Private-clip kick probabilities from trees fitted on all public clips (as Q4_work/make_v10_parts.py: HGB over 4 configs x 3 seeds averaged, ET x 3 seeds,
then (HGB + ET) / 2; mirror augmentation + mirror TTA). Features assembled exactly as in e_trees.py. usage: python trees_private.py TAG feat1 [feat2 ...]
-> cache/prv_trees_TAG.npy (pm order, columns left/middle/right)"""
import sys, pickle, numpy as np
from common import *
from cvlib import *

TAG, feats = sys.argv[1], sys.argv[2:]
base = dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
HC = [dict(), dict(max_depth=5, max_iter=200, min_samples_leaf=40), dict(max_depth=3, min_samples_leaf=50), dict(max_depth=3)]
parts = {"tr": [], "pr": []}
for f in feats:
    p = f"{W}/{f[2:]}.pkl" if f.startswith("W:") else f"{PC}/{f}.pkl"; K = pickle.load(open(p, "rb")); r = K["rows"]; dim = len(K["names"])
    if isinstance(next(iter(r.values()))[0], tuple): r = {c: (v[0][0], v[1][0]) for c, v in r.items()}
    parts["tr"].append(stack(r, gt["clip"], dim)); parts["pr"].append(stack(r, pm["clip"], dim))
Xa, Xb = (np.concatenate([q[j] for q in parts["tr"]], 1) for j in (0, 1)); Qa, Qb = (np.concatenate([q[j] for q in parts["pr"]], 1) for j in (0, 1))
keep = ~np.isnan(Xa).all(0); Xa, Xb, Qa, Qb = Xa[:, keep], Xb[:, keep], Qa[:, keep], Qb[:, keep]
Ph = sum(fit_predict(lambda s, c=c: HGB(random_state=s, **{**base, **c}), Xa, Xb, Qa, Qb) for c in HC) / len(HC)
Pe = fit_predict(et, Xa, Xb, Qa, Qb); P = (Ph + Pe) / 2
np.save(f"{PC}/prv_trees_{TAG}.npy", P); print(TAG, "private trees", P.shape, "pred dist", dict(zip(cl, np.bincount(P.argmax(1), minlength=3))))
