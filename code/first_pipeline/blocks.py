import numpy as np, sys
sys.argv = ["x"]
from cv import *
# blocks within the 268-dim vector (see feats.py order)
B = {"kpose_static": np.r_[0:90], "kpose_lines": np.r_[90:140], "k_velocity": np.r_[140:146], "ball_rel": np.r_[146:155],
     "keeper_motion": np.r_[155:171], "keeper_pose": np.r_[171:225], "keeper_lines": np.r_[225:255], "rel_geom": np.r_[255:261], "meta": np.r_[261:268]}
sel = lambda ks: np.concatenate([B[k] for k in ks])
foot_i = 261  # foot sign is first meta feature
tests = {"foot only": [foot_i]}
for k in B:
    if k != "meta": tests[k + " +foot"] = list(B[k]) + [foot_i]
tests["kpose_static+lines+vel+ball +foot"] = list(sel(["kpose_static", "kpose_lines", "k_velocity", "ball_rel"])) + [foot_i]
tests["velocity+ball+rel +foot"] = list(sel(["k_velocity", "ball_rel", "rel_geom"])) + [foot_i]
for nm, cs in tests.items():
    a, _, _ = cv(X[:, cs], Xm[:, cs], "kick", seeds=(0, 1))
    print(f"{nm:38s} kick {a:.3f}", flush=True)
