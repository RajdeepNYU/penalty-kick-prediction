import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from cv_kin import *
B = {
    "plant: placement (ank/toe/heel uv, foot-ball)": blk("pl_ank", "pl_toe", "pl_heel", "pl_foot_ball"),
    "plant: foot angle/length": blk("pl_footang", "pl_footlen"),
    "plant: shank/thigh": blk("pl_shank", "pl_thigh"),
    "kick: placement": blk("kk_ank_uv", "kk_toe_uv", "kk_heel_uv", "kk_foot_ball"),
    "kick: foot angle/length": blk("kk_footang", "kk_footlen"),
    "kick: shank/thigh (+angular vel)": blk("kk_shank", "kk_thigh"),
    "kick: toe/ankle velocity + rel. to plant": blk("kk_toe_vel", "kk_toe_speed", "kk_ank_vel", "kk_ank_speed", "kk_minus_pl", "foot_travel"),
    "hip axis": blk("hipax"), "shoulder axis": blk("shax"), "ear axis/head": blk("earax", "head"), "trunk": blk("trunk"),
    "approach (hip pos/vel)": blk("hip_uv", "hip_vel"), "meta+quality": blk("meta", "kpq"),
}
allc = np.arange(X.shape[1]); assert sum(len(v) for v in B.values()) + len(FOOT) == len(allc), (sum(len(v) for v in B.values()) + len(FOOT), len(allc))
Pr, cl = cv_clf(X, Xm, seeds=(0, 1)); base = (np.array(cl)[Pr.argmax(1)] == yk).mean(); print(f"ALL ({len(allc)} feats): kick acc {base:.3f}", flush=True)
for nm, ix in B.items():
    keep = np.setdiff1d(allc, ix); Pr, cl = cv_clf(X[:, keep], Xm[:, keep], seeds=(0, 1)); a = (np.array(cl)[Pr.argmax(1)] == yk).mean()
    print(f"drop {nm:46s} ({len(ix):3d} feats): {a:.3f}  ({a - base:+.3f})", flush=True)
print("--- single blocks (+foot sign) ---")
for nm, ix in B.items():
    cols = np.r_[ix, FOOT]; Pr, cl = cv_clf(X[:, cols], Xm[:, cols], seeds=(0, 1)); a = (np.array(cl)[Pr.argmax(1)] == yk).mean(); print(f"only {nm:46s}: {a:.3f}", flush=True)
