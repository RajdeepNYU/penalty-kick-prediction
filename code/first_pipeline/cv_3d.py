import numpy as np, pickle, sys, warnings
warnings.filterwarnings("ignore")
import cv_kin
from cv_kin import *
cv_kin.P = dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
variants = sys.argv[1:] or ["+1_2.5", "-1_2.5"]
def acc(P): return (np.array(["left", "middle", "right"])[P.argmax(1)] == yk).mean()
P0, _ = cv_clf(X, Xm, seeds=(0, 1, 2)); print(f"base 259 kinematic            kick {acc(P0):.3f}", flush=True)
for v in variants:
    K3 = pickle.load(open(f"{C}/kin3d_{v}.pkl", "rb")); r3 = K3["rows"]
    dim = len(K3["names"]); nan = np.full(dim, np.nan, np.float32); print(f"[{v}] clips missing 3D features: {sum(c not in r3 for c in gt['clip'])}")
    A = np.stack([r3.get(c, (nan, nan))[0] for c in gt["clip"]]); B = np.stack([r3.get(c, (nan, nan))[1] for c in gt["clip"]])
    keep = ~np.isnan(A).all(0) & ~np.isnan(B).all(0); print(f"[{v}] 3D dim {A.shape[1]} -> {keep.sum()} (all-NaN columns dropped: {[K3['names'][i] for i in np.where(~keep)[0]][:8]}); NaN share {np.isnan(A[:, keep]).mean():.3f}")
    A, B = A[:, keep], B[:, keep]
    P1, _ = cv_clf(np.c_[A, X[:, FOOT]], np.c_[B, Xm[:, FOOT]], seeds=(0, 1, 2)); print(f"3D only (+foot)  [{v}]         kick {acc(P1):.3f}", flush=True)
    P2, _ = cv_clf(np.c_[X, A], np.c_[Xm, B], seeds=(0, 1, 2)); print(f"base + 3D        [{v}]         kick {acc(P2):.3f}", flush=True)
    np.save(f"{C}/oof_kin3d_{v}.npy", P2)
