import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
import cv_kin
from cv_kin import *
Pk = np.load(f"{C}/oof_kin_bag.npy")                                  # kick OOF probs (left, middle, right)
gb = np.load(f"{C}/oof_final_gk_gbdt.npy"); cn = np.load(f"{C}/oof_gk_cnn30.npy")[:, [0, 2, 1]]   # both: left, right, stands
Pg = 0.5 * gb + 0.5 * cn; yg = gt["goalkeeper_direction"].values; yo = gt["goal_outcome"].values
cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean()
print("keeper OOF acc %.3f | kick OOF acc %.3f" % ((np.array(["left", "right", "stands in the middle"])[Pg.argmax(1)] == yg).mean(), acc(Pk)))
print("--- (1) multiplicative prior: P'(k) = P(k) * (1 + a * P_keeper(k)) for left/right ---")
for a in (0, 0.25, 0.5, 1.0, 2.0):
    P2 = Pk.copy(); P2[:, 0] *= 1 + a * Pg[:, 0]; P2[:, 2] *= 1 + a * Pg[:, 1]; print(f"   a={a:4.2f}: kick acc {acc(P2):.3f}")
print("--- (2) stacking: keeper probs as extra features (mirror swaps left/right) ---")
Xs = np.c_[X, Pg[:, 0], Pg[:, 1], Pg[:, 2]]; Xsm = np.c_[Xm, Pg[:, 1], Pg[:, 0], Pg[:, 2]]
cv_kin.P = dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
P0, _ = cv_clf(X, Xm, seeds=(0, 1, 2)); P1, _ = cv_clf(Xs, Xsm, seeds=(0, 1, 2)); print(f"   without keeper feats {acc(P0):.3f} | with keeper feats {acc(P1):.3f}")
np.save(f"{C}/oof_kin_stack.npy", P1)
print("--- (3) can outcome beat always-goal? ---")
pk = cl[Pk.argmax(1)]; pg = np.array(["left", "right", "stands in the middle"])[Pg.argmax(1)]; ng = (yo == "no_goal")
print("base no_goal rate %.3f" % ng.mean())
for thr in (0.0, 0.6, 0.7, 0.8, 0.9):
    m = (pk == pg) & (Pk.max(1) > thr) & (Pg.max(1) > 0.8); print(f"   pred kick side == pred keeper side & kick conf>{thr}: n={m.sum()} no_goal rate {ng[m].mean() if m.any() else float('nan'):.3f}")
