"""Shuffled-label refit control for final_assess.py, read-only on the caches: the ViTPose-L kick tree member (kin2_F_ball2 + kinR_F, HGB + ExtraTrees, the e_trees.py protocol:
source-grouped 5-fold, mirror rows and mirror TTA) trained on the real labels and on 3 random permutations of the labels (mirror labels follow), partition seed 0.
If the trees learned from the clips, the real labels give ~0.68 and the shuffled ones fall to the label-free level (~0.40-0.47).
usage: python cap.py assess_shuffle.py -> appended to logs/final/assess.txt"""
import pickle, numpy as np
from common import *
from cvlib import oof, hg, et, cl, yk, SW, grp, stack

Xa, Xb = [], []
for f in ("kin2_F_ball2", "kinR_F"):
    K = pickle.load(open(f"{PC}/{f}.pkl", "rb")); r = K["rows"]
    if isinstance(next(iter(r.values()))[0], tuple): r = {c: (v[0][0], v[1][0]) for c, v in r.items()}
    a, b = stack(r, gt["clip"], len(K["names"])); Xa.append(a); Xb.append(b)
Xa, Xb = np.concatenate(Xa, 1), np.concatenate(Xb, 1); keep = ~np.isnan(Xa).all(0); Xa, Xb = Xa[:, keep], Xb[:, keep]
L = open(f"{P}/logs/final/assess.txt", "a", encoding="utf-8"); pr = lambda *a: (print(*a, flush=True), print(*a, file=L, flush=True))
pr("\n== A7. shuffled-label refit (ViTPose-L kick trees, HGB + ExtraTrees, seed 0, source-grouped 5-fold): the same pipeline trained on labels with the clip link broken")
yk_ = np.asarray(yk, str); rng = np.random.default_rng(7); maj = max(np.mean(yk_ == c) for c in cl)
for name, y in [("real labels", yk_)] + [(f"shuffle {i + 1}", yk_[rng.permutation(len(yk_))]) for i in range(3)]:
    ym = np.array([SW[v] for v in y]); P_ = (oof(hg, Xa, Xb, y=y, y_m=ym, seeds=(0,)) + oof(et, Xa, Xb, y=y, y_m=ym, seeds=(0,))) / 2
    pr(f"   {name:12s}: kick right {np.mean(cl[P_.argmax(1)] == y):.4f} (label-free majority level {maj:.4f})")
