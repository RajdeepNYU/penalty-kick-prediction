"""Private keeper probabilities with the new-ball keeper features, as v10 (Q4_work/make_sub7.py + make_v10_parts.py): mean of {HGB x3 seeds, ET(leaf 2) x3 seeds}
on the RTMW keeper features, the same on the ViTPose-L keeper features, and the keeper CNN (cnn_gk_private.npy), i.e. (HGB_A + ET_A + HGB_F + ET_F + CNN) / 5.
-> cache/pg_private_b2.npy (pm order, columns left / right / stands in the middle)"""
import pickle, numpy as np
from common import *
from cvlib import fit_predict, hg, yg, ygm, stack, ExtraTreesClassifier, make_pipeline, SimpleImputer

etk = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=s))
TAG = os.environ.get("KEEP_TAG", "b2")       # b2: (HGB_A + ET_A + HGB_F + ET_F + CNN) / 5 ; b2G: + goal-plane trees (keepR.py) as two more members, (HGB+ET)/2 each -> / 7


def trees(f, both):
    K = pickle.load(open(f"{PC}/{f}.pkl", "rb")); r, n = K["rows"], len(K["names"])
    Xa, Xb = stack(r, gt["clip"], n); Qa, Qb = stack(r, pm["clip"], n); keep = ~np.isnan(Xa).all(0)
    P = [fit_predict(mk, Xa[:, keep], Xb[:, keep], Qa[:, keep], Qb[:, keep], y=yg, y_m=ygm)[:, [0, 2, 1]] for mk in (hg, etk)]
    return P if both else [(P[0] + P[1]) / 2]


parts = trees("keep_A_b2", True) + trees("keep_F_b2", True) + (trees("keepR_A", False) + trees("keepR_F", False) if TAG == "b2G" else [])
parts.append(np.load(f"{W}/cnn_gk_private.npy")[:, [0, 2, 1]])
Pg = sum(parts) / len(parts); np.save(f"{PC}/pg_private_{TAG}.npy", Pg); print(f"private keeper {TAG} ({len(parts)} members) pred counts (left / right / stands)", np.bincount(Pg.argmax(1), minlength=3))
if os.path.exists(f"{W}/pg_private_v7.npy"):
    old = (3 * np.load(f"{W}/pg_private_v7.npy") + np.load(f"{W}/keeper_private_F.npy")) / 5; print("private keeper: agreement with v10", round(float((Pg.argmax(1) == old.argmax(1)).mean()), 4))
