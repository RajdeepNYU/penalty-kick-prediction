"""Keeper features (Q4_work/kinkeep.build, unchanged) with the new ball, for the RTMW (kkeep) and ViTPose-L (kkeepF) keeper keypoints, then keeper CV
with the v10 protocol (HGB + ExtraTrees(min_samples_leaf=2), mirror aug/TTA) and the v10 keeper blend (2 RTMW trees : 2 ViTPose-L trees : 1 CNN).
-> cache/keep_{A,F}_b2.pkl, cache/keep_oof_{A,F}_b2.npy"""
import sys, pickle, warnings, numpy as np
from multiprocessing import Pool
from common import *
warnings.filterwarnings("ignore")
import kinkeep

BALLF = os.environ.get("BALLF", "ball2"); KTAG = {"ball2": "b2", "ball3": "b3"}[BALLF]          # BALLF=ball3 -> keep_{A,F}_b3 (mkball3.py)
B2 = pickle.load(open(f"{PC}/{BALLF}.pkl", "rb"))


def one(args):
    c, S = args
    b = load("ballspot", c); k0 = load("kkeep", c); ks = k0 if S == "A" else load("kkeepF", c); t = tracks()[c]
    if b is None or k0 is None or ks is None or t is None or t["keep"] is None: return c, None, None
    b = dict(b)
    if B2.get(c) is not None: b["ball_last"] = np.asarray(B2[c]["ball"], float)
    kk = dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"]); kinkeep.NAMES.clear(); a = kinkeep.build(b, kk, t, META[c], False); names = list(kinkeep.NAMES); kinkeep.NAMES.clear()
    m = kinkeep.build(b, kk, t, META[c], True)
    return c, (a, m) if a is not None and m is not None else None, names


if __name__ == "__main__":
    from cvlib import oof, hg, yg, ygm, stack, ExtraTreesClassifier, make_pipeline, SimpleImputer, SW, grp
    from sklearn.model_selection import StratifiedGroupKFold
    cg = np.array(["left", "middle", "right"]); cgW = np.array(["left", "right", "stands in the middle"])
    etk = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=s))
    accg = lambda P: float((cg[P.argmax(1)] == yg).mean())
    out = {}
    for S in "AF":
        with Pool(int(os.environ.get("NPOOL", "12"))) as p: res = p.map(one, [(c, S) for c in CLIPS], chunksize=20)
        rows = {c: r for c, r, _ in res if r is not None}; names = next(n for _, r, n in res if r is not None)
        pickle.dump(dict(rows=rows, names=names), open(f"{PC}/keep_{S}_{KTAG}.pkl", "wb"))
        Xa, Xb = stack(rows, gt["clip"], len(names)); keep = ~np.isnan(Xa).all(0); Xa, Xb = Xa[:, keep], Xb[:, keep]
        H = oof(hg, Xa, Xb, y=yg, y_m=ygm); E = oof(etk, Xa, Xb, y=yg, y_m=ygm); out[S] = (H + E) / 2
        np.save(f"{PC}/keep_oof_{S}_{KTAG}.npy", out[S][:, [0, 2, 1]])                   # saved in the v10 keeper column order (left, right, stands)
        print(f"keeper {S}: HGB {accg(H):.4f} ET {accg(E):.4f} trees {accg(out[S]):.4f}")
    ygW = gt["goalkeeper_direction"].values; ag = lambda P: float((cgW[P.argmax(1)] == ygW).mean()); nr, nf = out["A"][:, [0, 2, 1]], out["F"][:, [0, 2, 1]]
    if os.path.exists(f"{W}/oof_gk_resnet34.a1_in1k_0.npy"):                        # keeper-CNN OOF (Q4_work/train_cnn.py) for the blend
        cnn = np.load(f"{W}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]
        if os.path.exists(f"{W}/keep_oof_rtmw.npy"):
            kr, kf = np.load(f"{W}/keep_oof_rtmw.npy"), np.load(f"{W}/keep_oof_F.npy")
            print(f"v10 keeper trees RTMW {ag(kr):.4f} ViT-L {ag(kf):.4f} | v10 blend {ag((2 * kr + 2 * kf + cnn) / 5):.4f}")
        print(f"new-ball keeper trees RTMW {ag(nr):.4f} ViT-L {ag(nf):.4f} | blend {ag((2 * nr + 2 * nf + cnn) / 5):.4f}")
