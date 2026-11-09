"""Private-clip parts for the 4 Nov ideas (RULES_nov4.md), fitted on all labelled clips; each part also reproduces P4rkD's saved private part as a check.
trees -> trees_private.py recipe (HGB 4 configs x 3 seeds + ET x 3 seeds, mirror rows and TTA) on the original rows, read on the private clips' original rows, 25 fps grid rows (cache_g, idea 1c) and corrected-camera rows (kinR_{S}_camera2f, idea 3a) -> cache/nov4/prv_trees_{S}.npz
DC    -> pixel model on the CLAHE crops (idea 4) -> cache/nov4/prv_px_DC.npy
mix   -> pixel model trained on original + LR + LRJ crops, read on the private original crops (idea 1b) -> cache/nov4/prv_px_mix.npy
cam   -> P4rkD pixel model read with the corrected camera (idea 3a) -> cache/nov4/prv_px_cam.npy
usage: python cap.py nov4_private.py trees | DC | mix | cam"""
import os, sys, pickle, numpy as np
from common import *

OUT = f"{PC}/nov4"; os.makedirs(OUT, exist_ok=True); PART = sys.argv[1]; prv = list(pm["clip"])

if PART == "trees":
    from cvlib import HGB, et, stack, fit_predict, cl
    base = dict(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5); HC = [dict(), dict(max_depth=5, max_iter=200, min_samples_leaf=40), dict(max_depth=3, min_samples_leaf=50), dict(max_depth=3)]

    def rows(paths, clips):
        A, B = [], []
        for p in paths:
            K = pickle.load(open(p, "rb")); r = K["rows"]
            if isinstance(next(iter(r.values()))[0], tuple): r = {c: (v[0][0], v[1][0]) for c, v in r.items()}
            r = {c: v for c, v in r.items() if len(v[0]) == len(K["names"])}; a, b = stack(r, clips, len(K["names"])); A.append(a); B.append(b)
        return np.concatenate(A, 1), np.concatenate(B, 1)

    for S in "AFG":
        o = [f"{PC}/kin2_{S}_ball2.pkl", f"{PC}/kinR_{S}.pkl"]; V = dict(orig=o, grid=[f"{P}/cache_g/kin2_{S}_ball2.pkl", f"{P}/cache_g/kinR_{S}.pkl"], cam=[o[0], f"{PC}/kinR_{S}_camera2f.pkl"])
        Xa, Xb = rows(o, gt["clip"]); keep = ~np.isnan(Xa).all(0); Xa, Xb = Xa[:, keep], Xb[:, keep]; Q = {k: tuple(x[:, keep] for x in rows(v, prv)) for k, v in V.items()}; Qa, Qb = np.concatenate([q[0] for q in Q.values()]), np.concatenate([q[1] for q in Q.values()])
        Ph = sum(fit_predict(lambda s, c=c: HGB(random_state=s, **{**base, **c}), Xa, Xb, Qa, Qb) for c in HC) / len(HC); Pe = fit_predict(et, Xa, Xb, Qa, Qb); Pall = (Ph + Pe) / 2; R = {k: Pall[i * len(prv):(i + 1) * len(prv)] for i, k in enumerate(V)}
        print(f"{S}: check vs saved prv_trees_{S}_b2R: max abs difference {np.abs(R['orig'] - np.load(f'{PC}/prv_trees_{S}_b2R.npy')).max():.2e} | rows changed: grid {int((~np.isclose(Q['grid'][0], Q['orig'][0], equal_nan=True).all(1)).sum())}, cam {int((~np.isclose(Q['cam'][0], Q['orig'][0], equal_nan=True).all(1)).sum())}", flush=True)
        np.savez(f"{OUT}/prv_trees_{S}.npz", **R)

else:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import pxlib
    from pxlib import rows, cam, head, pub, cl, yk, SW
    VW = ("body_hm", "body_tor", "legs12"); K, C = 512, 0.001

    def fit(versions, tr, fa, cp, sp, yc, ycm, seed):   # as nov4_px.fit
        Xa, Xb = versions[0]; E = np.r_[Xa[tr], Xb[tr]]; sc = StandardScaler().fit(E); pc = PCA(K, random_state=seed).fit(sc.transform(E)); del E; q = pc.transform(sc.transform(np.r_[Xa[tr], Xb[tr]])).std(0) + 1e-6
        ex = lambda A, f, c_, s_: head(pc.transform(sc.transform(A)) / q, f, c_, s_)
        X = np.concatenate([np.r_[ex(a[tr], fa[tr], cp[tr], sp[tr]), ex(b[tr], -fa[tr], -cp[tr], sp[tr])] for a, b in versions]); Y = np.tile(np.r_[yc[tr], ycm[tr]], len(versions))
        m = LogisticRegression(C=C / len(versions), max_iter=4000).fit(X, Y); c_ = list(m.classes_); sw = [c_.index(SW[z]) for z in c_]; o_ = [c_.index(z) for z in cl]
        return lambda Aq, Bq, f, c, s: ((m.predict_proba(ex(Aq, f, c, s)) + m.predict_proba(ex(Bq, -f, -c, s))[:, sw]) / 2)[:, o_]

    tag = "DC" if PART == "DC" else "D"; Xa, Xb, ok_, fl = rows([f"e_{v}_vpbase{tag}" for v in VW], pub); fa, cp, sp = cam(pub, fl); yc = np.array([SW[v] if f else v for v, f in zip(yk, fl)]); ycm = np.array([SW[v] for v in yc])
    Qa, Qb, oq, flq = rows([f"e_{v}_vpbase{tag}" for v in VW], prv); fq, cq, sq = cam(prv, flq); unflip = lambda P: np.where(flq[:, None], P[:, ::-1], P)
    S0 = np.load(f"{PC}/Pp_e_body_hm_vpbaseD+e_body_tor_vpbaseD+e_legs12_vpbaseD_512_0.001.npy", allow_pickle=True).item()["P"]; tr = np.where(ok_)[0]; Po, Pv = np.zeros((len(prv), 3)), np.zeros((len(prv), 3))
    if PART == "cam":
        CF = pickle.load(open(f"{PC}/camera2f.pkl", "rb")); ph2 = np.radians(np.array([CF[c]["phi"] if CF.get(c) else 0.0 for c in prv])); cq2, sq2 = np.where(flq, -np.cos(ph2), np.cos(ph2)), np.sin(ph2); print("private camera rows changed:", int((~np.isclose(cq2, cq) | ~np.isclose(sq2, sq)).sum()), flush=True)
    if PART == "mix": L = [rows([f"e_{v}_vpbase{m}" for v in VW], pub)[:2] for m in ("LR", "LRJ")]; tr = np.where(ok_ & np.load(f"{PC}/cropsLR_ok.npy"))[0]
    for s in (0, 1, 2):
        if PART in ("DC", "cam"): f_ = fit([(Xa, Xb)], tr, fa, cp, sp, yc, ycm, s); Po += f_(Qa, Qb, fq, cq, sq) / 3; Pv += (f_(Qa, Qb, fq, cq2, sq2) if PART == "cam" else 0) / 3
        if PART == "mix": f_ = fit([(Xa, Xb)] + L, tr, fa, cp, sp, yc, ycm, s); Pv += f_(Qa, Qb, fq, cq, sq) / 3
        print("seed", s, flush=True)
    if PART == "DC": np.save(f"{OUT}/prv_px_DC.npy", unflip(Po))
    if PART == "cam": print(f"check vs saved P4rkD private pixel: max abs difference {np.abs(unflip(Po) - S0).max():.2e}", flush=True); np.save(f"{OUT}/prv_px_cam.npy", unflip(Pv))
    if PART == "mix": np.save(f"{OUT}/prv_px_mix.npy", unflip(Pv))
    print("saved", PART, flush=True)
