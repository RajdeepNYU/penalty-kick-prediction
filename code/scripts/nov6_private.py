"""Test-clip parts for the variants adopted by RULES_nov6.md, fitted on all 1300 labelled clips; every part re-derives a saved N4 part on the same code path as a check.
trees  -> K2: kick trees S_b2X (kin2 + kinR + kinX) with the trees_private.py recipe (HGB 4 configs x 3 seeds + ET x 3 seeds, mirror rows and TTA), read on the test clips'
          rows and on the corrected-camera rows (kinR_S_camera2f + kinX_S_camera2f, idea 3a of 4 Nov) -> cache/nov6/prv_trees_X_{S}.npz (orig, cam).
          Check: stream A with kin2 + kinR only must reproduce prv_trees_A_b2R.npy.
keeper -> G3: keep_private.py's seven members with keepD appended to the four tree members -> cache/nov6/pg_private_b2GD.npy.
          Check: the same without keepD must reproduce pg_private_b2G.npy.
pxT    -> Tm: the T pixel model (D's scaler and PCA, five alignment versions train with C / 5, the five averaged at test), three PCA seeds, read with the original camera
          terms and with the corrected camera (camera2f) -> cache/nov6/prv_px_T.npz (P, Pcam, D). Check: D from the same fit vs P4rkD's saved test-clip pixel.
usage: python cap.py nov6_private.py trees | keeper | pxT"""
import os, sys, pickle, numpy as np
from common import *

OUT = f"{PC}/nov6"; os.makedirs(OUT, exist_ok=True); PART = sys.argv[1]; prv = list(pm["clip"])

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

    def fitread(train, queries):
        Xa, Xb = rows(train, gt["clip"]); keep = ~np.isnan(Xa).all(0); Xa, Xb = Xa[:, keep], Xb[:, keep]; Q = [tuple(x[:, keep] for x in rows(q, prv)) for q in queries]
        Qa, Qb = np.concatenate([q[0] for q in Q]), np.concatenate([q[1] for q in Q])
        Ph = sum(fit_predict(lambda s, c=c: HGB(random_state=s, **{**base, **c}), Xa, Xb, Qa, Qb) for c in HC) / len(HC); Pe = fit_predict(et, Xa, Xb, Qa, Qb); Pall = (Ph + Pe) / 2
        return [Pall[i * len(prv):(i + 1) * len(prv)] for i in range(len(queries))], Q

    o = [f"{PC}/kin2_A_ball2.pkl", f"{PC}/kinR_A.pkl"]; (R0,), _ = fitread(o, [o]); print(f"check: A with kin2 + kinR vs saved prv_trees_A_b2R: max abs difference {np.abs(R0 - np.load(f'{PC}/prv_trees_A_b2R.npy')).max():.2e}", flush=True)
    for S in "AFG":
        tr = [f"{PC}/kin2_{S}_ball2.pkl", f"{PC}/kinR_{S}.pkl", f"{PC}/kinX_{S}.pkl"]; cm = [tr[0], f"{PC}/kinR_{S}_camera2f.pkl", f"{PC}/kinX_{S}_camera2f.pkl"]
        (Po, Pc), Q = fitread(tr, [tr, cm]); np.savez(f"{OUT}/prv_trees_X_{S}.npz", orig=Po, cam=Pc)
        print(f"{S}: saved | test rows changed by the corrected camera: {int((~np.isclose(Q[1][0], Q[0][0], equal_nan=True).all(1)).sum())} | argmax vs b2R trees differ on {int((Po.argmax(1) != np.load(f'{PC}/prv_trees_{S}_b2R.npy').argmax(1)).sum())}", flush=True)

if PART == "keeper":
    from cvlib import fit_predict, hg, yg, ygm, stack, ExtraTreesClassifier, make_pipeline, SimpleImputer
    etk = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=s))

    def trees(fs, both):
        A, B, QA, QB = [], [], [], []
        for f in fs:
            K = pickle.load(open(f"{PC}/{f}.pkl", "rb")); r, n = K["rows"], len(K["names"]); a, b = stack(r, gt["clip"], n); qa, qb = stack(r, pm["clip"], n); A.append(a); B.append(b); QA.append(qa); QB.append(qb)
        Xa, Xb, Qa, Qb = (np.concatenate(z, 1) for z in (A, B, QA, QB)); keep = ~np.isnan(Xa).all(0)
        P = [fit_predict(mk, Xa[:, keep], Xb[:, keep], Qa[:, keep], Qb[:, keep], y=yg, y_m=ygm)[:, [0, 2, 1]] for mk in (hg, etk)]
        return P if both else [(P[0] + P[1]) / 2]

    cnn = np.load(f"{W}/cnn_gk_private.npy")[:, [0, 2, 1]]
    for D, tag in ((False, "b2G"), (True, "b2GD")):
        ap = (lambda f, S: [f, f"keepD_{S}"]) if D else (lambda f, S: [f])
        parts = trees(ap("keep_A_b2", "A"), True) + trees(ap("keep_F_b2", "F"), True) + trees(ap("keepR_A", "A"), False) + trees(ap("keepR_F", "F"), False) + [cnn]; Pg = sum(parts) / len(parts)
        if not D: print(f"check: no keepD vs saved pg_private_b2G: max abs difference {np.abs(Pg - np.load(f'{PC}/pg_private_b2G.npy')).max():.2e}", flush=True)
        else: np.save(f"{OUT}/pg_private_b2GD.npy", Pg); print(f"saved pg_private_b2GD | predicted left / right / stands {np.bincount(Pg.argmax(1), minlength=3)}", flush=True)

if PART == "pxT":
    from pxlib import cam, pub, yk, SW, IX, OK3, FL3
    from nov6_px import fit, VW
    okH = np.load(f"{PC}/crops_hr_meta.npz")["ok"]; tags = ["D", "T1", "T2", "T3", "T4"]

    def load(tag, clips):
        J = np.array([IX[c] for c in clips]); G = [np.load(f"{PC}/ce_{v}_vpbase{tag}.npy", mmap_mode="r") for v in VW]
        return tuple(np.concatenate([np.asarray(g[J, k]).reshape(len(J), -1) for g in G], 1) for k in (0, 1))

    Jp, Jq = np.array([IX[c] for c in pub]), np.array([IX[c] for c in prv]); fl, flq = FL3[Jp], FL3[Jq]; ok_ = OK3[Jp] & okH[Jp]
    V, Vq = [load(t, pub) for t in tags], [load(t, prv) for t in tags]; fa, cp, sp = cam(pub, fl); fq, cq, sq = cam(prv, flq)
    CF = pickle.load(open(f"{PC}/camera2f.pkl", "rb")); ph2 = np.radians(np.array([CF[c]["phi"] if CF.get(c) else 0.0 for c in prv])); cq2, sq2 = np.where(flq, -np.cos(ph2), np.cos(ph2)), np.sin(ph2)
    print("test-clip camera terms changed by camera2f:", int((~np.isclose(cq2, cq) | ~np.isclose(sq2, sq)).sum()), flush=True)
    yc = np.array([SW[v] if f else v for v, f in zip(yk, fl)]); ycm = np.array([SW[v] for v in yc]); Pt, Ptc, Pd = (np.zeros((len(prv), 3)) for _ in range(3))
    for s in (0, 1, 2):
        pm_, pd_ = fit(V, np.where(ok_)[0], fa, cp, sp, yc, ycm, s); Pt += pm_(Vq, fq, cq, sq) / 3; Ptc += pm_(Vq, fq, cq2, sq2) / 3; Pd += pd_(Vq[:1], fq, cq, sq) / 3; print("seed", s, flush=True)
    uf = lambda X: np.where(flq[:, None], X[:, ::-1], X); R = dict(P=uf(Pt), Pcam=uf(Ptc), D=uf(Pd))
    FD = np.load(f"{PC}/Pp_e_body_hm_vpbaseD+e_body_tor_vpbaseD+e_legs12_vpbaseD_512_0.001.npy", allow_pickle=True).item()["P"]
    print(f"check: D from the T fit vs P4rkD test-clip pixel: max abs difference {np.abs(R['D'] - FD).max():.2e}, same argmax {np.mean(R['D'].argmax(1) == FD.argmax(1)):.4f}", flush=True)
    np.savez(f"{OUT}/prv_px_T.npz", **R); print("saved prv_px_T", flush=True)
