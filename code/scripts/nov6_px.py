"""Pixel model out of fold for RULES_nov6.md (variant Tm) on the fresh partition set 212223: the T job of RULES_px2 (archive/scripts/px2.py, unchanged logic).
D = P4rkD recipe (ViTPose+ base tokens of body_hm, body_tor, legs12 joined, scaler and PCA 512 on the training clips, C 0.001, head V3, crop and mirror rows, mirror TTA,
three partition seeds averaged); T = D's tokens plus the four alignment versions T1..T4, scaler and PCA from D's version, all five versions train (C / 5), the five
averaged at test. D is fit alongside T with the same scaler and PCA (the 5 Nov check: matches the saved P4rkD files to 8e-6). Tokens kept in float16. Per-seed checkpoint.
usage: python cap.py nov6_px.py 212223 -> cache/px2/px_T_212223.npz (P = T, D = D)"""
import os, sys, time, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from common import *
from pxlib import cam, head, pub, cl, yk, SW, grp, IX, OK3, FL3

OUT = f"{PC}/px2"; os.makedirs(OUT, exist_ok=True)
VW = ("body_hm", "body_tor", "legs12"); K, C = 512, 0.001; J = np.array([IX[c] for c in pub]); fl = FL3[J]
ok_ = OK3[J] & np.load(f"{PC}/crops_hr_meta.npz")["ok"][J]                    # the same training clips as RULES_px2 (all sets)


def load(tag):
    G = [np.load(f"{PC}/ce_{v}_vpbase{tag}.npy", mmap_mode="r") for v in VW]
    return tuple(np.concatenate([np.asarray(g[J, k]).reshape(len(J), -1) for g in G], 1) for k in (0, 1))


def fit(V, tr, fa, cp, sp, yc, ycm, seed):
    f32 = lambda A: A.astype(np.float32); E = f32(np.r_[V[0][0][tr], V[0][1][tr]]); sc = StandardScaler().fit(E); pc = PCA(K, random_state=seed).fit(sc.transform(E)); q = pc.transform(sc.transform(E)).std(0) + 1e-6; del E
    Z = lambda A: pc.transform(sc.transform(f32(A))) / q; Y = np.r_[yc[tr], ycm[tr]]
    R = [np.r_[head(Z(a[tr]), fa[tr], cp[tr], sp[tr]), head(Z(b[tr]), -fa[tr], -cp[tr], sp[tr])] for a, b in V]

    def lr(Xs):
        m = LogisticRegression(C=C / len(Xs), max_iter=4000).fit(np.concatenate(Xs), np.tile(Y, len(Xs))); c_ = list(m.classes_); sw = [c_.index(SW[z]) for z in c_]; o_ = [c_.index(z) for z in cl]
        return lambda Vq, f, c, s: (sum((m.predict_proba(head(Z(a), f, c, s)) + m.predict_proba(head(Z(b), -f, -c, s))[:, sw]) / 2 for a, b in Vq) / len(Vq))[:, o_]
    return lr(R), lr(R[:1])


if __name__ == "__main__":
    SET = sys.argv[1]; SEEDS = {"212223": (21, 22, 23)}[SET]
    V = [load(t) for t in ("D", "T1", "T2", "T3", "T4")]; fa, cp, sp = cam(pub, fl)                    # the ce_*.npy token files are float16 already
    yc = np.array([SW[v] if f else v for v, f in zip(yk, fl)]); ycm = np.array([SW[v] for v in yc]); Pc = {"P": np.zeros((len(pub), 3)), "D": np.zeros((len(pub), 3))}
    CK = f"{OUT}/px_T_{SET}_part.npz"; done = []
    if os.path.exists(CK): z = np.load(CK); Pc, done = {k: z[k].copy() for k in Pc}, list(z["done"]); print("resumed after seeds", done, flush=True)
    for s in SEEDS:
        if s in done: continue
        for f_, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(V[0][0], yk, grp)):
            t0 = time.time(); pm_, pd_ = fit(V, tr[ok_[tr]], fa, cp, sp, yc, ycm, s); Vq = [(a[te], b[te]) for a, b in V]
            Pc["P"][te] += pm_(Vq, fa[te], cp[te], sp[te]) / len(SEEDS); Pc["D"][te] += pd_(Vq[:1], fa[te], cp[te], sp[te]) / len(SEEDS)
            print(f"T {SET} seed {s} fold {f_} ({time.time() - t0:.0f}s)", flush=True)
        done.append(s); np.savez(CK, done=np.array(done), **Pc)
    R = {k: np.where(fl[:, None], v[:, ::-1], v) for k, v in Pc.items()}; np.savez(f"{OUT}/px_T_{SET}.npz", **R); print("saved", flush=True)
