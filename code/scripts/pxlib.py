"""Camera-aware pixel model on frozen DINOv3 tokens, for any set of cached views: OOF on the public clips and a fit on all public clips for the private ones. Head V3 of foot_sweep.py; scaler and PCA fitted on the training clips only, crop and mirror rows, mirror TTA.
View names: f_last, f_stack (foot_emb), z_foot35, z_foot80, z_body, z_upper, z_ball (zoom_emb), g_{body,upper,legs}_{0,1,2,3} (zoom3 grid, time index), c_{body,upper,legs}_{0,1,2,3} (zoom3 CLS), h_grid_{0,1,2}, h_pose_{0,1,2} (zoom4, 448 px)."""
import pickle, numpy as np
from common import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from cvlib import cl, yk, SW, grp

pub = list(gt["clip"]); prv = list(pm["clip"]); CAM = pickle.load(open(f"{PC}/camera2.pkl", "rb")); M3 = np.load(f"{PC}/zoom3_meta.npz"); IX = {c: j for j, c in enumerate(M3["clips"])}; OK3, FL3 = M3["ok"], M3["flip"]; _c = {}


def _src(nm):
    if nm not in _c:
        if nm == "f": Z = np.load(f"{PC}/foot_emb.npz"); _c[nm] = (dict(zip(list(Z["clips"]), range(len(Z["clips"])))), Z["ok"], Z["flip"], Z["grid"])
        if nm == "z": Z = np.load(f"{PC}/zoom_emb.npz"); _c[nm] = (dict(zip(list(Z["clips"]), range(len(Z["clips"])))), Z["ok"], Z["flip"], Z["grid"], list(Z["views"]))
        if nm == "3": _c[nm] = (IX, OK3, FL3, np.load(f"{PC}/zoom3_emb.npy", mmap_mode="r"), list(M3["views"]))
        if nm == "4": Z = np.load(f"{PC}/zoom4_meta.npz"); _c[nm] = (dict(zip(list(Z["clips"]), range(len(Z["clips"])))), Z["ok"], Z["flip"], np.load(f"{PC}/pose_tok.npy", mmap_mode="r"), np.load(f"{PC}/zoom4_grid.npy", mmap_mode="r"))
    return _c[nm]


def view(name, clips):
    """Returns crop rows, mirror rows, ok, flip for these clips. d_{a}-{b} = tokens of e_{a} minus tokens of e_{b}."""
    fam, rest = name.split("_", 1); n = len(clips)
    if fam == "d": a, b = rest.split("-"); Va, Vb = view(f"e_{a}", clips), view(f"e_{b}", clips); return Va[0] - Vb[0], Va[1] - Vb[1], Va[2] & Vb[2], Va[3]
    if fam == "f": ix, o, fl, G = _src("f"); J = np.array([ix[c] for c in clips]); t = {"last": 1, "stack": 0}[rest]; A, B = G[J, 0, t, 0], G[J, 0, t, 1]
    if fam == "z": ix, o, fl, G, vs = _src("z"); J = np.array([ix[c] for c in clips]); i = vs.index(rest); A, B = G[J, i, 0], G[J, i, 1]
    if fam in "gc" and len(fam) == 1:
        ix, o, fl, G, vs = _src("3"); J = np.array([ix[c] for c in clips]); v, t = rest.rsplit("_", 1); i, t = vs.index(v), int(t); sl = slice(1, None) if fam == "g" else slice(0, 1); A, B = np.asarray(G[J, i, t, 0][:, sl]), np.asarray(G[J, i, t, 1][:, sl])
    if fam == "h": ix, o, fl, Tk, Gr = _src("4"); J = np.array([ix[c] for c in clips]); kind, t = rest.rsplit("_", 1); G = Tk if kind == "pose" else Gr; t = int(t); A, B = np.asarray(G[J, t, 0]), np.asarray(G[J, t, 1])
    if fam == "e":  # e_{view}_{img}_{pool}, crop_emb.py; same clip order and flips as zoom3
        J = np.array([IX[c] for c in clips]); G = np.load(f"{PC}/ce_{rest}.npy", mmap_mode="r"); o, fl = OK3, FL3; A, B = np.asarray(G[J, 0]), np.asarray(G[J, 1])
        if rest.startswith("k"): o = np.load(f"{PC}/crops_k_meta.npz")["ok"]
    return A.astype(np.float32).reshape(n, -1), B.astype(np.float32).reshape(n, -1), o[J], fl[J]


def rows(names, clips):
    V = [view(v, clips) for v in names]; fl = V[0][3]; assert all((v[3] == fl).all() for v in V)
    return np.concatenate([v[0] for v in V], 1), np.concatenate([v[1] for v in V], 1), np.logical_and.reduce([v[2] for v in V]), fl


def cam(clips, fl):
    ph = np.radians(np.array([CAM[c]["phi"] if c in CAM else 0.0 for c in clips])); foot = np.array([1.0 if META[c]["foot"] == "right" else -1.0 for c in clips])
    return np.where(fl, -foot, foot), np.where(fl, -np.cos(ph), np.cos(ph)), np.sin(ph)


def head(Q, f, c, s): return np.column_stack([Q, Q * c[:, None], Q * s[:, None], 3 * f[:, None], c, s, Q * f[:, None], Q * (f * c)[:, None], Q * (f * s)[:, None]])


def _fit(Xa, Xb, tr, fa, cp, sp, yc, ycm, k, C, seed):
    E = np.r_[Xa[tr], Xb[tr]]; sc = StandardScaler().fit(E); pc = PCA(k, random_state=seed).fit(sc.transform(E)); q = pc.transform(sc.transform(E)).std(0) + 1e-6; ex = lambda A, f, c_, s_: head(pc.transform(sc.transform(A)) / q, f, c_, s_)
    m = LogisticRegression(C=C, max_iter=4000).fit(np.r_[ex(Xa[tr], fa[tr], cp[tr], sp[tr]), ex(Xb[tr], -fa[tr], -cp[tr], sp[tr])], np.r_[yc[tr], ycm[tr]]); c_ = list(m.classes_); sw = [c_.index(SW[z]) for z in c_]; o_ = [c_.index(z) for z in cl]
    return lambda Aq, Bq, f, c, s: ((m.predict_proba(ex(Aq, f, c, s)) + m.predict_proba(ex(Bq, -f, -c, s))[:, sw]) / 2)[:, o_]


def oof(names, k, C, seeds=(0, 1, 2)):
    Xa, Xb, ok_, fl = rows(names, pub); fa, cp, sp = cam(pub, fl); yc = np.array([SW[v] if f else v for v, f in zip(yk, fl)]); ycm = np.array([SW[v] for v in yc]); Pc = np.zeros((len(pub), 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp): Pc[te] += _fit(Xa, Xb, tr[ok_[tr]], fa, cp, sp, yc, ycm, k, C, s)(Xa[te], Xb[te], fa[te], cp[te], sp[te]) / len(seeds)
    return np.where(fl[:, None], Pc[:, ::-1], Pc)


def _fit_multi(V, tr, fa, cp, sp, yc, ycm, k, C, seed):
    """V = rows of several versions of one crop (same dimension): all train, all average at test. Scaler and PCA from the first version."""
    E = np.r_[V[0][0][tr], V[0][1][tr]]; sc = StandardScaler().fit(E); pc = PCA(k, random_state=seed).fit(sc.transform(E)); q = pc.transform(sc.transform(E)).std(0) + 1e-6; ex = lambda A, f, c_, s_: head(pc.transform(sc.transform(A)) / q, f, c_, s_)
    m = LogisticRegression(C=C / len(V), max_iter=4000).fit(np.concatenate([np.r_[ex(v[0][tr], fa[tr], cp[tr], sp[tr]), ex(v[1][tr], -fa[tr], -cp[tr], sp[tr])] for v in V]), np.tile(np.r_[yc[tr], ycm[tr]], len(V))); c_ = list(m.classes_); sw = [c_.index(SW[z]) for z in c_]; o_ = [c_.index(z) for z in cl]
    return lambda Vq, f, c, s: (sum((m.predict_proba(ex(v[0], f, c, s)) + m.predict_proba(ex(v[1], -f, -c, s))[:, sw]) / 2 for v in Vq) / len(Vq))[:, o_]


def oof_multi(names, k, C, seeds=(0, 1, 2)):
    V = [view(v, pub) for v in names]; fl = V[0][3]; ok_ = V[0][2]; fa, cp, sp = cam(pub, fl); yc = np.array([SW[v] if f else v for v, f in zip(yk, fl)]); ycm = np.array([SW[v] for v in yc]); Pc = np.zeros((len(pub), 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(V[0][0], yk, grp): Pc[te] += _fit_multi(V, tr[ok_[tr]], fa, cp, sp, yc, ycm, k, C, s)([(v[0][te], v[1][te]) for v in V], fa[te], cp[te], sp[te]) / len(seeds)
    return np.where(fl[:, None], Pc[:, ::-1], Pc)


def private_multi(names, k, C, seeds=(0, 1, 2)):
    V = [view(v, pub) for v in names]; fl = V[0][3]; ok_ = V[0][2]; fa, cp, sp = cam(pub, fl); yc = np.array([SW[v] if f else v for v, f in zip(yk, fl)]); ycm = np.array([SW[v] for v in yc]); Vq = [view(v, prv) for v in names]; flq = Vq[0][3]; fq, cq, sq = cam(prv, flq); Pc = np.zeros((len(prv), 3))
    for s in seeds: Pc += _fit_multi(V, np.where(ok_)[0], fa, cp, sp, yc, ycm, k, C, s)([(v[0], v[1]) for v in Vq], fq, cq, sq) / len(seeds)
    return np.where(flq[:, None], Pc[:, ::-1], Pc), Vq[0][2]


def private(names, k, C, seeds=(0, 1, 2)):
    """Fit on all public clips with features (three PCA seeds), predict the private clips. Returns probabilities and the private ok mask."""
    Xa, Xb, ok_, fl = rows(names, pub); fa, cp, sp = cam(pub, fl); yc = np.array([SW[v] if f else v for v, f in zip(yk, fl)]); ycm = np.array([SW[v] for v in yc]); Qa, Qb, oq, flq = rows(names, prv); fq, cq, sq = cam(prv, flq); Pc = np.zeros((len(prv), 3))
    for s in seeds: Pc += _fit(Xa, Xb, np.where(ok_)[0], fa, cp, sp, yc, ycm, k, C, s)(Qa, Qb, fq, cq, sq) / len(seeds)
    return np.where(flq[:, None], Pc[:, ::-1], Pc), oq
