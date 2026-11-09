"""Stage 2b: joint kicker/keeper PAIR ranker (motion, ball distance, pair geometry), source-grouped OOF on clicked clips.
Reads tracks_base.pkl (single-player ranker, track.py) as fallback and writes the final cache/tracks.pkl."""
import os  # notebook: paths from the environment
import pickle, os, sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]
ANCHOR, CONF, KMAX = 6, 0.3, 14
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
ck = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip")
src = dict(zip(gt["clip"], gt["source_id"]))
clips = list(gt["clip"]) + list(pm["clip"])
det = {c: pickle.load(open(f"{C}/det/{c}.pkl", "rb")) for c in clips}
old = pickle.load(open(f"{C}/tracks_base.pkl", "rb"))


def match(b, d, j):
    """centre displacement (px) of box b to its best match in frame j, normalised by box height; NaN if none."""
    f = d["frames"][j]; bb = f["pb"][f["pc"] >= 0.15]
    if len(bb) == 0: return np.nan, np.nan, np.nan
    h = max(b[3] - b[1], 1); c0 = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    cc = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]; dist = np.linalg.norm(cc - c0, axis=1) / h
    ratio = (bb[:, 3] - bb[:, 1]) / h; cost = dist + 1.5 * np.abs(np.log(ratio)); i = cost.argmin()
    if cost[i] > 1.5: return np.nan, np.nan, np.nan
    return (cc[i, 0] - c0[0]) / h, (cc[i, 1] - c0[1]) / h, 1.0


def cand_table(c):
    d = det[c]; f = d["frames"][ANCHOR]; w, h = d["wh"]; k = f["pc"] >= CONF
    b = f["pb"][k]; col = f["col"][k]; cf = f["pc"][k]
    if len(b) == 0: return None
    keep = np.argsort(-(cf * (b[:, 3] - b[:, 1])))[:KMAX]; b, col, cf = b[keep], col[keep], cf[keep]
    n = len(b); cx = (b[:, 0] + b[:, 2]) / 2 / w; cy = (b[:, 1] + b[:, 3]) / 2 / h
    bw = (b[:, 2] - b[:, 0]) / w; bh = (b[:, 3] - b[:, 1]) / h; cd = np.abs(col - np.median(col, axis=0)).sum(1)
    pos = np.c_[cx, cy]; Dm = np.linalg.norm(pos[:, None] - pos[None], axis=2); np.fill_diagonal(Dm, 9)
    mot = np.array([[*match(bx, d, 12)[:2], *match(bx, d, 3)[:2], match(bx, d, 8)[2] + match(bx, d, 12)[2] + match(bx, d, 3)[2] if True else 0] for bx in b], float)
    mot = np.nan_to_num(mot, nan=np.nan)
    sp = np.hypot(mot[:, 0], mot[:, 1]); sp_r = np.where(np.isfinite(sp), sp, -1).argsort().argsort() / n
    ball = d["frames"][ANCHOR]["bb"]; bc = d["frames"][ANCHOR]["bc"]
    if len(ball):
        feet = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]; bcen = np.c_[(ball[:, 0] + ball[:, 2]) / 2, (ball[:, 1] + ball[:, 3]) / 2]
        bd = np.linalg.norm(feet[:, None] - bcen[None], axis=2) / np.maximum(b[:, 3] - b[:, 1], 1)[:, None]; bdist = bd.min(1)
    else: bdist = np.full(n, np.nan)
    X = np.c_[cx, cy, bw, bh, cf, col, cd, Dm.min(1), np.full(n, n), cy.argsort().argsort() / n, cx.argsort().argsort() / n,
              bh / np.median(bh), mot[:, :4], sp, sp_r, mot[:, 4], bdist]
    return b, X, (cx, cy, bh)


def pairs(c, tab):
    b, X, (cx, cy, bh) = tab; n = len(b); I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); m = I != J; I, J = I[m], J[m]
    h = np.maximum(b[:, 3] - b[:, 1], 1)
    dx = ((b[J, 0] + b[J, 2]) / 2 - (b[I, 0] + b[I, 2]) / 2) / h[I]; dy = ((b[J, 1] + b[J, 3]) / 2 - (b[I, 1] + b[I, 3]) / 2) / h[I]
    P = np.c_[X[I], X[J], dx, dy, np.hypot(dx, dy), h[J] / h[I], np.abs(X[I][:, 5:15] - X[J][:, 5:15]).sum(1)]
    return I, J, P


def click_idx(c, b, who):
    w, h = det[c]["wh"]; r = ck.loc[c]; p = np.array([r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h])
    ins = (b[:, 0] <= p[0]) & (p[0] <= b[:, 2]) & (b[:, 1] <= p[1]) & (p[1] <= b[:, 3])
    if not ins.any(): return None
    ar = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); return int(np.where(ins)[0][ar[ins].argmin()])


tabs = {c: cand_table(c) for c in clips}
rows = []
for c in clips:
    t = tabs[c]
    if t is None or c not in ck.index: continue
    ki, gi = click_idx(c, t[0], "kicker"), click_idx(c, t[0], "keeper")
    I, J, P = pairs(c, t)
    y = ((I == ki) & (J == gi)).astype(int) if ki is not None and gi is not None and ki != gi else np.zeros(len(I), int)
    rows.append((c, P, y, ki is not None and gi is not None and ki != gi))
usable = [r for r in rows if r[3]]
print("clicked clips", len(rows), "with both players in the candidate pool:", len(usable))
cl = np.array([r[0] for r in usable]); grp = np.array([src[c] for c in cl])
fit = lambda sel: HGB(max_iter=250, learning_rate=0.06, max_depth=5, min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(
    np.vstack([usable[i][1] for i in sel]), np.concatenate([usable[i][2] for i in sel]))
oof_model = {}
for tr, te in GroupKFold(5).split(cl, groups=grp):
    m = fit(tr)
    for i in te: oof_model[cl[i]] = m
full = fit(np.arange(len(usable)))


def best_pair(c, model):
    t = tabs[c]; I, J, P = pairs(c, t); s = model.predict_proba(P)[:, 1]; q = s.argmax(); return t[0][I[q]], t[0][J[q]]


def follow(d, start_box, a0):
    n = len(d["frames"]); out = np.full((n, 4), np.nan); out[a0] = start_box
    for step in (1, -1):
        prev = start_box.copy(); j = a0 + step
        while 0 <= j < n:
            f = d["frames"][j]; b = f["pb"][f["pc"] >= 0.15]
            if len(b):
                hp = prev[3] - prev[1]; cp = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
                cc = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
                cost = np.linalg.norm(cc - cp, axis=1) / max(hp, 1) + 1.5 * np.abs(np.log((b[:, 3] - b[:, 1]) / max(hp, 1))); i = cost.argmin()
                if cost[i] < 1.5: out[j] = b[i]; prev = b[i]
            j += step
    return out


tracks = {}
for c in clips:
    if tabs[c] is None or len(tabs[c][0]) < 2: tracks[c] = old[c]; continue
    kb, gb = best_pair(c, oof_model.get(c, full))
    tracks[c] = dict(kick=follow(det[c], kb, ANCHOR), keep=follow(det[c], gb, ANCHOR), kick_a=kb, keep_a=gb)
pickle.dump(tracks, open(f"{C}/tracks.pkl", "wb"))


def hitrate(tr_):
    k = g = n = 0
    for c in ck.index:
        if c not in tr_: continue
        w, h = det[c]["wh"]; r = ck.loc[c]; n += 1
        for who, bx in (("kicker", tr_[c]["kick_a"]), ("keeper", tr_[c]["keep_a"])):
            p = (r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h); ok = bx[0] <= p[0] <= bx[2] and bx[1] <= p[1] <= bx[3]
            if who == "kicker": k += ok
            else: g += ok
    return k / n, g / n
print("OLD picker  kicker %.3f keeper %.3f" % hitrate(old))
print("NEW picker  kicker %.3f keeper %.3f" % hitrate(tracks))
chg = [c for c in clips if not np.allclose(tracks[c]["kick_a"], old[c]["kick_a"]) or not np.allclose(tracks[c]["keep_a"], old[c]["keep_a"])]
print("clips whose pick changed:", len(chg), "(train %d, private %d)" % (sum(c in src for c in chg), sum(c not in src for c in chg)))
pickle.dump(chg, open(f"{C}/changed_picks.pkl", "wb"))
