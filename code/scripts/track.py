"""Stage 2: choose kicker/keeper at the anchor frame (0.25 s before last frame) with a ranker trained on the
position clicks (out-of-fold by source for train clips), then track both across the 13 timed frames."""
import os  # notebook: paths from the environment
import pickle, os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]
ANCHOR = 6  # OFFS[6] == 0.25 s
CONF = 0.3
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
ck = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip")
src = dict(zip(gt["clip"], gt["source_id"]))
clips = [c for c in list(gt["clip"]) + list(pm["clip"]) if os.path.exists(f"{C}/det/{c}.pkl")]
det = {c: pickle.load(open(f"{C}/det/{c}.pkl", "rb")) for c in clips}


def cand_feats(d, j):
    """Per-candidate features for frame j; returns (X, boxes) after conf filter."""
    f = d["frames"][j]; w, h = d["wh"]; k = f["pc"] >= CONF
    b = f["pb"][k]; col = f["col"][k]; cf = f["pc"][k]
    if len(b) == 0: return np.zeros((0, 24)), b
    cx = (b[:, 0] + b[:, 2]) / 2 / w; cy = (b[:, 1] + b[:, 3]) / 2 / h
    bw = (b[:, 2] - b[:, 0]) / w; bh = (b[:, 3] - b[:, 1]) / h
    cd = np.abs(col - np.median(col, axis=0)).sum(1)
    pos = np.c_[cx, cy]; Dm = np.linalg.norm(pos[:, None] - pos[None], axis=2); np.fill_diagonal(Dm, 9)
    X = np.c_[cx, cy, bw, bh, cf, col, cd, Dm.min(1), np.full(len(b), len(b)),
              cy.argsort().argsort() / len(b), cx.argsort().argsort() / len(b), bh / np.median(bh)]
    return X, b


def click_labels(c, d, b, who):
    w, h = d["wh"]; r = ck.loc[c]; p = np.array([r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h])
    ins = (b[:, 0] <= p[0]) & (p[0] <= b[:, 2]) & (b[:, 1] <= p[1]) & (p[1] <= b[:, 3])
    lab = np.zeros(len(b), int)
    if ins.any():
        ar = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); lab[np.where(ins)[0][ar[ins].argmin()]] = 1
    return lab


# ---- training table from clicked train clips
rows = []
for c in clips:
    if c not in ck.index or c not in src: continue
    X, b = cand_feats(det[c], ANCHOR)
    if len(b) == 0: continue
    ky, gk = click_labels(c, det[c], b, "kicker"), click_labels(c, det[c], b, "keeper")
    for i in range(len(b)): rows.append((c, src[c], X[i], ky[i], gk[i]))
tc = np.array([r[0] for r in rows]); tg = np.array([r[1] for r in rows])
TX = np.stack([r[2] for r in rows]); TK = np.array([r[3] for r in rows]); TG = np.array([r[4] for r in rows])
okK = pd.Series(TK).groupby(tc).transform("max").values > 0; okG = pd.Series(TG).groupby(tc).transform("max").values > 0
fit = lambda m, y: HGB(max_iter=200, learning_rate=0.05, max_depth=4).fit(TX[m], y[m])
folds = list(GroupKFold(5).split(TX, groups=tg))
oof = {}  # clip -> (modelK, modelG) trained without its fold
for tri, tei in folds:
    m = np.zeros(len(TX), bool); m[tri] = True
    mk, mg = fit(m & okK, TK), fit(m & okG, TG)
    for c in set(tc[tei]): oof[c] = (mk, mg)
full = (fit(okK, TK), fit(okG, TG))


def pick(models, d):
    X, b = cand_feats(d, ANCHOR)
    if len(b) == 0: return None, None, b
    pk = models[0].predict_proba(X)[:, 1]; pg = models[1].predict_proba(X)[:, 1]
    ik = pk.argmax(); pg2 = pg.copy()
    if len(b) > 1: pg2[ik] = -1
    return ik, pg2.argmax(), b


def track(d, start_box, a0):
    """Follow a box from the anchor frame to every other frame (greedy nearest, size-consistent)."""
    n = len(d["frames"]); out = np.full((n, 4), np.nan); out[a0] = start_box
    for step in (1, -1):
        prev = start_box.copy(); j = a0 + step
        while 0 <= j < n:
            f = d["frames"][j]; b = f["pb"][f["pc"] >= 0.15]
            if len(b):
                hp = prev[3] - prev[1]; cp = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
                cc = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
                dist = np.linalg.norm(cc - cp, axis=1) / max(hp, 1); ratio = (b[:, 3] - b[:, 1]) / max(hp, 1)
                cost = dist + 1.5 * np.abs(np.log(ratio))
                i = cost.argmin()
                if cost[i] < 1.5: out[j] = b[i]; prev = b[i]
            j += step
    return out


tracks = {}
for c in clips:
    d = det[c]; models = oof.get(c, full)
    ik, ig, b = pick(models, d)
    if ik is None: tracks[c] = None; continue
    tracks[c] = dict(kick=track(d, b[ik], ANCHOR), keep=track(d, b[ig], ANCHOR) if ig is not None else None,
                     kick_a=b[ik], keep_a=b[ig])
pickle.dump(tracks, open(f"{C}/tracks_base.pkl", "wb"))

# ---- report picking accuracy on clicked clips (OOF)
okc = tot = okg = 0
for c, t in tracks.items():
    if t is None or c not in ck.index: continue
    w, h = det[c]["wh"]; r = ck.loc[c]; tot += 1
    for who, bx in (("kicker", t["kick_a"]), ("keeper", t["keep_a"])):
        p = np.array([r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h])
        hit = bx[0] <= p[0] <= bx[2] and bx[1] <= p[1] <= bx[3]
        if who == "kicker": okc += hit
        else: okg += hit
print(f"clips {len(tracks)}, with clicks {tot}: kicker {okc / tot:.3f} keeper {okg / tot:.3f}")
v = [t for t in tracks.values() if t is not None]
print("no tracks:", sum(t is None for t in tracks.values()),
      "| mean tracked frames of 13: kicker", np.mean([np.isfinite(t["kick"][:, 0]).sum() for t in v]).round(1),
      "keeper", np.mean([np.isfinite(t["keep"][:, 0]).sum() for t in v if t["keep"] is not None]).round(1))
