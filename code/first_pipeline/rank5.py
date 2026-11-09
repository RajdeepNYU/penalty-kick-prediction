"""Stage S2c: kicker/keeper picker v5 (ball-to-feet features; compares the joint pair scorer with a two-stage keeper->kicker decoder). v4 = (adds the TTA candidate source from scene2.py, lower confidence floors) with goal-box features and goal-zoomed extra candidates (scene.py). Trained on clicks, source-grouped OOF.
usage: python rank3.py [nogoal]   -> prints held-out pick accuracy; writes cache/picks3.pkl (anchor-frame boxes for every clip)."""
import pickle, os, sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
USE_GOAL = not (len(sys.argv) > 1 and sys.argv[1] == "nogoal"); ANCHOR, CONF, KMAX = 6, 0.1, 20
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); ck = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip")
src = dict(zip(gt["clip"], gt["source_id"])); clips = [c for c in list(gt["clip"]) + list(pm["clip"]) if os.path.exists(f"{C}/scene/{c}.pkl") and os.path.exists(f"{C}/scene3/{c}.pkl")]
det = {c: pickle.load(open(f"{C}/det/{c}.pkl", "rb")) for c in clips}; scn = {c: {**pickle.load(open(f"{C}/scene/{c}.pkl", "rb")), **pickle.load(open(f"{C}/scene3/{c}.pkl", "rb"))} for c in clips}
print("clips with scene info:", len(clips), "| goal detected in anchor frame:", np.mean([len(scn[c]["goal_a"]) > 0 for c in clips]).round(3), flush=True)


def match(b, d, j):
    f = d["frames"][j]; bb = f["pb"][f["pc"] >= 0.15]
    if len(bb) == 0: return np.nan, np.nan, np.nan
    h = max(b[3] - b[1], 1); c0 = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); cc = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
    dist = np.linalg.norm(cc - c0, axis=1) / h; ratio = (bb[:, 3] - bb[:, 1]) / h; cost = dist + 1.5 * np.abs(np.log(ratio)); i = cost.argmin()
    if cost[i] > 1.5: return np.nan, np.nan, np.nan
    return (cc[i, 0] - c0[0]) / h, (cc[i, 1] - c0[1]) / h, 1.0


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-6)


def cand_table(c):
    d, sc = det[c], scn[c]; f = d["frames"][ANCHOR]; w, h = d["wh"]; k = f["pc"] >= CONF
    B = [f["pb"][k]]; Cf = [f["pc"][k]]; Col = [f["col"][k]]; Src = [np.zeros(k.sum())]
    ex = sc["extra_a"];
    if len(ex): m = ex[:, 4] >= 0.15; B.append(ex[m, :4]); Cf.append(ex[m, 4]); Col.append(sc["extra_col"][m]); Src.append(np.ones(m.sum()))
    e2 = sc["extra2"]
    if len(e2): m = e2[:, 4] >= 0.05; B.append(e2[m, :4]); Cf.append(e2[m, 4]); Col.append(sc["extra2_col"][m]); Src.append(np.full(m.sum(), 2.0))
    b, cf, col, sr = np.concatenate(B), np.concatenate(Cf), np.concatenate(Col), np.concatenate(Src)
    if len(b) == 0: return None
    o = np.argsort(-cf); keep = []
    for i in o:
        if (b[i, 3] - b[i, 1]) < 10: continue
        if all(iou(b[i], b[j]) < 0.5 for j in keep): keep.append(i)
    b, cf, col, sr = b[keep], cf[keep], col[keep], sr[keep]; n0 = len(b)
    g = sc["goal_a"][0] if len(sc["goal_a"]) else None
    pri = cf * (b[:, 3] - b[:, 1])
    if g is not None and USE_GOAL:
        gw = g[2] - g[0]; ingoal = (b[:, 0] + b[:, 2]) / 2 > g[0] - 0.1 * gw; ingoal &= (b[:, 0] + b[:, 2]) / 2 < g[2] + 0.1 * gw; ingoal &= (b[:, 3] > g[1]) & (b[:, 3] < g[3] + 0.4 * (g[3] - g[1]))
        top = np.argsort(-(cf * ingoal))[:4]; boost = np.zeros(len(b)); boost[top[ingoal[top]]] = 1e6; pri = pri + boost          # keep only the 4 most confident people at the goal mouth
    sel = np.argsort(-pri)[:KMAX]; b, cf, col, sr = b[sel], cf[sel], col[sel], sr[sel]; n = len(b)
    cx = (b[:, 0] + b[:, 2]) / 2 / w; cy = (b[:, 1] + b[:, 3]) / 2 / h; bw = (b[:, 2] - b[:, 0]) / w; bh = (b[:, 3] - b[:, 1]) / h
    cd = np.abs(np.nan_to_num(col) - np.median(np.nan_to_num(col), axis=0)).sum(1); pos = np.c_[cx, cy]; Dm = np.linalg.norm(pos[:, None] - pos[None], axis=2); np.fill_diagonal(Dm, 9)
    mot = np.array([[*match(bx, d, 12)[:2], *match(bx, d, 3)[:2], match(bx, d, 8)[2] + match(bx, d, 12)[2] + match(bx, d, 3)[2]] for bx in b], float); sp = np.hypot(mot[:, 0], mot[:, 1]); sp_r = np.where(np.isfinite(sp), sp, -1).argsort().argsort() / n
    balls = [d["frames"][j]["bb"][d["frames"][j]["bc"] >= 0.1] for j in (ANCHOR - 1, ANCHOR, ANCHOR + 1)] + [sc["balls2"][:, :4] if len(sc["balls2"]) else np.zeros((0, 4))]
    ball = np.concatenate([x[:, :4] for x in balls if len(x)]) if any(len(x) for x in balls) else np.zeros((0, 4))
    if len(ball):
        feet = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]; bcen = np.c_[(ball[:, 0] + ball[:, 2]) / 2, (ball[:, 1] + ball[:, 3]) / 2]; bdist = (np.linalg.norm(feet[:, None] - bcen[None], axis=2) / np.maximum(b[:, 3] - b[:, 1], 1)[:, None]).min(1)
    else: bdist = np.full(n, np.nan)
    bd_rank = np.where(np.isfinite(bdist), np.nan_to_num(bdist, nan=99).argsort().argsort(), n) / n; nb = np.full(n, float(len(ball)))
    X = [cx, cy, bw, bh, cf, sr, *col.T, cd, Dm.min(1), np.full(n, n), cy.argsort().argsort() / n, cx.argsort().argsort() / n, bh / np.median(bh), *mot.T, sp, sp_r, bdist, bd_rank, (bd_rank == 0).astype(float), nb, np.log1p(bdist)]
    if USE_GOAL:
        if g is not None:
            gw, gh = g[2] - g[0], g[3] - g[1]; gcx = (g[0] + g[2]) / 2; fx, fy = (b[:, 0] + b[:, 2]) / 2, b[:, 3]; dxg, dyg = (fx - gcx) / gw, (fy - g[3]) / gw
            ins_x = ((fx > g[0] - 0.05 * gw) & (fx < g[2] + 0.05 * gw)).astype(float); ins_y = ((fy > g[1]) & (fy < g[3] + 0.35 * gh)).astype(float)
            X += [dxg, dyg, np.hypot(dxg, dyg), ins_x, ins_x * ins_y, (b[:, 3] - b[:, 1]) / gw, np.full(n, g[4]), np.full(n, gw / w), np.full(n, gw / max(gh, 1))]
        else: X += [np.full(n, np.nan)] * 9
    return b, np.array(X).T, h


def pairs(tab):
    b, X, h0 = tab; n = len(b); I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); m = I != J; I, J = I[m], J[m]; h = np.maximum(b[:, 3] - b[:, 1], 1)
    dx = ((b[J, 0] + b[J, 2]) / 2 - (b[I, 0] + b[I, 2]) / 2) / h[I]; dy = ((b[J, 1] + b[J, 3]) / 2 - (b[I, 1] + b[I, 3]) / 2) / h[I]
    return I, J, np.c_[X[I], X[J], dx, dy, np.hypot(dx, dy), h[J] / h[I]]


def click_idx(c, b, who):
    w, h = det[c]["wh"]; r = ck.loc[c]; p = np.array([r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h]); ins = (b[:, 0] <= p[0]) & (p[0] <= b[:, 2]) & (b[:, 1] <= p[1]) & (p[1] <= b[:, 3])
    if not ins.any(): return None
    ar = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); return int(np.where(ins)[0][ar[ins].argmin()])


tabs = {c: cand_table(c) for c in clips}; usable = []; n_clicked = n_pool = 0; POOL_FAIL = {}
for c in clips:
    t = tabs[c]
    if t is None or c not in ck.index: continue
    n_clicked += 1; ki, gi = click_idx(c, t[0], "kicker"), click_idx(c, t[0], "keeper")
    if ki is None or gi is None or ki == gi: POOL_FAIL[c] = ("kicker" if ki is None else "") + ("keeper" if gi is None else "") + ("same" if ki == gi and ki is not None else "")
    I, J, P = pairs(t); ok = ki is not None and gi is not None and ki != gi; n_pool += ok
    if ok: usable.append((c, P, ((I == ki) & (J == gi)).astype(int)))
print(f"clicked clips {n_clicked}: both players present in the candidate pool: {n_pool} ({n_pool / n_clicked:.3f}) | goal features {'ON' if USE_GOAL else 'OFF'}", flush=True)
cl = np.array([u[0] for u in usable]); grp = np.array([src[c] for c in cl])
fit = lambda sel: HGB(max_iter=250, learning_rate=0.06, max_depth=5, min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(np.vstack([usable[i][1] for i in sel]), np.concatenate([usable[i][2] for i in sel]))
oof = {}
for tr, te in GroupKFold(5).split(cl, groups=grp):
    m = fit(tr)
    for i in te: oof[cl[i]] = m
full = fit(np.arange(len(usable)))
picks = {}
for c in clips:
    t = tabs[c]
    if t is None or len(t[0]) < 2: picks[c] = None; continue
    I, J, P = pairs(t); s = (oof.get(c, full)).predict_proba(P)[:, 1]; q = s.argmax(); picks[c] = dict(kick=t[0][I[q]], keep=t[0][J[q]], score=float(s[q]), margin=float(s[q] - np.partition(s, -2)[-2]) if len(s) > 1 else 1.0)
hk = hg = hb = nn = 0
for c in clips:
    if c not in ck.index or picks[c] is None: continue
    w, h = det[c]["wh"]; r = ck.loc[c]; nn += 1; res = []
    for who, bx in (("kicker", picks[c]["kick"]), ("keeper", picks[c]["keep"])):
        p = (r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h); res.append(bx[0] <= p[0] <= bx[2] and bx[1] <= p[1] <= bx[3])
    hk += res[0]; hg += res[1]; hb += res[0] and res[1]
print(f"HELD-OUT picks on {nn} clicked clips: kicker {hk / nn:.3f} | keeper {hg / nn:.3f} | both {hb / nn:.3f}   (v4 pair picker: kicker 0.984, keeper 0.995, both 0.980)")
pickle.dump(picks, open(f"{C}/picks5{'' if USE_GOAL else '_nogoal'}.pkl", "wb")); pickle.dump(POOL_FAIL, open(f"{C}/pool_fail5.pkl", "wb")); print('pool failures:', len(POOL_FAIL), pd.Series(list(POOL_FAIL.values())).value_counts().to_dict())


# ================= two-stage decoder: keeper scorer -> kicker scorer =================
def rel(tab, g):
    b = tab[0]; h = np.maximum(b[:, 3] - b[:, 1], 1); dx = ((b[:, 0] + b[:, 2]) / 2 - (b[g, 0] + b[g, 2]) / 2) / h[g]; dy = ((b[:, 1] + b[:, 3]) / 2 - (b[g, 1] + b[g, 3]) / 2) / h[g]
    return np.c_[dx, dy, np.hypot(dx, dy), h / h[g]]
rowsK = []
for c in clips:
    t = tabs[c]
    if t is None or c not in ck.index: continue
    ki, gi = click_idx(c, t[0], "kicker"), click_idx(c, t[0], "keeper"); rowsK.append((c, t, ki, gi))
cl2 = np.array([r[0] for r in rowsK]); grp2 = np.array([src[c] for c in cl2]); hitK = hitG = hitB = 0; mk = lambda: HGB(max_iter=300, learning_rate=0.05, max_depth=5, min_samples_leaf=15, l2_regularization=1.0, random_state=0)
two = {}
for tr, te in GroupKFold(5).split(cl2, groups=grp2):
    Xg = np.vstack([rowsK[i][1][1] for i in tr if rowsK[i][3] is not None]); yg_ = np.concatenate([(np.arange(len(rowsK[i][1][0])) == rowsK[i][3]).astype(int) for i in tr if rowsK[i][3] is not None])
    mg = mk().fit(Xg, yg_)
    Xk = np.vstack([np.c_[rowsK[i][1][1], rel(rowsK[i][1], rowsK[i][3])] for i in tr if rowsK[i][2] is not None and rowsK[i][3] is not None and rowsK[i][2] != rowsK[i][3]])
    yk_ = np.concatenate([(np.arange(len(rowsK[i][1][0])) == rowsK[i][2]).astype(int) for i in tr if rowsK[i][2] is not None and rowsK[i][3] is not None and rowsK[i][2] != rowsK[i][3]])
    mkk = mk().fit(Xk, yk_)
    for i in te:
        c, t, ki, gi = rowsK[i]; b = t[0]
        if len(b) < 2: continue
        g = int(mg.predict_proba(t[1])[:, 1].argmax()); sk = mkk.predict_proba(np.c_[t[1], rel(t, g)])[:, 1]
        ov = np.array([iou(b[j], b[g]) for j in range(len(b))]); sk[ov > 0.3] = -1; k = int(sk.argmax()); two[c] = (b[k], b[g])
res = []
for c, (kb_, gb_) in two.items():
    w, h = det[c]["wh"]; r = ck.loc[c]; pk = (r["kicker_x_norm"] * w, r["kicker_y_norm"] * h); pg = (r["keeper_x_norm"] * w, r["keeper_y_norm"] * h)
    inb = lambda bx, q: bx[0] <= q[0] <= bx[2] and bx[1] <= q[1] <= bx[3]; res.append((inb(kb_, pk), inb(gb_, pg)))
res = np.array(res); print(f"TWO-STAGE held-out on {len(res)} clicked clips: kicker {res[:, 0].mean():.3f} | keeper {res[:, 1].mean():.3f} | both {(res[:, 0] & res[:, 1]).mean():.3f}")
pickle.dump(two, open(f"{C}/picks5_two.pkl", "wb"))

# ---- final picks for ALL clips: OOF for clicked clips (above), models fit on all clicked clips for the rest
Xg = np.vstack([r[1][1] for r in rowsK if r[3] is not None]); yg_ = np.concatenate([(np.arange(len(r[1][0])) == r[3]).astype(int) for r in rowsK if r[3] is not None]); mg = mk().fit(Xg, yg_)
ok_ = [r for r in rowsK if r[2] is not None and r[3] is not None and r[2] != r[3]]; mkk = mk().fit(np.vstack([np.c_[r[1][1], rel(r[1], r[3])] for r in ok_]), np.concatenate([(np.arange(len(r[1][0])) == r[2]).astype(int) for r in ok_]))
final = {}
for c in clips:
    t = tabs[c]
    if c in two: final[c] = dict(kick=two[c][0], keep=two[c][1], source="oof"); continue
    if t is None or len(t[0]) < 2: final[c] = None; continue
    b = t[0]; g = int(mg.predict_proba(t[1])[:, 1].argmax()); sk = mkk.predict_proba(np.c_[t[1], rel(t, g)])[:, 1]; sk[np.array([iou(b[j], b[g]) for j in range(len(b))]) > 0.3] = -1
    final[c] = dict(kick=b[int(sk.argmax())], keep=b[g], source="full")
pickle.dump(final, open(f"{C}/picks5_final.pkl", "wb")); print("final picks:", sum(v is not None for v in final.values()), "of", len(clips))
