p = r"C:\path\to\project-root\Q4_work"; s = open(f"{p}\\rank4.py", encoding="utf-8").read()
def rep(a, b):
    global s; assert a in s, a[:70]; s = s.replace(a, b, 1)
rep('Stage S2b: kicker/keeper PAIR picker v4', 'Stage S2c: kicker/keeper picker v5 (ball-to-feet features; compares the joint pair scorer with a two-stage keeper->kicker decoder). v4 =')
rep('**pickle.load(open(f"{C}/scene2/{c}.pkl", "rb"))}', '**pickle.load(open(f"{C}/scene3/{c}.pkl", "rb"))}')
rep('os.path.exists(f"{C}/scene2/{c}.pkl")]', 'os.path.exists(f"{C}/scene3/{c}.pkl")]')
# ball features
rep('    ball = d["frames"][ANCHOR]["bb"]\n    if len(ball):',
    '    balls = [d["frames"][j]["bb"][d["frames"][j]["bc"] >= 0.1] for j in (ANCHOR - 1, ANCHOR, ANCHOR + 1)] + [sc["balls2"][:, :4] if len(sc["balls2"]) else np.zeros((0, 4))]\n    ball = np.concatenate([x[:, :4] for x in balls if len(x)]) if any(len(x) for x in balls) else np.zeros((0, 4))\n    if len(ball):')
rep("    X = [cx, cy, bw, bh, cf, sr,", "    bd_rank = np.where(np.isfinite(bdist), np.nan_to_num(bdist, nan=99).argsort().argsort(), n) / n; nb = np.full(n, float(len(ball)))\n    X = [cx, cy, bw, bh, cf, sr,")
rep("*mot.T, sp, sp_r, bdist]", "*mot.T, sp, sp_r, bdist, bd_rank, (bd_rank == 0).astype(float), nb, np.log1p(bdist)]")
rep("    return b, np.array(X).T, h\n", "    return b, np.array(X).T, h\n")
rep("picks3{'' if USE_GOAL else '_nogoal'}.pkl", "picks5{'' if USE_GOAL else '_nogoal'}.pkl") if "picks3{'' if USE_GOAL" in s else rep("picks4{'' if USE_GOAL else '_nogoal'}.pkl", "picks5{'' if USE_GOAL else '_nogoal'}.pkl")
rep("pool_fail4.pkl", "pool_fail5.pkl"); rep("(v3 picker: kicker 0.981, keeper 0.993, both 0.974)", "(v4 pair picker: kicker 0.984, keeper 0.995, both 0.980)")
# two-stage experiment appended
s += '''

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
'''
open(f"{p}\\rank5.py", "w", encoding="utf-8").write(s); print("wrote rank5.py")
