p = r"C:\path\to\project-root\Q4_work"; s = open(f"{p}\\rank3.py", encoding="utf-8").read()
def rep(a, b):
    global s; assert a in s, a[:70]; s = s.replace(a, b, 1)
rep('Stage S2: kicker/keeper PAIR picker v3', 'Stage S2b: kicker/keeper PAIR picker v4 (adds the TTA candidate source from scene2.py, lower confidence floors)')
rep('ANCHOR, CONF, KMAX = 6, 0.2, 16', 'ANCHOR, CONF, KMAX = 6, 0.1, 20')
rep('scn = {c: pickle.load(open(f"{C}/scene/{c}.pkl", "rb")) for c in clips}', 'scn = {c: {**pickle.load(open(f"{C}/scene/{c}.pkl", "rb")), **pickle.load(open(f"{C}/scene2/{c}.pkl", "rb"))} for c in clips}')
rep('if os.path.exists(f"{C}/scene/{c}.pkl")]', 'if os.path.exists(f"{C}/scene/{c}.pkl") and os.path.exists(f"{C}/scene2/{c}.pkl")]')
rep('    b, cf, col, sr = np.concatenate(B), np.concatenate(Cf), np.concatenate(Col), np.concatenate(Src)',
    '    e2 = sc["extra2"]\n    if len(e2): m = e2[:, 4] >= 0.05; B.append(e2[m, :4]); Cf.append(e2[m, 4]); Col.append(sc["extra2_col"][m]); Src.append(np.full(m.sum(), 2.0))\n    b, cf, col, sr = np.concatenate(B), np.concatenate(Cf), np.concatenate(Col), np.concatenate(Src)')
rep("picks3{'' if USE_GOAL else '_nogoal'}.pkl", "picks4{'' if USE_GOAL else '_nogoal'}.pkl"); rep("pool_fail.pkl", "pool_fail4.pkl"); rep("(v2 picker: kicker 0.964, keeper 0.958)", "(v3 picker: kicker 0.981, keeper 0.993, both 0.974)")
open(f"{p}\\rank4.py", "w", encoding="utf-8").write(s); print("wrote rank4.py")
