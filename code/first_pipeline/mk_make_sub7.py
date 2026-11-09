p = r"C:\path\to\project-root\Q4_work"; s = open(f"{p}\\make_sub5.py", encoding="utf-8").read()
def rep(a, b):
    global s; assert a in s, a[:70]; s = s.replace(a, b, 1)
rep('"""v5 submission.', '"""v7 submission = v6 recipe on the improved detections (two-stage picker, TTA candidates, goal box, filled tracks, fused ball).\nv5 recipe (kept for reference):')
rep(r'OUTCSV = ROOT + r"\Q4_work\submission_v5.csv"', r'OUTCSV = ROOT + r"\Q4_work\submission_v7.csv"')
rep('"kick_direction": np.array(ck)[Pk.argmax(1)]', '"kick_direction": np.array(ck)[(0.5 * Pk + 0.5 * np.load(f"{C}/nn_private.npy")).argmax(1)]')
rep('np.save(f"{C}/pk_private_v5.npy", Pk); np.save(f"{C}/pg_private_v5.npy", Pg)', 'np.save(f"{C}/pk_private_v7.npy", Pk); np.save(f"{C}/pg_private_v7.npy", Pg)')
rep('v4 = pd.read_csv(ROOT + r"\\Q4_work\\submission_v4.csv")', 'v4 = pd.read_csv(ROOT + r"\\Q4_work\\submission_v6.csv")'); rep("agreement with v4", "agreement with v6")
open(f"{p}\\make_sub7.py", "w", encoding="utf-8").write(s); print("wrote make_sub7.py")
