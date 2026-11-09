import pickle, glob, numpy as np, os, sys
C = r"C:\path\to\project-root\Q4_work\cache"; a, b = sys.argv[1], sys.argv[2]; out = []
for f in sorted(glob.glob(f"{C}/{a}/*.pkl"))[:40]:
    c = os.path.basename(f); h = pickle.load(open(f, "rb")); g = pickle.load(open(f"{C}/{b}/{c}", "rb"))
    if h is None or g is None: continue
    d = np.linalg.norm(h["kp"][:, :17] - g["kp"][:, :17], axis=2); m = (h["sc"][:, :17] > 1.5) & (g["sc"][:, :17] > 1.5); out.append((np.median(d[m]) if m.any() else np.nan, m.mean()))
o = np.array(out); print(a, "vs", b, "median px diff", np.nanmedian(o[:, 0]).round(2), "both-confident frac", o[:, 1].mean().round(2), "n", len(o))
