"""Sequence channels for the ViTPose stream (cache/kptsE, 17-kp body model): seqfeat.build on pseudo-RTMW arrays. Writes cache/seqE.pkl."""
import pickle, numpy as np, pandas as pd
import seqfeat
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}; rows = {}; skipped = 0
for c in list(gt["clip"]) + list(pm["clip"]):
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k0 = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); ks = pickle.load(open(f"{C}/kptsE/{c}.pkl", "rb"))
    if b is None or k0 is None or ks is None or tracks[c] is None or tracks[c]["keep"] is None: skipped += 1; continue
    kb = dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"]); a = seqfeat.build(b, kb, tracks[c], meta[c], False); m = seqfeat.build(b, kb, tracks[c], meta[c], True)
    if a is None or m is None: skipped += 1; continue
    rows[c] = (a, m)
pickle.dump(rows, open(f"{C}/seqE.pkl", "wb")); print("stream E built", len(rows), "skipped", skipped)
