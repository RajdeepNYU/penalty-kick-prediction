"""Second measurement stream for the sequence network: seqfeat.build on the RTMW3D model's own 2D keypoints (independent estimate of the same joints). Writes cache/seqB.pkl."""
import pickle, numpy as np, pandas as pd
import seqfeat
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
meta = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}
rows = {}; skipped = 0
for c in list(gt["clip"]) + list(pm["clip"]):
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); k3 = pickle.load(open(f"{C}/kpts3d/{c}.pkl", "rb"))
    if b is None or k is None or k3 is None or tracks[c] is None or tracks[c]["keep"] is None: skipped += 1; continue
    nF = b["nF"]; kp = np.zeros((nF, 133, 2), np.float32); sc = np.zeros((nF, 133), np.float32); kp[:, :23] = k3["kp2"]; sc[:, :23] = k3["sc"] * 10
    kb = dict(kp=kp, sc=sc, box=k["box"]); a = seqfeat.build(b, kb, tracks[c], meta[c], False); m = seqfeat.build(b, kb, tracks[c], meta[c], True)
    if a is None or m is None: skipped += 1; continue
    rows[c] = (a, m)
pickle.dump(rows, open(f"{C}/seqB.pkl", "wb")); print("built", len(rows), "skipped", skipped)
