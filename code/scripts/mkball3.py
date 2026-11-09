"""Ball 3: the stationary ball sits on the penalty spot, so the NBJW camera (nbjw_cam.py) gives an independent ball estimate: the spot (x = +-41.5 m,
y = 0, ball centre 0.11 m up) projected into the last frame. Where the ball2 detection is weak (score in the public bottom third) and lands more than 1 m
from the spot on a usable calibration (reprojection error < 10 px), and the spot is nearer the kicker's feet than ball2, ball3 = the projected spot;
elsewhere ball3 = ball2. Uses only the clip's own frames.
-> cache/ball3.pkl (ball2 layout, method 'nbjw_spot' where replaced)"""
import pickle, numpy as np
from common import *
import kinN
from wasb_ball import foot_refs

DSPOT = float(os.environ.get("DSPOT", "1.0")); OUT = "ball3" if DSPOT == 1.0 else "ball3b"          # DSPOT=0.5: ball3b, any ball2 score from 0.5 m (pipeline27)
B2 = pickle.load(open(f"{PC}/ball2.pkl", "rb")); TLOW =float(np.percentile([B2[c]["score"] for c in gt["clip"] if B2.get(c)], 33.3)); out = {}; rows = []
for c in CLIPS:
    out[c] = B2.get(c); b = load("ballspot", c)
    if b is None or not B2.get(c) or not os.path.exists(f"{PC}/nbjw/{c}.pkl"): continue
    r = pickle.load(open(f"{PC}/nbjw/{c}.pkl", "rb")); cam = kinN.camera(c, b)
    if r is None or cam is None or r["rep_err"] >= 10: continue
    Pl = np.asarray(b["M"][r["frame"]], float) @ r["P"] if r["frame"] != len(b["F"]) - 1 else r["P"]
    q = Pl @ np.array([41.5 * cam["s"], 0.0, -0.11, 1.0]); spot = q[:2] / q[2]
    if cam["dspot"] > DSPOT and (B2[c]["score"] <= TLOW or DSPOT != 1.0):
        k = load("kpts", c); hL = max(float(k["box"][-1][3] - k["box"][-1][1]), 20.0); refs = foot_refs(b, c)
        df = lambda p: float(np.min(np.hypot(*(refs - p).T)) / hL) if len(refs) else np.nan
        if not df(spot) < df(np.asarray(B2[c]["ball"], float)): continue                    # arbiter: at contact - 0.02 s the ball is next to a foot; keep the nearer one
        out[c] = dict(B2[c], ball=spot, method="nbjw_spot", d_foot=df(spot), d_ball2=float(np.hypot(*(spot - np.asarray(B2[c]["ball"], float))) / hL))
        rows.append((c in SPLIT and SPLIT[c] == "train", cam["dspot"], df(np.asarray(B2[c]["ball"], float)), df(spot), out[c]["d_ball2"]))
pickle.dump(out, open(f"{PC}/{OUT}.pkl", "wb")); A = np.array(rows, float)
print(f"{OUT}: replaced {len(rows)} clips (public {int(A[:, 0].sum())}, private {int(len(A) - A[:, 0].sum())}) | ball2-to-spot median {np.median(A[:, 1]):.2f} m | "
      f"distance to the nearest foot (kicker heights): ball2 median {np.nanmedian(A[:, 2]):.3f}, spot {np.nanmedian(A[:, 3]):.3f}; spot closer in {np.nanmean(A[:, 3] < A[:, 2]):.2f} | moved {np.median(A[:, 4]):.2f} h")
