"""Stage S3b: classical blob detector for the ball on the temporal-median image (ballmed.py), fused with the YOLO-on-median and consensus estimates.
Writes cache/ball_final.pkl: clip -> dict(ball (x,y in last-frame coords), method, support, ...)."""
import os  # notebook: paths from the environment
import pickle, os, cv2, numpy as np, pandas as pd
ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root"); D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]; W = 192
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); clips = list(gt["clip"]) + list(pm["clip"])


def blobs(med):
    up = cv2.resize(med, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC); g = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))           # bright things smaller than ~31 px (ball ~12-18 px upscaled)
    m = ((th > 28) & (hsv[:, :, 1] < 90)).astype(np.uint8); n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8); out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 25 or a > 700 or max(w, h) > 2.2 * min(w, h) + 2 or a < 0.5 * w * h: continue                       # size, aspect (rejects lines), fill (rejects thin shapes)
        out.append((cen[i][0] / 3, cen[i][1] / 3, float(th[lab == i].mean()), a / 9.0))
    return out


res = {}; strong_agree = []; weak_info = []
for c in clips:
    fp = f"{C}/ballmed/{c}.pkl"; bp = f"{C}/ballspot/{c}.pkl"
    if not (os.path.exists(fp) and os.path.exists(bp)): continue
    m = pickle.load(open(fp, "rb")); b = pickle.load(open(bp, "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
    if m is None or b is None: continue
    hL = m["hL"]; bx = k["box"][-1]; feet = np.array([(bx[0] + bx[2]) / 2, bx[3] - 0.1 * hL]); s = W / (3.6 * hL)
    to_img = lambda u, v: np.array([u, v]) / s + feet - np.array([W / 2, W / 2]) / s
    bl = blobs(m["med"]); cand = []
    for u, v, resp, area in bl:
        pos = to_img(u, v); dfeet = np.hypot(*(pos - feet)) / hL; cand.append((pos, resp / (1 + (dfeet / 0.8) ** 2), "blob"))
    if np.isfinite(m["ball_med"][0]): cand.append((m["ball_med"], 25.0 * m["conf"] * 8 / (1 + (np.hypot(*(m["ball_med"] - feet)) / hL / 0.8) ** 2) + 5, "yolo_med"))
    blob_best = max([x for x in cand if x[2] == "blob"], key=lambda x: x[1], default=None); cons = b["ball_last"]; sup = b["support"]
    if sup >= 3 and np.isfinite(cons[0]):
        near = [x for x in cand if np.hypot(*(x[0] - cons)) / hL < 0.3]
        if blob_best is not None and sup >= 6: strong_agree.append(np.hypot(*(blob_best[0] - cons)) / hL < 0.3)
        final, method = (cons, "consensus")
    else:
        best = max(cand, key=lambda x: x[1], default=None)
        final, method = (best[0], best[2]) if best is not None else ((cons, "consensus_weak") if np.isfinite(cons[0]) else (feet + np.array([0.0, 0.0]), "feet_fallback"))
    d_foot = np.hypot(*(final - feet)) / hL
    res[c] = dict(ball=np.asarray(final, float), method=method, support=sup, d_feet=d_foot, hL=hL)
    if sup < 3: weak_info.append((method, d_foot))
pickle.dump(res, open(f"{C}/ball_final.pkl", "wb")); r = pd.DataFrame(res).T
print("clips:", len(r), "| method counts:", r["method"].value_counts().to_dict())
print("blob detector agrees with strong consensus (support>=6, within 0.3 body-heights): %.3f (n=%d)" % (np.mean(strong_agree), len(strong_agree)))
w = pd.DataFrame(weak_info, columns=["method", "d"]); print("weak-ball clips (support<3): %d; final ball within 0.6 body-heights of the feet: %.3f; by method:" % (len(w), (w.d < 0.6).mean()), w.groupby("method")["d"].agg(["size", lambda x: (x < 0.6).mean()]).round(2).to_dict("index"))
