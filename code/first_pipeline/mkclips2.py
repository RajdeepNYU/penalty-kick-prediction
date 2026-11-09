"""Stage E1: stabilised ball-centred goal-aligned clips. Each of the last T=12 frames of F is warped into last-frame coordinates (camera homography) and
cropped around the stationary ball, rotated so the ball->keeper direction points up (then image-left = kicker's left; h-flip = exact mirror).
Writes cache/clips2.npy [N,12,W,W,3] uint8 RGB and cache/clips2_meta.npy [N,12,2] (dt seconds, exists)."""
import cv2, os, pickle, time, numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
T, W = 12, 160
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
clips = list(gt["clip"]) + list(pm["clip"]); split = ["train"] * len(gt) + ["val"] * len(pm); N = len(clips)
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
mm = np.lib.format.open_memmap(f"{C}/clips2.npy", mode="w+", dtype=np.uint8, shape=(N, T, W, W, 3)); meta = np.zeros((N, T, 2), np.float32)
t0 = time.time()
for n, (c, sp) in enumerate(zip(clips, split)):
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); t = tracks[c]
    if b is None or k is None or t is None or t["keep"] is None: continue
    ball = b["ball_last"]; kb = t["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]
    if len(vv) == 0: continue
    if not np.isfinite(ball[0]):
        a = (k["box"][-1][[0, 2]].mean(), k["box"][-1][3]); ball = np.array(a)
    kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]); fv = kf - ball; ang = -np.pi / 2 - np.arctan2(fv[1], fv[0])
    hL = max(float(k["box"][-1][3] - k["box"][-1][1]), 20.0); side = 3.2 * hL; s = W / side
    cs, sn = np.cos(ang), np.sin(ang); R = np.array([[cs, -sn], [sn, cs]]) * s
    To = np.eye(3); To[:2, :2] = R; To[:2, 2] = np.array([W / 2, W / 2]) - R @ ball                       # last-frame coords -> output pixels
    cap = cv2.VideoCapture(f"{D}/{sp}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    nF = b["nF"]; idx = list(range(max(0, nF - T), nF))
    for q, i in enumerate(idx):
        A = To @ b["M"][i]; img = cv2.warpPerspective(fr[b["F"][i]], A, (W, W), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        slot = T - len(idx) + q; mm[n, slot] = img[:, :, ::-1]; meta[n, slot] = [b["ts"][i] - b["ts"][-1], 1.0]
    for slot in range(T - len(idx)): mm[n, slot] = mm[n, T - len(idx)]                                    # front-pad with the earliest frame (meta exists=0)
    if n % 200 == 0: print(n, f"{(time.time() - t0) / (n + 1):.2f}s/clip", flush=True)
mm.flush(); np.save(f"{C}/clips2_meta.npy", meta); print("done")
