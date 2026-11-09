"""Stage S3: ball from the TEMPORAL MEDIAN of homography-stabilised frames around the kicker's feet (moving legs vanish, the stationary ball stays), detected with YOLO11x (ball class) on the upscaled median image.
usage: python ballmed.py [N] -> cache/ballmed/<clip>.pkl : ball_med (x,y in last-frame coords) , conf, agreement with the consensus ball from ballspot.py"""
import os  # notebook: paths from the environment
import cv2, os, pickle, sys, time, numpy as np, pandas as pd
from ultralytics import YOLO
ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root"); D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]; OUT = f"{C}/ballmed"; os.makedirs(OUT, exist_ok=True)
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]
if len(sys.argv) > 1: todo = todo[::max(len(todo) // int(sys.argv[1]), 1)][:int(sys.argv[1])]
yolo = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt"); W = 192; t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
    if b is None or k is None: pickle.dump(None, open(fp, "wb")); continue
    bx = k["box"][-1]; hL = max(bx[3] - bx[1], 20.0); feet = np.array([(bx[0] + bx[2]) / 2, bx[3] - 0.1 * hL]); side = 3.6 * hL; s = W / side
    To = np.eye(3); To[0, 0] = To[1, 1] = s; To[:2, 2] = np.array([W / 2, W / 2]) - s * feet
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    stack = np.stack([cv2.warpPerspective(fr[b["F"][i]], To @ b["M"][i], (W, W), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for i in range(b["nF"])])
    med = np.median(stack, axis=0).astype(np.uint8); up = cv2.resize(med, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    r = yolo.predict(up, imgsz=576, conf=0.01, classes=[32], verbose=False, device=0)[0]; bb = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    best = None
    if len(bb):
        ctr = (bb[:, :2] + bb[:, 2:]) / 2 / 3; dist = np.hypot(ctr[:, 0] - W / 2, ctr[:, 1] - W / 2) / W; score = cf / (1 + (dist / 0.3) ** 2); q = int(score.argmax()); best = ctr[q] / s + (feet - np.array([W / 2, W / 2]) / s); bconf = float(cf[q])
    bl = b["ball_last"]; agree = float(np.hypot(*(best - bl)) / hL) if best is not None and np.isfinite(bl[0]) else np.nan
    pickle.dump(dict(ball_med=best if best is not None else np.array([np.nan, np.nan]), conf=bconf if best is not None else 0.0, agree=agree, hL=hL, med=med), open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
