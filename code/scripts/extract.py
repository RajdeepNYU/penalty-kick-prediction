"""Stage 1: person/ball detection at fixed time offsets before each clip's last frame.
Resumable; writes one pickle per clip into cache/det/."""
import os  # notebook: paths from the environment
import cv2, os, pickle, sys, time, numpy as np, pandas as pd
from ultralytics import YOLO

ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"
OUT = os.environ["Q4_CACHE"] + r"\det"
DET_W = ROOT + r"\Q1-Q3work\yolo11x.pt"   # COCO YOLO11x
OFFS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.06, 0.03, 0.0]  # seconds before last frame
os.makedirs(OUT, exist_ok=True)

tr = pd.read_csv(f"{D}/train_ground_truth.csv")["clip"].tolist()
pr = pd.read_csv(f"{D}/private_metadata.csv")["clip"].tolist()
todo = [("train", c) for c in tr] + [("val", c) for c in pr]
if len(sys.argv) > 1: todo = todo[:int(sys.argv[1])]
model = YOLO(DET_W)


def read_clip(path):
    cap = cv2.VideoCapture(path); fr, ts = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f); ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    return fr, np.array(ts)


def torso_colour(hsv, b):
    out = []
    for x0, y0, x1, y1 in b.astype(int):
        ty0 = y0 + (y1 - y0) // 5; ty1 = y0 + (y1 - y0) * 3 // 5
        tx0 = x0 + (x1 - x0) // 4; tx1 = x1 - (x1 - x0) // 4
        t = hsv[max(ty0, 0):max(ty1, ty0 + 1), max(tx0, 0):max(tx1, tx0 + 1)].reshape(-1, 3)
        if len(t) == 0: t = np.zeros((1, 3), np.uint8)
        hh = np.bincount((t[:, 0] // 23).clip(0, 7), weights=t[:, 1] / 255., minlength=8) / len(t)
        out.append(np.r_[hh, t[:, 1].mean() / 255, t[:, 2].mean() / 255])
    return np.array(out).reshape(-1, 10)


t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    fr, ts = read_clip(f"{D}/{split}/{c}")
    last = ts[-1]
    idx = [int(np.abs(ts - (last - o)).argmin()) for o in OFFS]
    ims = [fr[i] for i in idx]
    h, w = ims[0].shape[:2]
    res = model.predict(ims, imgsz=1280, conf=0.1, classes=[0, 32], verbose=False, device=0, half=True, batch=13)
    frames = []
    for im, r in zip(ims, res):
        b = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy()
        p = cl == 0
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        frames.append(dict(pb=b[p], pc=cf[p], col=torso_colour(hsv, b[p]), bb=b[~p], bc=cf[~p]))
    pickle.dump(dict(wh=(w, h), ts=ts[idx], idx=idx, offs=OFFS, nframes=len(fr), dur=float(last), frames=frames), open(fp, "wb"))
    n += 1
    if n % 50 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
