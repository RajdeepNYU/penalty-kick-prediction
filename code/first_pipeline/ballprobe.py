import cv2, numpy as np, pandas as pd, pickle, sys
from ultralytics import YOLO
R = r"C:\path\to\project-root"
gt = pd.read_csv(R + r"\Q4data\train_ground_truth.csv").sample(120, random_state=1)
tracks = pickle.load(open(R + r"\Q4_work\cache\tracks.pkl", "rb"))
m = YOLO(R + r"\Q1-Q3work\yolo11x.pt")
NL = 6; hit = np.zeros(NL); tot = 0
for _, r in gt.iterrows():
    t = tracks[r["clip"]]; kb = t["kick"][12] if np.isfinite(t["kick"][12][0]) else t["kick_a"]
    cap = cv2.VideoCapture(R + rf"\Q4data\train\{r['clip']}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    if len(fr) < NL: continue
    cx, cy = (kb[0] + kb[2]) / 2, (kb[1] + kb[3]) / 2; side = 3.0 * (kb[3] - kb[1])
    ims = []
    for k in range(NL - 1, -1, -1):
        im = fr[len(fr) - 1 - k]; pad = cv2.copyMakeBorder(im, 400, 400, 400, 400, cv2.BORDER_REPLICATE)
        x0, y0 = int(cx - side / 2) + 400, int(cy - side / 2) + 400
        ims.append(cv2.resize(pad[y0:y0 + int(side), x0:x0 + int(side)], (640, 640), interpolation=cv2.INTER_CUBIC))
    res = m.predict(ims, imgsz=640, conf=0.03, classes=[32], verbose=False, device=0)
    tot += 1
    for q, rr in enumerate(res): hit[q] += len(rr.boxes) > 0
print("clips", tot, "ball detected (conf>=0.03) in frame -5..-0 (oldest..last):", (hit / tot).round(2))
