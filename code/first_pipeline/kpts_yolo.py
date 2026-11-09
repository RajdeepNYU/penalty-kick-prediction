"""Stream H: YOLO11x-pose (bottom-up-ish single-shot detector head, different family from RTMW/ViTPose) run on a padded crop around the kicker box of every frame of F.
Same layout as kptsE (133-slot arrays, slots 0-16 = COCO, scores x5). usage: python kpts_yolo.py [rev]"""
import torch, cv2, os, sys, pickle, time, numpy as np, pandas as pd
from ultralytics import YOLO
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"; OUT = f"{C}/" + os.environ.get("YP_OUT", "kptsH"); os.makedirs(OUT, exist_ok=True)
model = YOLO(os.environ.get("YP_MODEL", "yolo11x-pose.pt")); IMG = int(os.environ.get("YP_IMG", 512))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]
if len(sys.argv) > 1 and sys.argv[1] == "rev": todo = todo[::-1]
if os.environ.get("YP_LIMIT"): todo = todo[:int(os.environ["YP_LIMIT"])]
t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
    if b is None or k is None: pickle.dump(None, open(fp, "wb")); continue
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    nF = b["nF"]; KP = np.zeros((nF, 133, 2), np.float32); SC = np.zeros((nF, 133), np.float32); H_, W_ = fr[0].shape[:2]; crops = []; offs = []
    for i, j in enumerate(b["F"]):
        bx = k["box"][i]; w = bx[2] - bx[0]; h = bx[3] - bx[1]; s = max(w, h) * 1.35; cx = (bx[0] + bx[2]) / 2; cy = (bx[1] + bx[3]) / 2
        x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2)); S_ = int(round(s)); M = np.zeros((S_, S_, 3), np.uint8)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + S_, W_), min(y0 + S_, H_)
        if xb > xa and yb > ya: M[ya - y0:yb - y0, xa - x0:xb - x0] = fr[j][ya:yb, xa:xb]
        crops.append(cv2.resize(M, (IMG, IMG))); offs.append((x0, y0, S_ / IMG))
    res = model.predict(crops, imgsz=IMG, verbose=False, conf=0.01, half=True)
    for i, r in enumerate(res):
        if r.keypoints is None or len(r.boxes) == 0: continue
        bxs = r.boxes.xyxy.cpu().numpy(); ctr = np.array([IMG / 2, IMG / 2]); score = r.boxes.conf.cpu().numpy() - 0.002 * np.linalg.norm((bxs[:, :2] + bxs[:, 2:]) / 2 - ctr, axis=1)
        a = int(score.argmax()); kp = r.keypoints.xy[a].cpu().numpy(); cf = r.keypoints.conf[a].cpu().numpy(); x0, y0, sc = offs[i]
        KP[i, :17] = kp * sc + np.array([x0, y0]); SC[i, :17] = cf * 5.0
    pickle.dump(dict(kp=KP, sc=SC, box=k["box"]), open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
