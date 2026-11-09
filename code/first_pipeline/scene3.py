"""Stage S1b: third person-candidate source: YOLO11x with test-time augmentation at conf 0.05 on the anchor frame (recovers blurred/small/low-contrast players).
Writes cache/scene2/<clip>.pkl: extra2 [k,5] boxes+conf, extra2_col [k,10]. usage: python scene2.py [rev]"""
import numpy as np, cv2, pandas as pd, pickle, os, sys, time
from ultralytics import YOLO
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"; OUT = f"{C}/scene3"; os.makedirs(OUT, exist_ok=True)
yolo = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt"); gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]
if len(sys.argv) > 1 and sys.argv[1] == "rev": todo = todo[::-1]
def torso_colour(hsv, b):
    out = []
    for x0, y0, x1, y1 in b.astype(int):
        ty0 = y0 + (y1 - y0) // 5; ty1 = y0 + (y1 - y0) * 3 // 5; tx0 = x0 + (x1 - x0) // 4; tx1 = x1 - (x1 - x0) // 4
        t = hsv[max(ty0, 0):max(ty1, ty0 + 1), max(tx0, 0):max(tx1, tx0 + 1)].reshape(-1, 3)
        if len(t) == 0: t = np.zeros((1, 3), np.uint8)
        hh = np.bincount((t[:, 0] // 23).clip(0, 7), weights=t[:, 1] / 255., minlength=8) / len(t); out.append(np.r_[hh, t[:, 1].mean() / 255, t[:, 2].mean() / 255])
    return np.array(out).reshape(-1, 10)
t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    im = fr[d["idx"][6]]; r = yolo.predict(im, imgsz=1280, conf=0.03, classes=[0, 32], augment=True, verbose=False, device=0)[0]; allb = r.boxes.xyxy.cpu().numpy(); allc = r.boxes.conf.cpu().numpy(); allk = r.boxes.cls.cpu().numpy(); b, cf = allb[allk == 0], allc[allk == 0]; balls = np.c_[allb[allk == 1], allc[allk == 1]] if (allk == 1).any() else np.zeros((0, 5), np.float32)
    col = torso_colour(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), b) if len(b) else np.zeros((0, 10))
    pickle.dump(dict(extra2=np.c_[b, cf] if len(b) else np.zeros((0, 5), np.float32), extra2_col=col, balls2=balls), open(fp, "wb")); n += 1
    if n % 200 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
