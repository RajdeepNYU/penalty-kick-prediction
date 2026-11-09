"""Stage 10: dense ball candidates over the final ~0.3 s, detected on upscaled kicker-centred crops (YOLO11x, COCO sports ball)."""
import cv2, os, pickle, time, numpy as np, pandas as pd
from ultralytics import YOLO

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUT = f"{C}/ball"; os.makedirs(OUT, exist_ok=True)
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
split = {c: "train" for c in gt["clip"]}; split.update({c: "val" for c in pm["clip"]})
m = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt")
cen = lambda b: np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
t0 = time.time(); n = 0
for c, t in tracks.items():
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); kb = t["kick"]; v = np.isfinite(kb[:, 0])
    if not v.any(): pickle.dump(None, open(fp, "wb")); continue
    cap = cv2.VideoCapture(f"{D}/{split[c]}/{c}"); fr, ts = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f); ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    ts = np.array(ts); last = ts[-1]
    sel = [i for i in range(len(fr)) if ts[i] >= last - 0.30][-14:]
    if len(sel) < 4: sel = list(range(max(0, len(fr) - 4), len(fr)))
    kc = cen(kb[v]); dts = np.array(d["ts"])[v]; o = np.argsort(dts)
    h0 = float(np.nanmedian((kb[:, 3] - kb[:, 1])[v])); side = 3.4 * h0
    ims, meta = [], []
    for i in sel:
        cx = np.interp(ts[i], dts[o], kc[o, 0]); cy = np.interp(ts[i], dts[o], kc[o, 1])
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        pad = cv2.copyMakeBorder(fr[i], 500, 500, 500, 500, cv2.BORDER_REPLICATE)
        ims.append(cv2.resize(pad[y0 + 500:y0 + 500 + int(side), x0 + 500:x0 + 500 + int(side)], (640, 640), interpolation=cv2.INTER_CUBIC))
        meta.append((cx, cy, x0, y0))
    res = m.predict(ims, imgsz=640, conf=0.03, classes=[32], verbose=False, device=0, batch=14)
    cands = []
    for r, (cx, cy, x0, y0) in zip(res, meta):
        b = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); s = side / 640
        if len(b): cands.append(np.c_[(b[:, 0] + b[:, 2]) / 2 * s + x0, (b[:, 1] + b[:, 3]) / 2 * s + y0, cf, (b[:, 2] - b[:, 0]) * s])
        else: cands.append(np.zeros((0, 4)))
    pickle.dump(dict(ts=ts[sel], cands=cands, kc=np.array([[a, b_] for a, b_, _, _ in meta]), h0=h0, wh=d["wh"]), open(fp, "wb"))
    n += 1
    if n % 200 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
