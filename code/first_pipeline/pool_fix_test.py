import pickle, cv2, numpy as np, pandas as pd, time
from ultralytics import YOLO
R = r"C:\path\to\project-root"; D = R + r"\Q4data"; C = R + r"\Q4_work\cache"
pf = pickle.load(open(f"{C}/pool_fail.pkl", "rb")); ck = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip")
y11 = YOLO(R + r"\Q1-Q3work\yolo11x.pt"); res = {k: 0 for k in ("det conf>=0.1", "det conf>=0.03", "yolo11x 1920 conf.05", "yolo11x 1920 +TTA conf.05", "yolo11x 1280 +TTA conf.05")}; n = 0; times = {}
for c, why in pf.items():
    who = "kicker" if "kicker" in why else "keeper"; d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); w, h = d["wh"]; r = ck.loc[c]; p = (r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h)
    cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    im = fr[d["idx"][6]]; n += 1
    inside = lambda b: bool(((b[:, 0] <= p[0]) & (p[0] <= b[:, 2]) & (b[:, 1] <= p[1]) & (p[1] <= b[:, 3])).any()) if len(b) else False
    fm = d["frames"][6]; res["det conf>=0.1"] += inside(fm["pb"][fm["pc"] >= 0.1])
    for key, kw in (("yolo11x 1920 conf.05", dict(imgsz=1920)), ("yolo11x 1920 +TTA conf.05", dict(imgsz=1920, augment=True)), ("yolo11x 1280 +TTA conf.05", dict(imgsz=1280, augment=True))):
        t = time.time(); b = y11.predict(im, conf=0.05, classes=[0], verbose=False, device=0, **kw)[0].boxes.xyxy.cpu().numpy(); times[key] = times.get(key, 0) + time.time() - t; res[key] += inside(b)
    res["det conf>=0.03"] += 0
print(f"pool-failure clips: {n}; click covered by a candidate box:"); [print(f"  {k:28s} {v}/{n}" + (f"   ({times[k] / n:.2f}s/frame)" if k in times else "")) for k, v in res.items() if k != "det conf>=0.03"]
