import numpy as np, pandas as pd
from ultralytics import YOLO
R = r"C:\path\to\project-root"; C = R + r"\Q4_work\cache"
gt = pd.read_csv(R + r"\Q4data\train_ground_truth.csv")
W = np.load(f"{C}/clips_wide.npy", mmap_mode="r")
m = YOLO(R + r"\Q1-Q3work\yolo11x.pt")
rng = np.random.RandomState(0); sel = rng.choice(len(gt), 300, replace=False)
rows = []
for i in sel:
    ims = [np.ascontiguousarray(W[i, q][:, :, ::-1]) for q in range(16)]     # BGR
    res = m.predict(ims, imgsz=448, conf=0.05, classes=[32], verbose=False, device=0, batch=16)
    pts = []
    for q, r in enumerate(res):
        if len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy(); j = c.argmax()
            pts.append((q, (b[j, 0] + b[j, 2]) / 2 / 224 - 0.5, (b[j, 1] + b[j, 3]) / 2 / 224 - 0.5, c[j]))
    rows.append((i, pts))
y = gt["kick_direction"].values
tot = 0; n_ok = 0; res_ = []
for i, pts in rows:
    if len(pts) < 3: continue
    p = np.array(pts); a, b = p[:3].mean(0), p[-3:].mean(0)    # early vs late ball position (crop coords: x right = kicker's right, y down)
    dx, dy = b[1] - a[1], b[2] - a[2]; dist = np.hypot(dx, dy)
    res_.append((i, dx, dy, dist, len(pts)))
d = pd.DataFrame(res_, columns=["i", "dx", "dy", "dist", "n"]); d["y"] = y[d["i"]]
print("clips with >=3 ball detections:", len(d), "of", len(rows))
for thr in (0.1, 0.2, 0.3):
    f = d[d.dist > thr]
    # predicted direction by horizontal component relative to upward flight
    ang = np.degrees(np.arctan2(f.dx, -f.dy))    # 0 = straight up (toward goal), + = to kicker's right
    pred = np.where(ang > 15, "right", np.where(ang < -15, "left", "middle"))
    print(f"moved > {thr}: n={len(f)}  agree with label {np.mean(pred == f.y.values):.3f}  | label dist {f.y.value_counts(normalize=True).round(2).to_dict()} | pred dist {pd.Series(pred).value_counts(normalize=True).round(2).to_dict()}")
    print("   up-flight (dy<0) share: %.2f" % np.mean(f.dy < 0))
