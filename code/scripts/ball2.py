"""Stage B2: re-estimate the stationary ball with a strong foot prior. Candidates: YOLO11x sports-ball detections on foot-centred crops of all F frames
(warped to last-frame coords), the temporal-median candidate and the blob candidate from the old pipeline, and the old consensus ball.
Clusters are scored by support x confidence x exp(-(d_foot / 0.3)^2), d_foot = distance (kicker heights) to the nearest foot keypoint in the last 3 frames.
Writes cache/ball2.pkl {clip: dict(ball, d_foot, support, method, d_old)}"""
import pickle, numpy as np, cv2, time
from ultralytics import YOLO
from common import *

yolo = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt")
bf = pickle.load(open(f"{W}/ball_final.pkl", "rb"))
FEET = [15, 16, 17, 18, 19, 20, 21, 22]


def foot_refs(b, c, hL):
    pts = []
    for sub, ids in (("kpts", FEET), ("kptsF", [15, 16]), ("kptsG", [15, 16])):
        k = load(sub, c)
        if k is None: continue
        for j in range(max(0, b["nF"] - 3), b["nF"]):
            sc = k["sc"][j, ids]; p = warp(b["M"][j], k["kp"][j, ids])[sc >= 1.5]
            if len(p): pts.append(p)
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def dfoot(p, refs, hL):
    return np.min(np.hypot(*(refs - p).T)) / hL if len(refs) else np.nan


out = {}; t0 = time.time()
for n, c in enumerate(CLIPS):
    b = load("ballspot", c); k = load("kpts", c)
    if b is None or k is None: out[c] = None; continue
    hL = max(float(k["box"][-1][3] - k["box"][-1][1]), 20.0); refs = foot_refs(b, c, hL)
    fr = read_frames(c, b["F"]); crops, meta = [], []
    for i in range(b["nF"]):
        bx = b["boxes"][i]; h = bx[3] - bx[1]; cx, cy = (bx[0] + bx[2]) / 2, bx[3] - 0.1 * h; side = 2.4 * h
        x0, y0 = int(cx - side / 2), int(cy - side / 2); pad = cv2.copyMakeBorder(fr[i], 600, 600, 600, 600, cv2.BORDER_REPLICATE)
        crops.append(cv2.resize(pad[y0 + 600:y0 + 600 + int(side), x0 + 600:x0 + 600 + int(side)], (640, 640), interpolation=cv2.INTER_CUBIC)); meta.append((x0, y0, side))
    res = yolo.predict(crops, imgsz=640, conf=0.01, classes=[32], verbose=False, device=0, batch=len(crops), half=True)
    cand = []                                                       # (frame, x, y, conf)
    for i, (r, (x0, y0, side)) in enumerate(zip(res, meta)):
        bb = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); s = side / 640
        for q in range(len(bb)):
            x, y = warp(b["M"][i], [(bb[q, 0] + bb[q, 2]) / 2 * s + x0, (bb[q, 1] + bb[q, 3]) / 2 * s + y0]); cand.append((i, x, y, float(cf[q])))
    extra = []
    if np.isfinite(b["ball_last"][0]): extra.append(("old", b["ball_last"], 0.3))
    f_ = bf.get(c)
    if f_ is not None and np.isfinite(np.asarray(f_["ball"], float)[0]): extra.append(("fused", np.asarray(f_["ball"], float), 0.3))
    m_ = load("ballmed", c)
    if m_ is not None and np.isfinite(np.asarray(m_["ball_med"], float)[0]): extra.append(("median", np.asarray(m_["ball_med"], float), 0.3 * min(1.0, 5 * m_["conf"])))
    rad = 0.12 * hL; best, bsc, bmeth = None, 0.0, None
    pts = [(np.array([x, y]), "yolo") for _, x, y, _ in cand] + [(p, nm) for nm, p, _ in extra]
    A = np.array([[i, x, y, cf] for i, x, y, cf in cand]) if cand else np.zeros((0, 4))
    for p, nm in pts:
        near = np.hypot(A[:, 1] - p[0], A[:, 2] - p[1]) < rad if len(A) else np.zeros(0, bool)
        sup = sum(A[near & (A[:, 0] == fi), 3].max() for fi in np.unique(A[near, 0])) if near.any() else 0.0
        sup += sum(w for nm2, q, w in extra if np.hypot(*(q - p)) < rad)
        sc = sup * np.exp(-(dfoot(p, refs, hL) / 0.3) ** 2) if len(refs) else sup
        if sc > bsc: bsc, best, bmeth = sc, p, nm
    if best is not None:
        near = np.hypot(A[:, 1] - best[0], A[:, 2] - best[1]) < rad if len(A) else np.zeros(0, bool)
        if near.sum() >= 2: w = A[near, 3]; best = np.array([(A[near, 1] * w).sum() / w.sum(), (A[near, 2] * w).sum() / w.sum()])
        out[c] = dict(ball=best, d_foot=dfoot(best, refs, hL), support=int(len(np.unique(A[near, 0]))) if len(A) else 0, score=bsc, method=bmeth,
                      d_old=float(np.hypot(*(best - b["ball_last"])) / hL) if np.isfinite(b["ball_last"][0]) else np.nan, d_foot_old=dfoot(b["ball_last"], refs, hL) if np.isfinite(b["ball_last"][0]) else np.nan)
    else: out[c] = dict(ball=b["ball_last"], d_foot=dfoot(b["ball_last"], refs, hL), support=0, score=0.0, method="none", d_old=0.0, d_foot_old=dfoot(b["ball_last"], refs, hL))
    if n % 200 == 0: print(n, f"{time.time() - t0:.0f}s", flush=True)
pickle.dump(out, open(f"{PC}/ball2.pkl", "wb"))
import pandas as pd
R = pd.DataFrame([dict(clip=c, **{a: v for a, v in d.items() if a != "ball"}) for c, d in out.items() if d])
print(R[["d_foot", "d_foot_old", "d_old", "support"]].describe(percentiles=[.5, .9, .95, .99]).round(3).T)
print("changed (> 0.15 heights):", int((R.d_old > 0.15).sum()), "| new ball far from feet (> 0.5):", int((R.d_foot > 0.5).sum()), "vs old", int((R.d_foot_old > 0.5).sum()), "| methods:", R.method.value_counts().to_dict())
