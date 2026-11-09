"""Stage A: per-clip frame set F (last 8 native frames + sparse frames back to 1.0 s), camera-motion homographies to the last frame,
interpolated kicker boxes, and a *stationary-ball* consensus location (ball detections from kicker-centred crops warped to the last frame).
usage: python ballspot.py [N]  (N = only first N clips, for testing). Writes cache/ballspot/<clip>.pkl"""
import cv2, os, pickle, sys, time, numpy as np, pandas as pd
from ultralytics import YOLO

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUT = f"{C}/ballspot"; os.makedirs(OUT, exist_ok=True)
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]
if len(sys.argv) > 1: todo = todo[::max(len(todo) // int(sys.argv[1]), 1)][:int(sys.argv[1])]
yolo = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt")
SPARSE = [0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0]


def frame_set(ts):
    L = len(ts) - 1; idx = set(range(max(0, L - 7), L + 1))
    for o in SPARSE:
        j = int(np.abs(ts - (ts[L] - o)).argmin())
        if j > 0 or ts[L] - o > ts[0] - 1e-6: idx.add(j)
    return sorted(idx)


def interp_box(t, d, tq):
    kb = t["kick"]; v = np.isfinite(kb[:, 0]); tt = np.array(d["ts"])[v]; o = np.argsort(tt)
    return np.array([np.interp(tq, tt[o], kb[v][o][:, k]) for k in range(4)])


def homog(g0, g1, box0):
    """homography mapping frame0 -> frame1 (full-res coords) from LK tracks on the background (kicker region masked)."""
    h, w = g0.shape; mask = np.full((h, w), 255, np.uint8)
    x0, y0, x1, y1 = [int(v / 2) for v in box0]; pad = int(0.4 * (y1 - y0))
    mask[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad] = 0
    p0 = cv2.goodFeaturesToTrack(g0, 500, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 20: return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 20: return np.eye(3), 0
    H, inl = cv2.findHomography(p0[ok].reshape(-1, 2) * 2, p1[ok].reshape(-1, 2) * 2, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 15: return np.eye(3), 0
    return H, int(inl.sum())


def warp_pt(M, x, y):
    p = M @ np.array([x, y, 1.0]); return p[:2] / p[2]


t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    t = tracks[c]
    if t is None or t["keep"] is None: pickle.dump(None, open(fp, "wb")); continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr, ts = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f); ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    ts = np.array(ts); F = frame_set(ts); nF = len(F)
    boxes = np.array([interp_box(t, d, ts[j]) for j in F])
    gray = [cv2.resize(cv2.cvtColor(fr[j], cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5) for j in F]
    # chain homographies to the last frame in F
    Hs, inl = [], []
    for i in range(nF - 1):
        H, k = homog(gray[i], gray[i + 1], boxes[i]); Hs.append(H); inl.append(k)
    M = [None] * nF; M[-1] = np.eye(3)
    for i in range(nF - 2, -1, -1): M[i] = M[i + 1] @ Hs[i]
    # ball candidates from kicker-centred crops, warped into last-frame coordinates
    crops, meta = [], []
    for i, j in enumerate(F):
        b = boxes[i]; h = b[3] - b[1]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2 + 0.15 * h; side = 3.2 * h
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        pad = cv2.copyMakeBorder(fr[j], 500, 500, 500, 500, cv2.BORDER_REPLICATE)
        crops.append(cv2.resize(pad[y0 + 500:y0 + 500 + int(side), x0 + 500:x0 + 500 + int(side)], (640, 640), interpolation=cv2.INTER_CUBIC)); meta.append((x0, y0, side))
    res = yolo.predict(crops, imgsz=640, conf=0.01, classes=[32], verbose=False, device=0, batch=nF)
    cands = []   # (frame i, x_last, y_last, conf, size_px)
    for i, (r, (x0, y0, side)) in enumerate(zip(res, meta)):
        bb = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); s = side / 640
        for k in range(len(bb)):
            x, y = warp_pt(M[i], (bb[k, 0] + bb[k, 2]) / 2 * s + x0, (bb[k, 1] + bb[k, 3]) / 2 * s + y0)
            cands.append((i, x, y, float(cf[k]), float((bb[k, 2] - bb[k, 0]) * s)))
    hL = boxes[-1][3] - boxes[-1][1]; rad = 0.14 * hL
    feet = np.array([(boxes[-1][0] + boxes[-1][2]) / 2, boxes[-1][3] - 0.1 * hL])
    best, bscore = None, 0.0
    if cands:
        A = np.array(cands)
        for k in range(len(A)):
            near = np.hypot(A[:, 1] - A[k, 1], A[:, 2] - A[k, 2]) < rad
            sc = 0.0
            for fi in np.unique(A[near, 0]): sc += A[near & (A[:, 0] == fi), 3].max()
            dprior = np.hypot(A[k, 1] - feet[0], A[k, 2] - feet[1]) / hL
            sc *= 1.0 / (1.0 + (dprior / 1.3) ** 2)
            if sc > bscore: bscore, best = sc, k
    if best is not None:
        A = np.array(cands); near = np.hypot(A[:, 1] - A[best, 1], A[:, 2] - A[best, 2]) < rad
        w = A[near, 3]; bl = np.array([(A[near, 1] * w).sum() / w.sum(), (A[near, 2] * w).sum() / w.sum()])
        support = int(len(np.unique(A[near, 0]))); bconf = float(A[near, 3].mean())
    else: bl, support, bconf = np.array([np.nan, np.nan]), 0, 0.0
    ball_f = np.array([warp_pt(np.linalg.inv(M[i]), bl[0], bl[1]) if np.isfinite(bl[0]) else [np.nan, np.nan] for i in range(nF)])
    pickle.dump(dict(F=F, ts=ts[F], M=np.array(M), boxes=boxes, ball_last=bl, ball_f=ball_f, support=support, conf=bconf, score=bscore,
                     inliers=inl, nF=nF, wh=d["wh"], fps_native=float(1 / np.median(np.diff(ts))) if len(ts) > 1 else 0.0), open(fp, "wb"))
    n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
