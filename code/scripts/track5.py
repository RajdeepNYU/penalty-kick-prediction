"""Stage S4: tracks from the v5 two-stage picks (picks5_final.pkl): follow each pick over the 13 timed frames (det persons, conf>=0.1), fill missing frames with
YOLO11x TTA detections (conf 0.05) matched to the neighbouring track. Writes cache/tracks_new.pkl (same format as tracks.pkl) and cache/changed5.pkl."""
import os  # notebook: paths from the environment
import pickle, os, cv2, time, numpy as np, pandas as pd
from ultralytics import YOLO
ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root"); D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]; ANCHOR = 6
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); clips = list(gt["clip"]) + list(pm["clip"]); split = {c: "train" for c in gt["clip"]}; split.update({c: "val" for c in pm["clip"]})
picks = pickle.load(open(f"{C}/picks5_final.pkl", "rb")); old = pickle.load(open(f"{C}/tracks.pkl", "rb")); yolo = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt")


def cost_to(prev, bb):
    hp = max(prev[3] - prev[1], 1); cp = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2]); cc = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
    return np.linalg.norm(cc - cp, axis=1) / hp + 1.5 * np.abs(np.log(np.maximum((bb[:, 3] - bb[:, 1]) / hp, 1e-3)))


def follow(d, start, a0, thr=0.1):
    n = len(d["frames"]); out = np.full((n, 4), np.nan); out[a0] = start
    for step in (1, -1):
        prev = start.copy(); j = a0 + step
        while 0 <= j < n:
            f = d["frames"][j]; bb = f["pb"][f["pc"] >= thr]
            if len(bb):
                cst = cost_to(prev, bb); i = cst.argmin()
                if cst[i] < 1.5: out[j] = bb[i]; prev = bb[i]
            j += step
    return out


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-6)


new = {}; changed = []; filled = 0; t0 = time.time()
for n_, c in enumerate(clips):
    p = picks[c]; o = old[c]
    if p is None: new[c] = o; continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); kt, gt_ = follow(d, np.asarray(p["kick"], float), ANCHOR), follow(d, np.asarray(p["keep"], float), ANCHOR)
    miss = [j for j in range(len(kt)) if not np.isfinite(kt[j, 0]) or not np.isfinite(gt_[j, 0])]
    if miss:                                                                   # fill gaps with TTA detections on those frames only
        cap = cv2.VideoCapture(f"{D}/{split[c]}/{c}"); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        for j in miss:
            r = yolo.predict(fr[d["idx"][j]], imgsz=1280, conf=0.05, classes=[0], augment=True, verbose=False, device=0)[0]; bb = r.boxes.xyxy.cpu().numpy()
            if not len(bb): continue
            for arr in (kt, gt_):
                if np.isfinite(arr[j, 0]): continue
                nb = [arr[k] for k in (j - 1, j + 1) if 0 <= k < len(arr) and np.isfinite(arr[k, 0])]
                if nb:
                    cst = cost_to(nb[0], bb); i = cst.argmin()
                    if cst[i] < 2.0: arr[j] = bb[i]; filled += 1
    new[c] = dict(kick=kt, keep=gt_, kick_a=np.asarray(p["kick"], float), keep_a=np.asarray(p["keep"], float))
    if o is None or o.get("keep") is None or iou(o["kick_a"], new[c]["kick_a"]) < 0.5 or iou(o["keep_a"], new[c]["keep_a"]) < 0.5: changed.append(c)
    if n_ % 300 == 0: print(n_, f"{time.time() - t0:.0f}s", flush=True)
pickle.dump(new, open(f"{C}/tracks_new.pkl", "wb")); pickle.dump(changed, open(f"{C}/changed5.pkl", "wb"))
kv = np.mean([np.isfinite(v["kick"][:, 0]).mean() for v in new.values() if v]); gv = np.mean([np.isfinite(v["keep"][:, 0]).mean() for v in new.values() if v])
kf = np.mean([np.isfinite(v["kick"][:, 0]).all() for v in new.values() if v]); gf = np.mean([np.isfinite(v["keep"][:, 0]).all() for v in new.values() if v])
print(f"tracks: kicker frames tracked {kv:.3f} (fully tracked clips {kf:.3f}) | keeper {gv:.3f} (fully {gf:.3f}) | gaps filled by TTA: {filled}")
print(f"clips whose kicker or keeper pick changed vs current tracks: {len(changed)} of {len(clips)} ({len(changed) / len(clips):.1%})")
