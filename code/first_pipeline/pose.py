"""Stage 3: crops of tracked kicker/keeper at the timed frames -> pose (YOLO11m-pose) + saved JPEG crops."""
import cv2, os, pickle, time, sys, numpy as np, pandas as pd
from ultralytics import YOLO

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
POSE_W = ROOT + r"\Q4_word_GPT\models\yolo11m-pose.pt"
OUT = f"{C}/pose"; os.makedirs(OUT, exist_ok=True)
S = 256
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
split = {c: "train" for c in gt["clip"]}; split.update({c: "val" for c in pm["clip"]})
model = YOLO(POSE_W)


def crop(im, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = 1.6 * max(box[3] - box[1], 1.2 * (box[2] - box[0]))
    sc = S / side
    M = np.array([[sc, 0, S / 2 - sc * cx], [0, sc, S / 2 - sc * cy]], np.float32)
    return cv2.warpAffine(im, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), M


def run(crops, Ms):
    res = model.predict(crops, imgsz=S, conf=0.05, verbose=False, device=0, half=True, batch=26)
    kps = np.full((len(crops), 17, 3), np.nan, np.float32)
    for i, r in enumerate(res):
        if r.keypoints is None or len(r.boxes) == 0: continue
        b = r.boxes.xyxy.cpu().numpy(); k = r.keypoints.data.cpu().numpy()
        # person closest to crop centre, preferring large boxes
        d = np.hypot((b[:, 0] + b[:, 2]) / 2 - S / 2, (b[:, 1] + b[:, 3]) / 2 - S / 2) / S
        sc = d - 0.3 * (b[:, 3] - b[:, 1]) / S
        j = sc.argmin(); p = k[j].copy()
        Mi = cv2.invertAffineTransform(Ms[i])
        p[:, :2] = p[:, :2] @ Mi[:, :2].T + Mi[:, 2]
        kps[i] = p
    return kps


t0 = time.time(); n = 0
for c, t in tracks.items():
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))
    if t is None:
        pickle.dump(None, open(fp, "wb")); continue
    cap = cv2.VideoCapture(f"{D}/{split[c]}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    res = {}
    for who in ("kick", "keep"):
        tb = t[who]
        if tb is None: res[who] = None; continue
        crops, Ms, js = [], [], []
        for j, i in enumerate(d["idx"]):
            if np.isfinite(tb[j, 0]):
                cr, M = crop(fr[i], tb[j]); crops.append(cr); Ms.append(M); js.append(j)
        kps = np.full((len(d["idx"]), 17, 3), np.nan, np.float32); jp = [None] * len(d["idx"])
        if crops:
            k = run(crops, Ms)
            for q, j in enumerate(js):
                kps[j] = k[q]; jp[j] = cv2.imencode(".jpg", crops[q], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        res[who] = dict(kp=kps, jpg=jp)
    pickle.dump(res, open(fp, "wb")); n += 1
    if n % 50 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
