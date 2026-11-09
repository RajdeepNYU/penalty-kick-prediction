"""Stage 7: goal-aligned crops. Each crop is rotated so the kicker->keeper (goal) direction points up in the crop;
then 'kicker's left' == screen-left and an h-flip is an exact left<->right mirror."""
import os  # notebook: paths from the environment
import cv2, os, pickle, time, numpy as np, pandas as pd

ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]
S = 224
OUT = f"{C}/align"; os.makedirs(OUT, exist_ok=True)
KF, GF = [12, 11, 10, 9, 8, 6], [12, 10, 8, 6]      # kicker: 0,.03,.06,.1,.15,.25 s ; keeper: 0,.06,.15,.25
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
split = {c: "train" for c in gt["clip"]}; split.update({c: "val" for c in pm["clip"]})
cen = lambda b: np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def crop(im, box, ang, side_mult=1.7):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = side_mult * max(box[3] - box[1], 1.2 * (box[2] - box[0])); s = S / side
    c, sn = np.cos(ang), np.sin(ang)
    R = np.array([[c, -sn], [sn, c]]) * s
    t = np.array([S / 2, S / 2]) - R @ np.array([cx, cy])
    M = np.c_[R, t].astype(np.float32)
    return cv2.warpAffine(im, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


t0 = time.time(); n = 0
for c, t in tracks.items():
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    if t is None or t["keep"] is None: pickle.dump(None, open(fp, "wb")); continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))
    kc, gc = cen(t["kick"]), cen(t["keep"]); v = np.isfinite(kc[:, 0]) & np.isfinite(gc[:, 0])
    if not v.any(): pickle.dump(None, open(fp, "wb")); continue
    fv = (gc[v] - kc[v]).mean(0); a = np.arctan2(fv[1], fv[0]); ang = -np.pi / 2 - a
    cap = cv2.VideoCapture(f"{D}/{split[c]}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    res = {}
    for who, FR, key in (("kick", KF, "kick"), ("keep", GF, "keep")):
        arr = np.zeros((len(FR), S, S, 3), np.uint8); miss = []
        for q, j in enumerate(FR):
            b = t[key][j]
            if not np.isfinite(b[0]):  # fall back to nearest valid frame
                vj = np.where(np.isfinite(t[key][:, 0]))[0]; j2 = vj[np.abs(vj - j).argmin()]; b = t[key][j2]; miss.append(True)
            else: miss.append(False)
            arr[q] = crop(fr[d["idx"][j]], b, ang)[:, :, ::-1]
        res[who] = arr; res[who + "_miss"] = miss
    res["f"] = fv / np.linalg.norm(fv); res["ang"] = ang
    pickle.dump(res, open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
