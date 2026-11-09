"""Stage B3: RTMW-x wholebody keypoints for the KEEPER on every frame of F (keeper boxes interpolated from the 13 timed tracks, refined once from keypoints).
Writes cache/kkeep/<clip>.pkl: kp [nF,133,2] image px, sc [nF,133], box [nF,4]."""
import torch
import cv2, os, pickle, sys, time, numpy as np, pandas as pd
from rtmlib.tools.pose_estimation import RTMPose

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUT = f"{C}/kkeep"; os.makedirs(OUT, exist_ok=True)
URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
pose = RTMPose(URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]


def interp_box(tb, d, tq):
    v = np.isfinite(tb[:, 0]); tt = np.array(d["ts"])[v]; o = np.argsort(tt)
    return np.array([np.interp(tq, tt[o], tb[v][o][:, k]) for k in range(4)])


def kbox(kp, sc, old):
    m = sc[:23] > 2.0
    if m.sum() < 8: return old
    p = kp[:23][m]; x0, y0 = p.min(0); x1, y1 = p.max(0); h = y1 - y0; w = x1 - x0
    return np.array([x0 - 0.12 * w, y0 - 0.12 * h, x1 + 0.12 * w, y1 + 0.08 * h])


t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    t = tracks[c]; b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb"))
    if t is None or t["keep"] is None or b is None: pickle.dump(None, open(fp, "wb")); continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    nF = b["nF"]; KP = np.zeros((nF, 133, 2), np.float32); SC = np.zeros((nF, 133), np.float32); BX = np.zeros((nF, 4), np.float32)
    for i, j in enumerate(b["F"]):
        box = interp_box(t["keep"], d, b["ts"][i]); h0 = box[3] - box[1]
        kp, sc = pose(fr[j], bboxes=[[float(v) for v in box]]); kp, sc = kp[0], sc[0]
        nb = kbox(kp, sc, box)
        if abs((nb[3] - nb[1]) - h0) > 0.15 * h0 or np.hypot((nb[0] + nb[2] - box[0] - box[2]) / 2, (nb[1] + nb[3] - box[1] - box[3]) / 2) > 0.1 * h0:
            kp, sc = pose(fr[j], bboxes=[[float(v) for v in nb]]); kp, sc = kp[0], sc[0]; box = nb
        KP[i], SC[i], BX[i] = kp, sc, box
    pickle.dump(dict(kp=KP, sc=SC, box=BX), open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
