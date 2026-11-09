"""Stage B5: two extra measurement streams of the kicker's pose from RTMW-x with perturbed boxes (C: 1.12x scale, shifted right/up; D: 0.9x scale, shifted left/down).
Writes cache/kptsC/<clip>.pkl and cache/kptsD/<clip>.pkl: kp [nF,133,2] image px, sc [nF,133]."""
import torch
import cv2, os, pickle, time, numpy as np, pandas as pd
from rtmlib.tools.pose_estimation import RTMPose
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
pose = RTMPose(URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
STREAMS = {"C": (1.12, 0.04, -0.03), "D": (0.90, -0.04, 0.03)}               # (scale, dx/h, dy/h)
for k in STREAMS: os.makedirs(f"{C}/kpts{k}", exist_ok=True)
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]; t0 = time.time(); n = 0
for split, c in todo:
    if all(os.path.exists(f"{C}/kpts{k}/{c}.pkl") for k in STREAMS): continue
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); kk = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
    if b is None or kk is None:
        for k in STREAMS: pickle.dump(None, open(f"{C}/kpts{k}/{c}.pkl", "wb"))
        continue
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    nF = b["nF"]; out = {k: (np.zeros((nF, 133, 2), np.float32), np.zeros((nF, 133), np.float32)) for k in STREAMS}
    for i, j in enumerate(b["F"]):
        bx = kk["box"][i]; h = bx[3] - bx[1]; cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2; w = bx[2] - bx[0]
        for k, (sc_, dx, dy) in STREAMS.items():
            nb = [cx + dx * h - sc_ * w / 2, cy + dy * h - sc_ * h / 2, cx + dx * h + sc_ * w / 2, cy + dy * h + sc_ * h / 2]
            kp, sc = pose(fr[j], bboxes=[[float(v) for v in nb]]); out[k][0][i], out[k][1][i] = kp[0], sc[0]
    for k in STREAMS: pickle.dump(dict(kp=out[k][0], sc=out[k][1], box=kk["box"]), open(f"{C}/kpts{k}/{c}.pkl", "wb"))
    n += 1
    if n % 200 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
