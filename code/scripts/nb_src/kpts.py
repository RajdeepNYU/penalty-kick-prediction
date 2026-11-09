"""Stage B: RTMW-x wholebody keypoints (133, incl. toes/heels) for the kicker on every frame of F (from ballspot). Box refined once from the keypoints.
usage: python kpts.py [N]. Writes cache/kpts/<clip>.pkl  (kp [nF,133,2] image coords, sc [nF,133], box [nF,4] used)"""
import os  # notebook: paths from the environment
import torch  # must come first: makes onnxruntime find the CUDA/cuDNN libraries bundled with torch
import cv2, os, pickle, sys, time, numpy as np, pandas as pd
from rtmlib.tools.pose_estimation import RTMPose

ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]
OUT = f"{C}/kpts"; os.makedirs(OUT, exist_ok=True)
URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
pose = RTMPose(URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]
if len(sys.argv) > 1: todo = todo[::max(len(todo) // int(sys.argv[1]), 1)][:int(sys.argv[1])]


def kbox(kp, sc, old):
    """box from confident body+foot keypoints (0-22), padded; falls back to old box."""
    m = sc[:23] > 0.3
    if m.sum() < 8: return old
    p = kp[:23][m]; x0, y0 = p.min(0); x1, y1 = p.max(0); h = y1 - y0; w = x1 - x0
    return np.array([x0 - 0.12 * w, y0 - 0.12 * h, x1 + 0.12 * w, y1 + 0.08 * h])


t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"; bp = f"{C}/ballspot/{c}.pkl"
    if os.path.exists(fp) or not os.path.exists(bp): continue
    b = pickle.load(open(bp, "rb"))
    if b is None: pickle.dump(None, open(fp, "wb")); continue
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    nF = b["nF"]; KP = np.zeros((nF, 133, 2), np.float32); SC = np.zeros((nF, 133), np.float32); BX = np.zeros((nF, 4), np.float32); reran = 0
    for i, j in enumerate(b["F"]):
        box = b["boxes"][i].copy()
        kp, sc = pose(fr[j], bboxes=[[float(v) for v in box]]); kp, sc = kp[0], sc[0]
        nb = kbox(kp, sc, box); hb = box[3] - box[1]
        if abs((nb[3] - nb[1]) - hb) > 0.15 * hb or np.hypot((nb[0] + nb[2] - box[0] - box[2]) / 2, (nb[1] + nb[3] - box[1] - box[3]) / 2) > 0.1 * hb:
            kp, sc = pose(fr[j], bboxes=[[float(v) for v in nb]]); kp, sc = kp[0], sc[0]; box = nb; reran += 1
        KP[i], SC[i], BX[i] = kp, sc, box
    pickle.dump(dict(kp=KP, sc=SC, box=BX, reran=reran), open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
