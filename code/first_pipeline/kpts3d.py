"""Stage B2: RTMW3D-x (3D wholebody) on the kicker box (refined, from kpts.py) at every frame of F. Keeps keypoints 0-22 (body + feet).
Writes cache/kpts3d/<clip>.pkl: kp3 [nF,23,3] (x,y in model-crop px, z depth in metres, root-relative), kp2 [nF,23,2] image px, sc [nF,23]."""
import torch  # first: lets onnxruntime find CUDA/cuDNN
import cv2, os, pickle, sys, time, numpy as np, pandas as pd
from rtmlib.tools.pose_estimation import RTMPose3d

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
OUT = f"{C}/kpts3d"; os.makedirs(OUT, exist_ok=True)
URL = "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
pose = RTMPose3d(URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]
t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"; bp = f"{C}/ballspot/{c}.pkl"; kp_ = f"{C}/kpts/{c}.pkl"
    if os.path.exists(fp): continue
    b = pickle.load(open(bp, "rb")); k = pickle.load(open(kp_, "rb"))
    if b is None or k is None: pickle.dump(None, open(fp, "wb")); continue
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    nF = b["nF"]; K3 = np.zeros((nF, 23, 3), np.float32); K2 = np.zeros((nF, 23, 2), np.float32); SC = np.zeros((nF, 23), np.float32)
    for i, j in enumerate(b["F"]):
        o = pose(fr[j], bboxes=[[float(v) for v in k["box"][i]]])
        K3[i], K2[i], SC[i] = o[0][0][:23], o[3][0][:23], o[1][0][:23]
    pickle.dump(dict(kp3=K3, kp2=K2, sc=SC), open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
