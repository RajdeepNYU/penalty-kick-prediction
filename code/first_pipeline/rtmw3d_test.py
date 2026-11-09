import torch, numpy as np, pickle, cv2, pandas as pd, time
from rtmlib.tools.pose_estimation import RTMPose3d
R = r"C:\path\to\project-root"; C = R + r"\Q4_work\cache"
URL = "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
pose = RTMPose3d(URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
import inspect; print("call signature:", inspect.signature(pose.__call__))
gt = pd.read_csv(R + r"\Q4data\train_ground_truth.csv").sample(4, random_state=3)
for _, r in gt.iterrows():
    c = r["clip"]; b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
    cap = cv2.VideoCapture(R + rf"\Q4data\train\{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    out = pose(fr[b["F"][-1]], bboxes=[[float(v) for v in k["box"][-1]]])
    print(c, "foot", r["preferred_foot"], "K", r["kick_direction"], "| n outputs", len(out), [getattr(o, "shape", None) for o in out])
    kp3 = out[0][0]; kp2 = out[3][0] if len(out) > 3 else None
    print("   3d x,y,z ranges (body kps 0-22):", np.round(kp3[:23].min(0), 2), np.round(kp3[:23].max(0), 2))
    if kp2 is not None: print("   2d vs rtmw 2d ankle (px):", np.round(kp2[15], 1), np.round(k["kp"][-1][15], 1), "| 3d ankle", np.round(kp3[15], 2))
t = time.time()
for _ in range(10): pose(fr[b["F"][-1]], bboxes=[[float(v) for v in k["box"][-1]]])
print("s/call", round((time.time() - t) / 10, 3))
