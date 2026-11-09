"""Stage B6: ViTPose++ base (usyd-community/vitpose-plus-base, COCO 17-kp head, top-down ViT) on the KEEPER boxes (cache/kkeep) of every frame of F: a third, architecturally different pose measurement.
Writes cache/<VIT_OUT>/<clip>.pkl in the same layout as kpts (133-slot arrays; only slots 0-16 filled; scores mapped to the RTMW scale x5). usage: python kpts_vit.py [rev]"""
import os  # notebook: paths from the environment
import torch, cv2, os, sys, pickle, time, numpy as np, pandas as pd
from transformers import AutoProcessor, VitPoseForPoseEstimation
ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root"); D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]; OUT = f"{C}/" + os.environ.get("VIT_OUT", "kptsE"); os.makedirs(OUT, exist_ok=True)
name = os.environ.get("VIT_MODEL", "usyd-community/vitpose-plus-base"); proc = AutoProcessor.from_pretrained(name); model = VitPoseForPoseEstimation.from_pretrained(name).cuda().eval()
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]
if len(sys.argv) > 1 and sys.argv[1] == "rev": todo = todo[::-1]
t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kkeep/{c}.pkl", "rb"))
    if b is None or k is None: pickle.dump(None, open(fp, "wb")); continue
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    nF = b["nF"]; KP = np.zeros((nF, 133, 2), np.float32); SC = np.zeros((nF, 133), np.float32); imgs = [cv2.cvtColor(fr[j], cv2.COLOR_BGR2RGB) for j in b["F"]]
    for i, im in enumerate(imgs):
        bx = k["box"][i]; box = np.array([[bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]]], np.float32)
        inp = proc(im, boxes=[box], return_tensors="pt").to("cuda")
        with torch.no_grad(): out = model(**inp, dataset_index=torch.tensor([0], device="cuda"))
        r = proc.post_process_pose_estimation(out, boxes=[box])[0][0]; KP[i, :17] = r["keypoints"].cpu().numpy(); SC[i, :17] = r["scores"].cpu().numpy() * 5.0
    pickle.dump(dict(kp=KP, sc=SC, box=k["box"]), open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
