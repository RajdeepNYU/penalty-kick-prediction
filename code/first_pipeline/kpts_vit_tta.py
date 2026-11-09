"""Stream I: ViTPose++ large with test-time augmentation (box scale 0.9/1.0/1.15, shifts, horizontal flip with COCO pair swap), keypoints averaged per frame in image space -> lower jitter than a single pass.
Same layout as kptsF. usage: python kpts_vit_tta.py [rev]"""
import torch, cv2, os, sys, pickle, time, numpy as np, pandas as pd
from transformers import AutoProcessor, VitPoseForPoseEstimation
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"; OUT = f"{C}/" + os.environ.get("VIT_OUT", "kptsI"); os.makedirs(OUT, exist_ok=True)
name = os.environ.get("VIT_MODEL", "usyd-community/vitpose-plus-large"); proc = AutoProcessor.from_pretrained(name); model = VitPoseForPoseEstimation.from_pretrained(name).cuda().eval()
FL = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]
if len(sys.argv) > 1 and sys.argv[1] == "rev": todo = todo[::-1]
if os.environ.get("VIT_LIMIT"): todo = todo[:int(os.environ["VIT_LIMIT"])]
VAR = [(1.0, 0, 0), (0.9, 0, 0), (1.15, 0, 0), (1.0, .05, .05), (1.0, -.05, -.05), (1.05, .05, -.05), (1.05, -.05, .05)]
def pose(im, boxes):
    inp = proc(im, boxes=[boxes], return_tensors="pt").to("cuda")
    with torch.no_grad(): out = model(**inp, dataset_index=torch.zeros(len(boxes), dtype=torch.long, device="cuda"))
    r = proc.post_process_pose_estimation(out, boxes=[boxes])[0]; return np.stack([q["keypoints"].cpu().numpy() for q in r]), np.stack([q["scores"].cpu().numpy() for q in r])
t0 = time.time(); n = 0
for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
    if b is None or k is None: pickle.dump(None, open(fp, "wb")); continue
    cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    nF = b["nF"]; KP = np.zeros((nF, 133, 2), np.float32); SC = np.zeros((nF, 133), np.float32); imgs = [cv2.cvtColor(fr[j], cv2.COLOR_BGR2RGB) for j in b["F"]]; W_ = imgs[0].shape[1]
    for i, im in enumerate(imgs):
        x0, y0, x1, y1 = k["box"][i]; w, h = x1 - x0, y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bs = np.array([[cx + dx * w - s * w / 2, cy + dy * h - s * h / 2, s * w, s * h] for s, dx, dy in VAR], np.float32)
        kp, sc = pose(im, bs)
        fb = np.array([[W_ - (cx + dx * w + s * w / 2), cy + dy * h - s * h / 2, s * w, s * h] for s, dx, dy in VAR[:3]], np.float32)
        kf, sf = pose(im[:, ::-1].copy(), fb); kf = kf[:, FL]; kf[..., 0] = W_ - kf[..., 0]; sf = sf[:, FL]
        K_ = np.concatenate([kp, kf]); S_ = np.concatenate([sc, sf]); wgt = S_[..., None] + 1e-3
        KP[i, :17] = (K_ * wgt).sum(0) / wgt.sum(0); SC[i, :17] = S_.mean(0) * 5.0
    pickle.dump(dict(kp=KP, sc=SC, box=k["box"]), open(fp, "wb")); n += 1
    if n % 20 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
