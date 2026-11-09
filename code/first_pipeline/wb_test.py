import torch, numpy as np, pickle, cv2, pandas as pd, sys
from transformers import AutoProcessor, VitPoseForPoseEstimation
R = r"C:\path\to\project-root"; C = R + r"\Q4_work\cache"
S = r"scratch"
name = "usyd-community/vitpose-plus-base"
proc = AutoProcessor.from_pretrained(name); model = VitPoseForPoseEstimation.from_pretrained(name).cuda().eval()
print("num_experts", getattr(model.config.backbone_config, "num_experts", None), "labels", len(model.config.id2label) if model.config.id2label else None)
gt = pd.read_csv(R + r"\Q4data\train_ground_truth.csv").sample(6, random_state=7)
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); rows = []
for _, r in gt.iterrows():
    c = r["clip"]; t = tracks[c]; d = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))
    cap = cv2.VideoCapture(R + rf"\Q4data\train\{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    tiles = []
    for j in (12, 8):
        b = t["kick"][j]
        if not np.isfinite(b[0]): continue
        im = fr[d["idx"][j]][:, :, ::-1].copy()
        box = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]]], np.float32)
        inp = proc(im, boxes=[box], return_tensors="pt").to("cuda")
        with torch.no_grad(): out = model(**inp, dataset_index=torch.tensor([5], device="cuda"))
        res = proc.post_process_pose_estimation(out, boxes=[box])[0][0]
        kp = res["keypoints"].cpu().numpy(); sc = res["scores"].cpu().numpy()
        x0, y0, x1, y1 = [int(v) for v in (b[0] - 0.3 * (b[2] - b[0]), b[1] - 0.1 * (b[3] - b[1]), b[2] + 0.3 * (b[2] - b[0]), b[3] + 0.15 * (b[3] - b[1]))]
        crop = fr[d["idx"][j]][max(y0, 0):y1, max(x0, 0):x1].copy(); sc_ = 400 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=sc_, fy=sc_)
        names = {15: (0, 255, 0), 16: (0, 255, 0), 17: (0, 0, 255), 18: (255, 0, 0), 19: (0, 255, 255), 20: (0, 0, 255), 21: (255, 0, 0), 22: (0, 255, 255)}
        for k, col in names.items():
            if sc[k] > 0.3: cv2.circle(crop, (int((kp[k, 0] - max(x0, 0)) * sc_), int((kp[k, 1] - max(y0, 0)) * sc_)), 4, col, -1)
        tiles.append(cv2.resize(crop, (300, 400)))
    if tiles:
        row = np.hstack(tiles); cv2.putText(row, f"{c[:4]} foot={r['preferred_foot']} K={r['kick_direction']}  (red=big toe blue=small toe yellow=heel green=ankle)", (3, 12), 0, 0.38, (255, 255, 255), 1); rows.append(row)
cv2.imwrite(f"{S}/wb_test.jpg", np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - len(rows) % 2, 2)]))
print("ok")
