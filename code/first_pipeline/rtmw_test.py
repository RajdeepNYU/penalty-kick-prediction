import numpy as np, pickle, cv2, pandas as pd, time
from rtmlib.tools.pose_estimation import RTMPose
R = r"C:\path\to\project-root"; C = R + r"\Q4_work\cache"
S = r"scratch"
URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
pose = RTMPose(URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
gt = pd.read_csv(R + r"\Q4data\train_ground_truth.csv").sample(6, random_state=11)
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); rows = []; t0 = time.time(); n = 0
for _, r in gt.iterrows():
    c = r["clip"]; t = tracks[c]; d = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))
    cap = cv2.VideoCapture(R + rf"\Q4data\train\{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    tiles = []
    for j in (12, 9):
        b = t["kick"][j]
        if not np.isfinite(b[0]): continue
        im = fr[d["idx"][j]]; kp, sc = pose(im, bboxes=[[float(v) for v in b]]); n += 1
        kp, sc = kp[0], sc[0]
        x0, y0 = int(max(b[0] - 0.3 * (b[2] - b[0]), 0)), int(max(b[1] - 0.1 * (b[3] - b[1]), 0)); x1, y1 = int(b[2] + 0.3 * (b[2] - b[0])), int(b[3] + 0.15 * (b[3] - b[1]))
        crop = im[y0:y1, x0:x1].copy(); s = 420 / crop.shape[0]; crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for k, col in {15: (0, 255, 0), 16: (0, 255, 0), 17: (0, 0, 255), 18: (255, 0, 0), 19: (0, 255, 255), 20: (0, 0, 255), 21: (255, 0, 0), 22: (0, 255, 255), 11: (255, 0, 255), 12: (255, 0, 255), 13: (0, 128, 255), 14: (0, 128, 255)}.items():
            if sc[k] > 0.3: cv2.circle(crop, (int((kp[k, 0] - x0) * s), int((kp[k, 1] - y0) * s)), 4, col, -1)
        tiles.append(cv2.resize(crop, (330, 420)))
    while tiles and len(tiles) < 2: tiles.append(np.zeros((420, 330, 3), np.uint8))
    if tiles:
        row = np.hstack(tiles); cv2.putText(row, f"{c[:4]} foot={r['preferred_foot']} K={r['kick_direction']}", (3, 12), 0, 0.4, (255, 255, 255), 1); rows.append(row)
print("per-call time", round((time.time() - t0) / max(n, 1), 3), "s incl. video decode")
cv2.imwrite(f"{S}/rtmw_test.jpg", np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - len(rows) % 2, 2)])); print("ok")
