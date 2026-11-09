import pickle, os, cv2, numpy as np, pandas as pd, sys
R = r"C:\path\to\project-root"; C = R + r"\Q4_work\cache"
S = r"scratch"
gt = pd.read_csv(R + r"\Q4data\train_ground_truth.csv"); split = set(gt["clip"])
files = sorted(os.listdir(f"{C}/ballspot")); recs = []
for f in files:
    b = pickle.load(open(f"{C}/ballspot/{f}", "rb"))
    if b is not None: recs.append((f[:-4], b))
sup = np.array([b["support"] for _, b in recs]); print("clips", len(recs), "support (frames agreeing) pctiles 10/25/50/75:", np.percentile(sup, [10, 25, 50, 75]), "| support>=3: %.2f  >=5: %.2f  zero: %.2f" % ((sup >= 3).mean(), (sup >= 5).mean(), (sup == 0).mean()))
print("mean nF", np.mean([b["nF"] for _, b in recs]).round(1), "| inliers min per clip (median):", np.median([min(b["inliers"]) if b["inliers"] else 0 for _, b in recs]))
sel = [r for r in recs if r[0] in split][::max(len(recs) // 12, 1)][:12]; tiles = []
for c, b in sel:
    cap = cv2.VideoCapture(R + rf"\Q4data\train\{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    for q in (b["nF"] - 1, b["nF"] - 6):
        im = fr[b["F"][q]].copy(); bx = b["boxes"][q]; h = bx[3] - bx[1]
        x0, y0 = int(max((bx[0] + bx[2]) / 2 - 1.1 * h, 0)), int(max(bx[1] - 0.15 * h, 0)); x1, y1 = int((bx[0] + bx[2]) / 2 + 1.1 * h), int(bx[3] + 0.3 * h)
        bp = b["ball_f"][q]
        if np.isfinite(bp[0]): cv2.circle(im, (int(bp[0]), int(bp[1])), max(int(0.06 * h), 4), (0, 0, 255), 2)
        crop = im[y0:y1, x0:x1]; s = 300 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=s, fy=s)
        pad = np.zeros((300, 300, 3), np.uint8); pad[:crop.shape[0], :crop.shape[1]] = crop; tiles.append(pad)
        if q == b["nF"] - 1: cv2.putText(pad, f"{c[:4]} sup={b['support']} conf={b['conf']:.2f}", (3, 14), 0, 0.45, (0, 255, 255), 1)
rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles) - len(tiles) % 6, 6)]
cv2.imwrite(f"{S}/ballview.jpg", np.vstack(rows)); print("ok")
