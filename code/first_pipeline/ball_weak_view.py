import pickle, cv2, numpy as np, pandas as pd
R = r"C:\path\to\project-root"; D = R + r"\Q4data"; C = R + r"\Q4_work\cache"
S = r"scratch"
cov = pd.read_pickle(f"{C}/audit_cov.pkl"); weak = cov[cov.ok & (cov.ball_support < 3) & (cov.part == "train")]["clip"].tolist()[::6][:12]; tiles = []
for c in weak:
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); m = pickle.load(open(f"{C}/ballmed_weak/{c}.pkl", "rb"))
    cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    im = fr[b["F"][-1]].copy(); bx = k["box"][-1]; hL = bx[3] - bx[1]; cx, cy = (bx[0] + bx[2]) / 2, bx[3] - 0.2 * hL; x0, y0, x1, y1 = int(cx - 1.4 * hL), int(cy - 1.2 * hL), int(cx + 1.4 * hL), int(cy + 0.7 * hL)
    bl = b["ball_last"]; bm = m["ball_med"]
    if np.isfinite(bl[0]): cv2.circle(im, (int(bl[0]), int(bl[1])), max(int(0.07 * hL), 5), (0, 0, 255), 2)
    if np.isfinite(bm[0]): cv2.circle(im, (int(bm[0]), int(bm[1])), max(int(0.07 * hL), 5), (0, 255, 0), 2)
    crop = cv2.resize(im[max(y0, 0):y1, max(x0, 0):x1], (360, 260)); med = cv2.resize(m["med"], (260, 260)); t = np.hstack([crop, med])
    cv2.putText(t, f"{c[:4]} sup={b['support']} red=consensus green=median", (4, 14), 0, 0.42, (0, 255, 255), 1); tiles.append(t)
cv2.imwrite(f"{S}/ball_weak.jpg", np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - len(tiles) % 2, 2)]))
