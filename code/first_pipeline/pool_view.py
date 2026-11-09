import pickle, cv2, numpy as np, pandas as pd
R = r"C:\path\to\project-root"; D = R + r"\Q4data"; C = R + r"\Q4_work\cache"
S = r"scratch"
pf = pickle.load(open(f"{C}/pool_fail.pkl", "rb")); ck = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip"); tiles = []
for c, why in list(pf.items())[:16]:
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); sc = pickle.load(open(f"{C}/scene/{c}.pkl", "rb")); cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    im = fr[d["idx"][6]].copy(); w, h = d["wh"]; fm = d["frames"][6]; r = ck.loc[c]
    for b, cf in zip(fm["pb"], fm["pc"]):
        if cf >= 0.1: cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0) if cf >= 0.2 else (0, 140, 255), 1)
    for e in sc["extra_a"]: cv2.rectangle(im, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (255, 200, 0), 1)
    if len(sc["goal_a"]): g = sc["goal_a"][0]; cv2.rectangle(im, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (255, 0, 255), 2)
    who = "kicker" if "kicker" in why else "keeper"; px, py = int(r[f"{who}_x_norm"] * w), int(r[f"{who}_y_norm"] * h)
    cv2.circle(im, (px, py), 14, (0, 0, 255), 3); y0, y1, x0, x1 = max(py - 150, 0), min(py + 150, h), max(px - 200, 0), min(px + 200, w); crop = cv2.resize(im[y0:y1, x0:x1], (480, 360))
    cv2.putText(crop, f"{c[:4]} missing {why} (red=click)", (6, 20), 0, 0.55, (0, 255, 255), 2); tiles.append(crop)
cv2.imwrite(f"{S}/pool_fail.jpg", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]))
