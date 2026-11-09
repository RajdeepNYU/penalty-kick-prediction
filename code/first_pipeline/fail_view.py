import pickle, cv2, numpy as np, pandas as pd, sys
R = r"C:\path\to\project-root"; D = R + r"\Q4data"; C = R + r"\Q4_work\cache"
S = r"scratch"
ck = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip"); tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); gt = pd.read_csv(f"{D}/train_ground_truth.csv")
who = sys.argv[1]; key = "kick_a" if who == "kicker" else "keep_a"; fails = []
for c in gt["clip"]:
    if c not in ck.index: continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); w, h = d["wh"]; r = ck.loc[c]; p = (r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h); bx = tracks[c][key]
    if not (bx[0] <= p[0] <= bx[2] and bx[1] <= p[1] <= bx[3]): fails.append(c)
print(who, "failures:", len(fails)); tiles = []
for c in fails[:20]:
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    im = fr[d["idx"][6]].copy(); w, h = d["wh"]; fm = d["frames"][6]; r = ck.loc[c]
    for b, cf in zip(fm["pb"], fm["pc"]):
        if cf >= 0.3: cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), 1)
    for key_, col in (("kick_a", (0, 255, 0)), ("keep_a", (255, 128, 0))):
        b = tracks[c][key_]; cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
    cv2.circle(im, (int(r[f"{who}_x_norm"] * w), int(r[f"{who}_y_norm"] * h)), 9, (0, 0, 255), 3)
    cv2.putText(im, f"{c[:4]} {who} click=red green=kicker pick orange=keeper pick", (6, 22), 0, 0.7, (0, 255, 255), 2); tiles.append(cv2.resize(im, (640, 360)))
while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f"{S}/fail_{who}.jpg", np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, min(len(tiles), 12), 2)]))
