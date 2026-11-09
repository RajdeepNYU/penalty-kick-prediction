import pickle, cv2, numpy as np, pandas as pd
R = r"C:\path\to\project-root"; D = R + r"\Q4data"; C = R + r"\Q4_work\cache"
S = r"scratch"
picks = pickle.load(open(f"{C}/picks4.pkl", "rb")); pf = pickle.load(open(f"{C}/pool_fail4.pkl", "rb")); ck = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip"); bad = []
for c, p in picks.items():
    if c not in ck.index or p is None: continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); w, h = d["wh"]; r = ck.loc[c]; res = []
    for who, bx in (("kicker", p["kick"]), ("keeper", p["keep"])):
        q = (r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h); res.append(bx[0] <= q[0] <= bx[2] and bx[1] <= q[1] <= bx[3])
    if not all(res): bad.append((c, res))
print("clicked clips with a wrong pick:", len(bad), "| of which pool failures:", sum(c in pf for c, _ in bad), "| kicker wrong:", sum(not r[0] for _, r in bad), "keeper wrong:", sum(not r[1] for _, r in bad)); tiles = []
for c, res in bad[:12]:
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    im = fr[d["idx"][6]].copy(); w, h = d["wh"]; r = ck.loc[c]; p = picks[c]
    cv2.rectangle(im, tuple(int(v) for v in p["kick"][:2]), tuple(int(v) for v in p["kick"][2:]), (0, 255, 0), 3); cv2.rectangle(im, tuple(int(v) for v in p["keep"][:2]), tuple(int(v) for v in p["keep"][2:]), (255, 128, 0), 3)
    for who in ("kicker", "keeper"): cv2.circle(im, (int(r[f"{who}_x_norm"] * w), int(r[f"{who}_y_norm"] * h)), 12, (0, 0, 255) if who == "kicker" else (0, 0, 160), 3)
    cv2.putText(im, f"{c[:4]} kick{'OK' if res[0] else 'X'} keep{'OK' if res[1] else 'X'} (green=kicker pick, blue=keeper pick, red=click)", (6, 24), 0, 0.8, (0, 255, 255), 2); tiles.append(cv2.resize(im, (640, 360)))
while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f"{S}/rank_fail.jpg", np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
