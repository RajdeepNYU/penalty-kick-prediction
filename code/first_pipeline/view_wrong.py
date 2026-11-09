import numpy as np, pandas as pd, cv2, pickle
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; cl = np.array(["left", "middle", "right"]); cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
tr = lambda n: (np.load(f"{C}/base_hgb{n}.npy") + np.load(f"{C}/base_et{n}.npy")) / 2; bag = lambda st: sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_st{st}.npy") for e, h in cfg) / 5
Pk = 0.5 * (tr("") + tr("_F") + tr("_G")) / 3 + 0.5 * (bag("EF") + bag("G")) / 2; pk = cl[Pk.argmax(1)]; conf = Pk.max(1)
idx = np.where((pk != yk) & (pk != "middle") & (yk != "middle") & (conf > 0.75))[0]; print(len(idx), "confident L/R swaps"); rng = np.random.RandomState(0); sel = rng.choice(idx, 6, replace=False); tiles = []
for i in sel:
    c = gt["clip"][i]; b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    f = fr[b["F"][-1]].copy(); t = cv2.resize(f, (640, 360)); cv2.putText(t, f"{c} true={yk[i]} pred={pk[i]} p={conf[i]:.2f} foot={gt['preferred_foot'][i]}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2); tiles.append(t); print(i, c, yk[i], pk[i], round(conf[i], 2), gt["preferred_foot"][i])
cv2.imwrite(ROOT + r"\Q4_work\wrong_montage.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6])]))
