import numpy as np, pickle, glob, pandas as pd, itertools
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean()
M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); trees = (M["hgb259"] + M["extratrees"]) / 2
v6 = {(e, h): np.load(f"{C}/oof_seq_{e}_{h}_k0_s1_conv_b1.npy") for e, h in [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]}
v7 = {k.split("oof_seq2_")[1][:-4]: np.load(k) for k in sorted(glob.glob(f"{C}/oof_seq2_20_96_b1_kc0_*.npy"))}
nn6 = sum(v6.values()) / len(v6); print(f"v6 NN bag (5 cfgs) {acc(nn6):.3f} | v6 final (0.5/0.5) {acc(0.5 * trees + 0.5 * nn6):.3f}")
for k, P in v7.items(): print(f"  variant {k}: alone {acc(P):.3f}")
nn_all = (sum(v6.values()) + sum(v7.values())) / (len(v6) + len(v7)); print(f"NN bag 5+{len(v7)} variants: alone {acc(nn_all):.3f} | final {acc(0.5 * trees + 0.5 * nn_all):.3f}")
for w in (0.4, 0.5, 0.6, 0.7): print(f"   trees {1 - w:.1f} / NN-bag {w:.1f}: {acc((1 - w) * trees + w * nn_all):.3f}")
print("hgb / ET / NN weights:", {(a, b): round(acc(a * M['hgb259'] + b * M['extratrees'] + (1 - a - b) * nn_all), 3) for a, b in [(0.25, 0.25), (0.15, 0.35), (0.35, 0.15), (0.1, 0.3), (0.2, 0.2)]})
