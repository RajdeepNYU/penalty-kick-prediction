import numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean(); lr = yk != "middle"
trees = (np.load(f"{C}/base_hgb.npy") + np.load(f"{C}/base_et.npy")) / 2; cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]; LR = lambda P: (cl[P.argmax(1)][lr] == yk[lr]).mean()
bags = {"A+B": sum(np.load(f"{C}/oof_seq_{e}_{h}_k0_s1_conv_b1.npy") for e, h in cfg) / 5}
for st in ("E", "F", "EF"): bags[f"A+B+{st}"] = sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_st{st}.npy") for e, h in cfg) / 5
print("trees alone:", round(acc(trees), 3))
for k, P in bags.items(): B = 0.5 * trees + 0.5 * P; print(f"NN bag {k:8s}: alone {acc(P):.3f} (L/R-only {LR(P):.3f}) | blend with trees {acc(B):.3f} (L/R-only {LR(B):.3f})")
allb = sum(bags[k] for k in ("A+B+E", "A+B+F", "A+B+EF")) / 3; print(f"average of the E, F, EF bags: alone {acc(allb):.3f} | blend with trees {acc(0.5 * trees + 0.5 * allb):.3f}")
