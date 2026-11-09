import numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean(); lr = yk != "middle"
trees = (np.load(f"{C}/base_hgb.npy") + np.load(f"{C}/base_et.npy")) / 2; cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
old = sum(np.load(f"{C}/oof_seq_{e}_{h}_k0_s1_conv_b1.npy") for e, h in cfg) / 5; new = sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_stE.npy") for e, h in cfg) / 5
LR = lambda P: (cl[P.argmax(1)][lr] == yk[lr]).mean()
print(f"NN bag (A+B):   alone {acc(old):.3f} (L/R-only {LR(old):.3f}) | blend with trees {acc(0.5 * trees + 0.5 * old):.3f} (L/R-only {LR(0.5 * trees + 0.5 * old):.3f})")
print(f"NN bag (A+B+E): alone {acc(new):.3f} (L/R-only {LR(new):.3f}) | blend with trees {acc(0.5 * trees + 0.5 * new):.3f} (L/R-only {LR(0.5 * trees + 0.5 * new):.3f})")
both = (old + new) / 2; print(f"both bags averaged: alone {acc(both):.3f} | blend with trees {acc(0.5 * trees + 0.5 * both):.3f}")
for w in (0.4, 0.5, 0.6, 0.7): print(f"   A+B+E bag weight {w}: {acc((1 - w) * trees + w * new):.3f}", end=" |")
print()
