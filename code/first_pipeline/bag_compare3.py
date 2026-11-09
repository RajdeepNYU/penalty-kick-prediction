import numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean(); lr = yk != "middle"
LR = lambda P: (cl[P.argmax(1)][lr] == yk[lr]).mean(); cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
tr = lambda n: (np.load(f"{C}/base_hgb{n}.npy") + np.load(f"{C}/base_et{n}.npy")) / 2
tA, tF, tG = tr(""), tr("_F"), tr("_G"); bags = {"A+B": sum(np.load(f"{C}/oof_seq_{e}_{h}_k0_s1_conv_b1.npy") for e, h in cfg) / 5}
for st in ("E", "F", "EF", "G", "EFG"): bags[f"A+B+{st}"] = sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_st{st}.npy") for e, h in cfg) / 5
print(f"trees: RTMW {acc(tA):.3f} | ViTPose-L {acc(tF):.3f} | ViTPose-H {acc(tG):.3f} | RTMW+L {acc((tA + tF) / 2):.3f} | RTMW+L+H {acc((tA + tF + tG) / 3):.3f}")
for k, P in bags.items(): print(f"NN bag {k:9s}: alone {acc(P):.3f} (L/R-only {LR(P):.3f})")
print("\nblends (0.5 trees + 0.5 NN bag):")
for tn, T in (("trees RTMW+L", (tA + tF) / 2), ("trees RTMW+L+H", (tA + tF + tG) / 3)):
    for bn in ("A+B+EF", "A+B+G", "A+B+EFG"): B = 0.5 * T + 0.5 * bags[bn]; print(f"  {tn:15s} + NN {bn:8s}: kick {acc(B):.3f} (L/R-only {LR(B):.3f})")
both = (bags["A+B+EF"] + bags["A+B+EFG"]) / 2; B = 0.5 * (tA + tF + tG) / 3 + 0.5 * both; print(f"  trees RTMW+L+H + NN (EF and EFG bags averaged): kick {acc(B):.3f} (L/R-only {LR(B):.3f})")
