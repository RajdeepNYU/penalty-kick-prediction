import numpy as np, pickle, glob, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean()
M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); trees = (M["hgb259"] + M["extratrees"]) / 2
fs = {f.split("oof_seq_")[1][:-4]: np.load(f) for f in sorted(glob.glob(f"{C}/oof_seq_*_k0_s1_conv_b1.npy"))}
for k, P in fs.items(): print(f"{k:28s} alone {acc(P):.3f} | trees+nn {acc(trees * 2 / 3 + P / 3):.3f}")
nn = sum(fs.values()) / len(fs); print(f"NN bag of {len(fs)}: alone {acc(nn):.3f}")
for w in (0.25, 0.33, 0.4, 0.5, 0.6): print(f"trees*{1 - w:.2f} + nnbag*{w:.2f}: {acc(trees * (1 - w) + nn * w):.3f}")
print("+ hgb alone / ET alone weights:", f"hgb {acc((M['hgb259'] + nn) / 2):.3f} et {acc((M['extratrees'] + nn) / 2):.3f} hgb+et+nn {acc((M['hgb259'] + M['extratrees'] + nn) / 3):.3f}")
