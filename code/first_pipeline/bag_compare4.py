import numpy as np, pandas as pd, itertools, sys
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean(); lr = yk != "middle"
LR = lambda P: (cl[P.argmax(1)][lr] == yk[lr]).mean(); cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
tr = lambda n: (np.load(f"{C}/base_hgb{n}.npy") + np.load(f"{C}/base_et{n}.npy")) / 2
bag = lambda st: sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_st{st}.npy") for e, h in cfg) / 5
T3 = (tr("") + tr("_F") + tr("_G")) / 3; Th = tr("_H"); print(f"trees: RTMW+L+H {acc(T3):.3f} | YOLO {acc(Th):.3f} | +YOLO(1/4 each) {acc((3 * T3 + Th) / 4):.3f}")
B = {s: bag(s) for s in ("EF", "G", "H", "FH", "GH", "EFGH")}
for k, P in B.items(): print(f"NN bag st{k:5s}: {acc(P):.3f} (L/R {LR(P):.3f})")
print(f"v10 blend: {acc(0.5 * T3 + 0.5 * (B['EF'] + B['G']) / 2):.3f} (L/R {LR(0.5 * T3 + 0.5 * (B['EF'] + B['G']) / 2):.3f})")
for nm, nns in (("EF+G+H", ("EF", "G", "H")), ("FH+GH", ("FH", "GH")), ("EF+GH", ("EF", "GH")), ("EFGH", ("EFGH",)), ("EF+G+FH+GH", ("EF", "G", "FH", "GH")), ("EF+G+FH+GH+EFGH", ("EF", "G", "FH", "GH", "EFGH"))):
    N = sum(B[s] for s in nns) / len(nns)
    for tn, T in (("T3", T3), ("T3+H", (3 * T3 + Th) / 4)): X = 0.5 * T + 0.5 * N; print(f"  trees {tn:5s} + NN {nm:18s}: {acc(X):.3f} (L/R {LR(X):.3f})")
